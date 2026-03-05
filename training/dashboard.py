"""TUI dashboard for ANE training (train_large). Uses blessed for terminal UI."""

import argparse, fcntl, json, math, os, re, select, signal, struct, subprocess, sys, time, threading
from collections import deque
from pathlib import Path

import numpy as np

try:
    from blessed import Terminal
except ImportError:
    print('pip install blessed')
    sys.exit(1)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

DIM, HIDDEN, HEADS, SEQ, VOCAB, NLAYERS = 768, 2048, 12, 256, 32000, 12
HD = DIM // HEADS
CKPT_PATH_STATIC = 'ane_stories110M_ckpt.bin'
CKPT_PATH_DYNAMIC = 'training_dynamic/ane_stories110M_dyn_ckpt.bin'
CKPT_PATH = CKPT_PATH_STATIC  # set in main() based on --dynamic
TOKENIZER_PATH = str(Path(__file__).resolve().parent.parent / 'assets' / 'models' / 'tokenizer.bin')


class State:
    def __init__(self):
        self.model_config = {}
        self.params = {}
        self.kernels = {}
        self.training = {}
        self.flops = {}
        self.step = 0
        self.total_steps = 0
        self.loss = 0.0
        self.best_loss = float('inf')
        self.loss_history = []
        self.ms_per_step = 0.0
        self.compile_pct = 0.0
        self.compiles = 0
        self.component_timing = {}
        self.power = {'ane': 0.0, 'cpu': 0.0, 'gpu': 0.0}
        self.power_history_ane = deque(maxlen=300)
        self.power_history_cpu = deque(maxlen=300)
        self.logs = deque(maxlen=2000)
        self.log_scroll = 0
        self.auto_scroll = True
        self.batch_num = 0
        self.efficiency = {}
        self.gen_text = ''
        self.gen_step = 0
        self.gen_status = 'idle'
        self.gen_lock = threading.Lock()
        self.cpu_pct_history = deque(maxlen=300)
        self.mem_mb_history = deque(maxlen=300)
        self.proc_mem_mb_history = deque(maxlen=300)
        self.train_pid = None
        self.step_timestamps = []  # (step, time.monotonic()) for running ms/step
        self.train_start = None    # wall clock when first step seen
        self.compile_ms = 0.0      # total compile time

S = State()


class Tokenizer:
    def __init__(self, path):
        self.vocab = []
        self.scores = []
        with open(path, 'rb') as f:
            max_len = struct.unpack('i', f.read(4))[0]
            for _ in range(VOCAB):
                score = struct.unpack('f', f.read(4))[0]
                slen = struct.unpack('i', f.read(4))[0]
                tok = f.read(slen).decode('utf-8', errors='replace')
                self.vocab.append(tok)
                self.scores.append(score)

    def decode(self, token_id):
        if 0 <= token_id < len(self.vocab):
            s = self.vocab[token_id]
            if s.startswith('<0x') and s.endswith('>'):
                try:
                    return chr(int(s[3:-1], 16))
                except:
                    return s
            return s
        return ''

_tokenizer = None
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = Tokenizer(TOKENIZER_PATH)
        except Exception as e:
            S.logs.append(f'[gen] tokenizer load failed: {e}')
            return None
    return _tokenizer


def load_weights_from_ckpt(path):
    try:
        with open(path, 'rb') as f:
            # CkptHdr: 96 bytes (verified with sizeof)
            hdr = f.read(96)
            if len(hdr) < 96:
                return None
            wq_sz = DIM * DIM
            wo_sz = DIM * DIM
            w1_sz = HIDDEN * DIM
            w2_sz = DIM * HIDDEN
            w3_sz = HIDDEN * DIM
            # Per-layer: weights + adam state (m,v for each)
            adam_per_layer = (wq_sz*2 + wq_sz*2 + wq_sz*2 + wo_sz*2 +
                              w1_sz*2 + w2_sz*2 + w3_sz*2 + DIM*2 + DIM*2)
            W = {}
            for L in range(NLAYERS):
                W[f'Wq{L}'] = np.frombuffer(f.read(wq_sz * 4), dtype=np.float32).reshape(DIM, DIM).copy()
                W[f'Wk{L}'] = np.frombuffer(f.read(wq_sz * 4), dtype=np.float32).reshape(DIM, DIM).copy()
                W[f'Wv{L}'] = np.frombuffer(f.read(wq_sz * 4), dtype=np.float32).reshape(DIM, DIM).copy()
                W[f'Wo{L}'] = np.frombuffer(f.read(wo_sz * 4), dtype=np.float32).reshape(DIM, DIM).copy()
                W[f'W1_{L}'] = np.frombuffer(f.read(w1_sz * 4), dtype=np.float32).reshape(HIDDEN, DIM).copy()
                W[f'W2_{L}'] = np.frombuffer(f.read(w2_sz * 4), dtype=np.float32).reshape(DIM, HIDDEN).copy()
                W[f'W3_{L}'] = np.frombuffer(f.read(w3_sz * 4), dtype=np.float32).reshape(HIDDEN, DIM).copy()
                W[f'rms1_{L}'] = np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy()
                W[f'rms2_{L}'] = np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy()
                # Skip adam state for this layer
                f.seek(adam_per_layer * 4, 1)
            W['rms_final'] = np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy()
            f.seek(DIM * 2 * 4, 1)  # skip rms_final adam
            W['embed'] = np.frombuffer(f.read(VOCAB * DIM * 4), dtype=np.float32).reshape(VOCAB, DIM).copy()
            return W
    except Exception as e:
        S.logs.append(f'[gen] ckpt load failed: {e}')
        return None


def rmsnorm(x, w):
    ss = np.mean(x * x) + 1e-5
    return x * (1.0 / math.sqrt(ss)) * w

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def generate_text(W, max_tokens=64, temperature=0.8):
    tokenizer = get_tokenizer()
    if tokenizer is None:
        return '[no tokenizer]'

    tokens = [1]
    text_parts = []

    # Precompute RoPE frequencies
    freqs = np.zeros((SEQ, HD // 2), dtype=np.float32)
    for pos in range(SEQ):
        for i in range(HD // 2):
            freq = 1.0 / (10000.0 ** (2.0 * i / HD))
            freqs[pos, i] = pos * freq

    for step in range(max_tokens):
        seq_len = len(tokens)
        if seq_len > SEQ:
            break

        x = W['embed'][tokens[-1]].copy()

        for L in range(NLAYERS):
            # RMSNorm + QKV
            xn = rmsnorm(x, W[f'rms1_{L}'])
            q = W[f'Wq{L}'] @ xn
            k = W[f'Wk{L}'] @ xn
            v = W[f'Wv{L}'] @ xn

            # RoPE
            pos = seq_len - 1
            for h in range(HEADS):
                for i in range(HD // 2):
                    freq = freqs[pos, i]
                    cos_v, sin_v = math.cos(freq), math.sin(freq)
                    qi, qi1 = q[h * HD + 2 * i], q[h * HD + 2 * i + 1]
                    q[h * HD + 2 * i] = qi * cos_v - qi1 * sin_v
                    q[h * HD + 2 * i + 1] = qi * sin_v + qi1 * cos_v
                    ki, ki1 = k[h * HD + 2 * i], k[h * HD + 2 * i + 1]
                    k[h * HD + 2 * i] = ki * cos_v - ki1 * sin_v
                    k[h * HD + 2 * i + 1] = ki * sin_v + ki1 * cos_v

            # Attention (single token)
            o = np.zeros(DIM, dtype=np.float32)
            for h in range(HEADS):
                qh = q[h * HD:(h + 1) * HD]
                kh = k[h * HD:(h + 1) * HD]
                vh = v[h * HD:(h + 1) * HD]
                score = np.dot(qh, kh) / math.sqrt(HD)
                o[h * HD:(h + 1) * HD] = vh

            # Residual + output projection
            x2 = x + W[f'Wo{L}'] @ o

            # FFN
            x2n = rmsnorm(x2, W[f'rms2_{L}'])
            h1 = W[f'W1_{L}'] @ x2n
            h3 = W[f'W3_{L}'] @ x2n
            # SiLU
            h1 = h1 * (1.0 / (1.0 + np.exp(-h1))) * h3
            ffn_out = W[f'W2_{L}'] @ h1

            x = x2 + ffn_out

        x = rmsnorm(x, W['rms_final'])

        # Logits
        logits = W['embed'] @ x

        if temperature < 0.01:
            next_tok = int(np.argmax(logits))
        else:
            logits = logits / temperature
            probs = softmax(logits)
            next_tok = int(np.random.choice(VOCAB, p=probs))

        if next_tok == 2:
            break
        tokens.append(next_tok)
        piece = tokenizer.decode(next_tok)
        text_parts.append(piece)

    return ''.join(text_parts)


def generation_thread():
    last_gen_step = -1
    while True:
        time.sleep(5)
        if S.step <= last_gen_step + 99:
            continue
        if not os.path.exists(CKPT_PATH):
            continue
        with S.gen_lock:
            S.gen_status = 'generating'
            S.gen_step = S.step
        try:
            W = load_weights_from_ckpt(CKPT_PATH)
            if W is None:
                with S.gen_lock:
                    S.gen_status = 'idle'
                continue
            text = generate_text(W, max_tokens=64, temperature=0.8)
            with S.gen_lock:
                S.gen_text = text
                S.gen_step = S.step
                S.gen_status = 'done'
            S.step  # just to reference
        except Exception as e:
            with S.gen_lock:
                S.gen_text = f'[error: {e}]'
                S.gen_status = 'done'
        last_gen_step = S.step


def sysmetrics_thread():
    while True:
        time.sleep(1)
        if not HAS_PSUTIL:
            continue
        now = time.monotonic()
        S.cpu_pct_history.append(psutil.cpu_percent(interval=None))
        mem = psutil.virtual_memory()
        S.mem_mb_history.append(mem.used / (1024 * 1024))
        pid = S.train_pid
        if pid:
            try:
                p = psutil.Process(pid)
                S.proc_mem_mb_history.append(p.memory_info().rss / (1024 * 1024))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass


RE_CONFIG = re.compile(r'dim=(\d+) hidden=(\d+) heads=(\d+) seq=(\d+) vocab=(\d+) layers=(\d+)')
RE_PARAMS = re.compile(r'Params: ([\d.]+)M \(transformer ([\d.]+)M \+ embed ([\d.]+)M\)')
RE_KERNELS = re.compile(r'Kernels: (\d+).*?(\d+) weight-bearing')
RE_KERNELS_DYN = re.compile(r'Kernels: (\d+) compiled, (\d+) weight-bearing')
RE_ACCUM = re.compile(r'Accum (\d+).*LR=([\d.e+-]+)')
RE_STEP = re.compile(r'step\s+(\d+)\s+loss=([\d.]+)(?:\s+lr=([\d.e+-]+))?(?:\s+([\d.]+)ms/step)?')
RE_BATCH = re.compile(r'\[batch (\d+): compile=([\d.]+)ms train=([\d.]+)ms \(([\d.]+)ms/step\) compiles=(\d+)\]')
RE_TIMING = re.compile(r'ane=([\d.]+) io=([\d.]+) cls=([\d.]+) elem=([\d.]+) rms=([\d.]+) cblas_wait=([\d.]+)')
RE_TIMING_DYN = re.compile(r'ane_fwd=([\d.]+) io_fwd=([\d.]+) rms=([\d.]+) ane_bwd=([\d.]+) io_bwd=([\d.]+) silu=([\d.]+) rms_bwd=([\d.]+) cls=([\d.]+) cblas_wait=([\d.]+) dw_copy=([\d.]+)')
RE_RESTART = re.compile(r'\[exec\(\) restart step (\d+)')
RE_RESUME = re.compile(r'\[RESUMED step (\d+), loss=([\d.]+)\]')
RE_FLOPS = re.compile(r'FLOPs/step: fwd=([\d.]+)M bwd_dx=([\d.]+)M bwd_dW=([\d.]+)M sdpa_bwd=([\d.]+)M total=([\d.]+)M')
RE_ANE_FLOPS = re.compile(r'ANE FLOPs/step: ([\d.]+)M')
RE_ANE_TFLOPS = re.compile(r'ANE TFLOPS:\s+([\d.]+)')
RE_ANE_UTIL = re.compile(r'ANE utilization:\s+([\d.]+)%')
RE_EFFICIENCY = re.compile(r'(Total steps|Wall time|Compile time|Compile|Train time|Avg compile|Avg train|ANE TFLOPS|Total TFLOPS|ANE utilization):?\s+(.+)')
RE_COMPILED = re.compile(r'Compiled (\d+) kernels in (\d+)ms')
RE_ANE_POWER = re.compile(r'ANE Power:\s+([\d.]+)\s*mW')
RE_CPU_POWER = re.compile(r'CPU Power:\s+([\d.]+)\s*mW')
RE_GPU_POWER = re.compile(r'GPU Power:\s+([\d.]+)\s*mW')

def parse_line(line):
    S.logs.append(line)
    # Parse JSON lines from static pipeline ({"type":"step",...} or {"type":"batch",...})
    stripped = line.strip()
    if stripped.startswith('{'):
        try:
            j = json.loads(stripped)
            jt = j.get('type')
            if jt == 'step':
                S.step, S.loss = j['step'], j['loss']
                S.loss_history.append((S.step, S.loss))
                S.best_loss = min(S.best_loss, S.loss)
                S.compiles = j.get('compiles', S.compiles)
                now = time.monotonic()
                if S.train_start is None:
                    S.train_start = now
                S.step_timestamps.append((S.step, now))
                if len(S.step_timestamps) >= 2:
                    dt = S.step_timestamps[-1][1] - S.step_timestamps[-2][1]
                    if dt > 0:
                        S.ms_per_step = dt * 1000
                # Extract component timing from JSON
                ct = {}
                for k in ('t_ane', 't_io', 't_cls', 't_elem', 't_rms', 't_cblas_wait'):
                    if k in j:
                        ct[k[2:]] = j[k]  # strip 't_' prefix
                if ct:
                    S.component_timing = ct
                return
            elif jt == 'batch':
                S.batch_num = j.get('batch', S.batch_num)
                compile_ms = j.get('compile_ms', 0)
                train_ms = j.get('train_ms', 0)
                S.ms_per_step = j.get('ms_per_step', S.ms_per_step)
                S.compile_ms += compile_ms
                S.compile_pct = 100 * S.compile_ms / (S.compile_ms + train_ms) if S.compile_ms + train_ms > 0 else 0
                return
            elif jt == 'perf':
                if 'ane_tflops' in j:
                    S.flops['ane_tflops'] = j['ane_tflops']
                if 'ane_util_pct' in j:
                    S.flops['ane_util'] = j['ane_util_pct']
                return
        except (json.JSONDecodeError, KeyError):
            pass
    m = RE_CONFIG.search(line)
    if m:
        S.model_config = dict(zip(['dim', 'hidden', 'heads', 'seq', 'vocab', 'layers'], map(int, m.groups())))
        return
    m = RE_PARAMS.search(line)
    if m:
        S.params = {'total': float(m[1]), 'transformer': float(m[2]), 'embed': float(m[3])}
        return
    m = RE_KERNELS_DYN.search(line) or RE_KERNELS.search(line)
    if m:
        S.kernels = {'total': int(m[1]), 'weight_bearing': int(m[2])}
        return
    m = RE_ACCUM.search(line)
    if m:
        S.training = {'accum': int(m[1]), 'lr': m[2]}
        return
    m = RE_FLOPS.search(line)
    if m:
        S.flops.update(fwd=float(m[1]), bwd_dx=float(m[2]), bwd_dw=float(m[3]),
                       sdpa_bwd=float(m[4]), total=float(m[5]))
        return
    m = RE_ANE_FLOPS.search(line)
    if m:
        S.flops['ane'] = float(m[1])
        return
    m = RE_STEP.search(line)
    if m:
        S.step, S.loss = int(m[1]), float(m[2])
        if m[3]:
            S.training['lr'] = m[3]
        if m[4]:
            S.ms_per_step = float(m[4])
        now = time.monotonic()
        if S.train_start is None:
            S.train_start = now
        S.step_timestamps.append((S.step, now))
        if not m[4] and len(S.step_timestamps) >= 2:
            dt = S.step_timestamps[-1][1] - S.step_timestamps[-2][1]
            if dt > 0:
                S.ms_per_step = dt * 1000
        S.loss_history.append((S.step, S.loss))
        S.best_loss = min(S.best_loss, S.loss)
        return
    m = RE_BATCH.search(line)
    if m:
        S.batch_num = int(m[1])
        compile_ms, train_ms = float(m[2]), float(m[3])
        S.ms_per_step = float(m[4])
        S.compiles = int(m[5])
        S.compile_pct = 100 * compile_ms / (compile_ms + train_ms) if compile_ms + train_ms > 0 else 0
        return
    m = RE_TIMING_DYN.search(line)
    if m:
        vals = list(map(float, m.groups()))
        S.component_timing = {
            'ane_fwd': vals[0], 'io_fwd': vals[1], 'rms': vals[2],
            'ane_bwd': vals[3], 'io_bwd': vals[4], 'silu': vals[5],
            'rms_bwd': vals[6], 'cls': vals[7], 'cblas_wait': vals[8], 'dw_copy': vals[9],
            '_dynamic': True
        }
        return
    m = RE_TIMING.search(line)
    if m:
        S.component_timing = dict(zip(['ane', 'io', 'cls', 'elem', 'rms', 'cblas_wait'], map(float, m.groups())))
        return
    m = RE_ANE_TFLOPS.search(line)
    if m:
        S.flops['ane_tflops'] = float(m[1])
        return
    m = RE_ANE_UTIL.search(line)
    if m:
        S.flops['ane_util'] = float(m[1])
        return
    m = RE_COMPILED.search(line)
    if m:
        S.compiles = int(m[1])
        S.compile_ms += float(m[2])
        return
    m = RE_EFFICIENCY.search(line)
    if m:
        S.efficiency[m[1].strip()] = m[2].strip()
        return


def parse_powermetrics_text(text):
    now = time.monotonic()
    m = RE_ANE_POWER.search(text)
    if m:
        S.power['ane'] = float(m[1]) / 1000.0
        S.power_history_ane.append((now, S.power['ane']))
    m = RE_CPU_POWER.search(text)
    if m:
        S.power['cpu'] = float(m[1]) / 1000.0
        S.power_history_cpu.append((now, S.power['cpu']))
    m = RE_GPU_POWER.search(text)
    if m:
        S.power['gpu'] = float(m[1]) / 1000.0


BRAILLE_BASE = 0x2800

BRAILLE_MAP = [
    [1, 8],
    [2, 16],
    [4, 32],
    [64, 128],
]

def braille_chart(values, width, height, label_fmt='{:.1f}', y_range=None):
    if not values or width < 8 or height < 2:
        return ['(no data)'] * max(1, height)
    chart_w = width - 6
    if chart_w < 2:
        return ['(no data)'] * max(1, height)
    points_x = chart_w * 2
    points_y = height * 4
    data = values[-points_x:] if len(values) > points_x else values
    lo, hi = min(data), max(data)
    if y_range:
        lo, hi = y_range
    if hi - lo < 0.001:
        lo, hi = lo - 0.5, hi + 0.5
    margin = (hi - lo) * 0.05
    lo -= margin
    hi += margin

    grid = [[0] * chart_w for _ in range(height)]

    def plot(px, py):
        px = max(0, min(points_x - 1, px))
        py = max(0, min(points_y - 1, py))
        grid[py // 4][px // 2] |= BRAILLE_MAP[py % 4][px % 2]

    def val_to_y(v):
        return int((1 - (v - lo) / (hi - lo)) * (points_y - 1))

    for i in range(len(data)):
        if i >= points_x:
            break
        y0 = val_to_y(data[i])
        plot(i, y0)
        if i > 0:
            y_prev = val_to_y(data[i - 1])
            y_lo, y_hi = min(y_prev, y0), max(y_prev, y0)
            for yy in range(y_lo, y_hi + 1):
                if y_hi != y_lo:
                    t = (yy - y_prev) / (y0 - y_prev)
                    xx = int(i - 1 + t)
                else:
                    xx = i
                plot(xx, yy)

    lines = []
    for r in range(height):
        if r == 0:
            label = label_fmt.format(hi)[:5].rjust(5)
        elif r == height - 1:
            label = label_fmt.format(lo)[:5].rjust(5)
        elif r == height // 2:
            label = label_fmt.format((hi + lo) / 2)[:5].rjust(5)
        else:
            label = '     '
        row_str = ''.join(chr(BRAILLE_BASE | grid[r][c]) for c in range(chart_w))
        lines.append(f'{label}\u2502{row_str}')

    lines.append('     \u2514' + '\u2500' * chart_w)
    return lines


def draw(term):
    w, h = term.width, term.height
    if w < 40 or h < 15:
        print(term.home + term.clear + 'Terminal too small', end='', flush=True)
        return

    buf = []

    def put(y, x, text, style=''):
        if 0 <= y < h and x < w:
            text = text[:w - x]
            if style:
                buf.append(term.move(y, x) + style + text + term.normal)
                return
            buf.append(term.move(y, x) + text)

    buf.append(term.home + term.clear)

    mid_x = w // 2
    right_w = w - mid_x - 1
    left_w = mid_x - 1

    row = 0

    # Model Config header
    hdr = '\u2500 Model Config '
    put(row, 0, '\u250c' + hdr + '\u2500' * max(0, w - len(hdr) - 2) + '\u2510', term.cyan)
    row += 1

    cfg = S.model_config
    if cfg:
        line1 = f"stories110M  dim={cfg.get('dim', '')} hidden={cfg.get('hidden', '')} heads={cfg.get('heads', '')} seq={cfg.get('seq', '')} layers={cfg.get('layers', '')}"
        put(row, 0, '\u2502', term.cyan)
        put(row, 2, line1)
        put(row, w - 1, '\u2502', term.cyan)
        row += 1
        p, k, t = S.params, S.kernels, S.training
        line2 = f"{p.get('total', '?')}M params ({p.get('transformer', '?')}M xfmr + {p.get('embed', '?')}M embed)"
        put(row, 0, '\u2502', term.cyan)
        put(row, 2, line2)
        put(row, w - 1, '\u2502', term.cyan)
        row += 1
        line3 = f"{k.get('total', '?')} kernels ({k.get('weight_bearing', '?')} wt-bearing) | Accum {t.get('accum', '?')} | Adam LR={t.get('lr', '?')}"
        put(row, 0, '\u2502', term.cyan)
        put(row, 2, line3)
        put(row, w - 1, '\u2502', term.cyan)
        row += 1
    else:
        put(row, 0, '\u2502', term.cyan)
        put(row, 2, 'Waiting for model config...')
        put(row, w - 1, '\u2502', term.cyan)
        row += 1

    remaining = h - row - 1
    # Allocate: loss curve ~40%, logs ~30%, power/cpu/mem/gen share rest
    power_h = max(3, remaining // 8)
    gen_h = max(2, remaining // 10)
    extra_panels = power_h + power_h + gen_h + 6  # power + cpu/mem + gen + dividers
    log_h_min = max(5, remaining // 5)
    curve_h = max(5, remaining - extra_panels - log_h_min)

    # Loss Curve + Training Stats divider
    put(row, 0, '\u251c\u2500 Loss Curve ' + '\u2500' * max(0, left_w - 13) + '\u252c\u2500 Training Stats ' + '\u2500' * max(0, right_w - 17) + '\u2524', term.cyan)
    row += 1

    # Loss curve
    loss_vals = [l for _, l in S.loss_history]
    curve_lines = braille_chart(loss_vals, left_w - 1, curve_h)
    for i, cl in enumerate(curve_lines):
        put(row + i, 0, '\u2502', term.cyan)
        put(row + i, 1, cl, term.green)
        put(row + i, mid_x, '\u2502', term.cyan)
        put(row + i, w - 1, '\u2502', term.cyan)

    # Training stats (right panel)
    sr = row
    step_str = f'{S.step}' + (f'/{S.total_steps}' if S.total_steps and S.total_steps < 999999 else '')
    # Elapsed time
    elapsed = 0.0
    if S.train_start:
        elapsed = time.monotonic() - S.train_start
    elapsed_str = f'{elapsed:.1f}s' if elapsed < 60 else f'{elapsed/60:.1f}m'
    put(sr, mid_x + 1, f' Step: {step_str}  Loss: {S.loss:.4f}  [{elapsed_str}]' if S.loss else ' Step: --', term.yellow)
    sr += 1
    # ms/step + steps/sec
    sps = 1000.0 / S.ms_per_step if S.ms_per_step > 0 else 0
    put(sr, mid_x + 1, f' Best: {S.best_loss:.4f}   {S.ms_per_step:.1f}ms/step ({sps:.1f} steps/s)' if S.best_loss < float('inf') else ' Best: --')
    sr += 1
    # TFLOPS
    ane_tflops = S.flops.get('ane_tflops', 0)
    ane_util = S.flops.get('ane_util', 0)
    total_tflops = 0
    if S.ms_per_step > 0 and S.flops.get('ane', 0) > 0:
        if not ane_tflops:
            ane_tflops = (S.flops['ane'] * 1e6) / (S.ms_per_step * 1e-3) / 1e12
        total_tflops = (S.flops.get('total', 0) * 1e6) / (S.ms_per_step * 1e-3) / 1e12
    if not ane_util and ane_tflops:
        ane_util = 100.0 * ane_tflops / 15.8
    compile_str = f'  Compile: {S.compile_ms/1000:.1f}s' if S.compile_ms > 0 else ''
    if ane_tflops:
        tflops_str = f' ANE: {ane_tflops:.2f}T'
        if total_tflops:
            tflops_str += f'  Total: {total_tflops:.2f}T'
        tflops_str += f'  Util: {ane_util:.1f}%{compile_str}'
        put(sr, mid_x + 1, tflops_str)
    elif compile_str:
        put(sr, mid_x + 1, f'{compile_str}')
    sr += 1
    ct = S.component_timing
    if ct:
        if ct.get('_dynamic'):
            put(sr, mid_x + 1, f' fwd={ct.get("ane_fwd",0):.1f} bwd={ct.get("ane_bwd",0):.1f} io={ct.get("io_fwd",0)+ct.get("io_bwd",0):.1f} silu={ct.get("silu",0):.1f}')
            sr += 1
            put(sr, mid_x + 1, f' cls={ct.get("cls",0):.1f} rms={ct.get("rms",0)+ct.get("rms_bwd",0):.1f} dw={ct.get("dw_copy",0):.1f} ms/step')
            sr += 1
        else:
            put(sr, mid_x + 1, f' ane={ct.get("ane", 0):.1f} io={ct.get("io", 0):.1f} cls={ct.get("cls", 0):.1f} elem={ct.get("elem", 0):.1f}')
            sr += 1
            put(sr, mid_x + 1, f' rms={ct.get("rms", 0):.1f} cblas_wait={ct.get("cblas_wait", 0):.1f} ms/step')
            sr += 1
    pw = S.power
    if any(pw.values()):
        put(sr, mid_x + 1, '\u2500 Power ' + '\u2500' * max(0, right_w - 9), term.cyan)
        sr += 1
        put(sr, mid_x + 1, f' ANE: {pw["ane"]:.1f}W  CPU: {pw["cpu"]:.1f}W  GPU: {pw["gpu"]:.1f}W', term.magenta)
        sr += 1
    if S.batch_num:
        put(sr, mid_x + 1, f' Batch: {S.batch_num}  Compiles: {S.compiles}')
        sr += 1

    # Fill vertical borders between loss curve and stats
    top_end = row + len(curve_lines)
    for r in range(row, max(top_end, sr)):
        if r >= top_end:
            put(r, 0, '\u2502', term.cyan)
        if r >= sr:
            put(r, mid_x, '\u2502', term.cyan)
        put(r, w - 1, '\u2502', term.cyan)
    row = max(top_end, sr)

    # Power charts
    has_power = len(S.power_history_ane) > 1 or len(S.power_history_cpu) > 1
    if has_power:
        put(row, 0, '\u251c\u2500 ANE Power (W) ' + '\u2500' * max(0, left_w - 16) + '\u252c\u2500 CPU Power (W) ' + '\u2500' * max(0, right_w - 17) + '\u2524', term.cyan)
        row += 1
        ane_vals = [v for _, v in S.power_history_ane]
        cpu_vals = [v for _, v in S.power_history_cpu]
        ane_lines = braille_chart(ane_vals, left_w - 1, power_h, label_fmt='{:.1f}')
        cpu_lines = braille_chart(cpu_vals, right_w - 1, power_h, label_fmt='{:.1f}')
        max_lines = max(len(ane_lines), len(cpu_lines))
        while len(ane_lines) < max_lines:
            ane_lines.append(' ' * (left_w - 1))
        while len(cpu_lines) < max_lines:
            cpu_lines.append(' ' * (right_w - 1))
        for i in range(max_lines):
            put(row + i, 0, '\u2502', term.cyan)
            put(row + i, 1, ane_lines[i], term.red)
            put(row + i, mid_x, '\u2502', term.cyan)
            put(row + i, mid_x + 1, cpu_lines[i], term.blue)
            put(row + i, w - 1, '\u2502', term.cyan)
        row += max_lines

    # CPU / Memory charts
    has_sysmetrics = len(S.cpu_pct_history) > 0
    if has_sysmetrics:
        put(row, 0, '\u251c\u2500 CPU % ' + '\u2500' * max(0, left_w - 8) + '\u252c\u2500 Memory (MB) ' + '\u2500' * max(0, right_w - 15) + '\u2524', term.cyan)
        row += 1
        cpu_vals = list(S.cpu_pct_history)
        mem_vals = list(S.proc_mem_mb_history) if S.proc_mem_mb_history else list(S.mem_mb_history)
        mem_label = 'proc' if S.proc_mem_mb_history else 'sys'
        cpu_lines = braille_chart(cpu_vals, left_w - 1, power_h, label_fmt='{:.0f}', y_range=(0, 100))
        mem_lines = braille_chart(mem_vals, right_w - 1, power_h, label_fmt='{:.0f}')
        max_lines = max(len(cpu_lines), len(mem_lines))
        while len(cpu_lines) < max_lines:
            cpu_lines.append(' ' * (left_w - 1))
        while len(mem_lines) < max_lines:
            mem_lines.append(' ' * (right_w - 1))
        for i in range(max_lines):
            put(row + i, 0, '\u2502', term.cyan)
            put(row + i, 1, cpu_lines[i], term.yellow)
            put(row + i, mid_x, '\u2502', term.cyan)
            put(row + i, mid_x + 1, mem_lines[i], term.magenta)
            put(row + i, w - 1, '\u2502', term.cyan)
        row += max_lines

    # Generated text
    with S.gen_lock:
        gen_text = S.gen_text
        gen_step = S.gen_step
        gen_status = S.gen_status
    if gen_text or gen_status == 'generating':
        status_tag = ' (generating...)' if gen_status == 'generating' else f' (step {gen_step})'
        put(row, 0, '\u251c\u2500 Generated Text' + status_tag + ' ' + '\u2500' * max(0, w - 20 - len(status_tag)) + '\u2524', term.cyan)
        row += 1
        if gen_text:
            line_w = w - 3
            text = gen_text.replace('\n', ' ')
            wrapped = [text[i:i + line_w] for i in range(0, len(text), line_w)]
            for i, tl in enumerate(wrapped[:gen_h]):
                put(row, 0, '\u2502', term.cyan)
                put(row, 2, tl, term.white)
                put(row, w - 1, '\u2502', term.cyan)
                row += 1
        else:
            put(row, 0, '\u2502', term.cyan)
            put(row, 2, '...')
            put(row, w - 1, '\u2502', term.cyan)
            row += 1

    # Logs
    log_h = h - row - 1
    scroll_hint = ' (scroll) ' if not S.auto_scroll else ' '
    put(row, 0, '\u251c\u2500 Logs' + scroll_hint + '\u2500' * max(0, w - 8 - len(scroll_hint)) + '\u2524', term.cyan)
    row += 1

    logs = list(S.logs)
    if log_h > 0 and logs:
        if S.auto_scroll:
            start = max(0, len(logs) - log_h)
        else:
            start = max(0, min(S.log_scroll, len(logs) - log_h))
        visible = logs[start:start + log_h]
        for i, line in enumerate(visible):
            put(row + i, 0, '\u2502', term.cyan)
            if RE_STEP.search(line):
                put(row + i, 1, line[:w - 2], term.yellow)
            elif line.strip().startswith('[batch'):
                put(row + i, 1, line[:w - 2], term.blue)
            elif 'FAIL' in line or 'error' in line.lower():
                put(row + i, 1, line[:w - 2], term.red)
            else:
                put(row + i, 1, line[:w - 2])
            put(row + i, w - 1, '\u2502', term.cyan)
        for i in range(len(visible), log_h):
            put(row + i, 0, '\u2502', term.cyan)
            put(row + i, w - 1, '\u2502', term.cyan)

    # Bottom border
    put(h - 1, 0, '\u2514' + '\u2500' * (w - 2) + '\u2518', term.cyan)

    sys.stdout.write(''.join(buf))
    sys.stdout.flush()


def set_nonblock(fd):
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

def spawn_training(resume=False, steps=10000, dynamic=False, ane=False, scratch=False,
                   lr=None, accum=None, no_ane_extras=False):
    if dynamic:
        cmd = 'cd training_dynamic && make 2>&1 && ./train'
    elif ane:
        cmd = 'make train_large_ane 2>&1 && ./train_large_ane'
    else:
        cmd = 'make train_large 2>&1 && ./train_large'
    if resume:
        cmd += ' --resume'
    if scratch and dynamic:
        cmd += ' --scratch'
    if lr is not None:
        cmd += f' --lr {lr}'
    if accum is not None and dynamic:
        cmd += f' --accum {accum}'
    if no_ane_extras and ane:
        cmd += ' --no-ane-extras'
    cmd += f' --steps {steps}'
    proc = subprocess.Popen(
        ['bash', '-c', cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)) or '.')
    set_nonblock(proc.stdout.fileno())
    return proc

def spawn_powermetrics():
    if not sys.stdin.isatty():
        return None
    try:
        proc = subprocess.Popen(
            ['sudo', 'powermetrics', '--samplers', 'cpu_power,gpu_power,ane_power', '-i', '1000'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        set_nonblock(proc.stdout.fileno())
        return proc
    except (FileNotFoundError, PermissionError):
        return None

def main():
    parser = argparse.ArgumentParser(description='ANE Training Dashboard (stories110M)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic weight pipeline (training_dynamic/)')
    parser.add_argument('--ane', action='store_true', help='PR#19: ANE-offloaded classifier/softmax/rmsnorm_bwd')
    parser.add_argument('--no-ane-extras', action='store_true', help='Disable ANE extras (use with --ane)')
    parser.add_argument('--scratch', action='store_true', help='Train from scratch (random init)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--accum', type=int, default=None, help='Gradient accumulation steps')
    parser.add_argument('--infinite', action='store_true', help='Train indefinitely')
    parser.add_argument('--no-powermetrics', action='store_true')
    parser.add_argument('--no-generate', action='store_true', help='Disable text generation')
    parser.add_argument('--steps', type=int, default=10000, help='Total steps (default: 10000)')
    args = parser.parse_args()

    if args.infinite:
        args.steps = 999999999
    S.total_steps = args.steps

    global CKPT_PATH
    CKPT_PATH = CKPT_PATH_DYNAMIC if args.dynamic else CKPT_PATH_STATIC

    term = Terminal()
    procs = []

    train_proc = spawn_training(resume=args.resume, steps=args.steps, dynamic=args.dynamic,
                                scratch=args.scratch, lr=args.lr, accum=args.accum,
                                ane=args.ane, no_ane_extras=args.no_ane_extras)
    S.train_pid = train_proc.pid
    procs.append(train_proc)

    if HAS_PSUTIL:
        psutil.cpu_percent(interval=None)  # prime the counter
        sys_t = threading.Thread(target=sysmetrics_thread, daemon=True)
        sys_t.start()

    pm_proc = None
    if not args.no_powermetrics:
        pm_proc = spawn_powermetrics()
        if pm_proc:
            procs.append(pm_proc)

    if not args.no_generate:
        gen_t = threading.Thread(target=generation_thread, daemon=True)
        gen_t.start()

    pm_buf = ''
    train_buf = ''

    def cleanup():
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass

    signal.signal(signal.SIGINT, lambda *a: cleanup())
    signal.signal(signal.SIGTERM, lambda *a: cleanup())

    resized = [False]
    def on_resize(*a):
        resized[0] = True

    signal.signal(signal.SIGWINCH, on_resize)

    with term.fullscreen(), term.cbreak(), term.hidden_cursor():
        draw(term)
        last_draw = time.monotonic()

        while True:
            fds = []
            fd_map = {}
            if train_proc and train_proc.stdout:
                fd = train_proc.stdout.fileno()
                fds.append(fd)
                fd_map[fd] = 'train'
            if pm_proc and pm_proc.stdout:
                fd = pm_proc.stdout.fileno()
                fds.append(fd)
                fd_map[fd] = 'pm'
            fds.append(sys.stdin.fileno())
            fd_map[sys.stdin.fileno()] = 'stdin'

            try:
                readable, _, _ = select.select(fds, [], [], 0.25)
            except (ValueError, OSError):
                continue

            need_draw = resized[0]
            resized[0] = False

            train_finished = False

            for fd in readable:
                kind = fd_map.get(fd)
                if kind == 'train':
                    try:
                        data = os.read(fd, 65536)
                    except BlockingIOError:
                        continue
                    except (OSError, ValueError):
                        data = b''
                    if not data:
                        if train_proc.poll() is not None:
                            try:
                                rest = train_proc.stdout.read()
                                if rest:
                                    for line in rest.decode('utf-8', errors='replace').split('\n'):
                                        if line:
                                            parse_line(line)
                            except Exception:
                                pass
                            S.logs.append('[dashboard] Training finished. Press q to exit.')
                            train_finished = True
                        continue
                    train_buf += data.decode('utf-8', errors='replace')
                    while '\n' in train_buf:
                        line, train_buf = train_buf.split('\n', 1)
                        parse_line(line)
                    need_draw = True

                elif kind == 'pm':
                    try:
                        data = os.read(fd, 65536).decode('utf-8', errors='replace')
                    except BlockingIOError:
                        continue
                    except (OSError, ValueError):
                        data = ''
                    if not data:
                        continue
                    pm_buf += data
                    while '\n\n' in pm_buf or '*** ' in pm_buf:
                        end = pm_buf.find('\n*** ', 1)
                        if end < 0:
                            end = pm_buf.find('\n\n', 1)
                            if end < 0:
                                break
                        chunk = pm_buf[:end]
                        pm_buf = pm_buf[end:]
                        parse_powermetrics_text(chunk)
                    if len(pm_buf) > 16384:
                        pm_buf = pm_buf[-8192:]
                    need_draw = True

                elif kind == 'stdin':
                    key = term.inkey(timeout=0)
                    if not key:
                        continue
                    if key == 'q':
                        cleanup()
                        return
                    elif key.name == 'KEY_UP':
                        S.auto_scroll = False
                        S.log_scroll = max(0, S.log_scroll - 1)
                        need_draw = True
                    elif key.name == 'KEY_DOWN':
                        S.log_scroll += 1
                        need_draw = True
                    elif key == 'p':
                        S.auto_scroll = not S.auto_scroll
                        if S.auto_scroll:
                            S.log_scroll = max(0, len(S.logs) - 10)
                        need_draw = True
                    elif key == 'r':
                        if train_proc:
                            train_proc.terminate()
                            train_proc.wait()
                        train_proc = spawn_training(resume=True, steps=args.steps, dynamic=args.dynamic,
                                                        lr=args.lr, accum=args.accum,
                                                        ane=args.ane, no_ane_extras=args.no_ane_extras)
                        S.train_pid = train_proc.pid
                        procs = [p for p in procs if p.poll() is None]
                        procs.append(train_proc)
                        S.logs.append('[dashboard] Restarted with --resume')
                        need_draw = True
                    elif key == 'g':
                        with S.gen_lock:
                            S.gen_status = 'generating'
                            S.gen_step = S.step
                        def force_gen():
                            try:
                                W = load_weights_from_ckpt(CKPT_PATH)
                                if W:
                                    text = generate_text(W, max_tokens=64, temperature=0.8)
                                    with S.gen_lock:
                                        S.gen_text = text
                                        S.gen_step = S.step
                                        S.gen_status = 'done'
                            except Exception as e:
                                with S.gen_lock:
                                    S.gen_text = f'[error: {e}]'
                                    S.gen_status = 'done'
                        threading.Thread(target=force_gen, daemon=True).start()
                        need_draw = True

            now = time.monotonic()
            if not need_draw and now - last_draw > 1.0:
                need_draw = True
            if need_draw and now - last_draw > 0.066:
                draw(term)
                last_draw = now

            if train_finished:
                draw(term)
                while True:
                    key = term.inkey(timeout=1)
                    if key == 'q':
                        cleanup()
                        return

if __name__ == '__main__':
    main()
