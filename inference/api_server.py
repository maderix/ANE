#!/usr/bin/env python3
"""HTTP API server for ANE inference.

Bridges HTTP requests to the qwen_ane Unix socket server. Handles tokenization
so clients can send plain text prompts and receive decoded responses.

Prerequisites:
    1. Start the ANE server:  ./qwen_ane qwen05b.bin --server /tmp/qwen_ane.sock
    2. Start this API:        python3 api_server.py [--port 8000]

Usage:
    curl http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"prompt": "What is 2+2?", "max_tokens": 50}'

    curl http://localhost:8000/health
"""
import argparse
import json
import os
import socket
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

DEFAULT_SOCK = "/tmp/qwen_ane.sock"
MODEL_DIR = Path.home() / "models" / "Qwen2.5-0.5B-Instruct"

tokenizer = None

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    return tokenizer


def query_ane(token_ids: list[int], max_tokens: int, sock_path: str) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(120)
    s.connect(sock_path)
    req = json.dumps({"tokens": token_ids, "max_tokens": max_tokens}) + "\n"
    s.sendall(req.encode())

    data = b""
    while True:
        chunk = s.recv(131072)
        if not chunk:
            break
        data += chunk
        if b"\n" in data:
            break
    s.close()
    return json.loads(data.decode().strip())


class ANEHandler(BaseHTTPRequestHandler):
    sock_path = DEFAULT_SOCK

    def log_message(self, format, *args):
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {format % args}\n")

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            alive = os.path.exists(self.sock_path)
            self._send_json(200, {"status": "ok" if alive else "no_backend", "socket": self.sock_path})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/completions":
            self._send_json(404, {"error": "not found, use POST /v1/completions"})
            return

        length = int(self.headers.get("Content-Length", 0))
        if length == 0 or length > 65536:
            self._send_json(400, {"error": "invalid content length"})
            return

        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        prompt = body.get("prompt", "")
        max_tokens = min(body.get("max_tokens", 50), 512)
        system_prompt = body.get("system", "You are a helpful assistant. Be concise.")

        if not prompt:
            self._send_json(400, {"error": "missing 'prompt' field"})
            return

        tok = get_tokenizer()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tok.encode(text)

        t0 = time.time()
        try:
            result = query_ane(input_ids, max_tokens, self.sock_path)
        except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
            self._send_json(503, {"error": f"ANE backend unavailable: {e}"})
            return

        elapsed = time.time() - t0

        output_ids = result.get("output", [])
        decoded = tok.decode(output_ids, skip_special_tokens=True) if output_ids else ""

        self._send_json(200, {
            "text": decoded,
            "output_tokens": output_ids,
            "prompt_tokens": len(input_ids),
            "gen_tokens": len(output_ids),
            "prefill_tps": result.get("prefill_tps", 0),
            "decode_tps": result.get("decode_tps", 0),
            "elapsed_s": round(elapsed, 3),
        })


def main():
    parser = argparse.ArgumentParser(description="HTTP API for ANE inference")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--sock", type=str, default=DEFAULT_SOCK)
    args = parser.parse_args()

    ANEHandler.sock_path = args.sock

    print(f"Loading tokenizer from {MODEL_DIR}...")
    get_tokenizer()
    print("Tokenizer ready.")

    if not os.path.exists(args.sock):
        print(f"WARNING: Socket {args.sock} not found. Start the ANE server first:")
        print(f"  ./qwen_ane qwen05b.bin --server {args.sock}")

    server = HTTPServer((args.host, args.port), ANEHandler)
    print(f"API server listening on http://{args.host}:{args.port}")
    print(f"  POST /v1/completions  {{\"prompt\": \"...\", \"max_tokens\": 50}}")
    print(f"  GET  /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
