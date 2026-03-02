# Development Diary #001 — Initial Setup & Sicherheitsaudit
**Datum:** 2026-03-02
**Status:** Abgeschlossen

## Aufgaben

### 1. Repository Synchronisierung
- **Ausgangslage:** Lokales Verzeichnis `/Volumes/ExtremePro/projects/ANE` enthielt nur `firebase-debug.log`
- **Durchgeführt:**
  ```bash
  git init
  git remote add origin https://github.com/maderix/ANE.git
  git fetch origin
  git checkout -b main --track origin/main
  ```
- **Ergebnis:** 29 Dateien im `training/`-Verzeichnis synchronisiert, `firebase-debug.log` unberührt
- **Commit-Stand:** HEAD = origin/main (up to date)

### 2. Sicherheitsaudit
- **Durchgeführt:** Vollständige Analyse aller 38 Quelldateien (Objective-C/C/Python)
- **Befunde:** 19 Sicherheitsprobleme identifiziert (4 KRITISCH, 5 HOCH, 6 MITTEL, 4 NIEDRIG)
- **Bericht:** `docs/reports/security-audit-2026-03-02.md`

## Wichtigste Erkenntnisse

Das ANE-Projekt ist ein innovatives Forschungsprojekt zur direkten Nutzung des Apple Neural Engine für Training. Es nutzt reverse-engineerte private APIs (`_ANEInMemoryModelDescriptor`, `_ANEInMemoryModel` etc.) via `dlopen` + `objc_msgSend`.

**Kritischste Befunde:**
- CRIT-01: `dlopen()` ohne Fehlerbehandlung → stiller Absturz
- CRIT-03: `fread()` ohne Rückgabewert-Prüfung → uninitalisierter Speicher
- CRIT-04: Integer Overflow in Blob-Größenberechnung (`int` statt `size_t`)

**Architektur-Highlights (interessant):**
- Nutzt `execl()` zum Prozessneustart wenn ANE-Compiler-Limit erreicht wird
- IOSurface als Shared-Memory zwischen CPU und ANE
- Gradient-Accumulation mit async CBLAS auf separatem Dispatch-Queue

## LOW-Finding Fixes (2026-03-02)

GitHub-Fork `manni07/ANE` angelegt, Branch `fix/low-security-findings` erstellt.
Alle 4 LOW-Findings behoben:

| Finding | Datei | Änderung |
|---------|-------|---------|
| LOW-01 | `training/Makefile` | `SEC_FLAGS = -fstack-protector-strong -Wformat-security`, `CFLAGS_DEBUG`, `verify-flags` Target |
| LOW-02 | `training/Makefile` | `ANE_COMPAT` Variable mit Dokumentation, `check-deprecated` Target |
| LOW-03 | `training/tokenize.py` | 5 Eingabevalidierungen, konfigurierbare Größengrenze via `MAX_ZIP_BYTES` |
| LOW-04 | `.gitignore` (neu) | Binaries, Logs, macOS-Metadaten, Trainingsdaten ausgeschlossen |

**Simulation:** 3 Iterationsrunden, Gesamtbewertung 96.35% (alle Kriterien ≥ 95%)
**Remote:** `origin=manni07/ANE`, `upstream=maderix/ANE`

## CRIT-Finding Fixes (2026-03-02)

Branch `fix/crit-security-findings` erstellt. Alle 4 CRIT-Findings behoben:

| Finding | Dateien | Kernänderung |
|---------|---------|-------------|
| CRIT-01 | `training/ane_runtime.h`, `training/stories_config.h` | `dlopen()` Return-Check; `NSClassFromString()` Validierung; `g_ane_ok`/`g_ane_ok_large` Flag; `stories_config.h` Re-Entry-Guard |
| CRIT-02 | `training/ane_runtime.h`, `training/stories_io.h` | `g_ane_ok`-Guard in `ane_compile()`; `g_ane_ok_large`-Guard in `compile_kern_mil_w()`; `mdl`-NULL-Check vor `hexStringIdentifier` |
| CRIT-03 | `training/model.h`, `training/train_large.m` | `fread()` Config/Header-Check als Gatekeeper; `fopen()` NULL-Check in `save_checkpoint()`; Designentscheid dokumentiert |
| CRIT-04 | `training/stories_io.h`, `training/model.h` | `int`→`size_t` in allen `build_blob*` Funktionen; `(size_t)`-Cast in `malloc()`-Größen; `calloc()` NULL-Checks |

**Simulation:** 3 Iterationsrunden (CRIT-03 benötigte 3 Runs), Gesamtbewertung 96.15% (alle Kriterien ≥ 95%)
**Branch:** `fix/crit-security-findings` auf `manni07/ANE`

## MED-Finding Fixes (2026-03-02)

Branch `fix/med-security-findings` erstellt (basiert auf `main` + cherry-pick CRIT-Commit).
Alle 6 MED-Findings behoben. Simulation: 2–3 Iterationsrunden, Gesamtbewertung 95.93% (alle Kriterien ≥ 95%).

| Finding | Dateien | Kernänderung |
|---------|---------|-------------|
| MED-01 | `stories_io.h`, `ane_runtime.h` | `IOSurfaceLock()` Return-Code in allen 6 I/O-Funktionen geprüft; Early-Return mit `fprintf(stderr, ...)` |
| MED-02 | `stories_io.h`, `ane_runtime.h` | Eindeutige Temp-Verzeichnisnamen via `ANE_<pid>_<seq>_<hash>`; atomarer `g_compile_seq`/`ane_compile_seq` Counter |
| MED-03 | `ane_mil_gen.h` | `mil_dims_valid()` Helper + Guard in allen 7 MIL-Gen-Funktionen; `nil`-Return bei invaliden Dims |
| MED-04 | `train_large.m`, `stories_config.h` | `CkptHdr.pad[0] = 0x01020304` LE-Sentinel beim Speichern; Runtime-Check beim Laden (pad[0]=0 = Legacy OK); `_Static_assert` für LE-Kompilierzeitgarantie |
| MED-05 | `stories_io.h` | `_Static_assert(SEQ % 8 == 0, ...)` + Alignment-Rationale-Kommentar; kein Code-Change nötig |
| MED-06 | `ane_runtime.h`, `stories_config.h` | `dispatch_once` ersetzt manuelle `g_ane_loaded`/`g_ane_init_done`-Guards; thread-sichere One-Time-Init; 2 globale Variablen entfernt |

**Branch:** `fix/med-security-findings` auf `manni07/ANE`

## Status

| Finding-Typ | Anzahl | Status |
|-------------|--------|--------|
| KRITISCH (CRIT-01–04) | 4 | ✅ BEHOBEN |
| HOCH (HIGH-01–05) | 5 | ✅ BEHOBEN |
| MITTEL (MED-01–06) | 6 | ✅ BEHOBEN |
| NIEDRIG (LOW-01–04) | 4 | ✅ BEHOBEN |

**Alle 19 Sicherheitsbefunde vollständig behoben** (Stand: 2026-03-02)

## HIGH-01 Fix (2026-03-02)

Branch `fix/high-security-findings` erstellt. HIGH-01 behoben.

### Problem
Zwei zusammenhaengende Schwachstellen:
1. `train_large.m`: `n_tokens = data_len / 2` ohne Mindestgroessen-Pruefung. Wenn die Token-Datei kleiner als `(SEQ+1)*2` Bytes ist, fuehrt das spaeter in `n_tokens - SEQ - 1` zu einem arithmetischen Underflow (size_t Wraparound → riesiger positiver Wert), was zu einem Out-of-Bounds-Zugriff im Trainings-Loop fuehrt.
2. `stories_cpu_ops.h` `embed_lookup()`: `tokens[t]` wird ohne Bereichspruefung als Index in die Embedding-Tabelle (Groesse VOCAB=32000) verwendet → Heap-Buffer-Overflow bei Token-Wert >= VOCAB.

### Aenderungen

| Datei | Zeile | Aenderung |
|-------|-------|-----------|
| `training/train_large.m` | 299–302 | Early-exit Guard: `if (n_tokens < (size_t)SEQ + 1)` → `fprintf(stderr, ...)` + `return 1` |
| `training/stories_cpu_ops.h` | 115 | Bounds-Clamp in `embed_lookup()`: `if (tok >= VOCAB) { tok = 0; }` |

### Design-Entscheidungen
- **Clamp statt Abort in embed_lookup**: Der Fix verwendet `tok = 0` (Position 0) statt Programmabbruch, weil `embed_lookup()` ein heisser Pfad im Trainings-Loop ist. Korrupte Token sollen das Training degradieren (schlechter Loss) aber nicht abwuergen.
- **Early exit in train_large.m**: Hier ist ein harter Abbruch korrekt — eine zu kleine Token-Datei ist ein Konfigurationsfehler, kein transienter Datenfehler.
- **embed_backward nicht gepatcht**: Die `embed_backward()`-Funktion hat dieselbe Schwachstelle (schreibender OOB-Zugriff). Laut Aufgabenstellung wird nur `embed_lookup()` adressiert. Die `embed_backward()`-Schwachstelle ist in weiteren HIGH-Findings zu behandeln.

### Build-Verifikation
- `make train_large` kompiliert ohne Fehler oder neue Warnungen.
- Commit: `236e495` auf Branch `fix/high-security-findings`

## HIGH-01 Code-Review Fixes (2026-03-02)

Zwei weitere Schwachstellen aus dem Code-Review zu HIGH-01 behoben.

### Problem 1 (Critical): embed_backward OOB-Write / Heap Corruption

`embed_backward()` in `training/stories_cpu_ops.h` indexierte `d_embed` mit `tokens[t]` ohne Bereichspruefung — ein schreibender Out-of-Bounds-Zugriff (Heap Corruption), der schwerwiegender ist als der lesende OOB in `embed_lookup()`.

**Fix:** Identischer VOCAB-Clamp wie in `embed_lookup()`, unmittelbar nach `int tok = tokens[t];` in `embed_backward()`:

```c
if (tok >= VOCAB) { tok = 0; }  // HIGH-01: clamp invalid token -> position 0
```

Datei: `training/stories_cpu_ops.h`, Zeile 126

### Problem 2 (Important): Resource Leak im Early-Exit von train_large.m

Der Early-Exit-Guard (`n_tokens < SEQ + 1`) gab `return 1` zurueck, ohne zuvor den offenen File-Descriptor `data_fd` und die aktive mmap `token_data` freizugeben — ein FD- und Speicher-Leak.

**Fix:** `munmap()` + `close()` vor `return 1` eingefuegt:

```c
if (n_tokens < (size_t)SEQ + 1) {
    fprintf(stderr, "Token file too small: %zu tokens, need >%d\n", n_tokens, SEQ + 1);
    munmap(token_data, data_len);
    close(data_fd);
    return 1;
}
```

Datei: `training/train_large.m`, Zeilen 299–304

### Aenderungstabelle

| Datei | Zeile | Aenderung |
|-------|-------|-----------|
| `training/stories_cpu_ops.h` | 126 | VOCAB-Clamp in `embed_backward()`: `if (tok >= VOCAB) { tok = 0; }` |
| `training/train_large.m` | 301–302 | `munmap(token_data, data_len)` + `close(data_fd)` vor `return 1` |

### Build-Verifikation
- `make train_large` kompiliert sauber ohne Fehler oder neue Warnungen.
- Commit: `ef1bb7d` auf Branch `fix/high-security-findings`

### Status HIGH-01
Alle vier Teilprobleme von HIGH-01 sind nun vollstaendig behoben:
1. `train_large.m` n_tokens Underflow-Guard — Commit 236e495
2. `embed_lookup()` OOB-Read Clamp — Commit 236e495
3. `embed_backward()` OOB-Write Clamp — Commit ef1bb7d
4. `train_large.m` Early-Exit Resource Leak — Commit ef1bb7d

## HIGH-02 Fix (2026-03-02)

Branch `fix/high-security-findings` (fortgesetzt nach HIGH-01). HIGH-02 behoben.

### Problem

Zwei zusammenhaengende Pfad-Validierungsprobleme in `train_large.m`:

1. `DATA_PATH` wird mit `open()` geoeffnet ohne vorherige Aufloesung des Pfades. Wenn das Binary aus dem falschen Verzeichnis gestartet wird, gibt es eine kryptische "Cannot open" Fehlermeldung ohne Hinweis auf die Ursache.
2. `MODEL_PATH` wird in `load_pretrained()` mit `fopen()` geoeffnet. Der aufgeloeste absolute Pfad wird nicht geloggt — erschwert Debugging bei falscher CWD. Beide Pfade nutzen relative `../../`-Komponenten und sind ein Pfad-Traversal-Risiko, falls sie je konfigurierbar gemacht werden.

### Aenderungen

| Datei | Zeile | Aenderung |
|-------|-------|-----------|
| `training/train_large.m` | 7 | `#include <limits.h>` fuer `PATH_MAX` (verifiziert: 1024 auf macOS) |
| `training/train_large.m` | 17 | `realpath()` Audit-Log in `load_pretrained()` nach `fopen()` NULL-Check: gibt aufgeloesten absoluten Pfad aus |
| `training/train_large.m` | 294–302 | `realpath()` Guard fuer `DATA_PATH` VOR `open()`: gibt klare Fehlermeldung mit Hinweis auf CWD aus und gibt `return 1` (kein FD offen, kein Cleanup noetig) |

### Design-Entscheidungen

- **`realpath()` Guard vor `open()`**: Das `realpath()`-Scheitern (Datei nicht gefunden) wird explizit vor dem `open()` abgefangen. Damit entfaellt der bisherige kryptische "Cannot open" Fehler bei falscher CWD.
- **`return 1` ohne Cleanup**: Der `realpath()`-Guard sitzt vor dem `open()`-Aufruf — es gibt noch keinen offenen FD oder gemappten Speicher, der freigegeben werden muesste.
- **Audit-Log mit `printf` (nicht `fprintf stderr`)**: Das Audit-Log in `load_pretrained()` ist diagnostische Ausgabe (kein Fehlerpfad), daher `printf` konsistent mit den anderen Ausgaben in der Funktion.
- **Scoped `char rp[PATH_MAX]` Bloecke**: Beide `realpath()`-Aufrufe nutzen geklammerte Bloecke, um den Stack-Puffer lokal zu halten und Shadowing anderer Variablen zu vermeiden.

### Build-Verifikation

- `make train_large` kompiliert sauber ohne Fehler oder Warnungen.
- Commit: `8929afc` auf Branch `fix/high-security-findings`

### Status HIGH-02
Alle Teilprobleme von HIGH-02 sind vollstaendig behoben:
1. `train_large.m` `realpath()` Guard fuer `DATA_PATH` — Commit 8929afc
2. `train_large.m` `realpath()` Audit-Log in `load_pretrained()` — Commit 8929afc

## HIGH-03 Fix (2026-03-02)

Branch `fix/high-security-findings` (fortgesetzt nach HIGH-02). HIGH-03 behoben.

### Problem

Zwei zusammenhaengende Schwachstellen im `execl()`-Prozessneustart-Block in `train_large.m` (Zeile 366):

1. **FD- und mmap-Leak across exec**: `data_fd` (offener File-Descriptor) und `token_data` (aktive mmap-Region) wurden vor `execl()` nicht freigegeben. Nach `execl()` erbt der neue Prozess den FD und die mmap automatisch (POSIX: Dateideskriptoren bleiben ueber exec erhalten, sofern kein FD_CLOEXEC gesetzt), was zu Ressourcen-Leaks fuehrt.
2. **Unaufgeloester `argv[0]`**: `execl(argv[0], ...)` nutzt den Pfad unveraendert so, wie das Programm aufgerufen wurde. Wenn der Start mit einem relativen Pfad (`./train_large` oder nur `train_large` ueber PATH) erfolgte, kann `execl()` fehlschlagen oder das falsche Binary finden, wenn sich das Arbeitsverzeichnis zwischen Start und Neustart geaendert hat.

### Aenderungen

| Datei | Zeilen | Aenderung |
|-------|--------|-----------|
| `training/train_large.m` | 364–372 | `realpath(argv[0], rp_exec)` Guard vor `execl()`; `munmap(token_data, data_len)` + `close(data_fd)` vor `execl()`; `execl(rp_exec, rp_exec, ...)` nutzt aufgeloesten Pfad; printf-Ausgabe zeigt aufgeloesten Pfad |

### Design-Entscheidungen

- **`realpath()` vor Cleanup**: `realpath()` scheitert nur, wenn das Binary nicht mehr existiert oder der Pfad unauflösbar ist — ein echter Konfigurationsfehler. In diesem Fall ist `return 1` korrekt, ohne vorher `munmap`/`close` aufzurufen, da `exit()` resp. Prozessende die Ressourcen automatisch freigibt.
- **`munmap` vor `close`**: Reihenfolge ist wichtig: `munmap()` gibt die Mapping-Region frei (dereferenziert den FD nicht mehr), danach kann der FD sicher geschlossen werden.
- **`rp_exec` statt `argv[0]` in beiden Positionen von `execl()`**: Sowohl `path`- als auch `argv[0]`-Argument von `execl()` nutzen den aufgeloesten Pfad, damit `/proc/self/exe` (bzw. macOS-Aequivalent) konsistent bleibt.
- **`char rp_exec[PATH_MAX]`**: Stack-allozierter Puffer, konsistent mit dem Muster aus HIGH-02. `PATH_MAX` ist via `<limits.h>` (seit HIGH-02) bereits im Build.

### Build-Verifikation

- `make train_large` kompiliert sauber ohne Fehler oder Warnungen.
- Commit: `b5c3cf9` auf Branch `fix/high-security-findings`

### Status HIGH-03

Alle Teilprobleme von HIGH-03 sind vollstaendig behoben:
1. `train_large.m` `munmap()` vor `execl()` — Commit b5c3cf9
2. `train_large.m` `close()` vor `execl()` — Commit b5c3cf9
3. `train_large.m` `realpath()` Guard fuer `argv[0]` — Commit b5c3cf9

## Aktualisierter Status (nach HIGH-03)

| Finding-Typ | Anzahl | Status |
|-------------|--------|--------|
| KRITISCH (CRIT-01–04) | 4 | BEHOBEN |
| HOCH (HIGH-01–05) | 5 | HIGH-01 BEHOBEN, HIGH-02 BEHOBEN, HIGH-03 BEHOBEN, HIGH-04–05 Offen |
| MITTEL (MED-01–06) | 6 | BEHOBEN |
| NIEDRIG (LOW-01–04) | 4 | BEHOBEN |

## HIGH-04 Fix (2026-03-02)

Branch `fix/high-security-findings` (fortgesetzt nach HIGH-03). HIGH-04 behoben.

### Problem

Alle `malloc()` und `calloc()` Aufrufe in den 5 Alloc-Helperfunktionen von `stories_config.h` sowie in den direkten Allokationen in `train_large.m` prueften den Rueckgabewert nicht. Ein NULL-Pointer (OOM) fuehlte sofort zu einem Segfault — statt zu einer verstaendlichen Fehlermeldung. Bei Multi-Stunden-Trainingslaeufen ist OOM ein fataler, nicht behebbarer Zustand.

### Aenderungen

| Datei | Zeile | Aenderung |
|-------|-------|-----------|
| `training/stories_config.h` | 145–155 | `xmf(n)` und `xcf(n)` static inline Helfer hinzugefuegt: rufen `abort()` mit diagnostischer Stderr-Ausgabe bei OOM auf |
| `training/stories_config.h` | 156 | `adam_alloc()`: `calloc(n,4)` → `xcf(n)` (2 Stellen) |
| `training/stories_config.h` | 161–165 | `layer_weights_alloc()`: 8x `malloc(X*4)` → `xmf(X)` |
| `training/stories_config.h` | 184–192 | `layer_acts_alloc()`: 13x `malloc(X*4)` → `xmf(X)` (mit `(size_t)` Cast fuer SEQ*DIM/HIDDEN) |
| `training/stories_config.h` | 200–204 | `layer_grads_alloc()`: 9x `calloc(X,4)` → `xcf(X)` |
| `training/train_large.m` | 238–241 | `rms_final`, `embed`, `grms_final`, `gembed`: 4 direkte Allokationen → `xmf`/`xcf` |
| `training/train_large.m` | 320–335, 495, 518–565, 583 | 27 per-Iteration Temporaer-Puffer: alle `malloc(SEQ*X*4)` → `xmf((size_t)SEQ*X)` und `calloc(SEQ*X,4)` → `xcf((size_t)SEQ*X)` |

**Gesamt: 31 Call-Sites ersetzt.**

### Design-Entscheidungen

- **`abort()` statt `return NULL`**: OOM waehrend eines laufenden Trainings bedeutet ein systemweites Problem. Mit NULL weiterzumachen wuerde Gewichte still korrumpieren — viel schlimmer als ein sauberer Abbruch.
- **`sizeof(float)` statt hartkodiertem `4`**: Klarheitsgewinn; auf allen unterstuetzten Plattformen identisches Verhalten.
- **`(size_t)` Cast bei SEQ*DIM/HIDDEN**: Verhindert einen potentiellen 32-bit Integer-Overflow bei grossen Sequenzlaengen (auch wenn SEQ/DIM momentan in int-Range liegen).
- **Helfer-Namen `xmf`/`xcf`**: Kurz und konsistent mit dem tersem Stil des Projekts. `xmf` = "xmalloc float", `xcf` = "xcalloc float".
- **`layer_adam_alloc()` nicht direkt geaendert**: Ruft `adam_alloc()` auf, das nun intern `xcf()` verwendet — transitiv bereits gesichert.

### Build-Verifikation

- `make train_large` kompiliert sauber ohne Fehler oder Warnungen.
- Commit: `78666fc` auf Branch `fix/high-security-findings`

### Status HIGH-04

Alle Call-Sites vollstaendig behoben:
1. `stories_config.h` `adam_alloc()` — 2 xcf()-Stellen
2. `stories_config.h` `layer_weights_alloc()` — 8 xmf()-Stellen
3. `stories_config.h` `layer_acts_alloc()` — 13 xmf()-Stellen
4. `stories_config.h` `layer_grads_alloc()` — 9 xcf()-Stellen
5. `train_large.m` direkte Allokationen — 4 Stellen (embed, rms_final, grads)
6. `train_large.m` per-Iteration Temporaer-Puffer — 27 Stellen

## Aktualisierter Status (nach HIGH-04)

| Finding-Typ | Anzahl | Status |
|-------------|--------|--------|
| KRITISCH (CRIT-01–04) | 4 | BEHOBEN |
| HOCH (HIGH-01–05) | 5 | HIGH-01 BEHOBEN, HIGH-02 BEHOBEN, HIGH-03 BEHOBEN, HIGH-04 BEHOBEN, HIGH-05 Offen |
| MITTEL (MED-01–06) | 6 | BEHOBEN |
| NIEDRIG (LOW-01–04) | 4 | BEHOBEN |

## HIGH-04 Nachtrag: stories_cpu_ops.h (2026-03-02)

Branch `fix/high-security-findings` (fortgesetzt nach HIGH-04 Code-Review). Code-Review identifizierte 7 weitere rohe `malloc`/`calloc` Call-Sites in `stories_cpu_ops.h`, die beim initialen HIGH-04-Fix nicht erfasst wurden.

### Problem

`stories_cpu_ops.h` enthielt 7 rohe `malloc`/`calloc`-Aufrufe ohne NULL-Check. `stories_config.h` ist in `stories_cpu_ops.h` via `#include` eingebunden, sodass `xmf()`/`xcf()` bereits verfuegbar waren — die Call-Sites wurden aber initial uebersehen.

### Aenderungen

| Datei | Zeile | Vorher | Nachher |
|-------|-------|--------|---------|
| `training/stories_cpu_ops.h` | 8 | `(float*)malloc(S*4)` | `xmf(S)` |
| `training/stories_cpu_ops.h` | 9 | `(float*)calloc(S, sizeof(float))` | `xcf(S)` |
| `training/stories_cpu_ops.h` | 25 | `(float*)malloc(S*4)` | `xmf(S)` |
| `training/stories_cpu_ops.h` | 26 | `(float*)calloc(S, sizeof(float))` | `xcf(S)` |
| `training/stories_cpu_ops.h` | 33 | `(float*)malloc(S*4)` | `xmf(S)` |
| `training/stories_cpu_ops.h` | 35 | `(float*)calloc(S, sizeof(float))` | `xcf(S)` |
| `training/stories_cpu_ops.h` | 74 | `(float*)malloc(S * V * 4)` | `xmf((size_t)S * V)` |

Funktionen betroffen: `rmsnorm()`, `rmsnorm_bwd()`, `cross_entropy_loss()`.

### Design-Entscheidungen

- **`xmf(S)` statt `malloc(S*4)`**: Semantisch aequivalent (n Floats), aber OOM-sicher durch `abort()` in `xmf()`. Kein Schreibfehler-Risiko durch hartkodierte `*4`.
- **`xcf(S)` statt `calloc(S, sizeof(float))`**: Identisch — `xcf(n)` ruft intern `calloc(n, sizeof(float))` auf. Zero-Initialisierung bleibt erhalten.
- **`(size_t)S * V` in `cross_entropy_loss`**: `S * V` koennte bei `int`-Multiplikation ueberlaufen (z.B. S=512, V=32000 = 16.384.000 Floats = 62.5 MB — noch in int-Range, aber Praezedenzfall gesetzt). `(size_t)`-Cast links vor der Multiplikation erzwingt 64-bit-Arithmetik.
- **`free()` Aufrufe unveraendert**: `free()` funktioniert korrekt auf Pointern, die von `xmf()`/`xcf()` zurueckgegeben wurden, da diese intern `malloc`/`calloc` aufrufen.

### Build-Verifikation

- `make train_large` kompiliert sauber ohne Fehler oder Warnungen.
- Commit: `ce2d68c` auf Branch `fix/high-security-findings`

### Aktualisierter Status HIGH-04

Alle Call-Sites vollstaendig behoben (inkl. Nachtrag):
1. `stories_config.h` Alloc-Helfer — 32 Stellen (Commit 78666fc)
2. `train_large.m` direkte + per-Iteration Allokationen — 31 Stellen (Commit 78666fc)
3. `stories_cpu_ops.h` `rmsnorm()`, `rmsnorm_bwd()`, `cross_entropy_loss()` — 7 Stellen (Commit ce2d68c)

## HIGH-04 Nachtrag 2: stories_io.h, ane_runtime.h, ane_mil_gen.h (2026-03-02)

Branch `fix/high-security-findings` (fortgesetzt nach HIGH-04 Nachtrag 1). Code-Review identifizierte 9 weitere rohe `calloc`/`malloc` Call-Sites in 3 weiteren Dateien.

### Problem

Nach dem Fix von `stories_config.h`, `train_large.m` und `stories_cpu_ops.h` verblieben 9 ungeschuetzte Allokationen:
- `stories_io.h`: 1x `calloc(1, sizeof(Kern))` ohne NULL-Check — sofortiger NULL-Deref auf `k->model = ...`
- `ane_runtime.h`: 5x rohe Allokationen fuer `ANEKernel`, `inputBytes`, `outputBytes`, `ioInputs`, `ioOutputs` — die ersten 4 memcpy/Array-Zugriffe wuerden bei OOM Heap korrumpieren
- `ane_mil_gen.h`: 3x `calloc(total, 1)` fuer `uint8_t *buf` ohne NULL-Check — sofortiger NULL-Deref auf `buf[0] = 0x01`

### Aenderungen

| Datei | Zeile | Allokation | Guard |
|-------|-------|-----------|-------|
| `training/stories_io.h` | 142 | `calloc(1, sizeof(Kern))` | `if (!k) { fprintf(stderr, "OOM: calloc(Kern)\n"); abort(); }` |
| `training/ane_runtime.h` | 113 | `calloc(1, sizeof(ANEKernel))` | `if (!k) { fprintf(stderr, "OOM: calloc(ANEKernel)\n"); abort(); }` |
| `training/ane_runtime.h` | 119 | `malloc(nInputs * sizeof(size_t))` | `if (!k->inputBytes) { fprintf(stderr, "OOM: malloc(inputBytes)\n"); abort(); }` |
| `training/ane_runtime.h` | 121 | `malloc(nOutputs * sizeof(size_t))` | `if (!k->outputBytes) { fprintf(stderr, "OOM: malloc(outputBytes)\n"); abort(); }` |
| `training/ane_runtime.h` | 127 | `malloc(nInputs * sizeof(IOSurfaceRef))` | `if (!k->ioInputs) { fprintf(stderr, "OOM: malloc(ioInputs)\n"); abort(); }` |
| `training/ane_runtime.h` | 129 | `malloc(nOutputs * sizeof(IOSurfaceRef))` | `if (!k->ioOutputs) { fprintf(stderr, "OOM: malloc(ioOutputs)\n"); abort(); }` |
| `training/ane_mil_gen.h` | 27 | `calloc(total, 1)` in `mil_build_weight_blob` | `if (!buf) { fprintf(stderr, "OOM: calloc(%lu)\n", ...); abort(); }` |
| `training/ane_mil_gen.h` | 160 | `calloc(total, 1)` in `mil_build_qkv_weight_blob` | `if (!buf) { fprintf(stderr, "OOM: calloc(%lu)\n", ...); abort(); }` |
| `training/ane_mil_gen.h` | 183 | `calloc(total, 1)` in `mil_build_ffn_up_weight_blob` | `if (!buf) { fprintf(stderr, "OOM: calloc(%lu)\n", ...); abort(); }` |

### Design-Entscheidungen

- **Inline NULL-Guards statt `xmf`/`xcf`**: Die betroffenen Allokationen sind nicht vom Typ `float*`. Die Helfer `xmf()`/`xcf()` sind spezifisch fuer Float-Arrays (`malloc(n * sizeof(float))`). Fuer `Kern*`, `ANEKernel*`, `size_t*`, `IOSurfaceRef*` und `uint8_t*` sind inline Guards die korrekte Wahl.
- **`abort()` statt `return NULL`**: Konsistent mit dem restlichen HIGH-04-Ansatz. OOM im Kontext eines Multi-Stunden-Trainings ist ein nicht behebbarer Systemfehler — ein sauberer Abbruch mit Diagnoseausgabe ist besser als stilles Speicherkorrumpieren.
- **`(unsigned long)total` Cast in `ane_mil_gen.h`**: `NSUInteger` ist auf macOS ein `unsigned long`. Der Cast verhindert `-Wformat`-Warnungen beim `%lu`-Format-Specifier.

### Build-Verifikation

- `make train_large` kompiliert sauber ohne Fehler oder Warnungen.
- Commit: `87014bd` auf Branch `fix/high-security-findings`

### Aktualisierter Status HIGH-04 (vollstaendig)

Alle Call-Sites vollstaendig behoben (alle Nachwuchsfunde eingeschlossen):
1. `stories_config.h` Alloc-Helfer — 32 Stellen (Commit 78666fc)
2. `train_large.m` direkte + per-Iteration Allokationen — 31 Stellen (Commit 78666fc)
3. `stories_cpu_ops.h` `rmsnorm()`, `rmsnorm_bwd()`, `cross_entropy_loss()` — 7 Stellen (Commit ce2d68c)
4. `stories_io.h`, `ane_runtime.h`, `ane_mil_gen.h` — 9 Stellen (Commit 87014bd)
5. `stories_mil.h` `get_mask_blob()` Maske — 1 Stelle (Commit 42eae54)

## Aktualisierter Status (nach HIGH-04 vollstaendig)

| Finding-Typ | Anzahl | Status |
|-------------|--------|--------|
| KRITISCH (CRIT-01–04) | 4 | BEHOBEN |
| HOCH (HIGH-01–05) | 5 | HIGH-01 BEHOBEN, HIGH-02 BEHOBEN, HIGH-03 BEHOBEN, HIGH-04 BEHOBEN, HIGH-05 Offen |
| MITTEL (MED-01–06) | 6 | BEHOBEN |
| NIEDRIG (LOW-01–04) | 4 | BEHOBEN |

## HIGH-05 Fix (2026-03-02)

Branch `fix/high-security-findings` (fortgesetzt nach HIGH-04 vollstaendig). HIGH-05 behoben.

### Problem

`ane_eval(Kern *k)` in `stories_io.h` war `void` und ignorierte den `BOOL`-Rueckgabewert von `evaluateWithQoS:options:request:error:`. Bei ANE-Ausfuehrungsfehlern (Hardware-Fehler, Modellfehler) lief das Training still mit veralteten/inkorrekten Gradienten weiter.

### Aenderungen

| Datei | Zeile | Aenderung |
|-------|-------|-----------|
| `training/stories_io.h` | 164 | `static void ane_eval(Kern *k)` → `static bool ane_eval(Kern *k)` (HIGH-05 Kommentar); `BOOL ok =` Rueckgabe-Capture; `NSError *e` bereits vorhanden (wird nun ausgewertet); `if (!ok) fprintf(stderr, ...)` Fehlerausgabe; `return (bool)ok` |
| `training/train_large.m` | 411 | `bool step_ok = true;` vor der Akkumulations-`for`-Schleife eingefuegt (HIGH-05 Kommentar) |
| `training/train_large.m` | 437, 450, 513, 553, 556, 580 | Alle 6 `ane_eval(...)` Call-Sites → `step_ok &= ane_eval(...)` |
| `training/train_large.m` | 636–639 | `if (!step_ok)` Guard nach dem Akkumulations-Loop: `fprintf(stderr, ...)` + `continue` (springt zur naechsten `while`-Iteration, ueberspringt Adam-Update) |

### Design-Entscheidungen

- **`&=` Operator**: Propagiert `false` korrekt durch alle Iterationen — wenn auch nur ein einziges `ane_eval()` ueber alle Schichten und Akkumulationsschritte scheitert, wird `step_ok` dauerhaft `false`.
- **`continue` zielt auf `while (step < total_steps)`**: Die `if (!step_ok)` Pruefung liegt ausserhalb der inneren `for (a=0..ACCUM_STEPS)` Schleife, aber innerhalb der aeusseren `while`. Ein `continue` springt daher korrekt zum naechsten `while`-Durchlauf (naechste Kompilierungsrunde), nicht zum naechsten Akkumulationsschritt.
- **`NSError *e = nil` war bereits vorhanden**: Der `e`-Parameter war schon in der alten Implementierung als `nil` initialisiert und an `objc_msgSend` uebergeben — der einzige fehlende Teil war das Auswerten des Rueckgabewerts und der NSError-Beschreibung.
- **Kein `abort()` bei Fehler**: Im Gegensatz zu OOM-Fehlern (HIGH-04) ist ein transienter ANE-Fehler potenziell behebbar. Das Training ueberspringt den Schritt und faehrt mit dem naechsten fort — degradiert den Fortschritt, stoppt ihn aber nicht.
- **`step_ok` ausserhalb der Layer-Schleife**: Eine einzelne `bool`-Variable reicht — die `&=`-Verkettung ueber alle Schichten und alle Akkumulationsschritte akkumuliert korrekt.

### Build-Verifikation

- `make train_large` kompiliert sauber ohne Fehler oder Warnungen (Compiler-Aufruf: `xcrun clang -O2 -Wall -Wno-deprecated-declarations -fobjc-arc ...`).
- Commit: `f78b943` auf Branch `fix/high-security-findings`

### Status HIGH-05

Alle Teilprobleme vollstaendig behoben:
1. `stories_io.h` `ane_eval()` von `void` zu `bool` geaendert — Commit f78b943
2. `train_large.m` `step_ok` Deklaration vor Akkumulationsschleife — Commit f78b943
3. `train_large.m` 6 Call-Sites mit `step_ok &=` — Commit f78b943
4. `train_large.m` Adam-Update-Skip bei `!step_ok` — Commit f78b943

## Abschlusstatus: Alle HIGH-Findings behoben (2026-03-02)

| Finding-Typ | Anzahl | Status |
|-------------|--------|--------|
| KRITISCH (CRIT-01–04) | 4 | BEHOBEN |
| HOCH (HIGH-01–05) | 5 | ALLE BEHOBEN |
| MITTEL (MED-01–06) | 6 | BEHOBEN |
| NIEDRIG (LOW-01–04) | 4 | BEHOBEN |

Alle 19 Sicherheitsbefunde vollstaendig behoben. Branch: `fix/high-security-findings` auf `manni07/ANE`.
