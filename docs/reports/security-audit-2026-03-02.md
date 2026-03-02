# Sicherheitsaudit: ANE (Apple Neural Engine Training Framework)
**Datum:** 2026-03-02
**Repository:** https://github.com/maderix/ANE
**Prüfer:** Claude Code (claude-sonnet-4-6)
**Scope:** Vollständige Codebase-Analyse (38 Quelldateien, Objective-C/C/Python)

---

## Executive Summary

Das ANE-Projekt implementiert Neural-Network-Training direkt auf Apples Neural Engine (ANE) via reverse-engineerter privater APIs. Es handelt sich um ein **Forschungs-/Experimental-Projekt** mit erheblichen inhärenten Sicherheitsrisiken durch die Nutzung undokumentierter Apple-Schnittstellen.

**Gesamtbewertung: HOHES RISIKO** für produktiven Einsatz.

| Kategorie | Anzahl |
|-----------|--------|
| KRITISCH  | 4      |
| HOCH      | 5      |
| MITTEL    | 6      |
| NIEDRIG   | 4      |
| **Gesamt**| **19** |

---

## KRITISCHE Befunde

### [CRIT-01] Keine Fehlerbehandlung bei `dlopen()` für Private Framework
**Datei:** `training/ane_runtime.h:26`, `api_exploration.m:15`
**Schweregrad:** KRITISCH
**Status: BEHOBEN** (2026-03-02, Branch `fix/crit-security-findings`)

```objc
// ane_runtime.h:26
dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
```

**Problem:**
- Der Rückgabewert von `dlopen()` wird nicht geprüft. Wenn das Framework nicht gefunden wird (nach macOS-Update oder auf nicht-Apple-Silicon-Hardware), gibt `dlopen()` NULL zurück — aber die Ausführung läuft weiter.
- Alle nachfolgenden `NSClassFromString()`-Aufrufe geben dann ebenfalls NULL zurück.
- `g_ane_loaded = true` wird gesetzt auch wenn das Laden fehlschlug.

**Folge:** Nullzeiger-Dereferenzierungen beim ersten API-Aufruf, unkontrollierter Absturz ohne aussagekräftige Fehlermeldung.

**Empfehlung:**
```objc
void *handle = dlopen("...", RTLD_NOW);
if (!handle) {
    fprintf(stderr, "ANE framework not found: %s\n", dlerror());
    abort();
}
if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
    fprintf(stderr, "ANE private classes not found (API changed?)\n");
    abort();
}
```

---

### [CRIT-02] Unsichere `objc_msgSend`-Casts ohne Typ-Validierung
**Dateien:** `training/ane_runtime.h:59-125`, `training/stories_io.h:90-117`
**Schweregrad:** KRITISCH
**Status: BEHOBEN** (2026-03-02, Branch `fix/crit-security-findings`)

```objc
// ane_runtime.h:59-61
id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
    g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
    milText, wdict, nil);
```

**Probleme:**
1. Die Klasse `g_ANEDesc` könnte NULL sein (wenn `dlopen` fehlschlug, s. CRIT-01)
2. Die Methodensignatur ist hardcodiert — bei Apple-API-Änderungen falsches Casting = undefiniertes Verhalten / Speicherkorruption
3. Kein `@try/@catch` um mögliche Objective-C Exceptions abzufangen
4. Globale Variablen `g_D`, `g_I`, `g_AIO`, `g_AR` in `stories_io.h` könnten NULL sein

**Folge:** Speicherkorruption, SIGBUS, unkontrollierter Absturz.

**Empfehlung:** Mindestens NULL-Checks vor jedem `objc_msgSend`:
```objc
if (!g_ANEDesc) { fprintf(stderr, "g_ANEDesc is NULL\n"); return NULL; }
```

---

### [CRIT-03] `fread()`-Rückgabewerte nie geprüft — uninitalisierter Speicher
**Dateien:** `training/model.h:81-146`, `training/train_large.m:17-55`
**Schweregrad:** KRITISCH
**Status: BEHOBEN** (2026-03-02, Branch `fix/crit-security-findings`)

```c
// model.h:81
fread(&m->cfg, sizeof(Config), 1, f);  // Rückgabewert ignoriert!

// train_large.m:29
fread(embed, 4, V * DIM, f);  // Kein Check ob V*DIM floats gelesen wurden
```

**Probleme:**
1. Wenn die Model-Datei kleiner als erwartet ist (korrupt, abgeschnitten), werden Structs mit Garbage-Werten befüllt
2. Kein Check ob `cfg.dim`, `cfg.hidden_dim`, `cfg.n_layers` plausibel sind bevor Speicher allokiert wird
3. `fread(embed, 4, V * DIM, f)` — bei V=32000, DIM=768: liest 98,304,000 Bytes. Keine Größenvalidierung.
4. In `load_checkpoint()`: wenn die Datei nach dem Header endet, werden Gewichte mit 0-Bytes befüllt ohne Warnung

**Empfehlung:**
```c
size_t n = fread(&m->cfg, sizeof(Config), 1, f);
if (n != 1) { fprintf(stderr, "Config read failed\n"); fclose(f); return -1; }
if (m->cfg.dim <= 0 || m->cfg.dim > 65536 || m->cfg.n_layers <= 0) {
    fprintf(stderr, "Invalid model config\n"); fclose(f); return -1;
}
```

---

### [CRIT-04] Integer Overflow in Speicher-Berechnung
**Dateien:** `training/stories_io.h:13-14`, `training/ane_mil_gen.h:12-13`
**Schweregrad:** KRITISCH
**Status: BEHOBEN** (2026-03-02, Branch `fix/crit-security-findings`)

```c
// stories_io.h:13-14
static NSData *build_blob(const float *w, int rows, int cols) {
    int ws = rows * cols * 2;   // INT-Multiplikation, kein size_t!
    int tot = 128 + ws;
```

**Problem:** Bei grösseren Modellen mit `dim >= 2048, hidden >= 16384` könnten Integer-Overflows entstehen. `*(uint32_t*)(chunk + 8) = (uint32_t)wsize;` — wenn `wsize` als `int` negativ wird (Overflow), wird ein negativer Wert als uint32 geschrieben = falsche Blob-Größe → ANE-Fehler oder Speicherkorruption.

**Empfehlung:** `size_t` für alle Speichergrößenberechnungen:
```c
size_t ws = (size_t)rows * cols * sizeof(_Float16);
size_t tot = 128 + ws;
```

---

## HOHE Befunde

### [HIGH-01] Keine Eingabevalidierung für Token-Indizes
**Datei:** `training/train_large.m:375-376`
**Schweregrad:** HOCH
**Status: BEHOBEN** (2026-03-02, Branch `fix/high-security-findings`)

```c
size_t max_pos = n_tokens - SEQ - 1;
size_t pos = (size_t)(drand48() * max_pos);
uint16_t *input_tokens = token_data + pos;
```

**Probleme:**
1. Token-Werte aus `token_data` werden direkt als Embedding-Indizes verwendet ohne Prüfung ob `token < VOCAB`
2. Wenn die `.bin`-Datei korrupte Token-Werte enthält (> 32000), entstehen Out-of-Bounds-Zugriffe auf `embed[]`
3. Kein Check ob `n_tokens >= SEQ + 1` vor der `max_pos`-Berechnung

**Folge:** Heap-Buffer-Overflow, korrupte `.bin`-Datei kann zu Speicherschäden führen.

---

### [HIGH-02] Checkpoint-Pfad mit relativer Verzeichnis-Navigation
**Datei:** `training/train_large.m:8-10`
**Schweregrad:** HOCH
**Status: BEHOBEN** (2026-03-02, Branch `fix/high-security-findings`)

```c
#define CKPT_PATH "ane_stories110M_ckpt.bin"
#define MODEL_PATH "../../assets/models/stories110M.bin"  // ← relativer Pfad!
#define DATA_PATH "tinystories_data00.bin"
```

**Probleme:**
1. `MODEL_PATH` enthält `../../` — relative Pfadnavigation. Wenn das Binary aus einem unerwarteten Verzeichnis gestartet wird, werden falsche Dateien gelesen.
2. Kein `realpath()`-Aufruf zur Normalisierung des Pfades
3. Manipulierter Checkpoint + `--resume` → unkontrollierte Binärdaten werden als Gewichte geladen

---

### [HIGH-03] `execl()` zur Prozessneustart ohne Argument-Validierung
**Datei:** `training/train_large.m:331`
**Schweregrad:** HOCH
**Status: BEHOBEN** (2026-03-02, Branch `fix/high-security-findings`)

```c
execl(argv[0], argv[0], "--resume", NULL);
```

**Probleme:**
1. `argv[0]` wird ohne Validierung übergeben. Via Symlink könnte ein beliebiges Binary gestartet werden.
2. `data_fd` (mmap'd Token-Datei) wird vor `execl()` nicht geschlossen — Dateideskriptor-Leak in neuen Prozess
3. `munmap(token_data)` wird vor `execl()` nicht aufgerufen

---

### [HIGH-04] Fehlende `malloc()`/`calloc()`-Rückgabewert-Prüfungen
**Dateien:** Alle `.m` und `.h` Dateien
**Schweregrad:** HOCH
**Status: BEHOBEN** (2026-03-02, Branch `fix/high-security-findings`)

```c
// train_large.m:219
float *embed = (float*)malloc(VOCAB*DIM*4);  // 32000*768*4 = 98MB — kein NULL-Check!
```

Keiner der `malloc()`/`calloc()`-Aufrufe prüft den Rückgabewert auf NULL. Bei Memory-Pressure (110M Model + Adam-State = mehrere GB) können Allokierungen fehlschlagen → Nullzeiger-Dereferenzierung.

---

### [HIGH-05] ANE-Inferenz ohne Fehlerprüfung im Trainings-Hot-Path
**Datei:** `training/stories_io.h:131-134`
**Schweregrad:** HOCH
**Status: BEHOBEN** (2026-03-02, Branch `fix/high-security-findings`)

```c
static void ane_run(Kern *k) {
    id mdl = (__bridge id)k->model; id req = (__bridge id)k->request; NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    // BOOL-Rückgabewert und NSError *e werden ignoriert!
}
```

**Problem:** ANE-Ausführung kann fehlschlagen (Thermal-Throttling, Hardware-Fehler, API-Änderungen). Stille Fehler führen zu unerkannter Gradientenkorruption.

---

## MITTLERE Befunde

### [MED-01] IOSurface Lock ohne Fehlerbehandlung
**Datei:** `training/stories_io.h:62-83`
**Schweregrad:** MITTEL
**Status: BEHOBEN** (2026-03-02, Branch `fix/med-security-findings`)

```c
IOSurfaceLock(s, 0, NULL);  // Return-Code ignoriert
```

`IOSurfaceLock()` gibt `kIOReturnSuccess` oder einen Fehlercode zurück. Bei Lock-Fehler wird trotzdem auf den Speicher zugegriffen — mögliche Data-Race-Condition.

---

### [MED-02] Temporäres Verzeichnis nicht sicher erstellt (TOCTOU-Risiko)
**Datei:** `training/ane_runtime.h:68-80`, `training/stories_io.h:94-100`
**Schweregrad:** MITTEL
**Status: BEHOBEN** (2026-03-02, Branch `fix/med-security-findings`)

```objc
NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
[milText writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
```

TOCTOU-Race zwischen `createDirectoryAtPath` und `writeToFile`. Der `hexStringIdentifier` könnte von einem anderen Prozess erraten und das Verzeichnis manipuliert werden.

---

### [MED-03] MIL-Text-Generierung ohne Parameter-Validierung
**Datei:** `training/ane_mil_gen.h:32-52`
**Schweregrad:** MITTEL
**Status: BEHOBEN** (2026-03-02, Branch `fix/med-security-findings`)

```objc
return [NSString stringWithFormat:
    @"...tensor<fp32, [1, %d, %d]> x...", in_ch, spatial, ...];
```

Negative oder extrem große `in_ch`/`out_ch`/`spatial`-Werte durch fehlerhafte Konfiguration erzeugen invalides MIL das an den undokumentierten ANE-Compiler übergeben wird.

---

### [MED-04] Keine Endianness-Prüfung bei Checkpoint-Serialisierung
**Datei:** `training/train_large.m:110-181`
**Schweregrad:** MITTEL
**Status: BEHOBEN** (2026-03-02, Branch `fix/med-security-findings`)

```c
h.magic = 0x424C5A54;
fwrite(&h, sizeof(h), 1, f);
```

Das `CkptHdr`-Struct wird als binärer Dump ohne Endianness-Marker geschrieben. Nicht portabel.

---

### [MED-05] NEON-Vektorisierung ohne Alignment-Garantie
**Datei:** `training/stories_io.h:41-58`
**Schweregrad:** MITTEL
**Status: BEHOBEN** (2026-03-02, Branch `fix/med-security-findings`)

```c
float16x8_t h = vld1q_f16((const __fp16*)(src + i));
```

Zeiger-Arithmetik mit `ch_off * sp` könnte das für NEON benötigte Alignment verletzen wenn `ch_off * sp` kein Vielfaches von 8 ist.

---

### [MED-06] Globale Variablen ohne Thread-Safety
**Datei:** `training/stories_io.h`, `training/stories_config.h`
**Schweregrad:** MITTEL
**Status: BEHOBEN** (2026-03-02, Branch `fix/med-security-findings`)

```c
static bool g_ane_loaded = false;
static int g_compile_count = 0;
```

`g_compile_count` wird via `__sync_fetch_and_add()` atomar inkrementiert, aber `g_ane_loaded` und Klassen-Variablen nicht atomar gesetzt — bei Multi-Thread-Nutzung Race-Condition in `ane_init()`.

---

## NIEDRIGE Befunde

### [LOW-01] Fehlende Compiler-Sicherheitsflags
**Datei:** `training/Makefile:2`
**Schweregrad:** NIEDRIG
**Status: BEHOBEN** (2026-03-02, Branch `fix/low-security-findings`)

```makefile
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc
```

Fehlende Flags: `-fstack-protector-strong`, `-D_FORTIFY_SOURCE=2`, `-Wformat=2`

**Fix:** `SEC_FLAGS = -fstack-protector-strong -Wformat-security` eingeführt. Hinweis:
`-D_FORTIFY_SOURCE=2` ist auf macOS (Apple LLVM) bei `-O2` implizit aktiv — explizite
Definition würde "macro redefinition"-Warnung erzeugen. `CFLAGS_DEBUG` mit
`-fsanitize=address,undefined` für Debug-Builds hinzugefügt. `make verify-flags`
zeigt aktive Flags.

---

### [LOW-02] `-Wno-deprecated-declarations` unterdrückt wichtige Warnungen
**Datei:** `training/Makefile:2`
**Schweregrad:** NIEDRIG
**Status: BEHOBEN** (2026-03-02, Branch `fix/low-security-findings`)

Unterdrückt Warnungen über veraltete API-Aufrufe — könnte wichtige Hinweise auf deprecated private APIs verstecken.

**Fix:** Flag in benannte Variable `ANE_COMPAT` extrahiert mit erklärendem Kommentar
(bewusste Unterdrückung wegen privater `_ANE*`-APIs via `objc_msgSend`). Neues Target
`make check-deprecated` baut ohne Unterdrückung und zeigt alle verborgenen Warnungen.

---

### [LOW-03] Python-Skript ohne Eingabevalidierung
**Datei:** `training/tokenize.py`
**Schweregrad:** NIEDRIG
**Status: BEHOBEN** (2026-03-02, Branch `fix/low-security-findings`)

Keine Validierung der Eingabedateigröße — bei sehr großen Eingaben Out-of-Memory möglich.

**Fix:** 5 Validierungen implementiert:
1. ZIP-Existenzprüfung mit hilfreicher Fehlermeldung
2. Konfigurierbare Größengrenze (Standard 10GB, via `MAX_ZIP_BYTES` env var überschreibbar)
3. Prüfung ob `data00.bin` im ZIP enthalten ist
4. Fehlerbehandlung bei `struct.unpack` wenn Output < 20 Bytes
5. Token-Range-Validierung (alle Token müssen < `VOCAB_SIZE=32000` sein)

---

### [LOW-04] Keine `.gitignore` für sensible Artefakte
**Datei:** Repository-Root
**Schweregrad:** NIEDRIG
**Status: BEHOBEN** (2026-03-02, Branch `fix/low-security-findings`)

Keine `.gitignore`-Datei. Binäre Artefakte (Checkpoints, Trainingsdaten, `firebase-debug.log`) könnten versehentlich committed werden.

**Fix:** `.gitignore` erstellt mit Regeln für: macOS-Metadaten (`.DS_Store`),
Log-Dateien (`*.log`), kompilierte Binaries (`training/train`, `training/train_large`,
alle Probe-Binaries), Trainingsdaten (`training/*.bin`), ANE-Artefakte
(`*.mlmodelc/`, `*.mlpackage/`), externe Assets (`assets/`).

---

## Positive Befunde (Stärken)

### Korrekte Speicherfreigabe
`ane_free()` (`ane_runtime.h:149-160`) und `free_kern()` (`stories_io.h:122-130`) implementieren vollständige Cleanup-Routinen mit `CFRelease()`, `unloadWithQoS:error:` und Temporärverzeichnis-Bereinigung.

### Magic-Byte Validierung in Checkpoints
```c
if (h.magic != 0x424C5A54 || h.version != 2) { fclose(f); return false; }
```
Grundlegender Schutz gegen korrupte Checkpoint-Dateien.

### Atomare Compile-Counter
```c
__sync_fetch_and_add(&g_compile_count, 1);
```
Thread-sicherer Zähler für ANE-Kompilierungsanzahl.

### Gradient-Accumulation mit async CBLAS
Korrekte Parallelisierung von CPU-Gewichtsgradienten-Berechnung via `dispatch_group_async`.

---

## Risikobewertung für Produktionseinsatz

| Aspekt | Bewertung |
|--------|-----------|
| Apple Silicon erforderlich | macOS 15+, M-Series only |
| Private API Stabilität | **SEHR GERING** — jedes macOS-Update kann brechen |
| Memory Safety | **MITTEL** — keine Bounds-Checks, keine Sanitizer |
| Input Validation | **GERING** — Dateien werden unkritisch gelesen |
| Error Handling | **GERING** — viele kritische Fehler werden ignoriert |
| Eignung für Produktion | **NEIN** — Forschungs-/Experimental-Projekt |

---

## Empfehlungen nach Priorität

### Sofortige Maßnahmen (KRITISCH)
1. `dlopen()` Rückgabewert prüfen und bei Fehler abbrechen
2. Alle `fread()`-Rückgabewerte prüfen + Dateigrößenvalidierung
3. NULL-Checks vor allen `objc_msgSend`-Aufrufen
4. `int` → `size_t` für alle Speichergrößenberechnungen

### Kurzfristige Maßnahmen (HOCH)
5. Token-Index-Validierung: `if (token >= VOCAB) abort()`
6. ANE-Inferenz-Rückgabewert und NSError prüfen
7. Compiler-Flags: `-fstack-protector-strong -D_FORTIFY_SOURCE=2`
8. `.gitignore` für binäre Artefakte erstellen

### Mittelfristige Maßnahmen (MITTEL)
9. IOSurface Lock-Rückgabewerte prüfen
10. `__atomic_store_n()` für `g_ane_loaded`
11. MIL-Parameter-Validierung vor Formatierung

---

*Dieser Bericht ist für das ANE-Forschungsprojekt erstellt. Das Projekt ist explizit als Proof-of-Concept/Forschungscode konzipiert und nicht für Produktionseinsatz gedacht.*
