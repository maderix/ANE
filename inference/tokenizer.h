// tokenizer.h -- Byte-level BPE tokenizer for Qwen2.5 in pure C
// Loads vocab.json + merges.txt from HuggingFace model directory.
// Implements GPT-style byte-level BPE (same algorithm as tiktoken/llama.cpp).
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define TOK_MAX_VOCAB    152000
#define TOK_MAX_MERGES   152000
#define TOK_MAX_TOKEN_LEN 256
#define TOK_HASH_SIZE    (1 << 20)  // ~1M buckets

// Special token IDs for Qwen2.5
#define TOK_IM_START     151644
#define TOK_IM_END       151645
#define TOK_ENDOFTEXT    151643

// --- Byte-to-unicode mapping (GPT-2 standard) ---
// Maps byte values 0-255 to unicode codepoints used in the BPE vocab.
// Printable ASCII stays the same; non-printable bytes map to U+0100..U+0143.

static int g_byte_to_unicode[256];
static int g_unicode_to_byte[65536];

static void tok_init_byte_mapping(void) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) {
            g_byte_to_unicode[b] = b;
        } else {
            g_byte_to_unicode[b] = 256 + n;
            n++;
        }
    }
    memset(g_unicode_to_byte, 0xFF, sizeof(g_unicode_to_byte));
    for (int b = 0; b < 256; b++)
        g_unicode_to_byte[g_byte_to_unicode[b]] = b;
}

// --- UTF-8 helpers ---

static int utf8_encode(int codepoint, char *out) {
    if (codepoint < 0x80) {
        out[0] = (char)codepoint;
        return 1;
    } else if (codepoint < 0x800) {
        out[0] = (char)(0xC0 | (codepoint >> 6));
        out[1] = (char)(0x80 | (codepoint & 0x3F));
        return 2;
    } else if (codepoint < 0x10000) {
        out[0] = (char)(0xE0 | (codepoint >> 12));
        out[1] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        out[2] = (char)(0x80 | (codepoint & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (codepoint >> 18));
    out[1] = (char)(0x80 | ((codepoint >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
    out[3] = (char)(0x80 | (codepoint & 0x3F));
    return 4;
}

static int utf8_decode(const char *s, int *codepoint) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) { *codepoint = c; return 1; }
    if ((c & 0xE0) == 0xC0) {
        *codepoint = ((c & 0x1F) << 6) | (s[1] & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0) {
        *codepoint = ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        return 3;
    }
    *codepoint = ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
    return 4;
}

// --- Hash map: string -> int ---

typedef struct {
    char *key;
    int value;
} TokHashEntry;

typedef struct {
    TokHashEntry *entries;
    int capacity;
} TokHashMap;

static unsigned int tok_hash(const char *s) {
    unsigned int h = 5381;
    while (*s) h = ((h << 5) + h) ^ (unsigned char)*s++;
    return h;
}

static void tok_hashmap_init(TokHashMap *m, int capacity) {
    m->capacity = capacity;
    m->entries = (TokHashEntry*)calloc(capacity, sizeof(TokHashEntry));
}

static void tok_hashmap_set(TokHashMap *m, const char *key, int value) {
    unsigned int idx = tok_hash(key) % m->capacity;
    while (m->entries[idx].key) {
        if (strcmp(m->entries[idx].key, key) == 0) {
            m->entries[idx].value = value;
            return;
        }
        idx = (idx + 1) % m->capacity;
    }
    m->entries[idx].key = strdup(key);
    m->entries[idx].value = value;
}

static int tok_hashmap_get(TokHashMap *m, const char *key, int default_val) {
    unsigned int idx = tok_hash(key) % m->capacity;
    while (m->entries[idx].key) {
        if (strcmp(m->entries[idx].key, key) == 0)
            return m->entries[idx].value;
        idx = (idx + 1) % m->capacity;
    }
    return default_val;
}

static void tok_hashmap_free(TokHashMap *m) {
    for (int i = 0; i < m->capacity; i++)
        if (m->entries[i].key) free(m->entries[i].key);
    free(m->entries);
    m->entries = NULL;
    m->capacity = 0;
}

// --- Merge pair ---

typedef struct {
    char *a;
    char *b;
} TokMerge;

// --- Tokenizer state ---

typedef struct {
    TokHashMap vocab;        // token string -> id
    char **id_to_token;      // id -> token string (for decoding)
    int vocab_size;
    TokMerge *merges;
    int n_merges;
    TokHashMap merge_rank;   // "a b" -> rank (lower = higher priority)

    // Special tokens
    int im_start;
    int im_end;
    int eos;
} Tokenizer;

// --- JSON string parsing (minimal, handles unicode escapes) ---

static int tok_parse_json_string(const char *s, char *out, int max_out) {
    if (*s != '"') return -1;
    s++;
    int n = 0;
    while (*s && *s != '"' && n < max_out - 1) {
        if (*s == '\\') {
            s++;
            switch (*s) {
                case '"': out[n++] = '"'; break;
                case '\\': out[n++] = '\\'; break;
                case '/': out[n++] = '/'; break;
                case 'n': out[n++] = '\n'; break;
                case 'r': out[n++] = '\r'; break;
                case 't': out[n++] = '\t'; break;
                case 'u': {
                    char hex[5] = {s[1], s[2], s[3], s[4], 0};
                    int cp = (int)strtol(hex, NULL, 16);
                    n += utf8_encode(cp, out + n);
                    s += 4;
                    break;
                }
                default: out[n++] = *s;
            }
        } else {
            out[n++] = *s;
        }
        s++;
    }
    out[n] = '\0';
    return n;
}

// --- Load vocab.json ---
// Format: {"token_string": id, ...}

static int tok_load_vocab(Tokenizer *t, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open vocab: %s\n", path); return -1; }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *data = (char*)malloc(fsize + 1);
    fread(data, 1, fsize, f);
    data[fsize] = '\0';
    fclose(f);

    tok_hashmap_init(&t->vocab, TOK_HASH_SIZE);
    t->id_to_token = (char**)calloc(TOK_MAX_VOCAB, sizeof(char*));
    t->vocab_size = 0;

    char *p = data;
    // Skip opening {
    while (*p && *p != '{') p++;
    if (*p) p++;

    char key_buf[TOK_MAX_TOKEN_LEN];
    while (*p) {
        while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',')) p++;
        if (*p == '}' || !*p) break;

        int klen = tok_parse_json_string(p, key_buf, sizeof(key_buf));
        if (klen < 0) break;

        // Skip past closing quote
        p++; // opening "
        while (*p) {
            if (*p == '\\') { p += 2; continue; }
            if (*p == '"') { p++; break; }
            p++;
        }

        // Skip colon and whitespace
        while (*p && (*p == ' ' || *p == ':')) p++;

        int id = (int)strtol(p, &p, 10);

        if (id >= 0 && id < TOK_MAX_VOCAB) {
            tok_hashmap_set(&t->vocab, key_buf, id);
            t->id_to_token[id] = strdup(key_buf);
            if (id >= t->vocab_size) t->vocab_size = id + 1;
        }
    }

    free(data);
    printf("  Vocab: %d tokens\n", t->vocab_size);
    return 0;
}

// --- Load merges.txt ---
// Format: one merge per line, "tokenA tokenB" (space-separated)
// First line may be a header starting with #

static int tok_load_merges(Tokenizer *t, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open merges: %s\n", path); return -1; }

    t->merges = (TokMerge*)malloc(TOK_MAX_MERGES * sizeof(TokMerge));
    tok_hashmap_init(&t->merge_rank, TOK_HASH_SIZE);
    t->n_merges = 0;

    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        // Strip newline
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;
        if (line[0] == '#') continue; // skip header

        // Split on first space
        char *space = strchr(line, ' ');
        if (!space) continue;
        *space = '\0';

        t->merges[t->n_merges].a = strdup(line);
        t->merges[t->n_merges].b = strdup(space + 1);

        // Store merge rank: "a b" -> rank
        *space = ' '; // restore
        tok_hashmap_set(&t->merge_rank, line, t->n_merges);

        t->n_merges++;
        if (t->n_merges >= TOK_MAX_MERGES) break;
    }

    fclose(f);
    printf("  Merges: %d rules\n", t->n_merges);
    return 0;
}

// --- Add special tokens ---

static void tok_add_special_tokens(Tokenizer *t) {
    struct { const char *text; int id; } specials[] = {
        {"<|endoftext|>", 151643},
        {"<|im_start|>",  151644},
        {"<|im_end|>",    151645},
    };
    for (int i = 0; i < 3; i++) {
        tok_hashmap_set(&t->vocab, specials[i].text, specials[i].id);
        if (specials[i].id < TOK_MAX_VOCAB) {
            if (t->id_to_token[specials[i].id]) free(t->id_to_token[specials[i].id]);
            t->id_to_token[specials[i].id] = strdup(specials[i].text);
        }
        if (specials[i].id >= t->vocab_size) t->vocab_size = specials[i].id + 1;
    }
    t->im_start = 151644;
    t->im_end = 151645;
    t->eos = 151643;
}

// --- Initialize tokenizer ---

static int tok_init(Tokenizer *t, const char *model_dir) {
    char path[4096];

    tok_init_byte_mapping();

    snprintf(path, sizeof(path), "%s/vocab.json", model_dir);
    if (tok_load_vocab(t, path) != 0) return -1;

    snprintf(path, sizeof(path), "%s/merges.txt", model_dir);
    if (tok_load_merges(t, path) != 0) return -1;

    tok_add_special_tokens(t);
    return 0;
}

static void tok_free(Tokenizer *t) {
    tok_hashmap_free(&t->vocab);
    tok_hashmap_free(&t->merge_rank);
    if (t->id_to_token) {
        for (int i = 0; i < t->vocab_size; i++)
            if (t->id_to_token[i]) free(t->id_to_token[i]);
        free(t->id_to_token);
    }
    if (t->merges) {
        for (int i = 0; i < t->n_merges; i++) {
            free(t->merges[i].a);
            free(t->merges[i].b);
        }
        free(t->merges);
    }
}

// --- BPE encoding ---

// Convert a raw byte string to its byte-level unicode representation (UTF-8).
// Each input byte is mapped through g_byte_to_unicode, then encoded as UTF-8.
static int tok_bytes_to_unicode_str(const char *input, int input_len, char *out, int max_out) {
    int n = 0;
    for (int i = 0; i < input_len && n < max_out - 4; i++) {
        unsigned char b = (unsigned char)input[i];
        int cp = g_byte_to_unicode[b];
        n += utf8_encode(cp, out + n);
    }
    out[n] = '\0';
    return n;
}

// A BPE word is a list of token strings (initially one per byte-level char).
typedef struct {
    char **tokens;
    int count;
    int capacity;
} BPEWord;

static void bpe_word_init(BPEWord *w) {
    w->capacity = 64;
    w->tokens = (char**)malloc(w->capacity * sizeof(char*));
    w->count = 0;
}

static void bpe_word_push(BPEWord *w, const char *s) {
    if (w->count >= w->capacity) {
        w->capacity *= 2;
        w->tokens = (char**)realloc(w->tokens, w->capacity * sizeof(char*));
    }
    w->tokens[w->count++] = strdup(s);
}

static void bpe_word_free(BPEWord *w) {
    for (int i = 0; i < w->count; i++) free(w->tokens[i]);
    free(w->tokens);
}

// Apply BPE merges to a word (list of token strings).
static void bpe_merge(BPEWord *w, Tokenizer *t) {
    while (w->count > 1) {
        // Find the pair with lowest merge rank
        int best_rank = t->n_merges + 1;
        int best_idx = -1;
        char pair_key[TOK_MAX_TOKEN_LEN * 2 + 2];

        for (int i = 0; i < w->count - 1; i++) {
            snprintf(pair_key, sizeof(pair_key), "%s %s", w->tokens[i], w->tokens[i+1]);
            int rank = tok_hashmap_get(&t->merge_rank, pair_key, t->n_merges + 1);
            if (rank < best_rank) {
                best_rank = rank;
                best_idx = i;
            }
        }

        if (best_idx < 0) break; // no more merges

        // Merge tokens[best_idx] and tokens[best_idx+1]
        char merged[TOK_MAX_TOKEN_LEN * 2 + 1];
        snprintf(merged, sizeof(merged), "%s%s", w->tokens[best_idx], w->tokens[best_idx+1]);
        free(w->tokens[best_idx]);
        free(w->tokens[best_idx+1]);
        w->tokens[best_idx] = strdup(merged);

        // Shift remaining tokens left
        for (int i = best_idx + 1; i < w->count - 1; i++)
            w->tokens[i] = w->tokens[i+1];
        w->count--;
    }
}

// Pre-tokenize: split on word boundaries (simplified GPT-style).
// Splits on transitions between: letters, digits, spaces, punctuation.
// Each "word" includes leading space if present (byte-level BPE convention).
typedef struct {
    char **words;
    int count;
    int capacity;
} WordList;

static void wordlist_init(WordList *wl) {
    wl->capacity = 256;
    wl->words = (char**)malloc(wl->capacity * sizeof(char*));
    wl->count = 0;
}

static void wordlist_push(WordList *wl, const char *s, int len) {
    if (wl->count >= wl->capacity) {
        wl->capacity *= 2;
        wl->words = (char**)realloc(wl->words, wl->capacity * sizeof(char*));
    }
    char *copy = (char*)malloc(len + 1);
    memcpy(copy, s, len);
    copy[len] = '\0';
    wl->words[wl->count++] = copy;
}

static void wordlist_free(WordList *wl) {
    for (int i = 0; i < wl->count; i++) free(wl->words[i]);
    free(wl->words);
}

static int is_letter(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
}

static int is_digit(unsigned char c) {
    return c >= '0' && c <= '9';
}

static void tok_pre_tokenize(const char *text, WordList *out) {
    wordlist_init(out);
    int len = (int)strlen(text);
    int i = 0;

    while (i < len) {
        int start = i;

        if (text[i] == ' ') {
            // Space + following word/punct
            i++;
            if (i < len && is_letter((unsigned char)text[i])) {
                while (i < len && is_letter((unsigned char)text[i])) i++;
            } else if (i < len && is_digit((unsigned char)text[i])) {
                while (i < len && is_digit((unsigned char)text[i])) i++;
            } else if (i < len && text[i] != ' ') {
                i++; // single punct after space
            }
            wordlist_push(out, text + start, i - start);
        } else if (is_letter((unsigned char)text[i])) {
            while (i < len && is_letter((unsigned char)text[i])) i++;
            wordlist_push(out, text + start, i - start);
        } else if (is_digit((unsigned char)text[i])) {
            while (i < len && is_digit((unsigned char)text[i])) i++;
            wordlist_push(out, text + start, i - start);
        } else if (text[i] == '\n' || text[i] == '\r') {
            while (i < len && (text[i] == '\n' || text[i] == '\r')) i++;
            wordlist_push(out, text + start, i - start);
        } else {
            i++;
            wordlist_push(out, text + start, 1);
        }
    }
}

// --- Main encode function ---
// Returns number of token IDs written. Caller provides output buffer.

static int tok_encode(Tokenizer *t, const char *text, int *ids, int max_ids) {
    int n_ids = 0;

    // Pre-tokenize into words
    WordList words;
    tok_pre_tokenize(text, &words);

    for (int w = 0; w < words.count && n_ids < max_ids; w++) {
        // Convert word bytes to byte-level unicode string
        char unicode_str[TOK_MAX_TOKEN_LEN * 4];
        int wlen = (int)strlen(words.words[w]);
        tok_bytes_to_unicode_str(words.words[w], wlen, unicode_str, sizeof(unicode_str));

        // Split unicode string into individual unicode chars
        BPEWord bpe;
        bpe_word_init(&bpe);

        const char *p = unicode_str;
        while (*p) {
            int cp;
            int cplen = utf8_decode(p, &cp);
            char single[8];
            int slen = utf8_encode(cp, single);
            single[slen] = '\0';
            bpe_word_push(&bpe, single);
            p += cplen;
        }

        // Apply BPE merges
        bpe_merge(&bpe, t);

        // Look up each resulting token in vocab
        for (int i = 0; i < bpe.count && n_ids < max_ids; i++) {
            int id = tok_hashmap_get(&t->vocab, bpe.tokens[i], -1);
            if (id >= 0) {
                ids[n_ids++] = id;
            } else {
                // Unknown token -- encode each byte-level char as individual token
                const char *bp = bpe.tokens[i];
                while (*bp && n_ids < max_ids) {
                    int bcp;
                    int bcplen = utf8_decode(bp, &bcp);
                    char single[8];
                    int slen = utf8_encode(bcp, single);
                    single[slen] = '\0';
                    int byte_id = tok_hashmap_get(&t->vocab, single, -1);
                    if (byte_id >= 0) ids[n_ids++] = byte_id;
                    bp += bcplen;
                }
            }
        }

        bpe_word_free(&bpe);
    }

    wordlist_free(&words);
    return n_ids;
}

// --- Encode with special tokens ---
// Splits text on special token patterns, encodes non-special parts with BPE.

static int tok_encode_with_special(Tokenizer *t, const char *text, int *ids, int max_ids) {
    struct { const char *text; int id; } specials[] = {
        {"<|im_start|>", TOK_IM_START},
        {"<|im_end|>",   TOK_IM_END},
        {"<|endoftext|>", TOK_ENDOFTEXT},
    };
    int n_specials = 3;
    int n_ids = 0;
    const char *p = text;

    while (*p && n_ids < max_ids) {
        // Check if current position matches a special token
        int matched = 0;
        for (int s = 0; s < n_specials; s++) {
            int slen = (int)strlen(specials[s].text);
            if (strncmp(p, specials[s].text, slen) == 0) {
                ids[n_ids++] = specials[s].id;
                p += slen;
                matched = 1;
                break;
            }
        }
        if (matched) continue;

        // Find next special token
        const char *next_special = NULL;
        for (int s = 0; s < n_specials; s++) {
            const char *found = strstr(p, specials[s].text);
            if (found && (!next_special || found < next_special))
                next_special = found;
        }

        // Encode the text up to the next special (or end)
        int chunk_len = next_special ? (int)(next_special - p) : (int)strlen(p);
        if (chunk_len > 0) {
            char *chunk = (char*)malloc(chunk_len + 1);
            memcpy(chunk, p, chunk_len);
            chunk[chunk_len] = '\0';
            n_ids += tok_encode(t, chunk, ids + n_ids, max_ids - n_ids);
            free(chunk);
        }
        p += chunk_len;
    }

    return n_ids;
}

// --- Decode token IDs to text ---

static int tok_decode(Tokenizer *t, const int *ids, int n_ids, char *out, int max_out) {
    int n = 0;
    for (int i = 0; i < n_ids; i++) {
        int id = ids[i];
        // Skip special tokens in output
        if (id == TOK_IM_START || id == TOK_IM_END || id == TOK_ENDOFTEXT)
            continue;
        if (id < 0 || id >= t->vocab_size || !t->id_to_token[id])
            continue;

        const char *tok_str = t->id_to_token[id];

        // Convert byte-level unicode token back to raw bytes
        const char *p = tok_str;
        while (*p && n < max_out - 1) {
            int cp;
            int cplen = utf8_decode(p, &cp);
            int byte_val = g_unicode_to_byte[cp < 65536 ? cp : 0];
            if (byte_val >= 0 && byte_val < 256) {
                out[n++] = (char)byte_val;
            } else {
                // Not a byte-mapped char, copy UTF-8 directly
                for (int j = 0; j < cplen && n < max_out - 1; j++)
                    out[n++] = p[j];
            }
            p += cplen;
        }
    }
    out[n] = '\0';
    return n;
}

// --- Chat template ---
// Formats: <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n

static int tok_apply_chat_template(const char *system_prompt, const char *user_prompt,
                                   char *out, int max_out) {
    if (!system_prompt) system_prompt = "You are a helpful assistant.";
    return snprintf(out, max_out,
        "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
        system_prompt, user_prompt);
}

// --- Convenience: encode a chat prompt ---

static int tok_encode_chat(Tokenizer *t, const char *system_prompt, const char *user_prompt,
                           int *ids, int max_ids) {
    char templated[65536];
    tok_apply_chat_template(system_prompt, user_prompt, templated, sizeof(templated));
    return tok_encode_with_special(t, templated, ids, max_ids);
}
