// data_validation.h — Shared token-data validation helpers
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum {
    TOKEN_DATA_VALID = 0,
    TOKEN_DATA_ERR_TOO_SHORT = 1,
    TOKEN_DATA_ERR_OOB_TOKEN = 2
} TokenDataValidationCode;

typedef struct {
    size_t required_tokens;
    size_t bad_index;
    uint16_t bad_token;
} TokenDataValidationError;

// Token files are 16-bit ids. Return false when byte length is misaligned.
static inline bool token_data_bytes_to_token_count(size_t n_bytes, size_t *n_tokens, size_t *extra_bytes) {
    size_t rem = n_bytes % sizeof(uint16_t);
    if (n_tokens) *n_tokens = n_bytes / sizeof(uint16_t);
    if (extra_bytes) *extra_bytes = rem;
    return rem == 0;
}

static inline bool token_data_has_min_tokens(size_t n_tokens, int seq, size_t *required_tokens) {
    if (seq < 0) return false;
    size_t needed = (size_t)seq + 1;
    if (required_tokens) *required_tokens = needed;
    return n_tokens >= needed;
}

static inline bool token_data_find_oob_token(const uint16_t *token_data, size_t n_tokens, int vocab,
                                             size_t *bad_index, uint16_t *bad_token) {
    if (!token_data || n_tokens == 0 || vocab <= 0) return false;
    for (size_t i = 0; i < n_tokens; i++) {
        if ((int)token_data[i] >= vocab) {
            if (bad_index) *bad_index = i;
            if (bad_token) *bad_token = token_data[i];
            return true;
        }
    }
    return false;
}

static inline TokenDataValidationCode token_data_validate(const uint16_t *token_data, size_t n_tokens,
                                                          int seq, int vocab,
                                                          TokenDataValidationError *err) {
    if (err) {
        err->required_tokens = 0;
        err->bad_index = 0;
        err->bad_token = 0;
    }

    size_t required = 0;
    if (!token_data_has_min_tokens(n_tokens, seq, &required)) {
        if (err) err->required_tokens = required;
        return TOKEN_DATA_ERR_TOO_SHORT;
    }

    size_t bad_index = 0;
    uint16_t bad_token = 0;
    if (token_data_find_oob_token(token_data, n_tokens, vocab, &bad_index, &bad_token)) {
        if (err) {
            err->bad_index = bad_index;
            err->bad_token = bad_token;
        }
        return TOKEN_DATA_ERR_OOB_TOKEN;
    }

    return TOKEN_DATA_VALID;
}
