// test_data_validation.c — Unit tests for token-data hardening helpers
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "data_validation.h"

typedef struct {
    int passed;
    int failed;
} TestStats;

#define CHECK_TRUE(stats, cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            (stats)->failed++; \
            return; \
        } \
    } while (0)

#define CHECK_EQ_INT(stats, got, want, msg) CHECK_TRUE((stats), (got) == (want), msg)
#define CHECK_EQ_SIZE(stats, got, want, msg) CHECK_TRUE((stats), (got) == (want), msg)

static uint32_t lcg_next(uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static void test_bytes_to_token_count_even(TestStats *stats) {
    size_t n_tokens = 0;
    size_t extra = 99;
    CHECK_TRUE(stats, token_data_bytes_to_token_count(1024, &n_tokens, &extra),
               "even byte length should map to token count");
    CHECK_EQ_SIZE(stats, n_tokens, 512, "1024 bytes should map to 512 tokens");
    CHECK_EQ_SIZE(stats, extra, 0, "even byte length should have zero remainder");
    stats->passed++;
}

static void test_bytes_to_token_count_odd(TestStats *stats) {
    size_t n_tokens = 0;
    size_t extra = 0;
    CHECK_TRUE(stats, !token_data_bytes_to_token_count(1025, &n_tokens, &extra),
               "odd byte length should fail alignment check");
    CHECK_EQ_SIZE(stats, n_tokens, 512, "odd byte length should still report floor token count");
    CHECK_EQ_SIZE(stats, extra, 1, "1025 bytes should report one extra byte");
    stats->passed++;
}

static void test_bytes_to_token_count_null_outputs(TestStats *stats) {
    CHECK_TRUE(stats, token_data_bytes_to_token_count(8, NULL, NULL),
               "alignment helper should work with null output pointers");
    CHECK_TRUE(stats, !token_data_bytes_to_token_count(9, NULL, NULL),
               "alignment helper should fail odd byte length with null outputs");
    stats->passed++;
}

static void test_min_tokens_boundary(TestStats *stats) {
    size_t required = 0;
    CHECK_TRUE(stats, token_data_has_min_tokens(257, 256, &required), "257 tokens should satisfy seq=256");
    CHECK_EQ_SIZE(stats, required, 257, "required tokens should be seq+1");
    stats->passed++;
}

static void test_min_tokens_short(TestStats *stats) {
    size_t required = 0;
    CHECK_TRUE(stats, !token_data_has_min_tokens(256, 256, &required), "256 tokens should fail seq=256");
    CHECK_EQ_SIZE(stats, required, 257, "required tokens should still be seq+1");
    stats->passed++;
}

static void test_min_tokens_negative_seq(TestStats *stats) {
    size_t required = 777;
    CHECK_TRUE(stats, !token_data_has_min_tokens(10, -1, &required), "negative seq should fail min-token check");
    CHECK_EQ_SIZE(stats, required, 777, "required token out param should remain unchanged for invalid seq");
    stats->passed++;
}

static void test_validate_too_short(TestStats *stats) {
    uint16_t tokens[2] = {1, 2};
    TokenDataValidationError err = {0};
    TokenDataValidationCode code = token_data_validate(tokens, 2, 4, 32000, &err);
    CHECK_EQ_INT(stats, code, TOKEN_DATA_ERR_TOO_SHORT, "too-short dataset should fail");
    CHECK_EQ_SIZE(stats, err.required_tokens, 5, "required token count should be reported");
    stats->passed++;
}

static void test_validate_too_short_precedes_oob(TestStats *stats) {
    uint16_t tokens[2] = {65000, 1};
    TokenDataValidationError err = {0};
    TokenDataValidationCode code = token_data_validate(tokens, 2, 4, 32000, &err);
    CHECK_EQ_INT(stats, code, TOKEN_DATA_ERR_TOO_SHORT, "too-short check should happen before OOB check");
    CHECK_EQ_SIZE(stats, err.required_tokens, 5, "required token count should still be reported");
    stats->passed++;
}

static void test_validate_too_short_with_null_err(TestStats *stats) {
    uint16_t tokens[2] = {1, 2};
    TokenDataValidationCode code = token_data_validate(tokens, 2, 4, 32000, NULL);
    CHECK_EQ_INT(stats, code, TOKEN_DATA_ERR_TOO_SHORT, "validation should work when err output is null");
    stats->passed++;
}

static void test_validate_oob_first(TestStats *stats) {
    uint16_t tokens[6] = {32000, 1, 2, 3, 4, 5};
    TokenDataValidationError err = {0};
    TokenDataValidationCode code = token_data_validate(tokens, 6, 4, 32000, &err);
    CHECK_EQ_INT(stats, code, TOKEN_DATA_ERR_OOB_TOKEN, "first token OOB should fail");
    CHECK_EQ_SIZE(stats, err.bad_index, 0, "bad index should point to first token");
    CHECK_EQ_INT(stats, err.bad_token, 32000, "bad token value should be reported");
    stats->passed++;
}

static void test_validate_oob_middle(TestStats *stats) {
    uint16_t tokens[7] = {1, 2, 3, 65535, 4, 5, 6};
    TokenDataValidationError err = {0};
    TokenDataValidationCode code = token_data_validate(tokens, 7, 4, 32000, &err);
    CHECK_EQ_INT(stats, code, TOKEN_DATA_ERR_OOB_TOKEN, "middle token OOB should fail");
    CHECK_EQ_SIZE(stats, err.bad_index, 3, "bad index should point to middle token");
    CHECK_EQ_INT(stats, err.bad_token, 65535, "bad token value should be reported");
    stats->passed++;
}

static void test_validate_oob_last(TestStats *stats) {
    uint16_t tokens[6] = {1, 2, 3, 4, 5, 40000};
    TokenDataValidationError err = {0};
    TokenDataValidationCode code = token_data_validate(tokens, 6, 4, 32000, &err);
    CHECK_EQ_INT(stats, code, TOKEN_DATA_ERR_OOB_TOKEN, "last token OOB should fail");
    CHECK_EQ_SIZE(stats, err.bad_index, 5, "bad index should point to last token");
    CHECK_EQ_INT(stats, err.bad_token, 40000, "bad token value should be reported");
    stats->passed++;
}

static void test_validate_ok(TestStats *stats) {
    uint16_t tokens[8] = {0, 1, 2, 3, 4, 5, 31998, 31999};
    TokenDataValidationError err;
    memset(&err, 0xA5, sizeof(err));
    TokenDataValidationCode code = token_data_validate(tokens, 8, 4, 32000, &err);
    CHECK_EQ_INT(stats, code, TOKEN_DATA_VALID, "valid dataset should pass");
    stats->passed++;
}

static void test_validate_vocab_boundary(TestStats *stats) {
    uint16_t valid_tokens[3] = {0, 0, 0};
    TokenDataValidationError err = {0};
    TokenDataValidationCode valid_code = token_data_validate(valid_tokens, 3, 2, 1, &err);
    CHECK_EQ_INT(stats, valid_code, TOKEN_DATA_VALID, "token 0 should be valid when vocab=1");

    uint16_t invalid_tokens[3] = {0, 1, 0};
    TokenDataValidationCode invalid_code = token_data_validate(invalid_tokens, 3, 2, 1, &err);
    CHECK_EQ_INT(stats, invalid_code, TOKEN_DATA_ERR_OOB_TOKEN, "token >= vocab should fail at vocab boundary");
    CHECK_EQ_SIZE(stats, err.bad_index, 1, "boundary OOB should report exact index");
    CHECK_EQ_INT(stats, err.bad_token, 1, "boundary OOB should report offending token");
    stats->passed++;
}

static void test_find_oob_empty(TestStats *stats) {
    size_t bad_index = 123;
    uint16_t bad_token = 456;
    CHECK_TRUE(stats, !token_data_find_oob_token(NULL, 0, 32000, &bad_index, &bad_token),
               "empty dataset should not report OOB token");
    CHECK_EQ_SIZE(stats, bad_index, 123, "bad index should remain unchanged for empty input");
    CHECK_EQ_INT(stats, bad_token, 456, "bad token should remain unchanged for empty input");
    stats->passed++;
}

static void test_find_oob_null_outputs(TestStats *stats) {
    uint16_t tokens[4] = {0, 1, 32000, 2};
    CHECK_TRUE(stats, token_data_find_oob_token(tokens, 4, 32000, NULL, NULL),
               "OOB scan should work with null output pointers");
    stats->passed++;
}

static void test_find_oob_invalid_vocab(TestStats *stats) {
    uint16_t tokens[3] = {0, 1, 2};
    CHECK_TRUE(stats, !token_data_find_oob_token(tokens, 3, 0, NULL, NULL),
               "OOB scan should reject non-positive vocab");
    CHECK_TRUE(stats, !token_data_find_oob_token(tokens, 3, -1, NULL, NULL),
               "OOB scan should reject negative vocab");
    stats->passed++;
}

static void test_find_oob_randomized_consistency(TestStats *stats) {
    uint32_t seed = 1;
    for (int iter = 0; iter < 512; iter++) {
        int vocab = (int)(lcg_next(&seed) % 128u) + 1;
        size_t n_tokens = (size_t)(lcg_next(&seed) % 64u);
        uint16_t tokens[64] = {0};

        bool expected_found = false;
        size_t expected_index = 0;
        uint16_t expected_token = 0;
        for (size_t i = 0; i < n_tokens; i++) {
            tokens[i] = (uint16_t)(lcg_next(&seed) % 256u);
            if (!expected_found && (int)tokens[i] >= vocab) {
                expected_found = true;
                expected_index = i;
                expected_token = tokens[i];
            }
        }

        size_t got_index = 0;
        uint16_t got_token = 0;
        bool got_found = token_data_find_oob_token(tokens, n_tokens, vocab, &got_index, &got_token);
        CHECK_EQ_INT(stats, got_found, expected_found, "randomized OOB scan should match reference result");
        if (expected_found) {
            CHECK_EQ_SIZE(stats, got_index, expected_index, "randomized OOB index should match reference");
            CHECK_EQ_INT(stats, got_token, expected_token, "randomized OOB token should match reference");
        }
    }
    stats->passed++;
}

int main(void) {
    TestStats stats = {0, 0};

    test_bytes_to_token_count_even(&stats);
    test_bytes_to_token_count_odd(&stats);
    test_bytes_to_token_count_null_outputs(&stats);
    test_min_tokens_boundary(&stats);
    test_min_tokens_short(&stats);
    test_min_tokens_negative_seq(&stats);
    test_validate_too_short(&stats);
    test_validate_too_short_precedes_oob(&stats);
    test_validate_too_short_with_null_err(&stats);
    test_validate_oob_first(&stats);
    test_validate_oob_middle(&stats);
    test_validate_oob_last(&stats);
    test_validate_ok(&stats);
    test_validate_vocab_boundary(&stats);
    test_find_oob_empty(&stats);
    test_find_oob_null_outputs(&stats);
    test_find_oob_invalid_vocab(&stats);
    test_find_oob_randomized_consistency(&stats);

    printf("test_data_validation: %d passed, %d failed\n", stats.passed, stats.failed);
    return stats.failed == 0 ? 0 : 1;
}
