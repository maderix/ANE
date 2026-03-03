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

static void test_validate_too_short(TestStats *stats) {
    uint16_t tokens[2] = {1, 2};
    TokenDataValidationError err = {0};
    TokenDataValidationCode code = token_data_validate(tokens, 2, 4, 32000, &err);
    CHECK_EQ_INT(stats, code, TOKEN_DATA_ERR_TOO_SHORT, "too-short dataset should fail");
    CHECK_EQ_SIZE(stats, err.required_tokens, 5, "required token count should be reported");
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

static void test_find_oob_empty(TestStats *stats) {
    size_t bad_index = 123;
    uint16_t bad_token = 456;
    CHECK_TRUE(stats, !token_data_find_oob_token(NULL, 0, 32000, &bad_index, &bad_token),
               "empty dataset should not report OOB token");
    CHECK_EQ_SIZE(stats, bad_index, 123, "bad index should remain unchanged for empty input");
    CHECK_EQ_INT(stats, bad_token, 456, "bad token should remain unchanged for empty input");
    stats->passed++;
}

int main(void) {
    TestStats stats = {0, 0};

    test_min_tokens_boundary(&stats);
    test_min_tokens_short(&stats);
    test_validate_too_short(&stats);
    test_validate_oob_first(&stats);
    test_validate_oob_middle(&stats);
    test_validate_oob_last(&stats);
    test_validate_ok(&stats);
    test_find_oob_empty(&stats);

    printf("test_data_validation: %d passed, %d failed\n", stats.passed, stats.failed);
    return stats.failed == 0 ? 0 : 1;
}
