#include "a5.h"

#define A5_MAX_BITS 256
#define A5_REG_BITS 8

static uint8_t reverseBits8(uint8_t x) {
    x = (x & 0xF0u) >> 4 | (x & 0x0Fu) << 4;
    x = (x & 0xCCu) >> 2 | (x & 0x33u) << 2;
    x = (x & 0xAAu) >> 1 | (x & 0x55u) << 1;
    return x;
}

static int parseTapsMask(const char* frag, uint8_t* taps_out) {
    long long bits_tmp[A5_MAX_BITS];
    size_t count = fragParseIntList(frag, bits_tmp, A5_MAX_BITS);
    if (count == 0) return 0;

    if (count == 1) {
        long long taps_value = bits_tmp[0];
        if (taps_value < 0 || taps_value > 255) return 0;
        *taps_out = (uint8_t)taps_value;
        return 1;
    }

    if (count != A5_REG_BITS) return 0;
    uint8_t mask = 0;
    for (size_t i = 0; i < count; ++i) {
        if (bits_tmp[i] != 0 && bits_tmp[i] != 1) return 0;
        if (bits_tmp[i]) mask |= (uint8_t)(1u << (A5_REG_BITS - 1 - i));
    }
    *taps_out = mask;
    return 1;
}

static uint8_t lfsrParity(uint8_t x) {
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    return x & 1u;
}

static void lfsrClockRight(uint8_t* state, uint8_t taps) {
    uint8_t feedback = lfsrParity((uint8_t)(*state & taps));
    *state = (uint8_t)((*state >> 1) | (uint8_t)(feedback << 7));
}

static void lfsrClockLeft(uint8_t* state, uint8_t taps) {
    uint8_t feedback = lfsrParity((uint8_t)(*state & taps));
    *state = (uint8_t)((*state << 1) | feedback);
}

static uint8_t regClockingBit(uint8_t state, int idx, int msb_indexing) {
    int bit = msb_indexing ? (A5_REG_BITS - 1 - idx) : idx;
    return (uint8_t)((state >> bit) & 1u);
}

static uint8_t regOutputBit(uint8_t state, int output_msb) {
    if (output_msb) return (uint8_t)((state >> 7) & 1u);
    return (uint8_t)(state & 1u);
}

static uint8_t a5_next_bit(uint8_t* s0, uint8_t* s1, uint8_t* s2, uint8_t taps,
                           int output_after_clock, int msb_indexing, int shift_left, int output_msb) {
    uint8_t c0 = regClockingBit(*s0, 1, msb_indexing);
    uint8_t c1 = regClockingBit(*s1, 2, msb_indexing);
    uint8_t c2 = regClockingBit(*s2, 3, msb_indexing);
    uint8_t maj = (uint8_t)((c0 + c1 + c2) >= 2 ? 1 : 0);

    uint8_t o0 = regOutputBit(*s0, output_msb);
    uint8_t o1 = regOutputBit(*s1, output_msb);
    uint8_t o2 = regOutputBit(*s2, output_msb);

    if (c0 == maj) {
        if (shift_left) lfsrClockLeft(s0, taps);
        else lfsrClockRight(s0, taps);
    }
    if (c1 == maj) {
        if (shift_left) lfsrClockLeft(s1, taps);
        else lfsrClockRight(s1, taps);
    }
    if (c2 == maj) {
        if (shift_left) lfsrClockLeft(s2, taps);
        else lfsrClockRight(s2, taps);
    }

    if (output_after_clock) {
        o0 = regOutputBit(*s0, output_msb);
        o1 = regOutputBit(*s1, output_msb);
        o2 = regOutputBit(*s2, output_msb);
    }

    return (uint8_t)((o0 ^ o1 ^ o2) & 1u);
}

static int scorePlaintext(const uint8_t* bytes, size_t n, const char* alph) {
    int score = 0;
    if (alph && *alph) {
        for (size_t i = 0; i < n; ++i) {
            if (strchr(alph, (char)bytes[i])) score += 2;
            else if (bytes[i] >= 32 && bytes[i] <= 126) score += 1;
            else score -= 2;
        }
        return score;
    }
    for (size_t i = 0; i < n; ++i) {
        if ((bytes[i] >= 'A' && bytes[i] <= 'Z') || bytes[i] == ' ') score += 2;
        else if (bytes[i] >= 32 && bytes[i] <= 126) score += 1;
        else score -= 2;
    }
    return score;
}

const char* a5Entry(const char* alph, const char* encText, const char* frag) {
    static char* out = NULL;
    if (out) { free(out); out = NULL; }

    if (!encText || !*encText) return strdup("missing input");
    if (!frag || !*frag) return strdup("missing -frag");

    int cipher_len = 0;
    int* cipher_bytes = parse_frag_array(encText, &cipher_len);
    if (!cipher_bytes || cipher_len <= 0) {
        if (cipher_bytes) free(cipher_bytes);
        return strdup("invalid ciphertext");
    }

    uint8_t taps_mask = 0;
    if (!parseTapsMask(frag, &taps_mask)) {
        free(cipher_bytes);
        return strdup("invalid coef(feedback taps)");
    }

    for (int i = 0; i < cipher_len; ++i) {
        if (cipher_bytes[i] < 0 || cipher_bytes[i] > 255) {
            free(cipher_bytes);
            return strdup("invalid ciphertext byte");
        }
    }

    uint8_t taps_candidates[2] = { taps_mask, reverseBits8(taps_mask) };
    int taps_count = (taps_candidates[0] == taps_candidates[1]) ? 1 : 2;

    uint8_t* best_bytes = (uint8_t*)malloc((size_t)cipher_len);
    uint8_t* tmp_bytes = (uint8_t*)malloc((size_t)cipher_len);
    if (!best_bytes || !tmp_bytes) {
        free(cipher_bytes);
        free(best_bytes);
        free(tmp_bytes);
        return strdup("alloc failed");
    }

    int best_score = -1000000;
    for (int t = 0; t < taps_count; ++t) {
        uint8_t taps = taps_candidates[t];
        for (int output_after = 0; output_after <= 1; ++output_after) {
            for (int msb_indexing = 0; msb_indexing <= 1; ++msb_indexing) {
                for (int shift_left = 0; shift_left <= 1; ++shift_left) {
                    for (int output_msb = 0; output_msb <= 1; ++output_msb) {
                        uint8_t s0 = taps;
                        uint8_t s1 = taps;
                        uint8_t s2 = taps;
                        for (int i = 0; i < cipher_len; ++i) {
                            uint8_t keystream_byte = 0;
                            for (int b = 0; b < 8; ++b) {
                                keystream_byte = (uint8_t)((keystream_byte << 1) |
                                    a5_next_bit(&s0, &s1, &s2, taps, output_after, msb_indexing,
                                                shift_left, output_msb));
                            }
                            tmp_bytes[i] = (uint8_t)cipher_bytes[i] ^ keystream_byte;
                        }
                        int score = scorePlaintext(tmp_bytes, (size_t)cipher_len, alph);
                        if (score > best_score) {
                            best_score = score;
                            memcpy(best_bytes, tmp_bytes, (size_t)cipher_len);
                        }
                    }
                }
            }
        }
    }

    int* plain_bytes = (int*)malloc((size_t)cipher_len * sizeof(int));
    if (!plain_bytes) {
        free(cipher_bytes);
        free(best_bytes);
        free(tmp_bytes);
        return strdup("alloc failed");
    }
    for (int i = 0; i < cipher_len; ++i) {
        plain_bytes[i] = (int)best_bytes[i];
    }
    out = numbersToBytes(plain_bytes, (size_t)cipher_len);

    free(cipher_bytes);
    free(best_bytes);
    free(tmp_bytes);
    free(plain_bytes);
    return out ? out : strdup("alloc failed");
}
