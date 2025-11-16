//============================A1/5 Not DONE=======================================
#include "stream.h"
#include "../enhancements/lith/lithuanian.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* Reverse 8-bit integer bit order */
static uint8_t reverse_bits8(uint8_t x) {
    x = (x & 0xF0) >> 4 | (x & 0x0F) << 4;
    x = (x & 0xCC) >> 2 | (x & 0x33) << 2;
    x = (x & 0xAA) >> 1 | (x & 0x55) << 1;
    return x;
}

/* Two step variants: right-shift LFSR and left-shift LFSR (parity taps) */
static uint8_t lfsr_next_bit_right(uint8_t *state, uint8_t taps) {
    uint8_t s = *state;
    uint8_t out = s & 1u;
    uint8_t x = s & taps;
    x ^= x >> 4; x ^= x >> 2; x ^= x >> 1;
    uint8_t newbit = x & 1u;
    s = (s >> 1) | (uint8_t)(newbit << 7);
    *state = s;
    return out;
}
static uint8_t lfsr_next_bit_left(uint8_t *state, uint8_t taps) {
    uint8_t s = *state;
    uint8_t out = (s >> 7) & 1u;
    uint8_t x = s & taps;
    x ^= x >> 4; x ^= x >> 2; x ^= x >> 1;
    uint8_t newbit = x & 1u;
    s = (uint8_t)((s << 1) | newbit);
    *state = s;
    return out;
}

/* collect 8 bits to a byte either MSB-first or LSB-first */
typedef uint8_t (*next_bit_fn)(uint8_t*, uint8_t);
static uint8_t lfsr_next_byte(uint8_t *state, uint8_t taps, next_bit_fn f, int msb_first) {
    uint8_t b = 0;
    if (msb_first) {
        for (int i = 0; i < 8; ++i) {
            uint8_t bit = f(state, taps) & 1u;
            b = (uint8_t)((b << 1) | bit);
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            uint8_t bit = f(state, taps) & 1u;
            b |= (uint8_t)(bit << i);
        }
    }
    return b;
}

static char* decrypt_with_variant(const int* cbytes, int n, uint8_t taps, uint8_t init_state,
                                  int shift_right, int msb_first, const char* alph) {
    if (!cbytes || n <= 0) return NULL;
    uint8_t st = init_state;
    char* out = (char*)malloc((size_t)n + 1);
    if (!out) return NULL;
    next_bit_fn f = shift_right ? lfsr_next_bit_right : lfsr_next_bit_left;
    for (int i = 0; i < n; ++i) {
        uint8_t ks = lfsr_next_byte(&st, taps, f, msb_first);
        uint8_t p = (uint8_t)cbytes[i] ^ ks;
        if (alph) {
            int ok = 0;
            for (const char* q = alph; *q; ++q) {
                if ((unsigned char)*q == p) {
                    ok = 1;
                    break;
                }
            }
            out[i] = ok ? (char)p : '?';
        } else {
            if ((p >= 'A' && p <= 'Z') || p == ' ') out[i] = (char)p;
            else out[i] = '?';
        }
    }
    out[n] = '\0';
    return out;
}

static int parse_int_token(const char* s) {
    if (!s || !*s) return -1;
    if (s[0] == '0' && s[1] == 'b') {
        int val = 0;
        for (const char* p = s + 2; *p; ++p) {
            if (*p == '0' || *p == '1') val = (val << 1) | (*p - '0');
            else return -1;
        }
        return val;
    } else {
        char* end = NULL;
        long v = strtol(s, &end, 10);
        if (end == s) return -1;
        if (v < 0 || v > 0xFF) return -1;
        return (int)v;
    }
}

static void write_candidate(FILE* f, uint8_t taps, uint8_t state, int variant_id, const char* s) {
    if (!f) return;
    int shift_right = variant_id & 1;
    int msb_first = (variant_id >> 1) & 1;
    int rev_taps = (variant_id >> 2) & 1;
    fprintf(f, "taps=%u (0b", (unsigned)taps);
    for (int i = 7; i >= 0; --i) fputc((taps & (1u<<i)) ? '1' : '0', f);
    fprintf(f, "); state=%u (0b", (unsigned)state);
    for (int i = 7; i >= 0; --i) fputc((state & (1u<<i)) ? '1' : '0', f);
    fprintf(f, ") variant=sr:%d,msb:%d,rev:%d => %s\n", shift_right, msb_first, rev_taps, s ? s : "");
}

const char* streamEntry(const char* alph, const char* encText, const char* frag) {
    (void)alph;
    if (!encText) return strdup("[no input]");
    if (!frag || !*frag) return strdup("[no frag provided]");

    int bigN = 0;
    int* cbytes = parse_frag_array(encText, &bigN);
    if (!cbytes || bigN <= 0) {
        if (cbytes) free(cbytes);
        return strdup("[invalid ciphertext format]");
    }
    const char* s = frag;
    if (strncmp(s, "lfsr:", 6) == 0) s += 6;
    char tmp[128];
    strncpy(tmp, s, sizeof(tmp)-1); tmp[sizeof(tmp)-1] = '\0';
    char* tokN = strtok(tmp, ";");
    char* tok2 = strtok(NULL, ";");

    if (!tokN) {
        free(cbytes);
        return strdup("[invalid frag]");
    }
    int N = strtol(tokN, NULL, 10);
    if (N != 8) {
        free(cbytes);
        return strdup("[unsupported LFSR size; only 8 supported]");
    }

    int sem_count = 0;
    for (const char* p = frag; *p; ++p) if (*p == ';') ++sem_count;

    if (sem_count >= 2) {
        const char* firstsemi = strchr(frag, ';');
        if (!firstsemi) {
            free(cbytes);
            return strdup("[invalid frag]");
        }
        char tail[128];
        strncpy(tail, firstsemi + 1, sizeof(tail)-1); tail[sizeof(tail)-1] = '\0';
        char* part1 = strtok(tail, ";");
        char* part2 = strtok(NULL, ";");
        if (!part1 || !part2) {
            free(cbytes);
            return strdup("[invalid taps/state]");
        }
        int taps = parse_int_token(part1);
        int state = parse_int_token(part2);
        if (taps < 0 || state < 0) {
            free(cbytes);
            return strdup("[invalid taps/state]");
        }

        const char* allowed_alph = (alph && *alph) ? alph : "ABCDEFGHIJKLMNOPQRSTUVWXYZ ";
        char* chosen = NULL;
        for (int rev = 0; rev <= 1 && !chosen; ++rev) {
            uint8_t taps_used = rev ? reverse_bits8((uint8_t)taps) : (uint8_t)taps;
            for (int shift_right = 0; shift_right <= 1 && !chosen; ++shift_right) {
                for (int msb_first = 0; msb_first <= 1 && !chosen; ++msb_first) {
                    char* dec = decrypt_with_variant(cbytes, bigN, taps_used, (uint8_t)state,
                                                    shift_right, msb_first, allowed_alph);
                    if (!dec) continue;
                    int ok = 1;
                    for (int i = 0; i < bigN; ++i) {
                        if (dec[i] == '?') {
                            ok = 0;
                            break;
                        }
                    }
                    if (ok) chosen = dec;
                    else free(dec);
                }
            }
        }
        free(cbytes);
        if (!chosen) return strdup("[decryption failed or no variant produced fully valid plaintext]");
        static char* static_out = NULL;
        if (static_out) {
            free(static_out);
            static_out = NULL;
        }
        static_out = strdup(chosen);
        free(chosen);
        return static_out;
    } else {

        const char* known_prefix = NULL;
        if (tok2 && *tok2 && strcmp(tok2, "brute") != 0) known_prefix = tok2;

        static char fname[128];
        const char* base = "stream_lfsr-";
        int p = 0;
        while (base[p] && p < (int)sizeof(fname)-1) {
            fname[p] = base[p];
            ++p;
        }
        fname[p] = '\0';
        if (!append_time_txt(fname, (int)sizeof fname)) {
            const char* fb = "unknown.txt";
            int i = 0;
            while (fb[i] && p + i < (int)sizeof(fname)-1) {
                fname[p+i] = fb[i];
                ++i;
            }
            fname[p+i] = '\0';
        }
        FILE* f = fopen(fname, "wb");
        if (!f) {
            free(cbytes);
            return strdup("[error opening output file]");
        }

        const char* allowed_alph = (alph && *alph) ? alph : "ABCDEFGHIJKLMNOPQRSTUVWXYZ ";

        int found_any = 0;
        for (int taps = 1; taps < 256; ++taps) {
            for (int rev = 0; rev <= 1; ++rev) {
                uint8_t taps_used = rev ? reverse_bits8((uint8_t)taps) : (uint8_t)taps;
                for (int state = 1; state < 256; ++state) {
                    for (int shift_right = 0; shift_right <= 1; ++shift_right) {
                        for (int msb_first = 0; msb_first <= 1; ++msb_first) {
                            int variant_id = (rev<<2) | (msb_first<<1) | (shift_right<<0);
                            char* dec = decrypt_with_variant(cbytes, bigN, taps_used, (uint8_t)state,
                                                             shift_right, msb_first, allowed_alph);
                            if (!dec) continue;
                            if (known_prefix) {
                                int okpref = 1;
                                size_t klen = strlen(known_prefix);
                                if ((int)klen > bigN) okpref = 0;
                                else {
                                    for (size_t z = 0; z < klen; ++z) {
                                        if (dec[z] != known_prefix[z]) {
                                            okpref = 0;
                                            break;
                                        }
                                    }
                                }
                                if (okpref) {
                                    int ok = 1;
                                    for (int i = 0; i < bigN; ++i) {
                                        if (dec[i] == '?') {
                                            ok = 0;
                                            break;
                                        }
                                    }
                                    if (ok) {
                                        write_candidate(f, (uint8_t)taps_used, (uint8_t)state, variant_id, dec);
                                        found_any = 1;
                                    }
                                }
                            } else {
                                int ok = 1;
                                for (int i = 0; i < bigN; ++i) {
                                    if (dec[i] == '?') {
                                        ok = 0;
                                        break;
                                    }
                                }
                                if (ok) {
                                    write_candidate(f, (uint8_t)taps_used, (uint8_t)state, variant_id, dec);
                                    found_any = 1;
                                }
                            }
                            free(dec);
                        }
                    }
                }
            }
        }
        fclose(f);
        free(cbytes);
        if (!found_any) return strdup("[no candidate found]");
        static char* static_fname = NULL;
        if (static_fname) {
            free(static_fname);
            static_fname = NULL;
        }
        static_fname = strdup(fname);
        return static_fname;
    }
}
