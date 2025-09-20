#include "hill.h"

static char* hillDecrypt2x2(const char* alph, const char* enc, const int invK[4]) {
    int alph_bytes = slen(alph);
    uint32_t* alph_u32 = (uint32_t*)malloc(sizeof(uint32_t) * (alph_bytes + 1));
    if (!alph_u32) return NULL;
    int m = utf8_to_u32(alph, alph_u32, alph_bytes + 1);
    if (m <= 0) { free(alph_u32); return NULL; }

    int enc_bytes = slen(enc);
    uint32_t* enc_u32 = (uint32_t*)malloc(sizeof(uint32_t) * (enc_bytes + 1));
    if (!enc_u32) { free(alph_u32); return NULL; }
    int enc_n = utf8_to_u32(enc, enc_u32, enc_bytes + 1);

    int out_cap = (enc_n + 1) * 4 + 1;
    char* out = (char*)malloc(out_cap);
    if (!out) { free(alph_u32); free(enc_u32); return NULL; }

    int oi = 0, buf[2], bcount = 0;

    for (int i = 0; i < enc_n; ++i) {
        uint32_t cp = enc_u32[i];
        int idx = u32_index_of(alph_u32, m, cp);
        if (idx < 0) {
            char tmp[4];
            int len = utf8_encode_one(cp, tmp);
            if (oi + len < out_cap) { for (int t = 0; t < len; ++t) out[oi++] = tmp[t]; }
            continue;
        }
        buf[bcount++] = idx;
        if (bcount == 2) {
            int x0 = buf[0], x1 = buf[1];
            int y0 = mod(invK[0]*x0 + invK[2]*x1, m);
            int y1 = mod(invK[1]*x0 + invK[3]*x1, m);
            char tmp[4];
            int len = utf8_encode_one(alph_u32[y0], tmp);
            if (oi + len < out_cap) { for (int t = 0; t < len; ++t) out[oi++] = tmp[t]; }
            len = utf8_encode_one(alph_u32[y1], tmp);
            if (oi + len < out_cap) { for (int t = 0; t < len; ++t) out[oi++] = tmp[t]; }
            bcount = 0;
        }
    }

    if (bcount == 1) {
        int x0 = buf[0], x1 = 0;
        int y0 = mod(invK[0]*x0 + invK[2]*x1, m);
        int y1 = mod(invK[1]*x0 + invK[3]*x1, m);
        char tmp[4];
        int len = utf8_encode_one(alph_u32[y0], tmp);
        if (oi + len < out_cap) { for (int t = 0; t < len; ++t) out[oi++] = tmp[t]; }
        len = utf8_encode_one(alph_u32[y1], tmp);
        if (oi + len < out_cap) { for (int t = 0; t < len; ++t) out[oi++] = tmp[t]; }
    }

    if (oi < out_cap) out[oi] = 0; else out[out_cap - 1] = 0;

    free(alph_u32);
    free(enc_u32);
    return out;
}


static const char* pieceHill(const char* alph, const char* encText, const char* keyCSV) {
    if (!alph || !encText || !keyCSV) return NULL;
    int K[4];
    if (parse_frag(keyCSV, K, 4) != 4) return NULL;

    int alph_bytes = slen(alph);
    uint32_t* alph_u32 = (uint32_t*)malloc(sizeof(uint32_t) * (alph_bytes + 1));
    if (!alph_u32) return NULL;
    int m = utf8_to_u32(alph, alph_u32, alph_bytes + 1);
    free(alph_u32);
    if (m <= 0) return NULL;

    int Kmod[4];
    for (int i = 0; i < 4; ++i) Kmod[i] = mod(K[i], m);

    int invK[4];
    if (!inv2x2mod(Kmod, m, invK)) return NULL;

    return hillDecrypt2x2(alph, encText, invK);
}


static const char* bruteHill(const char* alph, const char* encText, const char* knownFrag) {
    if (!alph || !encText) return NULL;
    int alph_bytes = slen(alph);
    uint32_t* alph_u32 = (uint32_t*)malloc(sizeof(uint32_t) * (alph_bytes + 1));
    if (!alph_u32) return NULL;
    int m = utf8_to_u32(alph, alph_u32, alph_bytes + 1);
    free(alph_u32);
    if (m <= 0) return NULL;

    size_t bestLen = 0;
    char* best = NULL;

    int writeToFile = (knownFrag == NULL) || (knownFrag && *knownFrag == 0);

    char fname[64];
    FILE* f = NULL;
    if (writeToFile) {
        strcpy(fname, "hill_");
        append_time_txt(fname, sizeof(fname));
        f = fopen(fname, "w");
        if (!f) f = NULL;
    }

    for (int a = 0; a < m; ++a) {
        for (int b = 0; b < m; ++b) {
            for (int c = 0; c < m; ++c) {
                for (int d = 0; d < m; ++d) {
                    int K[4] = { a, b, c, d };
                    int invK[4];
                    if (!inv2x2mod(K, m, invK)) continue;

                    char* dec = hillDecrypt2x2(alph, encText, invK);
                    if (!dec) continue;

                    int ok = 1;
                    if (knownFrag && *knownFrag) ok = strstr(dec, knownFrag) != NULL;

                    if (ok) {
                        if (writeToFile) {
                            if (f) {
                                fprintf(f, "%d,%d,%d,%d: %s\n", a, b, c, d, dec);
                                fflush(f);
                            }
                            free(dec);
                        } else {
                            if (!knownFrag || !*knownFrag) {
                                free(dec);
                                if (f) fclose(f);
                                return NULL;
                            }
                            size_t dl = slen(dec);
                            if (!best || dl > bestLen) {
                                if (best) free(best);
                                best = dec;
                                bestLen = dl;
                            } else {
                                free(dec);
                            }
                        }
                    } else {
                        free(dec);
                    }
                }
            }
        }
    }

    if (f) fclose(f);

    if (writeToFile) {
        if (f) return strdup(fname);
        return NULL;
    }

    return best;
}

const char* hillEntry(const char* alph, const char* encText, const char* frag) {
    if (!frag || !*frag) return bruteHill(alph, encText, frag);
    return pieceHill(alph, encText, frag);
}