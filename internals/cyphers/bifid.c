#include "bifid.h"

static char* build_key_alph(const char* alph, const char* keyword) {
    if (!alph) return NULL;
    int m = (int)strlen(alph);
    char* key = malloc(m + 1);
    if (!key) return NULL;
    int used_len = 0;
    int used_mask_cap = (m + 7) / 8;
    unsigned char* used = calloc(used_mask_cap, 1);
    if (!used) { free(key); return NULL; }

    auto_mark:
    (void)0;

    for (const char* p = keyword ? keyword : ""; *p; ++p) {
        const char* q = strchr(alph, *p);
        if (!q) continue;
        int idx = (int)(q - alph);
        int byte = idx >> 3;
        int bit = idx & 7;
        if (!(used[byte] & (1 << bit))) {
            used[byte] |= (1 << bit);
            key[used_len++] = *p;
            if (used_len == m) break;
        }
    }
    for (int i = 0; i < m && used_len < m; ++i) {
        int byte = i >> 3;
        int bit = i & 7;
        if (!(used[byte] & (1 << bit))) {
            used[byte] |= (1 << bit);
            key[used_len++] = alph[i];
        }
    }
    key[used_len] = '\0';
    free(used);
    return key;
}

static int idx_in_key(const char* key_alph, int n, char ch) {
    if (!key_alph) return -1;
    for (int i = 0; i < n * n; ++i) {
        if (key_alph[i] == ch) return i;
    }
    return -1;
}

static int filter_and_map(const char* alph, const char* text, int* out_map, char* out_chars, int maxlen) {
    if (!alph || !text || !out_map || !out_chars) return 0;
    int m = (int)strlen(alph);
    int j = 0;
    for (const unsigned char* p = (const unsigned char*)text; *p && j < maxlen; ++p) {
        char ch = (char)*p;
        const char* q = strchr(alph, ch);
        if (q) {
            out_map[j] = (int)(q - alph);
            out_chars[j] = ch;
        } else {
            out_map[j] = -1;
            out_chars[j] = ch;
        }
        j++;
    }
    return j;
}

static void build_alph_to_key_map(const char* alph, const char* key_alph, int n, int* map) {
    int m = (int)strlen(alph);
    for (int i = 0; i < m; ++i) {
        char ch = alph[i];
        int pos = -1;
        for (int k = 0; k < n * n; ++k) if (key_alph[k] == ch) { pos = k; break; }
        map[i] = pos;
    }
}

static char* bifid_decrypt_block(const int* filtered_idx, int len, const char* key_alph, int n) {
    if (!filtered_idx || len <= 0 || !key_alph) return NULL;
    int L = len;
    int *rows = malloc(sizeof(int) * L);
    int *cols = malloc(sizeof(int) * L);
    if (!rows || !cols) { free(rows); free(cols); return NULL; }
    for (int i = 0; i < L; ++i) {
        int pos = filtered_idx[i];
        rows[i] = pos / n;
        cols[i] = pos % n;
    }

    int *S = malloc(sizeof(int) * (2 * L));
    if (!S) { free(rows); free(cols); free(S); return NULL; }
    for (int i = 0; i < L; ++i) {
        S[2 * i] = rows[i];
        S[2 * i + 1] = cols[i];
    }

    int *R = malloc(sizeof(int) * L);
    int *C = malloc(sizeof(int) * L);
    if (!R || !C) { free(rows); free(cols); free(S); free(R); free(C); return NULL; }
    for (int i = 0; i < L; ++i) R[i] = S[i];
    for (int i = 0; i < L; ++i) C[i] = S[L + i];
    char* out = malloc((size_t)L + 1);
    if (!out) { free(rows); free(cols); free(S); free(R); free(C); return NULL; }
    for (int i = 0; i < L; ++i) {
        int rr = R[i];
        int cc = C[i];
        int pos = rr * n + cc;
        out[i] = key_alph[pos];
    }
    out[L] = '\0';

    free(rows); free(cols); free(S); free(R); free(C);
    return out;
}

const char* bifidEntry(const char* alph, const char* encText, const char* frag) {
    if (!encText) return strdup("[no input]");
    if (!alph || strlen(alph) == 0) return strdup("[no alphabet]");
    int m = (int)strlen(alph);
    int n = 0;
    for (int s = 1; s * s <= m; ++s) if (s * s == m) n = s;
    if (n == 0) return strdup("[alphabet length not a perfect square]");

    char* keyword = NULL;
    int period = 0;
    if (frag && frag[0]) {
        const char* semi = strchr(frag, ';');
        if (semi) {
            int kwlen = (int)(semi - frag);
            keyword = malloc((size_t)kwlen + 1);
            if (keyword) { memcpy(keyword, frag, kwlen); keyword[kwlen] = '\0'; }
            char* after = (char*)(semi + 1);
            long p = strtol(after, NULL, 10);
            if (p > 0) period = (int)p;
        } else {
            char* endptr = NULL;
            long v = strtol(frag, &endptr, 10);
            if (endptr != frag && *endptr == '\0' && v > 0) {
                period = (int)v;
            } else {
                keyword = strdup(frag);
            }
        }
    }

    char* key_alph = build_key_alph(alph, keyword);
    if (keyword) free(keyword);
    if (!key_alph) return strdup("[oom]");

    int *alph_to_key = malloc(sizeof(int) * m);
    if (!alph_to_key) { free(key_alph); return strdup("[oom]"); }
    for (int i = 0; i < m; ++i) {
        char ch = alph[i];
        int pos = -1;
        for (int k = 0; k < n * n; ++k) if (key_alph[k] == ch) { pos = k; break; }
        alph_to_key[i] = pos;
    }

    int cap = (int)strlen(encText) + 4;
    int* map_idx = malloc(sizeof(int) * cap);
    char* chars = malloc((size_t)cap);
    if (!map_idx || !chars) { free(key_alph); free(alph_to_key); free(map_idx); free(chars); return strdup("[oom]"); }
    int text_len = 0;
    for (const unsigned char* p = (const unsigned char*)encText; *p && text_len < cap; ++p) {
        char ch = (char)*p;
        const char* posptr = strchr(alph, ch);
        if (posptr) {
            int aidx = (int)(posptr - alph);
            map_idx[text_len] = alph_to_key[aidx];
            chars[text_len] = ch;
        } else {
            map_idx[text_len] = -1;
            chars[text_len] = ch;
        }
        ++text_len;
    }

    free(alph_to_key);

    if (text_len == 0) { free(key_alph); free(map_idx); free(chars); return strdup("[no text]"); }

    if (period <= 0) period = text_len;

    char* out = malloc((size_t)text_len + 1);
    if (!out) { free(key_alph); free(map_idx); free(chars); return strdup("[oom]"); }

    int pos = 0;

    int i = 0;
    while (i < text_len) {
        int block_idx_cap = period;
        int *block_keypos = malloc(sizeof(int) * block_idx_cap);
        int *block_positions = malloc(sizeof(int) * block_idx_cap);
        if (!block_keypos || !block_positions) {
            free(key_alph); free(map_idx); free(chars); free(out); free(block_keypos); free(block_positions);
            return strdup("[oom]");
        }
        int blk_len = 0;
        while (i < text_len && blk_len < period) {
            if (map_idx[i] >= 0) {
                block_keypos[blk_len] = map_idx[i];
                block_positions[blk_len] = i;
                blk_len++;
            }
            i++;
        }
        if (blk_len > 0) {
            char* decrypted_block = bifid_decrypt_block(block_keypos, blk_len, key_alph, n);
            if (!decrypted_block) {
                free(block_keypos); free(block_positions);
                free(key_alph); free(map_idx); free(chars); free(out);
                return strdup("[decryption failed]");
            }
            for (int k = 0; k < blk_len; ++k) {
                out[block_positions[k]] = decrypted_block[k];
            }
            free(decrypted_block);
        }
        free(block_keypos);
        free(block_positions);
    }

    for (int j = 0; j < text_len; ++j) {
        if (map_idx[j] == -1) out[j] = chars[j];
    }
    out[text_len] = '\0';

    free(key_alph);
    free(map_idx);
    free(chars);

    static char* static_out = NULL;
    if (static_out) { free(static_out); static_out = NULL; }
    static_out = strdup(out);
    free(out);
    return static_out;
}
