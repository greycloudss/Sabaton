#include "fleissner.h"
#include "../lithuanian.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#define MAX_BRUTE_N 6

static int rotate_index(int n, int r, int c) {
    return c * n + (n - 1 - r);
}

static int mask_is_valid(const unsigned char* mask, int n) {
    if (!mask) return 0;
    if ((n & 1) || (n <= 0)) return 0;
    int total = 0;
    int N = n * n;
    for (int i = 0; i < N; ++i) total += (mask[i] ? 1 : 0);
    if (total * 4 != N) return 0;
    int *seen = calloc(N, sizeof(int));
    if (!seen) return 0;
    int ok = 1;
    unsigned char *cur = malloc(N);
    if (!cur) { free(seen); return 0; }
    for (int i = 0; i < N; ++i) cur[i] = mask[i];
    for (int rot = 0; rot < 4 && ok; ++rot) {
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                int idx = r * n + c;
                if (cur[idx]) {
                    int br = r, bc = c;
                    for (int rr = 0; rr < rot; ++rr) {
                        int nbr = n - 1 - bc;
                        int nbc = br;
                        br = nbr; bc = nbc;
                    }
                    int bidx = br * n + bc;
                    if (seen[bidx]) { ok = 0; break; }
                    seen[bidx] = 1;
                }
            }
            if (!ok) break;
        }
        unsigned char *tmp = malloc(N);
        if (!tmp) { ok = 0; break; }
        for (int rr = 0; rr < n; ++rr)
            for (int cc = 0; cc < n; ++cc)
                tmp[cc * n + (n - 1 - rr)] = cur[rr * n + cc];
        memcpy(cur, tmp, N);
        free(tmp);
    }
    if (ok) for (int i = 0; i < N; ++i) if (!seen[i]) ok = 0;
    free(cur);
    free(seen);
    return ok;
}

static char* decrypt_with_mask_u32(const uint32_t* cipher_cp, int cipher_len, int n, const unsigned char* mask) {
    if (!cipher_cp || cipher_len <= 0 || n <= 0 || !mask) return NULL;
    int N = n * n;
    int blockSize = N;
    int totalBlocks = (cipher_len + blockSize - 1) / blockSize;
    uint32_t* grid = malloc(sizeof(uint32_t) * N);
    if (!grid) return NULL;
    int letters_needed = cipher_len;
    uint32_t* out_cp = malloc(sizeof(uint32_t) * (letters_needed + 1));
    if (!out_cp) { free(grid); return NULL; }
    int out_idx = 0;
    unsigned char *cur = malloc(N);
    if (!cur) { free(grid); free(out_cp); return NULL; }

    for (int b = 0; b < totalBlocks; b++) {
        for (int i = 0; i < N; ++i) {
            int pos = b * blockSize + i;
            grid[i] = (pos < cipher_len) ? cipher_cp[pos] : ' ';
        }
        memcpy(cur, mask, N);
        for (int rot = 0; rot < 4; ++rot) {
            for (int r = 0; r < n; ++r)
                for (int c = 0; c < n; ++c) {
                    int idx = r * n + c;
                    if (cur[idx] && out_idx < letters_needed)
                        out_cp[out_idx++] = grid[idx];
                }
            unsigned char *tmp = malloc(N);
            if (!tmp) { free(grid); free(out_cp); free(cur); return NULL; }
            for (int rr = 0; rr < n; ++rr)
                for (int cc = 0; cc < n; ++cc)
                    tmp[cc * n + (n - 1 - rr)] = cur[rr * n + cc];
            memcpy(cur, tmp, N);
            free(tmp);
        }
    }
    out_cp[out_idx] = 0;
    int out_buf_cap = (out_idx + 1) * 4 + 8;
    char* out_utf8 = malloc(out_buf_cap);
    if (!out_utf8) { free(grid); free(out_cp); free(cur); return NULL; }
    u32_to_utf8(out_cp, out_idx, out_utf8, out_buf_cap);
    free(grid);
    free(out_cp);
    free(cur);
    return out_utf8;
}

static void write_candidate_file(FILE* f, const unsigned char* mask, int n, const char* s) {
    if (!f) return;
    fprintf(f, "[");
    for (int i = 0; i < n * n; ++i) {
        if (i) fputc(',', f);
        fprintf(f, "%d", mask[i] ? 1 : 0);
    }
    fprintf(f, "] %s\n", s ? s : "");
}

static int brute_masks_and_write(const uint32_t* cipher_cp, int cipher_len, int n, FILE* f) {
    if (n <= 0 || n > MAX_BRUTE_N) return -1;
    int N = n * n;
    if (N > 24) return -1;
    unsigned long limit = 1UL << N;
    unsigned char *mask = malloc(N);
    if (!mask) return -1;
    for (unsigned long m = 0; m < limit; ++m) {
        for (int i = 0; i < N; ++i) mask[i] = ((m >> i) & 1) ? 1 : 0;
        //if (!mask_is_valid(mask, n)) continue;
        char* dec = decrypt_with_mask_u32(cipher_cp, cipher_len, n, mask);
        if (dec) { write_candidate_file(f, mask, n, dec); free(dec); }
    }
    free(mask);
    return 0;
}

const char* fleissnerEntry(const char* alph, const char* encText, const char* frag) {
    (void)alph;
    if (!encText) return strdup("[no input]");
    if (!frag || !frag[0]) return strdup("[no frag provided]");
    int cap_text = (int)strlen(encText) + 1;
    uint32_t* text_cp = malloc(sizeof(uint32_t) * cap_text);
    if (!text_cp) return strdup("[oom]");
    int ncp = utf8_to_u32(encText, text_cp, cap_text);
    if (ncp <= 0) { free(text_cp); return strdup("[utf8 decode error]"); }
    char *semi = strchr(frag, ';');
    if (semi) {
        int N = 0;
        char tmp[64];
        int read = 0;
        if (sscanf(frag, "%d%n", &N, &read) != 1 || N <= 0) { free(text_cp); return strdup("[invalid N]"); }
        const char* maskstr = semi + 1;
        int need = N * N;
        if ((int)strlen(maskstr) < need) { free(text_cp); return strdup("[mask too short]"); }
        unsigned char *mask = malloc(need);
        if (!mask) { free(text_cp); return strdup("[oom]"); }
        for (int i = 0; i < need; ++i) mask[i] = (maskstr[i] == '1') ? 1 : 0;
        char* dec = decrypt_with_mask_u32(text_cp, ncp, N, mask);
        free(mask);
        free(text_cp);
        if (!dec) return strdup("[decryption failed]");
        static char* static_out = NULL;
        if (static_out) { free(static_out); static_out = NULL; }
        static_out = strdup(dec);
        free(dec);
        return static_out;
    } else {
        long Nlong = strtol(frag, NULL, 10);
        if (Nlong <= 0 || Nlong > MAX_BRUTE_N) { free(text_cp); return strdup("[invalid or too big N for brute]"); }
        int N = (int)Nlong;
        static char fname[128];
        const char* base = "fleissner-";
        int p = 0;
        while (base[p] && p < (int)sizeof(fname) - 1) { fname[p] = base[p]; ++p; }
        fname[p] = '\0';
        if (!append_time_txt(fname, (int)sizeof fname)) {
            const char* fb = "unknown.txt";
            int i = 0;
            while (fb[i] && p + i < (int)sizeof(fname) - 1) { fname[p + i] = fb[i]; ++i; }
            fname[p + i] = '\0';
        }
        FILE* f = fopen(fname, "wb");
        if (!f) { free(text_cp); return strdup("[error opening output file]"); }
        int rc = brute_masks_and_write(text_cp, ncp, N, f);
        fclose(f);
        free(text_cp);
        if (rc != 0) return strdup("[brute failed or too big]");
        static char* static_fname = NULL;
        if (static_fname) { free(static_fname); static_fname = NULL; }
        static_fname = strdup(fname);
        return static_fname;
    }
}
