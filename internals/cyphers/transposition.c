#include "transposition.h"

#define MAX_PERM_COLS 9

static void swap_int(int* a, int* b) { int t = *a; *a = *b; *b = t; }

static char* decrypt_with_perm_u32(const uint32_t* text_cp, int n, int cols, const int* perm) {
    if (!text_cp || n <= 0 || cols <= 0 || !perm) return NULL;
    int rows = (n + cols - 1) / cols;
    int short_cols = cols * rows - n;

    int* col_len = malloc(sizeof(int) * cols);
    if (!col_len) return NULL;
    for (int c = 0; c < cols; ++c)
        col_len[c] = (c < cols - short_cols) ? rows : (rows - 1);

    uint32_t** cols_buf = malloc(sizeof(uint32_t*) * cols);
    if (!cols_buf) { free(col_len); return NULL; }
    for (int c = 0; c < cols; ++c) {
        cols_buf[c] = malloc(sizeof(uint32_t) * (col_len[c] > 0 ? col_len[c] : 1));
        if (!cols_buf[c]) {
            for (int t = 0; t < c; ++t) free(cols_buf[t]);
            free(cols_buf); free(col_len); return NULL;
        }
    }

    int read_pos = 0;
    for (int p = 0; p < cols; ++p) {
        int col_index = perm[p];
        int L = col_len[col_index];
        for (int i = 0; i < L; ++i) cols_buf[col_index][i] = (read_pos < n) ? text_cp[read_pos++] : 0;
    }

    uint32_t* out_cp = malloc(sizeof(uint32_t) * (n + 1));
    if (!out_cp) { for (int c = 0; c < cols; ++c) free(cols_buf[c]); free(cols_buf); free(col_len); return NULL; }
    int idx = 0;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            if (r < col_len[c]) out_cp[idx++] = cols_buf[c][r];
    out_cp[idx] = 0;

    int out_buf_cap = (idx + 1) * 4 + 8;
    char* out_utf8 = malloc(out_buf_cap);
    if (!out_utf8) { free(out_cp); for (int c = 0; c < cols; ++c) free(cols_buf[c]); free(cols_buf); free(col_len); return NULL; }
    u32_to_utf8(out_cp, idx, out_utf8, out_buf_cap);

    free(out_cp);
    for (int c = 0; c < cols; ++c) free(cols_buf[c]);
    free(cols_buf);
    free(col_len);
    return out_utf8;
}

static int permute_and_write(const uint32_t* text_cp, int n, int cols, FILE* f) {
    int *a = malloc(sizeof(int) * cols);
    int *c = calloc(cols, sizeof(int));
    if (!a || !c) { free(a); free(c); return -1; }
    for (int i = 0; i < cols; ++i) a[i] = i;

    char* s = decrypt_with_perm_u32(text_cp, n, cols, a);
    if (s) { fprintf(f, "["); for (int i = 0; i < cols; ++i) { if (i) fputc(',', f); fprintf(f, "%d", a[i]); } fprintf(f, "] %s\n", s); free(s); }

    int i = 0;
    while (i < cols) {
        if (c[i] < i) {
            if ((i & 1) == 0) swap_int(&a[0], &a[i]); else swap_int(&a[c[i]], &a[i]);
            s = decrypt_with_perm_u32(text_cp, n, cols, a);
            if (s) { fprintf(f, "["); for (int k = 0; k < cols; ++k) { if (k) fputc(',', f); fprintf(f, "%d", a[k]); } fprintf(f, "] %s\n", s); free(s); }
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i++;
        }
    }

    free(a); free(c);
    return 0;
}

static char* decrypt_with_key_u32(const uint32_t* text_cp, int n, const uint32_t* key_cp, int key_len) {
    if (!text_cp || n <= 0 || !key_cp || key_len <= 0) return NULL;
    int cols = key_len;
    int rows = (n + cols - 1) / cols;
    int short_cols = cols * rows - n;

    int* col_len = malloc(sizeof(int) * cols);
    if (!col_len) return NULL;
    for (int c = 0; c < cols; ++c) col_len[c] = (c < cols - short_cols) ? rows : (rows - 1);

    int* order = malloc(sizeof(int) * cols);
    if (!order) { free(col_len); return NULL; }
    for (int i = 0; i < cols; ++i) order[i] = i;
    for (int i = 0; i < cols - 1; ++i) {
        for (int j = i + 1; j < cols; ++j) {
            if (key_cp[order[i]] > key_cp[order[j]] || (key_cp[order[i]] == key_cp[order[j]] && order[i] > order[j])) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
        }
    }

    uint32_t** cols_buf = malloc(sizeof(uint32_t*) * cols);
    if (!cols_buf) { free(order); free(col_len); return NULL; }
    for (int c = 0; c < cols; ++c) {
        cols_buf[c] = malloc(sizeof(uint32_t) * (col_len[c] > 0 ? col_len[c] : 1));
        if (!cols_buf[c]) { for (int t = 0; t < c; ++t) free(cols_buf[t]); free(cols_buf); free(order); free(col_len); return NULL; }
    }

    int read_pos = 0;
    for (int p = 0; p < cols; ++p) {
        int col_index = order[p];
        int L = col_len[col_index];
        for (int i = 0; i < L; ++i) cols_buf[col_index][i] = (read_pos < n) ? text_cp[read_pos++] : 0;
    }

    uint32_t* out_cp = malloc(sizeof(uint32_t) * (n + 1));
    if (!out_cp) { for (int c = 0; c < cols; ++c) free(cols_buf[c]); free(cols_buf); free(order); free(col_len); return NULL; }
    int idx = 0;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            if (r < col_len[c]) out_cp[idx++] = cols_buf[c][r];
    out_cp[idx] = 0;

    int out_buf_cap = (idx + 1) * 4 + 8;
    char* out_utf8 = malloc(out_buf_cap);
    if (!out_utf8) { free(out_cp); for (int c = 0; c < cols; ++c) free(cols_buf[c]); free(cols_buf); free(order); free(col_len); return NULL; }
    u32_to_utf8(out_cp, idx, out_utf8, out_buf_cap);

    free(out_cp);
    for (int c = 0; c < cols; ++c) free(cols_buf[c]);
    free(cols_buf);
    free(order);
    free(col_len);
    return out_utf8;
}

const char* transpositionEntry(const char* alph, const char* encText, const char* frag) {
    (void)alph;
    if (!encText) return strdup("[no input]");
    if (!frag || !frag[0]) return strdup("[no key provided]");

    char* endptr = NULL;
    long maybe_cols = strtol(frag, &endptr, 10);
    if (endptr != frag && *endptr == '\0' && maybe_cols > 0) {
        int cols = (int)maybe_cols;
        if (cols > MAX_PERM_COLS) {
            char tmp[128];
            snprintf(tmp, sizeof(tmp), "[cols > %d; too many permutations]", MAX_PERM_COLS);
            return strdup(tmp);
        }

        int cap_text = (int)strlen(encText) + 1;
        uint32_t* text_cp = malloc(sizeof(uint32_t) * cap_text);
        if (!text_cp) return strdup("[oom]");
        int n = utf8_to_u32(encText, text_cp, cap_text);
        if (n <= 0) { free(text_cp); return strdup("[utf8 decode error]"); }

        int rows = (n + cols - 1) / cols;
        time_t t = time(NULL);
        char fname[128];
        if (t != (time_t)-1) snprintf(fname, sizeof(fname), "transposition_cols%d_%lld.txt", cols, (long long)t);
        else snprintf(fname, sizeof(fname), "transposition_cols%d.txt", cols);

        FILE* f = fopen(fname, "w");
        if (!f) { free(text_cp); return strdup("[error opening output file]"); }
        fprintf(f, "# brute transposition results cols=%d rows=%d permutations=%d\n", cols, rows, 0);

        int rc = permute_and_write(text_cp, n, cols, f);
        fclose(f);
        free(text_cp);
        if (rc != 0) return strdup("[error during permutation]");

        static char* static_fname = NULL;
        if (static_fname) { free(static_fname); static_fname = NULL; }
        static_fname = strdup(fname);
        return static_fname;
    }

    int cap_text = (int)strlen(encText) + 1;
    uint32_t* text_cp = malloc(sizeof(uint32_t) * cap_text);
    if (!text_cp) return strdup("[oom]");
    int n = utf8_to_u32(encText, text_cp, cap_text);
    if (n <= 0) { free(text_cp); return strdup("[utf8 decode error]"); }

    int cap_key = (int)strlen(frag) + 1;
    uint32_t* key_cp = malloc(sizeof(uint32_t) * cap_key);
    if (!key_cp) { free(text_cp); return strdup("[oom]"); }
    int key_len = utf8_to_u32(frag, key_cp, cap_key);
    if (key_len <= 0) { free(text_cp); free(key_cp); return strdup("[invalid key]"); }

    char* res = decrypt_with_key_u32(text_cp, n, key_cp, key_len);
    free(text_cp); free(key_cp);
    if (!res) return strdup("[decryption failed]");

    static char* static_out = NULL;
    if (static_out) { free(static_out); static_out = NULL; }
    static_out = strdup(res);
    free(res);
    return static_out;
}
