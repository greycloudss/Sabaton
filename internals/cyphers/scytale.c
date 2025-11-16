#include "scytale.h"

static char* decryptBrute(const char* encrypted_text, int rows, int k) {
    if (!encrypted_text || rows <= 0) return NULL;

    int cap = (int)strlen(encrypted_text) + 1;
    uint32_t* in = malloc(sizeof(uint32_t) * cap);
    if (!in) return NULL;
    int n = utf8_to_u32(encrypted_text, in, cap);
    if (n == 0) {
        free(in);
        return strdup("");
    }

    int cols = (int)((n + (size_t)rows - 1u) / (size_t)rows);
    int short_cols = cols * rows - n;

    uint32_t* out = malloc(sizeof(uint32_t) * (n + 1));
    if (!out) {
        free(in);
        return NULL;
    }

    int out_i = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int col_len = (j < cols - short_cols) ? rows : rows - 1;
            if (i < col_len) {
                int pos = 0;
                for (int x = 0; x < j; ++x) pos += (x < cols - short_cols) ? rows : rows - 1;
                pos += i;
                if (pos < n) out[out_i++] = in[pos];
            }
        }
    }
    out[out_i] = 0;

    int out_buf_cap = (out_i + 1) * 4 + 8;
    char* utf8 = malloc(out_buf_cap);
    if (!utf8) {
        free(in);
        free(out);
        return NULL;
    }
    u32_to_utf8(out, out_i, utf8, out_buf_cap);

    free(in);
    free(out);
    return utf8;
}

static char* decryptK(const char* encrypted_text, int num) {
    int k = num;
    return decryptBrute(encrypted_text, k, k);
}

static const char* decryptEntry(const char* encrypted_text, int num) {
    static char* output = NULL;
    if (output) {
        free(output);
        output = NULL;
    }

    if (!encrypted_text) return strdup("no input");

    if (num != 0) {
        char* cand = decryptK(encrypted_text, num);
        if (cand) {
            output = strdup(cand);
            free(cand);
        }
    } else {
        for (int k = 1; k < 30; ++k) {
            char* cand = decryptBrute(encrypted_text, k, k);
            if (!output && cand) output = strdup(cand);
            free(cand);
        }
    }

    if (!output) output = strdup("no output");
    return output;
}

const char* scytaleEntry(const char* alph, const char* encText, const char* frag) {
    (void)alph;
    if (!encText) return strdup("no input");

    if (frag && frag[0]) {
        char* endptr = NULL;
        long val = strtol(frag, &endptr, 10);
        if (endptr != frag && *endptr == '\0' && val > 0) {
            return decryptEntry(encText, (int)val);
        }
    }

    static char fname[128];
    const char* base = "scytaleBrute-";
    int p = 0;

    while (base[p] && p < (int)sizeof(fname) - 1) {
        fname[p] = base[p];
        ++p;
    }
    
    fname[p] = '\0';
    if (!append_time_txt(fname, (int)sizeof fname)) {
        const char* fb = "unknown.txt";
        int i = 0;
        while (fb[i] && p + i < (int)sizeof(fname) - 1) {
            fname[p + i] = fb[i];
            ++i;
        }
        fname[p + i] = '\0';
    }
    FILE* f = fopen(fname, "w");
    if (!f) return strdup("error opening output file");

    for (int k = 1; k < 30; ++k) {
        char* cand = decryptBrute(encText, k, k);
        if (!cand) continue;
        fprintf(f, "[k %d] %s\n", k, cand);
        free(cand);
    }
    fclose(f);

    static char* static_fname = NULL;
    if (static_fname) {
        free(static_fname);
        static_fname = NULL;
    }
    static_fname = strdup(fname);
    return static_fname;
}
