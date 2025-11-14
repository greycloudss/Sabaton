#include "block.h"
#include "../enhancements/lith/lithuanian.h"


static unsigned char g_fflag = 2;

static inline uint8_t Ffunc(uint8_t r, uint8_t k) {
    switch (g_fflag) {
    case 0:
        return (uint8_t)(((r | k) ^ (((r >> 4) &k))) &0xFF);
    case 1:
        return (uint8_t)(((r ^ k) &(((k >> 4) | r))) &0xFF);
    case 2:
        return (uint8_t)(((r | k) ^ (((k >> 4) &r))) &0xFF);
    case 3:
        return (uint8_t)(((r ^ k) &(((r >> 4) | k))) &0xFF);
    default:
        return (uint8_t)(((r | k) ^ (((k >> 4) &r))) &0xFF);
    }
}

static inline void enc_block_3(uint8_t inL, uint8_t inR,
    const int* keys, uint8_t* outL, uint8_t* outR) {
    uint8_t L = inL, R = inR;
    for (int i = 0; i < 3; i++) {
        uint8_t nL = R;
        uint8_t nR = (uint8_t)(L ^ Ffunc(R, (uint8_t) keys[i]));
        L = nL;
        R = nR;
    }
    uint8_t t = L;
    L = R;
    R = t;
   *outL = L;
   *outR = R;
}

static inline void dec_block_3(uint8_t inL, uint8_t inR,
    const int* keys,
        uint8_t* outL, uint8_t* outR) {
    uint8_t L = inL, R = inR;
    {
        uint8_t t = L;
        L = R;
        R = t;
    }
    for (int i = 2; i >= 0; i--) {
        uint8_t nL = (uint8_t)(R ^ Ffunc(L, (uint8_t) keys[i])), nR = L;
        L = nL;
        R = nR;
    }
   * outL = L;* outR = R;
}

static uint8_t* read_cipher(const char* encText, int* bigN) {
    int* encInt = parse_frag_array(encText, bigN);
    if (!encInt ||* bigN <= 0) {
        if (encInt) free(encInt);
        return NULL;
    }
    uint8_t* c = (uint8_t* ) malloc((size_t)* bigN);
    if (!c) {
        free(encInt);
        return NULL;
    }
    for (int i = 0; i <* bigN; i++) {
        int v = encInt[i];
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        c[i] = (uint8_t) v;
    }
    free(encInt);
    return c;
}

static void ascii_line(FILE* f, const uint8_t* buf, size_t len) {
    if (!f || !buf || !len) return;
    uint8_t* tmp = (uint8_t* ) malloc(len);
    if (!tmp) return;
    size_t j = 0;
    int sp = 1;
    for (size_t i = 0; i < len; i++) {
        uint8_t v = buf[i];
        if (v >= 'a' && v <= 'z') v = (uint8_t)(v - 32);
        if ((v >= 'A' && v <= 'Z') || v == ' ') {
            if (v == ' ') {
                if (!sp) {
                    tmp[j++] = ' ';
                    sp = 1;
                }
            } else {
                tmp[j++] = v;
                sp = 0;
            }
        } else {
            if (!sp) {
                tmp[j++] = ' ';
                sp = 1;
            }
        }
    }
    while (j > 0 && tmp[j - 1] == ' ') j--;
    if (j) fwrite(tmp, 1, j, f);
    fputc('\n', f);
    free(tmp);
}

static uint8_t* do_ecb(const uint8_t* c, int bigN,
    const int* keys, size_t* out_len) {
    if (!c || bigN <= 0 || (bigN &1)) return NULL;
    uint8_t* p = (uint8_t* ) malloc((size_t) bigN);
    if (!p) return NULL;
    size_t pos = 0;
    for (int i = 0; i < bigN; i += 2) {
        uint8_t L, R;
        dec_block_3(c[i], c[i + 1], keys, &L, &R);
        p[pos++] = L;
        p[pos++] = R;
    }
   * out_len = pos;
    return p;
}

static uint8_t* do_cbc_from_stream(const uint8_t* c, int bigN,
    const int* keys, size_t* out_len) {
    if (!c || bigN <= 3 || (bigN &1)) return NULL;
    uint8_t ivL = c[0], ivR = c[1], prevL = ivL, prevR = ivR;
    uint8_t* p = (uint8_t* ) malloc((size_t)(bigN - 2));
    if (!p) return NULL;
    size_t pos = 0;
    for (int i = 2; i < bigN; i += 2) {
        uint8_t dL, dR;
        dec_block_3(c[i], c[i + 1], keys, &dL, &dR);
        p[pos++] = (uint8_t)(dL ^ prevL);
        p[pos++] = (uint8_t)(dR ^ prevR);
        prevL = c[i];
        prevR = c[i + 1];
    }
   * out_len = pos;
    return p;
}

static uint8_t* do_cfb_from_stream(const uint8_t* c, int bigN,
    const int* keys, size_t* out_len) {
    if (!c || bigN <= 3 || (bigN &1)) return NULL;
    uint8_t sL = c[0], sR = c[1];
    uint8_t* p = (uint8_t* ) malloc((size_t)(bigN - 2));
    if (!p) return NULL;
    size_t pos = 0;
    for (int i = 2; i < bigN; i += 2) {
        uint8_t keL, keR;
        enc_block_3(sL, sR, keys, &keL, &keR);
        p[pos++] = (uint8_t)(c[i] ^ keL);
        p[pos++] = (uint8_t)(c[i + 1] ^ keR);
        sL = c[i];
        sR = c[i + 1];
    }
   * out_len = pos;
    return p;
}

static uint8_t* do_crt(const uint8_t* c, int bigN,
    const int* keys, size_t* out_len) {
    if (!c || bigN <= 0 || (bigN &1)) return NULL;
    uint8_t* p = (uint8_t* ) malloc((size_t) bigN);
    if (!p) return NULL;
    size_t pos = 0;
    for (int i = 0; i < bigN; i += 2) {
        uint8_t a = (uint8_t) Ffunc((uint8_t)(i / 2), (uint8_t) keys[0]), keL, keR;
        enc_block_3(a, a, keys, &keL, &keR);
        p[pos++] = (uint8_t)(c[i] ^ keL);
        p[pos++] = (uint8_t)(c[i + 1] ^ keR);
    }
   * out_len = pos;
    return p;
}

static void write_all_modes(FILE* fptr,
    const uint8_t* c, int bigN,
        const int* keys) {
    size_t n = 0;
    uint8_t* a = NULL;
    a = do_ecb(c, bigN, keys, &n);
    if (a) {
        ascii_line(fptr, a, n);
        free(a);
    }
    a = do_cbc_from_stream(c, bigN, keys, &n);
    if (a) {
        ascii_line(fptr, a, n);
        free(a);
    }
    a = do_cfb_from_stream(c, bigN, keys, &n);
    if (a) {
        ascii_line(fptr, a, n);
        free(a);
    }
    a = do_crt(c, bigN, keys, &n);
    if (a) {
        ascii_line(fptr, a, n);
        free(a);
    }
}

static
const char* run_once(const char* encText,
    const int* keys, size_t nkeys) {
    int N = 0;
    uint8_t* c = read_cipher(encText, &N);
    if (!c) return "";
    static char fname[128];
    const char* base = "block-";
    int p = 0;
    while (base[p] && p < (int) sizeof(fname) - 1) {
        fname[p] = base[p];
        p++;
    }
    fname[p] = '\0';
    if (!append_time_txt(fname, (int) sizeof fname)) {
        const char* fb = "unknown.txt";
        int i = 0;
        while (fb[i] && p + i < (int) sizeof(fname) - 1) {
            fname[p + i] = fb[i];
            i++;
        }
        fname[p + i] = '\0';
    }
    FILE* fptr = fopen(fname, "wb");
    if (!fptr) {
        free(c);
        return "";
    }
    if (nkeys >= 3) {
        write_all_modes(fptr, c, N, keys);
    }
    fclose(fptr);
    free(c);
    return recognEntry(fname);
}

static void rec_gen_keys(const char* encText, int* keys, size_t nkeys, size_t idx) {
    if (idx == nkeys) {
        run_once(encText, keys, nkeys);
        return;
    }
    if (keys[idx] == -1) {
        for (int v = 0; v < 256; v++) {
            keys[idx] = v;
            rec_gen_keys(encText, keys, nkeys, idx + 1);
        }
        keys[idx] = -1;
    } else rec_gen_keys(encText, keys, nkeys, idx + 1);
}

const char* blockEntry(const char* encText,
    const char* frag, char flag) {
    g_fflag = (unsigned char) flag;
    int n = 0;
    int* keys = parse_frag_array(frag, &n);
    if (!keys || n <= 0) {
        int defk[3] = { 217, 108, 80};
        return run_once(encText, defk, 3);
    }
    int has_missing = 0;
    for (int i = 0; i < n; i++) {
        if (keys[i] == -1) {
            has_missing = 1;
            break;
        }
    }
    if (has_missing) {
        rec_gen_keys(encText, keys, (size_t) n, 0);
        free(keys);
        return "";
    }
    const char* res = run_once(encText, keys, (size_t) n);
    free(keys);
    return res ? res : "";
}