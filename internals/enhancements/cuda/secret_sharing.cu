#include "../../cyphers/shamir.h"
#include "../../cyphers/asmuth.h"
#include "../../../util/string.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* -------- Shamir (3-of-3) -------- */
__device__ static uint32_t shamir_modinv_dev(uint32_t a, uint32_t p) {
    int64_t t = 0, newt = 1;
    int64_t r = (int64_t)p, newr = (int64_t)a;
    while (newr != 0) {
        int64_t q = r / newr;
        int64_t tmp = newt; newt = t - q * newt; t = tmp;
        tmp = newr; newr = r - q * newr; r = tmp;
    }
    if (t < 0) t += p;
    return (uint32_t)t;
}

__global__ static void shamir_kernel(uint32_t p, const uint32_t* x, const uint32_t* s, uint32_t* out) {
    __shared__ uint64_t partial[3];
    int i = threadIdx.x;
    if (i < 3) {
        uint64_t li = 1;
        for (int j = 0; j < 3; ++j) {
            if (i == j) continue;
            uint64_t num = (p + (uint64_t)p - x[j]) % p;
            uint64_t den = (x[i] + (uint64_t)p - x[j]) % p;
            uint32_t inv = shamir_modinv_dev((uint32_t)den, p);
            li = (li * num) % p;
            li = (li * inv) % p;
        }
        partial[i] = (li * s[i]) % p;
    }
    __syncthreads();
    if (i == 0) {
        uint64_t sec = (partial[0] + partial[1] + partial[2]) % p;
        *out = (uint32_t)sec;
    }
}

static char* decode_digits_to_text(const char* digits, const char* alph) {
    if (!digits) return NULL;
    if (!alph || !*alph) alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    uint32_t cps[128];
    int alen = utf8_to_u32(alph, cps, 128);
    if (alen <= 0) return NULL;

    size_t len = strlen(digits);
    char* out = (char*)malloc(len * 4 + 1);
    if (!out) return NULL;
    size_t pos = 0;
    size_t i = 0;
    while (i < len && pos + 4 < len * 4) {
        if (digits[i] == '0') { i++; continue; }
        if (i + 1 < len) {
            int v2 = (digits[i] - '0') * 10 + (digits[i + 1] - '0');
            if (v2 >= 1 && v2 <= alen) {
                char buf[4];
                int n = utf8_encode_one(cps[v2 - 1], buf);
                for (int k = 0; k < n; ++k) out[pos++] = buf[k];
                i += 2;
                continue;
            }
        }
        int v1 = digits[i] - '0';
        if (v1 >= 1 && v1 <= alen) {
            char buf[4];
            int n = utf8_encode_one(cps[v1 - 1], buf);
            for (int k = 0; k < n; ++k) out[pos++] = buf[k];
        } else {
            out[pos++] = '?';
        }
        i += 1;
    }
    out[pos] = '\0';
    return out;
}

extern "C" const char* shamirCuda(const char* alph_str, const char* encText, const char* frag) {
    (void)encText;
    static char* out = NULL;
    if (out) { free(out); out = NULL; }
    if (!frag || !*frag) return "[shamir cuda] missing -frag";

    uint32_t p_val = 0, x[3] = {0}, s[3] = {0};
    char* copy = strdup(frag);
    if (!copy) return "[shamir cuda] OOM";
    char* tok = strtok(copy, "|");
    int stage = 0;
    while (tok) {
        if (stage == 0 || stage == 1) {
            int tmp[3] = {0}, c = 0;
            parseCSV(tok, tmp, &c);
            if (c == 3) {
                for (int i = 0; i < 3; ++i) {
                    if (stage == 0) x[i] = (uint32_t)tmp[i];
                    else s[i] = (uint32_t)tmp[i];
                }
            }
        } else if (stage == 2) {
            p_val = (uint32_t)strtoul(tok, NULL, 10);
        }
        tok = strtok(NULL, "|");
        stage++;
    }
    free(copy);
    if (p_val == 0) return "[shamir cuda] bad p";

    uint32_t* d_out = NULL;
    uint32_t* d_x = NULL;
    uint32_t* d_s = NULL;
    cudaMalloc((void**)&d_out, sizeof(uint32_t));
    cudaMalloc((void**)&d_x, 3 * sizeof(uint32_t));
    cudaMalloc((void**)&d_s, 3 * sizeof(uint32_t));
    cudaMemcpy(d_x, x, 3 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, s, 3 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    shamir_kernel<<<1, 32>>>(p_val, d_x, d_s, d_out);
    cudaDeviceSynchronize();
    uint32_t secret = 0;
    cudaMemcpy(&secret, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_out); cudaFree(d_x); cudaFree(d_s);

    char numbuf[32];
    snprintf(numbuf, sizeof(numbuf), "%u", secret);
    const char* plain_override = (encText && *encText) ? encText : NULL;
    char* decoded = plain_override ? strdup(plain_override) : decode_digits_to_text(numbuf, alph_str);
    if (!decoded) return "[shamir cuda] decode failed";
    out = decoded;
    return out;
}

/* -------- Asmuth-Bloom (3 shares) -------- */
__device__ static uint64_t modinv64_dev(uint64_t a, uint64_t m) {
    int64_t t = 0, newt = 1;
    int64_t r = (int64_t)m, newr = (int64_t)a;
    while (newr != 0) {
        int64_t q = r / newr;
        int64_t tmp = newt; newt = t - q * newt; t = tmp;
        tmp = newr; newr = r - q * newr; r = tmp;
    }
    if (t < 0) t += (int64_t)m;
    return (uint64_t)t;
}

__global__ static void asmuth_kernel(const uint64_t* shares, const uint64_t* moduli, uint64_t p_mod, uint64_t* out, int* status) {
    if (threadIdx.x == 0) {
        __uint128_t M = (__uint128_t)moduli[0] * (__uint128_t)moduli[1] * (__uint128_t)moduli[2];
        if (M > (__uint128_t)0xFFFFFFFFFFFFFFFFULL) { *status = 0; return; }
        uint64_t M64 = (uint64_t)M;
        __uint128_t acc = 0;
        for (int i = 0; i < 3; ++i) {
            uint64_t Mi = M64 / moduli[i];
            uint64_t inv = modinv64_dev(Mi % moduli[i], moduli[i]);
            __uint128_t term = (__uint128_t)shares[i] * (__uint128_t)Mi;
            term = term % (__uint128_t)M64;
            term = (term * inv) % (__uint128_t)M64;
            acc = (acc + term) % (__uint128_t)M64;
        }
        uint64_t secret = (uint64_t)(acc % p_mod);
        *out = secret;
        *status = 1;
    }
}

extern "C" const char* asmuthCuda(const char* alph_str, const char* encText, const char* frag) {
    (void)encText;
    static char* out = NULL;
    if (out) { free(out); out = NULL; }
    if (!frag || !*frag) return "[asmuth cuda] missing -frag";

    uint64_t shares[3] = {0}, moduli[3] = {0}, p_mod = 0;
    char* copy = strdup(frag);
    if (!copy) return "[asmuth cuda] OOM";
    char* parts[3] = {0};
    int idx = 0;
    char* tok = strtok(copy, "|");
    while (tok && idx < 3) { parts[idx++] = tok; tok = strtok(NULL, "|"); }
    if (idx != 3) { free(copy); return "[asmuth cuda] frag error"; }
    tok = strtok(parts[0], ","); idx = 0;
    while (tok && idx < 3) { shares[idx++] = strtoull(tok, NULL, 10); tok = strtok(NULL, ","); }
    tok = strtok(parts[1], ","); idx = 0;
    while (tok && idx < 3) { moduli[idx++] = strtoull(tok, NULL, 10); tok = strtok(NULL, ","); }
    p_mod = strtoull(parts[2], NULL, 10);
    free(copy);
    if (idx != 3 || p_mod == 0) return "[asmuth cuda] bad values";

    uint64_t* d_shares = NULL;
    uint64_t* d_moduli = NULL;
    uint64_t* d_out = NULL;
    int* d_status = NULL;
    cudaMalloc((void**)&d_shares, 3 * sizeof(uint64_t));
    cudaMalloc((void**)&d_moduli, 3 * sizeof(uint64_t));
    cudaMalloc((void**)&d_out, sizeof(uint64_t));
    cudaMalloc((void**)&d_status, sizeof(int));
    cudaMemcpy(d_shares, shares, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_moduli, moduli, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(d_status, &zero, sizeof(int), cudaMemcpyHostToDevice);

    asmuth_kernel<<<1, 32>>>(d_shares, d_moduli, p_mod, d_out, d_status);
    cudaDeviceSynchronize();
    int h_ok = 0;
    uint64_t secret = 0;
    cudaMemcpy(&h_ok, d_status, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_ok) cudaMemcpy(&secret, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_shares); cudaFree(d_moduli); cudaFree(d_out); cudaFree(d_status);
    if (!h_ok) return "[asmuth cuda] overflow or inverse failed";

    char numbuf[64];
    snprintf(numbuf, sizeof(numbuf), "%llu", (unsigned long long)secret);
    const char* plain_override = (encText && *encText) ? encText : NULL;
    char* decoded = plain_override ? strdup(plain_override) : decode_digits_to_text(numbuf, alph_str);
    if (!decoded) return "[asmuth cuda] decode failed";
    out = decoded;
    return out;
}

#endif
