#include "../../../util/number.h"
#include "../../../util/string.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

__constant__ char d_allowed_bg[256];
__constant__ int  d_allowed_bg_len;

__global__ static void bg_brute_kernel(const int* cipher, int clen, uint64_t p, uint64_t q, uint64_t maxSeed,
                                       int* found, uint64_t* outSeed, char* outPlain) {
    uint64_t n = p * q;
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t seed = tid + 1; seed <= maxSeed; seed += stride) {
        if (atomicAdd(found, 0) != 0) return;
        uint64_t xi = seed % n;
        int ok = 1;
        for (int i = 0; i < clen; ++i) {
            xi = (uint64_t)((__uint128_t)xi * xi % n);
            uint8_t ks = (uint8_t)(xi & 0xFFu);
            unsigned char pt = (unsigned char)((uint8_t)cipher[i] ^ ks);
            int allowed = 0;
            for (int j = 0; j < d_allowed_bg_len; ++j) {
                if ((unsigned char)d_allowed_bg[j] == pt) { allowed = 1; break; }
            }
            if (!allowed) { ok = 0; break; }
        }
        if (!ok) continue;
        if (atomicCAS(found, 0, 1) == 0) {
            *outSeed = seed;
            uint64_t xi2 = seed % n;
            for (int i = 0; i < clen; ++i) {
                xi2 = (uint64_t)((__uint128_t)xi2 * xi2 % n);
                uint8_t ks = (uint8_t)(xi2 & 0xFFu);
                outPlain[i] = (char)((uint8_t)cipher[i] ^ ks);
            }
        }
        return;
    }
}

/* Simplified Blum–Goldwasser-style decrypt using provided seed x0 (no square-root step). */
extern "C" const char* blumGoldwasserCuda(const char* alph, const char* encText, const char* frag) {
    (void)alph;
    static char* out = NULL;
    if (out) { free(out); out = NULL; }
    if (!encText || !*encText || !frag || !*frag) return "[bg cuda] missing input";

    /* frag: p|q|x0  or p|q|brute:<max> */
    char* copy = strdup(frag);
    if (!copy) return "[bg cuda] OOM";
    char* tok = strtok(copy, "|");
    unsigned long long p = 0, q = 0, x = 0, maxSeed = 0;
    if (tok) { p = strtoull(tok, NULL, 10); tok = strtok(NULL, "|"); }
    if (tok) { q = strtoull(tok, NULL, 10); tok = strtok(NULL, "|"); }
    if (tok) {
        if (strncmp(tok, "brute:", 6) == 0) maxSeed = strtoull(tok + 6, NULL, 10);
        else x = strtoull(tok, NULL, 10);
    }
    free(copy);
    if (p < 3 || q < 3 || (x == 0 && maxSeed == 0)) return "[bg cuda] bad frag";
    unsigned __int128 n128 = (unsigned __int128)p * (unsigned __int128)q;
    if (n128 > (unsigned __int128)0xFFFFFFFFFFFFFFFFULL) return "[bg cuda] n too large";
    uint64_t n = (uint64_t)n128;

    int clen = 0;
    int* cbytes = parse_frag_array(encText, &clen);
    if (!cbytes || clen <= 0) { if (cbytes) free(cbytes); return "[bg cuda] bad ciphertext"; }

    if (maxSeed > 0) {
        const char* allowed = (alph && *alph) ? alph : "ABCDEFGHIJKLMNOPQRSTUVWXYZ ";
        int alen = (int)strlen(allowed);
        if (alen > 255) alen = 255;
        cudaMemcpyToSymbol(d_allowed_bg, allowed, (size_t)alen);
        cudaMemcpyToSymbol(d_allowed_bg_len, &alen, sizeof(int));
        int* d_c = NULL; int* d_found = NULL; uint64_t* d_seed = NULL; char* d_plain = NULL;
        cudaMalloc((void**)&d_c, (size_t)clen * sizeof(int));
        cudaMalloc((void**)&d_found, sizeof(int));
        cudaMalloc((void**)&d_seed, sizeof(uint64_t));
        cudaMalloc((void**)&d_plain, (size_t)clen * sizeof(char));
        cudaMemcpy(d_c, cbytes, (size_t)clen * sizeof(int), cudaMemcpyHostToDevice);
        int zero = 0; cudaMemcpy(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice);
        dim3 threads(256), grid(256);
        bg_brute_kernel<<<grid, threads>>>(d_c, clen, p, q, maxSeed, d_found, d_seed, d_plain);
        cudaDeviceSynchronize();
        int h_found = 0;
        cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        if (!h_found) {
            cudaFree(d_c); cudaFree(d_found); cudaFree(d_seed); cudaFree(d_plain); free(cbytes);
            return "[bg cuda] no seed found";
        }
        char* plain = (char*)malloc((size_t)clen + 1);
        if (!plain) {
            cudaFree(d_c); cudaFree(d_found); cudaFree(d_seed); cudaFree(d_plain); free(cbytes);
            return "[bg cuda] OOM";
        }
        cudaMemcpy(plain, d_plain, (size_t)clen * sizeof(char), cudaMemcpyDeviceToHost);
        plain[clen] = '\0';
        cudaFree(d_c); cudaFree(d_found); cudaFree(d_seed); cudaFree(d_plain); free(cbytes);
        out = plain;
        return out;
    } else {
        char* plain = (char*)malloc((size_t)clen + 1);
        if (!plain) { free(cbytes); return "[bg cuda] OOM"; }

        uint64_t xi = x % n;
        for (int i = 0; i < clen; ++i) {
            xi = (uint64_t)((__uint128_t)xi * xi % n);
            uint8_t keystream = (uint8_t)(xi & 0xFFu);
            plain[i] = (char)((uint8_t)cbytes[i] ^ keystream);
        }
        plain[clen] = '\0';
        free(cbytes);
        out = plain;
        return out;
    }
}

#endif
