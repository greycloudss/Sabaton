#include "../../cyphers/a5.h"
#include "../lith/lithuanian.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

__constant__ unsigned char d_allowed[256];
__constant__ int d_allowed_len;

__device__ __forceinline__ uint8_t parity8(uint8_t x) {
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    return x & 1u;
}

__device__ __forceinline__ uint8_t lfsr_clock(uint8_t s, uint8_t taps, int shift_left) {
    uint8_t fb = parity8((uint8_t)(s & taps));
    if (shift_left) return (uint8_t)((s << 1) | fb);
    else return (uint8_t)((s >> 1) | (uint8_t)(fb << 7));
}

__device__ __forceinline__ uint8_t reg_clock_bit(uint8_t s, int idx, int msb_indexing) {
    int bit = msb_indexing ? (7 - idx) : idx;
    return (uint8_t)((s >> bit) & 1u);
}

__device__ __forceinline__ uint8_t reg_out(uint8_t s, int out_msb) {
    return out_msb ? (uint8_t)((s >> 7) & 1u) : (uint8_t)(s & 1u);
}

__device__ __forceinline__ uint8_t reverse_bits8_dev(uint8_t x) {
    x = (x & 0xF0u) >> 4 | (x & 0x0Fu) << 4;
    x = (x & 0xCCu) >> 2 | (x & 0x33u) << 2;
    x = (x & 0xAAu) >> 1 | (x & 0x55u) << 1;
    return x;
}

__device__ uint8_t a5_next_bit_dev(uint8_t* s0, uint8_t* s1, uint8_t* s2, uint8_t taps,
                                   int output_after, int msb_indexing, int shift_left, int output_msb) {
    uint8_t c0 = reg_clock_bit(*s0, 1, msb_indexing);
    uint8_t c1 = reg_clock_bit(*s1, 2, msb_indexing);
    uint8_t c2 = reg_clock_bit(*s2, 3, msb_indexing);
    uint8_t maj = (uint8_t)((c0 + c1 + c2) >= 2 ? 1 : 0);

    uint8_t o0 = reg_out(*s0, output_msb);
    uint8_t o1 = reg_out(*s1, output_msb);
    uint8_t o2 = reg_out(*s2, output_msb);

    if (c0 == maj) *s0 = lfsr_clock(*s0, taps, shift_left);
    if (c1 == maj) *s1 = lfsr_clock(*s1, taps, shift_left);
    if (c2 == maj) *s2 = lfsr_clock(*s2, taps, shift_left);

    if (output_after) {
        o0 = reg_out(*s0, output_msb);
        o1 = reg_out(*s1, output_msb);
        o2 = reg_out(*s2, output_msb);
    }
    return (uint8_t)((o0 ^ o1 ^ o2) & 1u);
}

__device__ int score_bytes_dev(const uint8_t* p, int n) {
    int score = 0;
    for (int i = 0; i < n; ++i) {
        uint8_t c = p[i];
        int ok = 0;
        for (int j = 0; j < d_allowed_len; ++j) {
            if (d_allowed[j] == c) { ok = 1; break; }
        }
        if (ok) score += 2;
        else if (c >= 32 && c <= 126) score += 1;
        else score -= 2;
    }
    return score;
}

__global__ void a5BruteKernel(const int* cipher, int n, int* bestScore, int* bestMeta, uint8_t* bestPlain) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    const unsigned int totalComb = 255u * 16u; // taps 1..255, 4 config bits (output_after, msb_index, shift_dir, out_msb)

    for (unsigned int idx = tid; idx < totalComb; idx += stride) {
        unsigned int tmp = idx;
        unsigned int cfg = tmp % 16u; tmp /= 16u;
        unsigned int taps_idx = tmp % 255u;
        uint8_t taps = (uint8_t)(1u + taps_idx);

        uint8_t taps_use = (cfg & 8u) ? reverse_bits8_dev(taps) : taps;
        int output_after = (cfg >> 0) & 1u;
        int msb_indexing = (cfg >> 1) & 1u;
        int shift_left =  (cfg >> 2) & 1u;
        int output_msb =  (cfg >> 3) & 1u;

        uint8_t s0 = taps_use, s1 = taps_use, s2 = taps_use;
        uint8_t plain_local[256];
        if (n > 256) n = 256; // safety cap
        for (int i = 0; i < n; ++i) {
            uint8_t ks = 0;
            for (int b = 0; b < 8; ++b) {
                ks = (uint8_t)((ks << 1) | a5_next_bit_dev(&s0, &s1, &s2, taps_use, output_after, msb_indexing, shift_left, output_msb));
            }
            plain_local[i] = (uint8_t)((cipher[i] & 0xFF) ^ ks);
        }
        int sc = score_bytes_dev(plain_local, n);
        int prev = atomicMax(bestScore, sc);
        if (sc > prev) {
            bestMeta[0] = taps_use;
            bestMeta[1] = cfg;
            for (int i = 0; i < n; ++i) bestPlain[i] = plain_local[i];
        }
    }
}

extern "C" const char* a5Cuda(const char* alph, const char* encText, const char* frag) {
    (void)frag; // GPU brute ignores provided taps; searches all
    static char* out = NULL;
    if (out) { free(out); out = NULL; }

    int n = 0;
    int* cipher = parse_frag_array(encText, &n);
    if (!cipher || n <= 0) {
        if (cipher) free(cipher);
        return "[cuda-a5] invalid ciphertext";
    }
    if (n > 256) n = 256; // align with kernel cap

    const char* allowed = (alph && *alph) ? alph : "ABCDEFGHIJKLMNOPQRSTUVWXYZ ";
    int allowed_len = (int)strlen(allowed);
    if (allowed_len > 255) allowed_len = 255;
    cudaMemcpyToSymbol(d_allowed, allowed, allowed_len);
    cudaMemcpyToSymbol(d_allowed_len, &allowed_len, sizeof(int));

    int* d_cipher = NULL;
    int* d_bestScore = NULL;
    int* d_bestMeta = NULL;
    uint8_t* d_bestPlain = NULL;
    cudaMalloc((void**)&d_cipher, (size_t)n * sizeof(int));
    cudaMalloc((void**)&d_bestScore, sizeof(int));
    cudaMalloc((void**)&d_bestMeta, 2 * sizeof(int));
    cudaMalloc((void**)&d_bestPlain, (size_t)n * sizeof(uint8_t));
    cudaMemcpy(d_cipher, cipher, (size_t)n * sizeof(int), cudaMemcpyHostToDevice);
    int initScore = -1000000;
    cudaMemcpy(d_bestScore, &initScore, sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(256);
    dim3 blocks(64);
    a5BruteKernel<<<blocks, threads>>>(d_cipher, n, d_bestScore, d_bestMeta, d_bestPlain);
    cudaDeviceSynchronize();

    int h_score = -1000000;
    int h_meta[2] = {0};
    uint8_t* h_plain = (uint8_t*)malloc((size_t)n);
    if (!h_plain) {
        cudaFree(d_cipher); cudaFree(d_bestScore); cudaFree(d_bestMeta); cudaFree(d_bestPlain); free(cipher);
        return "[cuda-a5] alloc failed";
    }
    cudaMemcpy(&h_score, d_bestScore, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_meta, d_bestMeta, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_plain, d_bestPlain, (size_t)n * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    int* plain_bytes = (int*)malloc((size_t)n * sizeof(int));
    if (!plain_bytes) {
        free(h_plain); free(cipher);
        cudaFree(d_cipher); cudaFree(d_bestScore); cudaFree(d_bestMeta); cudaFree(d_bestPlain);
        return "[cuda-a5] alloc failed";
    }
    for (int i = 0; i < n; ++i) plain_bytes[i] = (int)h_plain[i];
    out = numbersToBytes(plain_bytes, (size_t)n);

    free(h_plain); free(cipher); free(plain_bytes);
    cudaFree(d_cipher); cudaFree(d_bestScore); cudaFree(d_bestMeta); cudaFree(d_bestPlain);
    return out ? out : "[cuda-a5] decode failed";
}

#endif
