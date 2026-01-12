#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "../../cyphers/knapsack.h"

#ifdef USE_CUDA

__global__ void knapKernel(const unsigned long long* key, int nBits, unsigned long long target, int* found, unsigned int* outMask) {
    unsigned long long tid = (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x + (unsigned long long)threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * (unsigned long long)blockDim.x;
    unsigned long long total = 1ULL << nBits;

    for (unsigned long long mask = tid; mask < total; mask += stride) {
        if (atomicAdd(found, 0) != 0) return;
        unsigned long long sum = 0;
        unsigned long long m = mask;
        for (int i = 0; i < nBits; ++i) {
            if (m & 1ULL) sum += key[i];
            m >>= 1ULL;
        }
        if (sum == target) {
            if (atomicCAS(found, 0, 1) == 0) {
                *outMask = (unsigned int)mask;
            }
            return;
        }
    }
}

static char* g_knap_out = NULL;
static size_t g_knap_cap = 0;

static char* ensure_knap_buf(size_t need) {
    if (g_knap_cap < need) {
        free(g_knap_out);
        g_knap_out = (char*)malloc(need);
        if (!g_knap_out) {
            g_knap_cap = 0;
            return NULL;
        }
        g_knap_cap = need;
    }
    return g_knap_out;
}

extern "C" const char* merkleBruteCuda(const char* alph, const char* encText, const char* frag) {
    (void)alph;
    unsigned long long* key = NULL; int nBits = 0;
    unsigned long long p = 0, w1 = 0; /* ignored for brute */
    extractKnapsackValues(frag, &key, &nBits, &p, &w1);
    if (!key || nBits <= 0) { if (key) free(key); return "[knap cuda] bad frag"; }
    if (nBits > 24) { free(key); return "[knap cuda] nBits too large (max 24)"; }

    int cN = 0; int* cArr = parse_frag_array(encText, &cN);
    if (!cArr || cN <= 0) { if (key) free(key); if (cArr) free(cArr); return "[knap cuda] bad ciphertext"; }

    unsigned long long* d_key = NULL;
    int* d_found = NULL;
    unsigned int* d_mask = NULL;
    cudaMalloc((void**)&d_key, (size_t)nBits * sizeof(unsigned long long));
    cudaMalloc((void**)&d_found, sizeof(int));
    cudaMalloc((void**)&d_mask, sizeof(unsigned int));
    cudaMemcpy(d_key, key, (size_t)nBits * sizeof(unsigned long long), cudaMemcpyHostToDevice);

    int* bytes = (int*)malloc((size_t)cN * sizeof(int));
    if (!bytes) {
        cudaFree(d_key); cudaFree(d_found); cudaFree(d_mask);
        free(key); free(cArr);
        return "[knap cuda] alloc failed";
    }

    dim3 threads(256);
    dim3 blocks(256);

    for (int k = 0; k < cN; ++k) {
        int zero = 0;
        cudaMemcpy(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice);
        knapKernel<<<blocks, threads>>>(d_key, nBits, (unsigned long long)(unsigned int)cArr[k], d_found, d_mask);
        cudaDeviceSynchronize();
        int h_found = 0;
        cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        if (!h_found) { bytes[k] = '?'; continue; }
        unsigned int h_mask = 0;
        cudaMemcpy(&h_mask, d_mask, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        int bits[32] = {0};
        for (int i = 0; i < nBits; ++i) bits[i] = (int)((h_mask >> i) & 1U);
        bytes[k] = bits_to_byte_msb(bits, nBits);
    }

    char* text = numbersToBytes(bytes, (size_t)cN);
    free(bytes);
    cudaFree(d_key); cudaFree(d_found); cudaFree(d_mask);
    free(key); free(cArr);
    if (!text) return "[knap cuda] output alloc failed";

    char* outBuf = ensure_knap_buf(strlen(text) + 1);
    if (!outBuf) { free(text); return "[knap cuda] alloc failed"; }
    strcpy(outBuf, text);
    free(text);
    return outBuf;
}

#endif
