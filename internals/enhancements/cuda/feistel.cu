#include "../../cyphers/feistel.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

extern "C" char g_funcFlag;

__device__ int is_valid_char_dev(unsigned char c) {
    return (c == ' ' || (c >= 'A' && c <= 'Z'));
}

__device__ uint8_t selFuncDev(char flag, uint8_t m, uint8_t k) {
    switch (flag) {
        case 0: return (uint8_t)((m | k) ^ ((m / 16) & k));
        case 1: return (uint8_t)((m & k) | ((k % 16) ^ m));
        case 2: return (uint8_t)((m | k) ^ ((k / 16) & m));
        case 3: return (uint8_t)((m ^ k) & ((k / 16) | m));
        default: return (uint8_t)((m & k) ^ ((k % 16) | m));
    }
}

__device__ void feistel_block_decrypt_dev(const int* encInt, int idx, const int* keyBytes, int rounds, char funcFlag, unsigned char* outL, unsigned char* outR) {
    uint8_t R = (uint8_t)encInt[idx];
    uint8_t L = (uint8_t)encInt[idx + 1];
    for (int j = rounds - 1; j >= 0; --j) {
        uint8_t t = (uint8_t)(R ^ selFuncDev(funcFlag, L, (uint8_t)keyBytes[j]));
        R = L;
        L = t;
    }
    *outL = (unsigned char)L;
    *outR = (unsigned char)R;
}

__global__ void feistelKernel(const int* encInt, int bigN, int rounds, char funcFlag, unsigned char* outPlain, int* found, int* foundKey) {
    unsigned long long tid = (unsigned long long) blockIdx.x * (unsigned long long) blockDim.x + (unsigned long long) threadIdx.x;
    unsigned long long stride = (unsigned long long) gridDim.x * (unsigned long long) blockDim.x;

    unsigned long long totalComb = 1ULL;
    for (int i = 0; i < rounds; ++i) totalComb *= 256ULL;

    for (unsigned long long comb = tid; comb < totalComb; comb += stride) {
        if (atomicAdd(found, 0) != 0) return;

        int keyBytes[4] = {0, 0, 0, 0};
        unsigned long long tmp = comb;
        for (int j = 0; j < rounds; ++j) {
            keyBytes[j] = (int)(tmp & 0xFFULL);
            tmp >>= 8;
        }

        int valid = 1;
        for (int i = 0; i < bigN; i += 2) {
            unsigned char cL;
            unsigned char cR;
            feistel_block_decrypt_dev(encInt, i, keyBytes, rounds, funcFlag, &cL, &cR);
            if (!is_valid_char_dev(cL) || !is_valid_char_dev(cR)) {
                valid = 0;
                break;
            }
        }
        if (!valid) continue;

        if (atomicCAS(found, 0, 1) == 0) {
            for (int j = 0; j < rounds; ++j) {
                foundKey[j] = keyBytes[j];
            }
            for (int i = 0; i < bigN; i += 2) {
                unsigned char cL;
                unsigned char cR;
                feistel_block_decrypt_dev(encInt, i, keyBytes, rounds, funcFlag, &cL, &cR);
                outPlain[i] = cL;
                outPlain[i + 1] = cR;
            }
        }
        return;
    }
}

static char* g_plain_buf = NULL;
static size_t g_plain_cap = 0;

static char* ensure_plain_buf(size_t len) {
    if (g_plain_cap < len + 1) {
        free(g_plain_buf);
        g_plain_cap = len + 1;
        g_plain_buf = (char*)malloc(g_plain_cap);
        if (!g_plain_buf) {
            g_plain_cap = 0;
            return NULL;
        }
    }
    return g_plain_buf;
}

static int gpu_bruteforce_for_rounds(const int* encInt, int bigN, int rounds, char funcFlag, int* outKey, unsigned char* outPlain) {
    int* d_encInt = NULL;
    unsigned char* d_outPlain = NULL;
    int* d_found = NULL;
    int* d_foundKey = NULL;

    cudaMalloc((void**)&d_encInt, bigN * sizeof(int));
    cudaMalloc((void**)&d_outPlain, bigN * sizeof(unsigned char));
    cudaMalloc((void**)&d_found, sizeof(int));
    cudaMalloc((void**)&d_foundKey, 4 * sizeof(int));

    cudaMemcpy(d_encInt, encInt, bigN * sizeof(int), cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = 256;
    feistelKernel<<<blocks, threads>>>(d_encInt, bigN, rounds, funcFlag, d_outPlain, d_found, d_foundKey);
    cudaDeviceSynchronize();

    int h_found = 0;
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_found) {
        cudaMemcpy(outPlain, d_outPlain, bigN * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        int tmpKey[4] = {0, 0, 0, 0};
        cudaMemcpy(tmpKey, d_foundKey, 4 * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < rounds; ++i) outKey[i] = tmpKey[i];
    }

    cudaFree(d_encInt);
    cudaFree(d_outPlain);
    cudaFree(d_found);
    cudaFree(d_foundKey);

    return h_found;
}

extern "C" const char* feistelBrute(const char* alph, const char* encText, const char* frag) {
    (void)alph;
    (void)frag;

    int bigN = 0;
    int* encInt = parse_frag_array(encText, &bigN);
    if (!encInt || bigN <= 0 || (bigN & 1)) {
        if (encInt) free(encInt);
        return "";
    }

    unsigned char* tmpPlain = (unsigned char*)malloc((size_t)bigN);
    if (!tmpPlain) {
        free(encInt);
        return "";
    }

    char funcFlag = g_funcFlag;
    char* bestPlain = NULL;
    int bestRounds = 0;

    for (int rounds = 1; rounds <= 4; ++rounds) {
        int key[4] = {0, 0, 0, 0};
        int found = gpu_bruteforce_for_rounds(encInt, bigN, rounds, funcFlag, key, tmpPlain);
        if (!found) continue;

        char fname[64];
        int w = snprintf(fname, sizeof(fname), "feistel-keys-%d.txt", rounds);
        if (w > 0 && w < (int)sizeof(fname)) {
            FILE* f = fopen(fname, "a");
            if (f) {
                for (int i = 0; i < rounds; ++i) {
                    fprintf(f, "%d%s", key[i], (i + 1 < rounds) ? " " : "\n");
                }
                fclose(f);
            }
        }

        char* buf = ensure_plain_buf((size_t)bigN);
        if (buf) {
            for (int i = 0; i < bigN; ++i) {
                unsigned char c = tmpPlain[i];
                if (c == '\0') c = ' ';
                buf[i] = (char)c;
            }
            buf[bigN] = '\0';
            bestPlain = buf;
            bestRounds = rounds;
        }
    }
    print("Feistel brute-force completed, the output stdout output might be wrong, instead - check the keys in the .txt files.\n");
    free(encInt);
    free(tmpPlain);

    if (!bestPlain || bestRounds == 0) return "";
    return bestPlain;
}