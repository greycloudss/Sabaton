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

__global__ void feistelKernel(const int* encInt, int bigN, const int* baseKey, const int* unknownPos, int unknownCount, unsigned int totalComb, int rounds, char funcFlag, unsigned char* outPlain, int* found, int* foundKey) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    for (unsigned int comb = tid; comb < totalComb; comb += stride) {
        if (atomicAdd(found, 0) != 0) return;

        int keyBytes[3];
        keyBytes[0] = baseKey[0];
        keyBytes[1] = baseKey[1];
        keyBytes[2] = baseKey[2];

        unsigned int tmp = comb;
        for (int u = 0; u < unknownCount; ++u) {
            int pos = unknownPos[u];
            int val = (int)(tmp & 0xFFu);
            keyBytes[pos] = val;
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

extern "C" const char* feistelBrute(const char* alph, const char* encText, const char* frag) {
    (void)alph;

    int bigN = 0;
    int* encInt = parse_frag_array(encText, &bigN);
    if (!encInt || bigN <= 0 || (bigN & 1)) {
        if (encInt) free(encInt);
        return "";
    }

    int keyInput[3] = { -1, -1, -1 };
    int keyCount = 3;

    if (frag && *frag) {
        int n = 0;
        int* parsed = parse_frag_array(frag, &n);
        if (parsed && n > 0) {
            if (n > 3) n = 3;
            keyCount = n;
            for (int i = 0; i < n; ++i) {
                keyInput[i] = parsed[i];
            }
        }
        if (parsed) free(parsed);
    }

    int baseKey[3] = { 0, 0, 0 };
    int unknownPos[3];
    int unknownCount = 0;

    for (int i = 0; i < keyCount; ++i) {
        if (keyInput[i] >= 0 && keyInput[i] <= 255) {
            baseKey[i] = keyInput[i] & 0xFF;
        } else {
            baseKey[i] = 0;
            unknownPos[unknownCount++] = i;
        }
    }
    for (int i = keyCount; i < 3; ++i) {
        baseKey[i] = 0;
    }

    unsigned int totalComb = 1;
    for (int i = 0; i < unknownCount; ++i) {
        totalComb *= 256u;
    }
    if (totalComb == 0) totalComb = 1;

    int* d_encInt = NULL;
    int* d_baseKey = NULL;
    int* d_unknownPos = NULL;
    unsigned char* d_outPlain = NULL;
    int* d_found = NULL;
    int* d_foundKey = NULL;

    cudaMalloc((void**)&d_encInt, bigN * sizeof(int));
    cudaMalloc((void**)&d_baseKey, 3 * sizeof(int));
    cudaMalloc((void**)&d_unknownPos, 3 * sizeof(int));
    cudaMalloc((void**)&d_outPlain, bigN * sizeof(unsigned char));
    cudaMalloc((void**)&d_found, sizeof(int));
    cudaMalloc((void**)&d_foundKey, 3 * sizeof(int));

    cudaMemcpy(d_encInt, encInt, bigN * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_baseKey, baseKey, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_unknownPos, unknownPos, 3 * sizeof(int), cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice);

    char funcFlag = g_funcFlag;

    int threads = 256;
    int blocks = 256;
    feistelKernel<<<blocks, threads>>>(d_encInt, bigN, d_baseKey, d_unknownPos, unknownCount, totalComb, keyCount, funcFlag, d_outPlain, d_found, d_foundKey);
    cudaDeviceSynchronize();

    int h_found = 0;
    int h_foundKey[3] = { 0, 0, 0 };
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_found) {
        cudaMemcpy(h_foundKey, d_foundKey, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    }

    const char* ret = "";

    if (h_found) {
        char* buf = ensure_plain_buf((size_t)bigN);
        if (buf) {
            unsigned char* tmpPlain = (unsigned char*)malloc((size_t)bigN);
            if (tmpPlain) {
                cudaMemcpy(tmpPlain, d_outPlain, bigN * sizeof(unsigned char), cudaMemcpyDeviceToHost);
                for (int i = 0; i < bigN; ++i) {
                    unsigned char c = tmpPlain[i];
                    if (c == '\0') c = ' ';
                    buf[i] = (char)c;
                }
                buf[bigN] = '\0';
                free(tmpPlain);
                ret = buf;
            }

            char fname[64];
            int w = snprintf(fname, sizeof(fname), "feistel-keys-%d.txt", keyCount);
            if (w > 0 && w < (int)sizeof(fname)) {
                FILE* f = fopen(fname, "a");
                if (f) {
                    for (int i = 0; i < keyCount; ++i) {
                        fprintf(f, "%d%s", h_foundKey[i], (i + 1 < keyCount) ? " " : "\n");
                    }
                    fclose(f);
                }
            }
        }
    }

    cudaFree(d_encInt);
    cudaFree(d_baseKey);
    cudaFree(d_unknownPos);
    cudaFree(d_outPlain);
    cudaFree(d_found);
    cudaFree(d_foundKey);
    free(encInt);

    return ret;
}
