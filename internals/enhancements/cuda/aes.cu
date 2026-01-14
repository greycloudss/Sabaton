#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "../../cyphers/aes.h"

#ifdef USE_CUDA

extern "C" char* decryptAESV(const int* cipher, int nBlocks, int p, int a, int b, const int T[4], const int K1[4], int rounds);

// Allowed alphabet for plaintext validation.
__constant__ char d_allowed_aes[256];
__constant__ int  d_allowed_aes_len;

// Device helpers
__device__ __forceinline__ int mod_dev(int x, int p) {
    int r = x % p;
    return (r < 0) ? r + p : r;
}

__device__ __forceinline__ int modinv_table(int x, const int* invTable, int p) {
    if (x <= 0 || x >= p) return 0;
    return invTable[x];
}

__device__ int get_subkey_dev(const int key[4], int p, int a, int b, const int* invTable, int out[4]) {
    int k22 = mod_dev(key[3], p);
    int t;
    if (k22 == 0) {
        t = mod_dev(b, p);
    } else {
        int inv_k22 = modinv_table(k22, invTable, p);
        if (inv_k22 == 0) return 0;
        long long tmp = (long long)a * inv_k22 + b;
        t = mod_dev((int)tmp, p);
    }
    out[0] = mod_dev(key[0] + t, p);
    out[1] = mod_dev(key[1] + out[0], p);
    out[2] = mod_dev(key[2] + out[1], p);
    out[3] = mod_dev(key[3] + out[2], p);
    return 1;
}

__device__ int decrypt_round_dev(const int* block, const int* key, const int* tInv, int aInv, int b, int p, const int* invTable, int* out) {
    // layer 4
    for (int i = 0; i < 4; ++i) out[i] = mod_dev(block[i] - key[i], p);
    // layer 3
    for (int i = 0; i < 2; ++i) {
        int v0 = out[i];
        int v1 = out[2 + i];
        int new0 = mod_dev(tInv[0]*v0 + tInv[1]*v1, p);
        int new1 = mod_dev(tInv[2]*v0 + tInv[3]*v1, p);
        out[i]     = new0;
        out[2 + i] = new1;
    }
    // layer 2 swap
    int tmp = out[2]; out[2] = out[3]; out[3] = tmp;
    // layer 1
    for (int i = 0; i < 4; ++i) {
        if (out[i] == b) { out[i] = 0; continue; }
        int val = mod_dev((int)((long long)aInv * mod_dev(out[i] - b, p)), p);
        int invVal = modinv_table(val, invTable, p);
        if (invVal == 0) return 0;
        out[i] = invVal;
    }
    return 1;
}

__device__ int decrypt_full_dev(const int in[4], const int Kstart[4], int rounds, int p, int a, int b, int aInv, const int TInv[4], const int* invTable, int* out) {
    int keys[5][4]; // rounds max 5 (guarded on host)
    for (int i = 0; i < 4; ++i) keys[0][i] = Kstart[i];
    for (int r = 1; r < rounds; ++r) {
        if (!get_subkey_dev(keys[r-1], p, a, b, invTable, keys[r])) return 0;
    }

    int cur[4];
    for (int i = 0; i < 4; ++i) cur[i] = in[i];

    int tmp[4];
    for (int r = rounds - 1; r >= 0; --r) {
        if (!decrypt_round_dev(cur, keys[r], TInv, aInv, b, p, invTable, tmp)) return 0;
        for (int i = 0; i < 4; ++i) cur[i] = tmp[i];
    }
    for (int i = 0; i < 4; ++i) out[i] = cur[i];
    return 1;
}

__device__ __forceinline__ int is_allowed_char(uint8_t c) {
    for (int i = 0; i < d_allowed_aes_len; ++i) {
        if ((uint8_t)d_allowed_aes[i] == c) return 1;
    }
    return 0;
}

// Search over (maxKey+1)^4 key space (bounded on host).
__global__ void aesBruteKernel(const int* cblock, int p, int a, int b, int aInv, const int* tInv, const int* invTable, int maxKey, int rounds, int* found, int* outKey) {
    unsigned long long tid = (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x + (unsigned long long)threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * (unsigned long long)blockDim.x;
    unsigned long long base = (unsigned long long)(maxKey + 1);
    unsigned long long total = base * base * base * base;

    for (unsigned long long idx = tid; idx < total; idx += stride) {
        if (atomicAdd(found, 0) != 0) return;

        unsigned long long tmp = idx;
        int key[4];
        for (int i = 0; i < 4; ++i) {
            key[i] = (int)(tmp % base);
            tmp /= base;
        }

        int plain[4];
        if (!decrypt_full_dev(cblock, key, rounds, p, a, b, aInv, tInv, invTable, plain)) continue;

        // Map to ASCII like CPU decryptAESV does.
        int ok = 1;
        for (int i = 0; i < 4; ++i) {
            uint8_t c = (uint8_t)(mod_dev(plain[i], p) & 0xFF);
            if (!is_allowed_char(c)) { ok = 0; break; }
        }
        if (!ok) continue;

        if (atomicCAS(found, 0, 1) == 0) {
            for (int i = 0; i < 4; ++i) outKey[i] = key[i];
        }
        return;
    }
}

// Host wrapper
static int parse_csv4(const char* s, int out[4]) {
    int count = 0;
    const char* p = s;
    while (*p && (*p == '[' || *p == ' ')) ++p;
    while (*p && count < 4) {
        while (*p == ' ' || *p == ',') ++p;
        if (!*p) break;
        char* end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        out[count++] = (int)v;
        p = end;
        while (*p && *p != ',' && *p != ' ' && *p != ']') ++p;
    }
    return count;
}

extern "C" const char* aesBruteCuda(const char* alph, const char* encText, const char* frag) {
    static char* out = NULL;
    if (out) { free(out); out = NULL; }

    if (!frag || strncmp(frag, "brute", 5) != 0) return "[aes cuda] frag must start with brute";
    if (!encText || !*encText) return "[aes cuda] missing ciphertext";

    int p = 0, a = 0, b = 0, rounds = 3, maxKey = 15;
    int T[4] = {0}, Tcount = 0;
    // tokens split by '|'
    char* copy = strdup(frag);
    if (!copy) return "[aes cuda] OOM";
    char* tok = strtok(copy, "|");
    while (tok) {
        while (*tok == ' ') ++tok;
        if (strncmp(tok, "p:", 2) == 0) p = stoi(tok + 2);
        else if (strncmp(tok, "a:", 2) == 0) a = stoi(tok + 2);
        else if (strncmp(tok, "b:", 2) == 0) b = stoi(tok + 2);
        else if (strncmp(tok, "T:", 2) == 0) Tcount = parse_csv4(tok + 2, T);
        else if (strncmp(tok, "R:", 2) == 0) rounds = stoi(tok + 2);
        else if (strncmp(tok, "max:", 4) == 0) maxKey = stoi(tok + 4);
        tok = strtok(NULL, "|");
    }
    free(copy);

    int nCipher = 0;
    int* cipher = parse_frag_array(encText, &nCipher);
    if (!cipher || nCipher < 4 || (nCipher % 4) != 0) {
        if (cipher) free(cipher);
        return "[aes cuda] ciphertext must be multiple of 4 ints";
    }

    if (p <= 0 || a == 0 || b == 0 || Tcount != 4) {
        free(cipher);
        return "[aes cuda] missing p,a,b or T";
    }
    if (rounds <= 0 || rounds > 5) {
        free(cipher);
        return "[aes cuda] rounds must be 1..5";
    }
    if (maxKey < 0 || maxKey > 31) {
        free(cipher);
        return "[aes cuda] max key must be 0..31";
    }
    unsigned long long totalComb = 1;
    for (int i = 0; i < 4; ++i) totalComb *= (unsigned long long)(maxKey + 1);
    if (totalComb > 100000000ULL) {
        free(cipher);
        return "[aes cuda] key space too large";
    }

    int tInv[4];
    if (!inv2x2mod(T, p, tInv)) {
        free(cipher);
        return "[aes cuda] T not invertible";
    }
    int aInv = 0;
    if (!modinv(a, p, &aInv)) {
        free(cipher);
        return "[aes cuda] a not invertible mod p";
    }

    // Build inverse table for mod p
    int* invTable = (int*)malloc((size_t)p * sizeof(int));
    if (!invTable) {
        free(cipher);
        return "[aes cuda] OOM";
    }
    invTable[0] = 0;
    for (int i = 1; i < p; ++i) {
        int inv;
        invTable[i] = modinv(i, p, &inv) ? inv : 0;
    }

    const char* allowed = (alph && *alph) ? alph : "ABCDEFGHIJKLMNOPQRSTUVWXYZ ";
    int allowed_len = (int)strlen(allowed);
    if (allowed_len > 255) allowed_len = 255;
    cudaMemcpyToSymbol(d_allowed_aes, allowed, (size_t)allowed_len);
    cudaMemcpyToSymbol(d_allowed_aes_len, &allowed_len, sizeof(int));

    int* d_block = NULL;
    int* d_tInv = NULL;
    int* d_invTable = NULL;
    int* d_found = NULL;
    int* d_key = NULL;

    cudaMalloc((void**)&d_block, 4 * sizeof(int));
    cudaMalloc((void**)&d_tInv, 4 * sizeof(int));
    cudaMalloc((void**)&d_invTable, (size_t)p * sizeof(int));
    cudaMalloc((void**)&d_found, sizeof(int));
    cudaMalloc((void**)&d_key, 4 * sizeof(int));

    cudaMemcpy(d_block, cipher, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tInv, tInv, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_invTable, invTable, (size_t)p * sizeof(int), cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(256), blocks(256);
    aesBruteKernel<<<blocks, threads>>>(d_block, p, a, b, aInv, d_tInv, d_invTable, maxKey, rounds, d_found, d_key);
    cudaDeviceSynchronize();

    int h_found = 0;
    int h_key[4] = {0,0,0,0};
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_found) {
        cudaMemcpy(h_key, d_key, 4 * sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_block); cudaFree(d_tInv); cudaFree(d_invTable);
    cudaFree(d_found); cudaFree(d_key);
    free(invTable);

    if (!h_found) {
        free(cipher);
        return "[aes cuda] no key found";
    }

    // Decrypt full ciphertext on CPU using found key.
    char* text = decryptAESV(cipher, nCipher / 4, p, a, b, T, h_key, rounds);
    free(cipher);
    if (!text) return "[aes cuda] decrypt failed";

    size_t needed = strlen(text) + 64;
    out = (char*)malloc(needed);
    if (!out) {
        free(text);
        return "[aes cuda] alloc failed";
    }
    snprintf(out, needed, "K=[%d,%d,%d,%d]\n%s", h_key[0], h_key[1], h_key[2], h_key[3], text);
    free(text);
    return out;
}

#endif
