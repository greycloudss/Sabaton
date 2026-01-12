#include "../../cyphers/stream.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>



__constant__ char d_allowed[256];
__constant__ int  d_allowed_len;
__constant__ char d_prefix[128];
__constant__ int  d_prefix_len;

__device__ __forceinline__ uint8_t reverse_bits8_dev(uint8_t x) {
    x = (x & 0xF0u) >> 4 | (x & 0x0Fu) << 4;
    x = (x & 0xCCu) >> 2 | (x & 0x33u) << 2;
    x = (x & 0xAAu) >> 1 | (x & 0x55u) << 1;
    return x;
}

__device__ __forceinline__ uint8_t lfsr_next_bit_right_dev(uint8_t *state, uint8_t taps) {
    uint8_t s = *state;
    uint8_t out = s & 1u;
    uint8_t x = s & taps;
    x ^= x >> 4; x ^= x >> 2; x ^= x >> 1;
    uint8_t newbit = x & 1u;
    s = (uint8_t)((s >> 1) | (newbit << 7));
    *state = s;
    return out;
}

__device__ __forceinline__ uint8_t lfsr_next_bit_left_dev(uint8_t *state, uint8_t taps) {
    uint8_t s = *state;
    uint8_t out = (s >> 7) & 1u;
    uint8_t x = s & taps;
    x ^= x >> 4; x ^= x >> 2; x ^= x >> 1;
    uint8_t newbit = x & 1u;
    s = (uint8_t)((s << 1) | newbit);
    *state = s;
    return out;
}

__device__ __forceinline__ uint8_t lfsr_next_byte_dev(uint8_t *state, uint8_t taps, int shift_right, int msb_first) {
    uint8_t b = 0;
    if (msb_first) {
        for (int i = 0; i < 8; ++i) {
            uint8_t bit = shift_right ? lfsr_next_bit_right_dev(state, taps) : lfsr_next_bit_left_dev(state, taps);
            b = (uint8_t)((b << 1) | (bit & 1u));
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            uint8_t bit = shift_right ? lfsr_next_bit_right_dev(state, taps) : lfsr_next_bit_left_dev(state, taps);
            b |= (uint8_t)((bit & 1u) << i);
        }
    }
    return b;
}

__device__ __forceinline__ int is_allowed(uint8_t c) {
    for (int i = 0; i < d_allowed_len; ++i) {
        if ((uint8_t)d_allowed[i] == c) return 1;
    }
    return 0;
}

__global__ void streamBruteKernel(const int* cipher, int n, char* outPlain, int* found, int* meta) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    const unsigned int totalComb = 255u * 255u * 8u;

    for (unsigned int idx = tid; idx < totalComb; idx += stride) {
        if (atomicAdd(found, 0) != 0) return;

        unsigned int tmp = idx;
        unsigned int variant = tmp % 8u; tmp /= 8u;
        unsigned int state_idx = tmp % 255u; tmp /= 255u;
        unsigned int taps_idx = tmp % 255u;

        uint8_t taps = (uint8_t)(1u + taps_idx);
        uint8_t state = (uint8_t)(1u + state_idx);

        int shift_right = (variant & 1u);
        int msb_first  = (variant >> 1) & 1u;
        int rev        = (variant >> 2) & 1u;

        uint8_t taps_use = rev ? reverse_bits8_dev(taps) : taps;

        uint8_t st = state;
        int ok = 1;
        for (int i = 0; i < n; ++i) {
            uint8_t ks = lfsr_next_byte_dev(&st, taps_use, shift_right, msb_first);
            uint8_t p = (uint8_t)((cipher[i] & 0xFF) ^ ks);
            if (i < d_prefix_len) {
                if (p != (uint8_t)d_prefix[i]) { ok = 0; break; }
            } else if (!is_allowed(p)) {
                ok = 0;
                break;
            }
        }
        if (!ok) continue;

        if (atomicCAS(found, 0, 1) == 0) {
            uint8_t st_out = state;
            for (int i = 0; i < n; ++i) {
                uint8_t ks = lfsr_next_byte_dev(&st_out, taps_use, shift_right, msb_first);
                uint8_t p = (uint8_t)((cipher[i] & 0xFF) ^ ks);
                outPlain[i] = (char)p;
            }
            meta[0] = (int)taps_use;
            meta[1] = (int)state;
            meta[2] = (int)variant;
            return;
        }
    }
}

static char* g_stream_plain = NULL;
static size_t g_stream_plain_cap = 0;

static char* ensure_stream_plain(size_t len) {
    if (g_stream_plain_cap < len) {
        free(g_stream_plain);
        g_stream_plain = (char*)malloc(len);
        if (!g_stream_plain) {
            g_stream_plain_cap = 0;
            return NULL;
        }
        g_stream_plain_cap = len;
    }
    return g_stream_plain;
}

static int parse_prefix(const char* frag, char* prefix, int* prefix_len) {
    *prefix_len = 0;
    if (!frag || !*frag) return 0;

    const char* s = frag;
    if (strncmp(s, "lfsr:", 5) == 0) s += 5;

    int sem_count = 0;
    for (const char* p = s; *p; ++p) if (*p == ';') ++sem_count;
    if (sem_count >= 2) return -1;

    char buf[192];
    strncpy(buf, s, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    char* tokN = strtok(buf, ";");
    char* tok2 = strtok(NULL, ";");

    if (!tokN) return 0;
    int N = stoi(tokN);
    if (N != 8) return -1;

    if (tok2 && strcmp(tok2, "brute") != 0) {
        int n = (int)strlen(tok2);
        if (n > 127) n = 127;
        memcpy(prefix, tok2, (size_t)n);
        *prefix_len = n;
    }
    return 0;
}

extern "C" const char* streamBruteCuda(const char* alph, const char* encText, const char* frag) {
    int bigN = 0;
    int* cbytes = parse_frag_array(encText, &bigN);
    if (!cbytes || bigN <= 0) {
        if (cbytes) free(cbytes);
        return "[cuda-stream] invalid ciphertext";
    }

    char prefix[128];
    int prefix_len = 0;
    if (parse_prefix(frag, prefix, &prefix_len) != 0) {
        free(cbytes);
        return "[cuda-stream] only lfsr:8 is supported";
    }

    const char* allowed = (alph && *alph) ? alph : "ABCDEFGHIJKLMNOPQRSTUVWXYZ ";
    int allowed_len = (int)strlen(allowed);
    if (allowed_len > 255) allowed_len = 255;

    cudaMemcpyToSymbol(d_allowed, allowed, (size_t)allowed_len);
    cudaMemcpyToSymbol(d_allowed_len, &allowed_len, sizeof(int));
    cudaMemcpyToSymbol(d_prefix, prefix, (size_t)prefix_len);
    cudaMemcpyToSymbol(d_prefix_len, &prefix_len, sizeof(int));

    int* d_cipher = NULL;
    char* d_plain = NULL;
    int* d_found = NULL;
    int* d_meta = NULL;

    cudaMalloc((void**)&d_cipher, (size_t)bigN * sizeof(int));
    cudaMalloc((void**)&d_plain, (size_t)bigN * sizeof(char));
    cudaMalloc((void**)&d_found, sizeof(int));
    cudaMalloc((void**)&d_meta, 3 * sizeof(int));

    cudaMemcpy(d_cipher, cbytes, (size_t)bigN * sizeof(int), cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(256);
    dim3 blocks(256);
    streamBruteKernel<<<blocks, threads>>>(d_cipher, bigN, d_plain, d_found, d_meta);
    cudaDeviceSynchronize();

    int h_found = 0;
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);

    static char metaOut[96];
    const char* ret = "[cuda-stream] no candidate found";
    char* plainTmp = NULL;

    if (h_found) {
        plainTmp = (char*)malloc((size_t)bigN + 1);
        if (!plainTmp) {
            ret = "[cuda-stream] allocation failed";
        } else {
            cudaMemcpy(plainTmp, d_plain, (size_t)bigN * sizeof(char), cudaMemcpyDeviceToHost);
            plainTmp[bigN] = '\0';

            int h_meta[3] = {0, 0, 0};
            cudaMemcpy(h_meta, d_meta, 3 * sizeof(int), cudaMemcpyDeviceToHost);
            int sr  = h_meta[2] & 1;
            int msb = (h_meta[2] >> 1) & 1;
            int rev = (h_meta[2] >> 2) & 1;
            snprintf(metaOut, sizeof(metaOut), "taps=%d state=%d sr=%d msb=%d rev=%d\n",
                     h_meta[0], h_meta[1], sr, msb, rev);

            size_t total_len = strlen(metaOut) + (size_t)bigN + 1;
            char* outBuf = ensure_stream_plain(total_len + 1);
            if (outBuf) {
                snprintf(outBuf, total_len + 1, "%s%s", metaOut, plainTmp);
                ret = outBuf;
            } else {
                ret = "[cuda-stream] allocation failed";
            }
        }
    }

    cudaFree(d_cipher);
    cudaFree(d_plain);
    cudaFree(d_found);
    cudaFree(d_meta);
    free(cbytes);
    if (plainTmp) free(plainTmp);

    return ret;
}

#endif
