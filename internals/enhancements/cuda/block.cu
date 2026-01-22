#include "../../cyphers/block.h"
#include "../../../util/string.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

__device__ __forceinline__ uint8_t Ffunc_dev(uint8_t r, uint8_t k, unsigned char flag) {
    switch (flag) {
    case 0: return (uint8_t)(((r | k) ^ (((r >> 4) & k))) & 0xFF);
    case 1: return (uint8_t)(((r ^ k) & (((k >> 4) | r))) & 0xFF);
    case 2: return (uint8_t)(((r | k) ^ (((k >> 4) & r))) & 0xFF);
    case 3: return (uint8_t)(((r ^ k) & (((r >> 4) | k))) & 0xFF);
    default:return (uint8_t)(((r | k) ^ (((k >> 4) & r))) & 0xFF);
    }
}

__device__ __forceinline__ void dec_block3_dev(uint8_t inL, uint8_t inR, const int* keys, uint8_t* oL, uint8_t* oR, unsigned char flag) {
    uint8_t L = inL, R = inR;
    uint8_t t = L; L = R; R = t;
    for (int i = 2; i >= 0; --i) {
        uint8_t nL = (uint8_t)(R ^ Ffunc_dev(L, (uint8_t)keys[i], flag));
        uint8_t nR = L;
        L = nL; R = nR;
    }
    *oL = L; *oR = R;
}

__global__ void blockKernel(const uint8_t* c, int pairs, const int* keys, unsigned char flag,
                            uint8_t* out_ecb, uint8_t* out_cbc, uint8_t* out_cfb, uint8_t* out_crt) {
    // single thread does sequential modes; kernel used just to stay on GPU build
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // ECB
    for (int i = 0; i < pairs; ++i) {
        dec_block3_dev(c[2*i], c[2*i+1], keys, &out_ecb[2*i], &out_ecb[2*i+1], flag);
    }

    // CBC (first pair is IV)
    if (pairs >= 1) {
        uint8_t prevL = c[0], prevR = c[1];
        for (int i = 1; i < pairs; ++i) {
            uint8_t pL, pR;
            dec_block3_dev(c[2*i], c[2*i+1], keys, &pL, &pR, flag);
            out_cbc[2*(i-1)]   = (uint8_t)(pL ^ prevL);
            out_cbc[2*(i-1)+1] = (uint8_t)(pR ^ prevR);
            prevL = c[2*i]; prevR = c[2*i+1];
        }
    }

    // CFB (first pair is IV)
    if (pairs >= 1) {
        uint8_t sL = c[0], sR = c[1];
        for (int i = 1; i < pairs; ++i) {
            uint8_t keL, keR;
            dec_block3_dev(sL, sR, keys, &keL, &keR, flag);
            out_cfb[2*(i-1)]   = (uint8_t)(c[2*i]   ^ keL);
            out_cfb[2*(i-1)+1] = (uint8_t)(c[2*i+1] ^ keR);
            sL = c[2*i]; sR = c[2*i+1];
        }
    }

    // CRT (CTR) keystream from block index using key0 only (same as block.c)
    for (int i = 0; i < pairs; ++i) {
        uint8_t a = (uint8_t)Ffunc_dev((uint8_t)i, (uint8_t)keys[0], flag);
        uint8_t keL, keR;
        dec_block3_dev(a, a, keys, &keL, &keR, flag);
        out_crt[2*i]   = (uint8_t)(c[2*i]   ^ keL);
        out_crt[2*i+1] = (uint8_t)(c[2*i+1] ^ keR);
    }
}

static void ascii_line_host(const uint8_t* buf, int bytes, char* out, int* pos) {
    int sp = 1;
    for (int i = 0; i < bytes; ++i) {
        uint8_t v = buf[i];
        if (v >= 'a' && v <= 'z') v = (uint8_t)(v - 32);
        if ((v >= 'A' && v <= 'Z') || v == ' ') {
            if (v == ' ') {
                if (!sp) {
                    out[(*pos)++] = ' ';
                    sp = 1;
                }
            } else {
                out[(*pos)++] = (char)v;
                sp = 0;
            }
        } else {
            if (!sp) {
                out[(*pos)++] = ' ';
                sp = 1;
            }
        }
    }
    while (*pos > 0 && out[*pos - 1] == ' ') (*pos)--;
    out[(*pos)++] = '\n';
}

extern "C" const char* blockCuda(const char* encText, const char* frag, char flag) {
    static char* out = NULL;
    if (out) { free(out); out = NULL; }

    int bigN = 0;
    int* encInt = parse_frag_array(encText, &bigN);
    if (!encInt || bigN <= 0 || (bigN & 1)) {
        if (encInt) free(encInt);
        return "[cuda-block] invalid ciphertext";
    }

    int keyN = 0;
    int* keys = parse_frag_array(frag, &keyN);
    if (!keys || keyN < 3) {
        if (encInt) free(encInt);
        if (keys) free(keys);
        return "[cuda-block] invalid keys";
    }

    int pairs = bigN / 2;
    uint8_t* cbytes = (uint8_t*)malloc((size_t)bigN);
    if (!cbytes) { free(encInt); free(keys); return "[cuda-block] alloc fail"; }
    for (int i = 0; i < bigN; ++i) {
        int v = encInt[i];
        if (v < 0) v = 0; if (v > 255) v = 255;
        cbytes[i] = (uint8_t)v;
    }

    uint8_t *d_c = NULL, *d_ecb = NULL, *d_cbc = NULL, *d_cfb = NULL, *d_crt = NULL;
    int* d_keys = NULL;
    cudaMalloc((void**)&d_c, (size_t)bigN);
    cudaMalloc((void**)&d_ecb, (size_t)bigN);
    cudaMalloc((void**)&d_cbc, (size_t)bigN);
    cudaMalloc((void**)&d_cfb, (size_t)bigN);
    cudaMalloc((void**)&d_crt, (size_t)bigN);
    cudaMalloc((void**)&d_keys, 3 * sizeof(int));
    cudaMemcpy(d_c, cbytes, (size_t)bigN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_keys, keys, 3 * sizeof(int), cudaMemcpyHostToDevice);

    blockKernel<<<1,32>>>(d_c, pairs, d_keys, (unsigned char)flag, d_ecb, d_cbc, d_cfb, d_crt);
    cudaDeviceSynchronize();

    uint8_t* h_ecb = (uint8_t*)malloc((size_t)bigN);
    uint8_t* h_cbc = (uint8_t*)malloc((size_t)bigN);
    uint8_t* h_cfb = (uint8_t*)malloc((size_t)bigN);
    uint8_t* h_crt = (uint8_t*)malloc((size_t)bigN);
    if (!h_ecb || !h_cbc || !h_cfb || !h_crt) {
        free(cbytes); free(keys);
        free(h_ecb); free(h_cbc); free(h_cfb); free(h_crt);
        cudaFree(d_c); cudaFree(d_ecb); cudaFree(d_cbc); cudaFree(d_cfb); cudaFree(d_crt); cudaFree(d_keys);
        return "[cuda-block] alloc fail";
    }

    cudaMemcpy(h_ecb, d_ecb, (size_t)bigN, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cbc, d_cbc, (size_t)bigN, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cfb, d_cfb, (size_t)bigN, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_crt, d_crt, (size_t)bigN, cudaMemcpyDeviceToHost);

    size_t outcap = (size_t)bigN * 4 + 16;
    out = (char*)malloc(outcap);
    if (!out) {
        free(cbytes); free(keys); free(h_ecb); free(h_cbc); free(h_cfb); free(h_crt);
        cudaFree(d_c); cudaFree(d_ecb); cudaFree(d_cbc); cudaFree(d_cfb); cudaFree(d_crt); cudaFree(d_keys);
        return "[cuda-block] alloc fail";
    }
    int pos = 0;
    ascii_line_host(h_ecb, bigN, out, &pos);
    if (pairs > 1) ascii_line_host(h_cbc, bigN - 2, out, &pos); else out[pos++] = '\n';
    if (pairs > 1) ascii_line_host(h_cfb, bigN - 2, out, &pos); else out[pos++] = '\n';
    ascii_line_host(h_crt, bigN, out, &pos);
    out[pos] = '\0';

    free(cbytes); free(keys); free(h_ecb); free(h_cbc); free(h_cfb); free(h_crt);
    cudaFree(d_c); cudaFree(d_ecb); cudaFree(d_cbc); cudaFree(d_cfb); cudaFree(d_crt); cudaFree(d_keys);
    return out;
}
#endif
