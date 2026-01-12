#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef USE_CUDA

__global__ void factorKernel(uint64_t n, uint64_t limit, int* found, uint64_t* outP) {
    uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;

    for (uint64_t d = 2 + tid; d <= limit; d += stride) {
        if (atomicAdd(found, 0) != 0) return;
        if (n % d == 0) {
            if (atomicCAS(found, 0, 1) == 0) {
                *outP = d;
            }
            return;
        }
    }
}

static int factor64_cuda(uint64_t n, uint64_t limit, uint64_t* p_out) {
    int* d_found = NULL;
    uint64_t* d_p = NULL;
    cudaMalloc((void**)&d_found, sizeof(int));
    cudaMalloc((void**)&d_p, sizeof(uint64_t));
    int zero = 0;
    cudaMemcpy(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(256);
    dim3 blocks(256);
    factorKernel<<<blocks, threads>>>(n, limit, d_found, d_p);
    cudaDeviceSynchronize();

    int h_found = 0;
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_found) cudaMemcpy(p_out, d_p, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_found);
    cudaFree(d_p);
    return h_found;
}

static uint64_t parse_u64(const char* s, int* ok) {
    *ok = 0;
    if (!s) return 0;
    while (*s == ' ' || *s == '\t') ++s;
    if (!*s) return 0;
    uint64_t v = 0;
    for (; *s; ++s) {
        if (*s < '0' || *s > '9') return 0;
        uint64_t nv = v * 10 + (uint64_t)(*s - '0');
        if (nv < v) return 0;
        v = nv;
    }
    *ok = 1;
    return v;
}

static uint64_t modexp64(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t res = 1 % mod;
    uint64_t b = base % mod;
    uint64_t e = exp;
    while (e) {
        if (e & 1) res = (uint64_t)((__uint128_t)res * b % mod);
        b = (uint64_t)((__uint128_t)b * b % mod);
        e >>= 1;
    }
    return res;
}

static uint64_t egcd_inv64(uint64_t a, uint64_t m, int* ok) {
    int64_t t = 0, newt = 1;
    int64_t r = (int64_t)m, newr = (int64_t)a;
    while (newr != 0) {
        int64_t q = r / newr;
        int64_t tmp = newt; newt = t - q * newt; t = tmp;
        tmp = newr; newr = r - q * newr; r = tmp;
    }
    if (r > 1) { *ok = 0; return 0; }
    if (t < 0) t += (int64_t)m;
    *ok = 1;
    return (uint64_t)t;
}

static uint64_t gcd64(uint64_t a, uint64_t b) {
    while (b) { uint64_t t = b; b = a % b; a = t; }
    return a;
}

static int parse_pair(const char* frag, uint64_t* a, uint64_t* b) {
    char tmp[256];
    if (!frag) return 0;
    const char* p = frag;
    if (strncmp(p, "brute:", 6) == 0) p += 6;
    strncpy(tmp, p, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';
    char* tok1 = strtok(tmp, ",[]");
    char* tok2 = strtok(NULL, ",[]");
    int ok1 = 0, ok2 = 0;
    if (tok1) *a = parse_u64(tok1, &ok1);
    if (tok2) *b = parse_u64(tok2, &ok2);
    return ok1 && ok2;
}

extern "C" const char* rsaBruteCuda(const char* alph, const char* encText, const char* frag) {
    (void)alph;
    static char outbuf[256];
    uint64_t n = 0, e = 0;
    if (!parse_pair(frag, &n, &e) || !encText) return "[rsa cuda] frag must be brute:[n,e] and ciphertext int";

    int okC = 0;
    uint64_t c = parse_u64(encText, &okC);
    if (!okC) return "[rsa cuda] ciphertext must be small integer";

    if (n < 4 || e == 0) return "[rsa cuda] invalid n or e";

    double rootd = sqrt((double)n);
    uint64_t limit = (uint64_t)rootd + 1;
    uint64_t p = 0;
    if (!factor64_cuda(n, limit, &p)) return "[rsa cuda] factor not found (too large)";
    if (p <= 1 || p >= n || n % p != 0) return "[rsa cuda] bad factor result";
    uint64_t q = n / p;
    uint64_t phi = (p - 1) * (q - 1);
    if (gcd64(e, phi) != 1) return "[rsa cuda] e not coprime with phi";
    int ok = 0;
    uint64_t d = egcd_inv64(e % phi, phi, &ok);
    if (!ok) return "[rsa cuda] inverse failed";
    uint64_t m = modexp64(c, d, n);
    snprintf(outbuf, sizeof(outbuf), "p=%llu q=%llu d=%llu m=%llu",
             (unsigned long long)p, (unsigned long long)q,
             (unsigned long long)d, (unsigned long long)m);
    return outbuf;
}

static void rabin_crt(uint64_t p, uint64_t q, uint64_t c, uint64_t* out_m) {
    uint64_t n = p * q;
    uint64_t exp_p = (p + 1) / 4;
    uint64_t exp_q = (q + 1) / 4;
    uint64_t r_p = modexp64(c, exp_p, p);
    uint64_t r_q = modexp64(c, exp_q, q);
    int ok = 0;
    uint64_t inv_p = egcd_inv64(p % q, q, &ok);
    if (!ok) { *out_m = 0; return; }
    uint64_t diff = (r_q + q - r_p) % q;
    uint64_t h = (uint64_t)((__uint128_t)diff * inv_p % q);
    uint64_t x = r_p + p * h;
    *out_m = x % n;
}

extern "C" const char* rabinBruteCuda(const char* alph, const char* encText, const char* frag) {
    (void)alph;
    static char outbuf[256];
    if (!encText || !frag) return "[rabin cuda] need encText and frag";
    int ok = 0;
    uint64_t n = parse_u64(frag + (strncmp(frag, "brute:", 6) == 0 ? 6 : 0), &ok);
    if (!ok || n < 9) return "[rabin cuda] frag must be brute:n (small)";
    int okC = 0;
    uint64_t c = parse_u64(encText, &okC);
    if (!okC) return "[rabin cuda] ciphertext must be small integer";

    double rootd = sqrt((double)n);
    uint64_t limit = (uint64_t)rootd + 1;
    uint64_t p = 0;
    if (!factor64_cuda(n, limit, &p)) return "[rabin cuda] factor not found (too large)";
    uint64_t q = n / p;
    if (p % 4 != 3 || q % 4 != 3) {
        snprintf(outbuf, sizeof(outbuf), "[rabin cuda] factors not 3 mod 4: p=%llu q=%llu",
                 (unsigned long long)p, (unsigned long long)q);
        return outbuf;
    }
    uint64_t m = 0;
    rabin_crt(p, q, c, &m);
    snprintf(outbuf, sizeof(outbuf), "p=%llu q=%llu m=%llu",
             (unsigned long long)p, (unsigned long long)q, (unsigned long long)m);
    return outbuf;
}

#endif
