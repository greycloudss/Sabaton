#ifdef __cplusplus
extern "C" {
#endif
#include "../../cyphers/elliptic.h"
#include "../../cyphers/elgamal.h"
#ifdef __cplusplus
}
#endif
#include "../../../util/fragmentation.h"
#include "../../../util/number.h"
#include "../../../util/string.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Device helpers for small 64-bit EC arithmetic */
__device__ __forceinline__ unsigned long long mod_add64(unsigned long long a, unsigned long long b, unsigned long long m) {
    unsigned long long res = a + b;
    if (res >= m || res < a) res = res % m;
    return res % m;
}

__device__ __forceinline__ unsigned long long mod_sub64(unsigned long long a, unsigned long long b, unsigned long long m) {
    return (a >= b) ? (a - b) % m : (m - ((b - a) % m)) % m;
}

__device__ __forceinline__ unsigned long long mod_mul64(unsigned long long a, unsigned long long b, unsigned long long m) {
    unsigned __int128 prod = (unsigned __int128)a * (unsigned __int128)b;
    return (unsigned long long)(prod % m);
}

__device__ static unsigned long long modinv64_dev(unsigned long long a, unsigned long long m) {
    long long t = 0, newt = 1;
    long long r = (long long)m, newr = (long long)a;
    while (newr != 0) {
        long long q = r / newr;
        long long tmp = newt; newt = t - q * newt; t = tmp;
        tmp = newr; newr = r - q * newr; r = tmp;
    }
    if (t < 0) t += (long long)m;
    return (unsigned long long)t;
}

typedef struct { unsigned long long x, y; int inf; } ECPoint_dev;

__device__ static ECPoint_dev ec_inf_dev(void) { ECPoint_dev R; R.x = 0; R.y = 0; R.inf = 1; return R; }
__device__ static ECPoint_dev ec_xy_dev(unsigned long long x, unsigned long long y) { ECPoint_dev P; P.x = x; P.y = y; P.inf = 0; return P; }

__device__ static ECPoint_dev ec_add_dev(unsigned long long q, unsigned long long a_mod, ECPoint_dev P, ECPoint_dev Q) {
    if (P.inf) return Q;
    if (Q.inf) return P;

    if (P.x == Q.x) {
        if ((P.y + Q.y) % q == 0ULL) return ec_inf_dev();
        if (P.y != Q.y) return ec_inf_dev();
    }

    unsigned long long lambda;
    if (P.x == Q.x && P.y == Q.y) {
        if (P.y == 0ULL) return ec_inf_dev();
        unsigned long long x2 = mod_mul64(P.x, P.x, q);
        unsigned long long num = mod_add64(mod_mul64(3ULL, x2, q), a_mod, q);
        unsigned long long den = mod_mul64(2ULL, P.y, q);
        unsigned long long inv = modinv64_dev(den, q);
        lambda = mod_mul64(num, inv, q);
    } else {
        unsigned long long num = mod_sub64(Q.y, P.y, q);
        unsigned long long den = mod_sub64(Q.x, P.x, q);
        unsigned long long inv = modinv64_dev(den, q);
        lambda = mod_mul64(num, inv, q);
    }

    unsigned long long x3 = mod_sub64(mod_sub64(mod_mul64(lambda, lambda, q), P.x, q), Q.x, q);
    unsigned long long y3 = mod_sub64(mod_mul64(lambda, mod_sub64(P.x, x3, q), q), P.y, q);
    return ec_xy_dev(x3, y3);
}

__device__ static ECPoint_dev ec_mul_dev(unsigned long long q, unsigned long long a_mod, unsigned long long k, ECPoint_dev P) {
    ECPoint_dev R = ec_inf_dev();
    ECPoint_dev Q = P;
    while (k) {
        if (k & 1ULL) R = ec_add_dev(q, a_mod, R, Q);
        Q = ec_add_dev(q, a_mod, Q, Q);
        k >>= 1ULL;
    }
    return R;
}

typedef struct {
    unsigned long long Rx, Ry;
    unsigned long long c1, c2;
} ECBlock;

__global__ static void mv_decrypt_kernel(const ECBlock* blocks, int count,
                                         unsigned long long q, unsigned long long a_mod,
                                         unsigned long long n, unsigned long long priv_r,
                                         unsigned long long* out_m1, unsigned long long* out_m2,
                                         int* status) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    ECBlock b = blocks[idx];
    ECPoint_dev R = ec_xy_dev(b.Rx % q, b.Ry % q);
    ECPoint_dev S = ec_mul_dev(q, a_mod, priv_r, R);
    unsigned long long k1 = S.x % n;
    unsigned long long k2 = S.y % n;
    if (k1 == 0 || k2 == 0) { status[idx] = 0; return; }
    unsigned long long invk1 = modinv64_dev(k1, n);
    unsigned long long invk2 = modinv64_dev(k2, n);
    unsigned long long m1 = mod_mul64(b.c1 % n, invk1, n);
    unsigned long long m2 = mod_mul64(b.c2 % n, invk2, n);
    out_m1[idx] = m1;
    out_m2[idx] = m2;
    status[idx] = 1;
}

static char* build_utf8_from_pairs(const unsigned long long* m1, const unsigned long long* m2, int count) {
    char* out = (char*)malloc((size_t)count * 8 + 1);
    if (!out) return NULL;
    size_t pos = 0;
    for (int i = 0; i < count; ++i) {
        unsigned long long vals[2] = { m1[i], m2[i] };
        for (int j = 0; j < 2; ++j) {
            uint32_t cp = (vals[j] <= 0x10FFFFULL) ? (uint32_t)vals[j] : (uint32_t)'?';
            char buf[4];
            int n = utf8_encode_one(cp, buf);
            for (int k = 0; k < n; ++k) out[pos++] = buf[k];
        }
    }
    out[pos] = '\0';
    return out;
}

extern "C" const char* ellipticCuda(const char* alph, const char* encText, const char* frag) {
    (void)alph;
    static char* out = NULL;
    if (out) { free(out); out = NULL; }
    if (!frag || !*frag || !encText || !*encText) return "[elliptic cuda] missing input";

    FragMap map = fragmapParse(frag, '|', ':', '=');
    int ok = 0;
    long long q_ll = fragGetLongLong(&map, "q", 0, &ok);
    long long a_ll = fragGetLongLong(&map, "a", 0, &ok);
    long long b_ll = fragGetLongLong(&map, "b", 0, &ok);
    long long n_ll = fragGetLongLong(&map, "n", 0, &ok);
    long long r_ll = fragGetLongLong(&map, "r", 0, &ok);
    const char* Pstr = fragGetScalar(&map, "P");
    const char* mode = fragGetScalar(&map, "mode");
    fragmapFree(&map);
    if (!q_ll || !a_ll || !b_ll || !n_ll || !r_ll || !Pstr) return "[elliptic cuda] bad params";
    if (mode && mode[0] && !(mode[0] == 'm' || mode[0] == 'd')) {
        /* non-decrypt modes fall back to CPU */
        return ellipticEntry(alph, encText, frag);
    }

    int pcnt = 0;
    unsigned long long* parr = parse_ull_array(Pstr, &pcnt);
    if (!parr || pcnt < 2) { if (parr) free(parr); return "[elliptic cuda] bad P"; }
    unsigned long long Px = parr[0], Py = parr[1];
    free(parr);

    unsigned long long q = (unsigned long long)q_ll;
    unsigned long long a_mod = ((unsigned long long)((a_ll % (long long)q) + (long long)q)) % q;
    unsigned long long n = (unsigned long long)n_ll;
    unsigned long long priv_r = (unsigned long long)r_ll;
    (void)b_ll; /* unused but kept for parity */
    if ((Px | Py) == 0ULL) { /* appease warnings */ }

    int cnt = 0;
    unsigned long long* nums = parse_ull_array(encText, &cnt);
    if (!nums || cnt <= 0 || (cnt % 4) != 0) { if (nums) free(nums); return "[elliptic cuda] bad ciphertext"; }
    int blocks = cnt / 4;
    ECBlock* h_blocks = (ECBlock*)malloc((size_t)blocks * sizeof(ECBlock));
    if (!h_blocks) { free(nums); return "[elliptic cuda] OOM"; }
    for (int i = 0, bi = 0; bi < blocks; ++bi, i += 4) {
        h_blocks[bi].Rx = nums[i + 0];
        h_blocks[bi].Ry = nums[i + 1];
        h_blocks[bi].c1 = nums[i + 2];
        h_blocks[bi].c2 = nums[i + 3];
    }
    free(nums);

    ECBlock* d_blocks = NULL;
    unsigned long long* d_m1 = NULL;
    unsigned long long* d_m2 = NULL;
    int* d_status = NULL;
    cudaMalloc((void**)&d_blocks, (size_t)blocks * sizeof(ECBlock));
    cudaMalloc((void**)&d_m1, (size_t)blocks * sizeof(unsigned long long));
    cudaMalloc((void**)&d_m2, (size_t)blocks * sizeof(unsigned long long));
    cudaMalloc((void**)&d_status, (size_t)blocks * sizeof(int));
    cudaMemcpy(d_blocks, h_blocks, (size_t)blocks * sizeof(ECBlock), cudaMemcpyHostToDevice);
    cudaMemset(d_status, 0, (size_t)blocks * sizeof(int));

    int threads = 256;
    int grid = (blocks + threads - 1) / threads;
    mv_decrypt_kernel<<<grid, threads>>>(d_blocks, blocks, q, a_mod, n, priv_r, d_m1, d_m2, d_status);
    cudaDeviceSynchronize();

    int* h_status = (int*)malloc((size_t)blocks * sizeof(int));
    unsigned long long* h_m1 = (unsigned long long*)malloc((size_t)blocks * sizeof(unsigned long long));
    unsigned long long* h_m2 = (unsigned long long*)malloc((size_t)blocks * sizeof(unsigned long long));
    if (!h_status || !h_m1 || !h_m2) {
        cudaFree(d_blocks); cudaFree(d_m1); cudaFree(d_m2); cudaFree(d_status);
        free(h_blocks); if (h_status) free(h_status); if (h_m1) free(h_m1); if (h_m2) free(h_m2);
        return "[elliptic cuda] OOM";
    }
    cudaMemcpy(h_status, d_status, (size_t)blocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_m1, d_m1, (size_t)blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_m2, d_m2, (size_t)blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_blocks); cudaFree(d_m1); cudaFree(d_m2); cudaFree(d_status);
    free(h_blocks);

    for (int i = 0; i < blocks; ++i) { if (!h_status[i]) { free(h_status); free(h_m1); free(h_m2); return "[elliptic cuda] decrypt failed"; } }

    char* txt = build_utf8_from_pairs(h_m1, h_m2, blocks);
    free(h_status); free(h_m1); free(h_m2);
    if (!txt) return "[elliptic cuda] alloc failed";
    out = txt;
    return out;
}

/* ElGamal: heavy big-int use -> fallback to CPU path to preserve correctness */
static unsigned long long modexp64_host(unsigned long long base, unsigned long long exp, unsigned long long mod) {
    unsigned long long res = 1 % mod;
    base %= mod;
    while (exp) {
        if (exp & 1ULL) res = (unsigned long long)((__uint128_t)res * base % mod);
        base = (unsigned long long)((__uint128_t)base * base % mod);
        exp >>= 1ULL;
    }
    return res;
}

extern "C" const char* elgamalCuda(const char* alph, const char* encText, const char* frag) {
    (void)alph;
    static char* out = NULL;
    if (out) { free(out); out = NULL; }
    if (!encText || !*encText || !frag || !*frag) return "[elgamal cuda] missing input";

    /* Expect frag tokens p:<prime>|g:<gen>|a:<priv> */
    unsigned long long p = 0, g = 0, a = 0;
    char* copy = strdup(frag);
    if (!copy) return "[elgamal cuda] OOM";
    char* tok = strtok(copy, "|");
    while (tok) {
        while (*tok == ' ') ++tok;
        if (strncmp(tok, "p:", 2) == 0) p = strtoull(tok + 2, NULL, 10);
        else if (strncmp(tok, "g:", 2) == 0) g = strtoull(tok + 2, NULL, 10);
        else if (strncmp(tok, "a:", 2) == 0) a = strtoull(tok + 2, NULL, 10);
        tok = strtok(NULL, "|");
    }
    free(copy);
    if (p < 5 || g == 0 || a == 0) return "[elgamal cuda] bad params";

    int n = 0;
    unsigned long long* arr = parse_ull_array(encText, &n);
    if (!arr || (n % 2) != 0 || n == 0) { if (arr) free(arr); return "[elgamal cuda] bad ciphertext"; }

    char* buf = (char*)malloc((size_t)(n / 2) + 1);
    if (!buf) { free(arr); return "[elgamal cuda] OOM"; }
    int pos = 0;
    for (int i = 0; i < n; i += 2) {
        unsigned long long c1 = arr[i] % p;
        unsigned long long c2 = arr[i + 1] % p;
        unsigned long long inv = modexp64_host(c1, p - 1 - (a % (p - 1)), p);
        unsigned long long m = (unsigned long long)((__uint128_t)c2 * inv % p);
        buf[pos++] = (char)(m & 0xFFu);
    }
    buf[pos] = '\0';
    free(arr);
    out = buf;
    return out;
}

#endif
