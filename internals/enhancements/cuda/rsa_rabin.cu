#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef USE_CUDA

__device__ __forceinline__ uint64_t gcd64_dev(uint64_t a, uint64_t b) {
    while (b) { uint64_t t = b; b = a % b; a = t; }
    return a;
}

__device__ __forceinline__ uint64_t mulmod64_dev(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)((__uint128_t)a * (__uint128_t)b % m);
}

__device__ __forceinline__ uint64_t rho_f(uint64_t x, uint64_t c, uint64_t m) {
    return (mulmod64_dev(x, x, m) + c) % m;
}

/* Bounded Pollard Rho scan: many seeds in parallel, each with limited iterations. */
__global__ void pollardRhoKernel(uint64_t n, uint64_t c_base, int max_iters, int* found, uint64_t* outP) {
    uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if ((n & 1ULL) == 0ULL) {
        if (atomicCAS(found, 0, 1) == 0) *outP = 2;
        return;
    }

    uint64_t c = (c_base ^ tid) | 1ULL;      /* ensure odd/non-zero */
    uint64_t x = tid + 2ULL;
    uint64_t y = x;

    for (int i = 0; i < max_iters; ++i) {
        if (atomicAdd(found, 0) != 0) return;
        x = rho_f(x, c, n);
        y = rho_f(rho_f(y, c, n), c, n);
        uint64_t d = gcd64_dev(x > y ? x - y : y - x, n);
        if (d > 1 && d < n) {
            if (atomicCAS(found, 0, 1) == 0) *outP = d;
            return;
        }
    }
}

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

static int factor64_cuda_rho(uint64_t n, uint64_t* p_out) {
    int* d_found = NULL;
    uint64_t* d_p = NULL;
    cudaMalloc((void**)&d_found, sizeof(int));
    cudaMalloc((void**)&d_p, sizeof(uint64_t));
    int zero = 0;
    cudaMemcpy(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(256);
    dim3 blocks(256);
    pollardRhoKernel<<<blocks, threads>>>(n, 1, 4096, d_found, d_p);
    cudaDeviceSynchronize();

    int h_found = 0;
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_found) cudaMemcpy(p_out, d_p, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_found);
    cudaFree(d_p);
    return h_found;
}

static int factor64_cuda_trial(uint64_t n, uint64_t limit, uint64_t* p_out) {
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

static int factor64_cuda(uint64_t n, uint64_t limit, uint64_t* p_out) {
    /* Try bounded Pollard Rho first, fall back to trial division. */
    if (factor64_cuda_rho(n, p_out)) return 1;
    return factor64_cuda_trial(n, limit, p_out);
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

/* ---------------- Known-key RSA modexp (<=128-bit) ---------------- */
__device__ __forceinline__ unsigned __int128 modmul128(unsigned __int128 a, unsigned __int128 b, unsigned __int128 m) {
    return (a * b) % m;
}

__device__ unsigned __int128 modexp128(unsigned __int128 base, unsigned __int128 exp, unsigned __int128 mod) {
    unsigned __int128 res = 1 % mod;
    base %= mod;
    while (exp) {
        if (exp & 1) res = modmul128(res, base, mod);
        base = modmul128(base, base, mod);
        exp >>= 1;
    }
    return res;
}

__global__ void rsaModexpKernel(uint64_t n_lo, uint64_t n_hi, uint64_t d_lo, uint64_t d_hi,
                                uint64_t c_lo, uint64_t c_hi, uint64_t* out_lo, uint64_t* out_hi) {
    unsigned __int128 n = ((unsigned __int128)n_hi << 64) | (unsigned __int128)n_lo;
    unsigned __int128 d = ((unsigned __int128)d_hi << 64) | (unsigned __int128)d_lo;
    unsigned __int128 c = ((unsigned __int128)c_hi << 64) | (unsigned __int128)c_lo;
    unsigned __int128 m = modexp128(c, d, n);
    *out_lo = (uint64_t)(m & 0xFFFFFFFFFFFFFFFFULL);
    *out_hi = (uint64_t)(m >> 64);
}

static int parse_u128_host(const char* s, unsigned __int128* out) {
    if (!s || !*s) return 0;
    unsigned __int128 v = 0;
    while (*s == ' ' || *s == '\t') ++s;
    for (; *s; ++s) {
        if (*s < '0' || *s > '9') break;
        v = v * 10 + (unsigned __int128)(*s - '0');
    }
    *out = v;
    return 1;
}

static int parse_u128_list(const char* s, unsigned __int128* out, int max_count) {
    if (!s || !*s || !out || max_count <= 0) return 0;
    int count = 0;
    const char* p = s;
    while (*p && count < max_count) {
        while (*p && (*p < '0' || *p > '9')) ++p;
        if (!*p) break;
        unsigned __int128 v = 0;
        while (*p >= '0' && *p <= '9') {
            v = v * 10 + (unsigned __int128)(*p - '0');
            ++p;
        }
        out[count++] = v;
    }
    return count;
}

static int parse_triplet_ned(const char* frag, unsigned __int128* n, unsigned __int128* e, unsigned __int128* d) {
    if (!frag) return 0;
    const char* p = strrchr(frag, '[');
    if (!p) return 0;
    ++p;
    char buf[256]; int bi = 0;
    unsigned __int128 vals[3]; int vi = 0;
    while (*p && *p != ']') {
        if (*p == ',' || *p == ']') {
            buf[bi] = '\0';
            if (bi > 0) {
                if (vi < 3) parse_u128_host(buf, &vals[vi]);
                vi++;
            }
            bi = 0;
        } else {
            if (bi < (int)sizeof(buf) - 1) buf[bi++] = *p;
        }
        ++p;
    }
    if (bi > 0 && vi < 3) {
        buf[bi] = '\0';
        parse_u128_host(buf, &vals[vi]);
        vi++;
    }
    if (vi < 3) return 0;
    *n = vals[0]; *e = vals[1]; *d = vals[2];
    return 1;
}

static void u128_to_dec(unsigned __int128 v, char* out, size_t cap) {
    char tmp[64]; int pos = 0;
    if (v == 0) {
        snprintf(out, cap, "0");
        return;
    }
    while (v && pos < (int)sizeof(tmp)) {
        unsigned __int128 q = v / 10;
        unsigned int rem = (unsigned int)(v - q * 10);
        tmp[pos++] = (char)('0' + rem);
        v = q;
    }
    size_t k = 0;
    while (pos > 0 && k + 1 < cap) out[k++] = tmp[--pos];
    out[k] = '\0';
}

static const char* decode_u64_text(uint64_t m, const char* alph) {
    static char outbuf[128];
    const char* a = (alph && *alph) ? alph : "ABCDEFGHIJKLMNOPQRSTUVWXYZ ";
    int alen = (int)strlen(a);
    char dec[32];
    snprintf(dec, sizeof(dec), "%llu", (unsigned long long)m);
    char tmp[34];
    const char* p = dec;
    int len = (int)strlen(dec);
    if (len % 2 != 0) {
        tmp[0] = '0';
        strncpy(tmp + 1, dec, sizeof(tmp) - 2);
        tmp[sizeof(tmp) - 1] = '\0';
        p = tmp;
        len++;
    }
    char plain[64];
    int pos = 0;
    for (int i = 0; i + 1 < len && pos < (int)sizeof(plain) - 1; i += 2) {
        int d1 = p[i] - '0';
        int d2 = p[i + 1] - '0';
        if (d1 < 0 || d1 > 9 || d2 < 0 || d2 > 9) { plain[pos++] = '?'; continue; }
        int code = d1 * 10 + d2;
        if (code >= 1 && code <= alen) plain[pos++] = a[code - 1];
        else plain[pos++] = '?';
    }
    plain[pos] = '\0';
    snprintf(outbuf, sizeof(outbuf), "%s", plain);
    return outbuf;
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
    const char* txt = decode_u64_text(m, alph);
    snprintf(outbuf, sizeof(outbuf), "p=%llu q=%llu d=%llu m=%llu\n%s",
             (unsigned long long)p, (unsigned long long)q,
             (unsigned long long)d, (unsigned long long)m, txt);
    return outbuf;
}

static int rabin_roots64(uint64_t p, uint64_t q, uint64_t c, uint64_t roots[4]) {
    uint64_t n = p * q;
    uint64_t exp_p = (p + 1) / 4;
    uint64_t exp_q = (q + 1) / 4;
    uint64_t r_p = modexp64(c, exp_p, p);
    uint64_t r_q = modexp64(c, exp_q, q);

    int ok = 0;
    uint64_t inv_p_mod_q = egcd_inv64(p % q, q, &ok); /* p^-1 mod q */
    if (!ok) return 0;
    uint64_t inv_q_mod_p = egcd_inv64(q % p, p, &ok); /* q^-1 mod p */
    if (!ok) return 0;

    int idx = 0;
    for (int sp = 0; sp < 2; ++sp) {
        uint64_t ap = (sp == 0) ? r_p : ((p - r_p) % p);
        for (int sq = 0; sq < 2; ++sq) {
            uint64_t aq = (sq == 0) ? r_q : ((q - r_q) % q);

            __uint128_t t1 = (__uint128_t)ap * q;
            t1 = (t1 % n) * (__uint128_t)inv_q_mod_p % n;

            __uint128_t t2 = (__uint128_t)aq * p;
            t2 = (t2 % n) * (__uint128_t)inv_p_mod_q % n;

            uint64_t x = (uint64_t)((t1 + t2) % n);
            roots[idx++] = x;
        }
    }
    return idx;
}

extern "C" const char* rabinBruteCuda(const char* alph, const char* encText, const char* frag) {
    (void)alph;
    static char outbuf[2048];
    if (!encText || !frag) return "[rabin cuda] need encText and frag";
    int ok = 0;
    uint64_t n = parse_u64(frag + (strncmp(frag, "brute:", 6) == 0 ? 6 : 0), &ok);
    if (!ok || n < 9) return "[rabin cuda] frag must be brute:n (small)";

    uint64_t cvals[128];
    int ccount = 0;
    {
        const char* p = encText;
        while (*p && ccount < 128) {
            while (*p && (*p < '0' || *p > '9')) ++p;
            if (!*p) break;
            uint64_t v = 0;
            while (*p >= '0' && *p <= '9') { v = v * 10 + (uint64_t)(*p - '0'); ++p; }
            cvals[ccount++] = v;
        }
    }
    if (ccount == 0) return "[rabin cuda] ciphertext must be integer or list";

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

    char text[1024]; text[0] = '\0';
    for (int i = 0; i < ccount; ++i) {
        uint64_t roots[4] = {0,0,0,0};
        int rcount = rabin_roots64(p, q, cvals[i], roots);
        if (rcount == 0) continue;
        char decoded[4][128];
        for (int k = 0; k < rcount; ++k) {
            const char* txt = decode_u64_text(roots[k], alph);
            strncpy(decoded[k], txt, sizeof(decoded[k]) - 1);
            decoded[k][sizeof(decoded[k]) - 1] = '\0';
        }
        int best = 0;
        int bestScore = -1;
        for (int k = 0; k < rcount; ++k) {
            if (strchr(decoded[k], '?')) continue;
            int spaces = 0, vowels = 0;
            for (const char* t = decoded[k]; *t; ++t) {
                if (*t == ' ') spaces++;
                if (strchr("AEIOUY", *t)) vowels++;
            }
            int score = spaces * 100 + vowels;
            if (score > bestScore) {
                bestScore = score;
                best = k;
            }
        }
        if (bestScore < 0 && rcount > 0) best = 0; /* fall back */
        strncat(text, decoded[best], sizeof(text) - strlen(text) - 1);
    }

    snprintf(outbuf, sizeof(outbuf), "p=%llu q=%llu\n%s",
             (unsigned long long)p, (unsigned long long)q, text);
    return outbuf;
}

extern "C" const char* rsaModexpCuda(const char* alph, const char* encText, const char* frag) {
    (void)alph;
    static char outbuf[2048];
    unsigned __int128 n=0,e=0,d=0;
    if (!parse_triplet_ned(frag, &n, &e, &d)) return "[rsa cuda modexp] frag must be [n,e,d]:[...]";
    unsigned __int128 cvals[128];
    int ccount = parse_u128_list(encText, cvals, 128);
    if (ccount == 0) return "[rsa cuda modexp] bad ciphertext";
    uint64_t n_lo = (uint64_t)n, n_hi = (uint64_t)(n >> 64);
    uint64_t d_lo = (uint64_t)d, d_hi = (uint64_t)(d >> 64);
    uint64_t *d_out_lo=NULL, *d_out_hi=NULL;
    cudaMalloc((void**)&d_out_lo, sizeof(uint64_t));
    cudaMalloc((void**)&d_out_hi, sizeof(uint64_t));

    char text[1024]; text[0] = '\0';
    char last_mdec[80]; last_mdec[0] = '\0';

    for (int i = 0; i < ccount; ++i) {
        uint64_t c_lo = (uint64_t)cvals[i], c_hi = (uint64_t)(cvals[i] >> 64);
        rsaModexpKernel<<<1,1>>>(n_lo, n_hi, d_lo, d_hi, c_lo, c_hi, d_out_lo, d_out_hi);
        cudaDeviceSynchronize();
        uint64_t h_lo=0,h_hi=0;
        cudaMemcpy(&h_lo, d_out_lo, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_hi, d_out_hi, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        unsigned __int128 m = ((unsigned __int128)h_hi << 64) | (unsigned __int128)h_lo;
        u128_to_dec(m, last_mdec, sizeof(last_mdec));
        if (h_hi == 0) {
            const char* t = decode_u64_text(h_lo, alph);
            strncat(text, t, sizeof(text) - strlen(text) - 1);
        }
    }
    cudaFree(d_out_lo); cudaFree(d_out_hi);

    snprintf(outbuf, sizeof(outbuf), "n=%llu... d=%llu...\n%s\n%s",
             (unsigned long long)n_lo, (unsigned long long)d_lo, last_mdec, text);
    return outbuf;
}

#endif
