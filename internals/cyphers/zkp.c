#include "zkp.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <stdarg.h>

#define QUAD_ROUNDS 5

static uint64_t next_rand64(uint64_t* state) {
    *state = (*state * 6364136223846793005ULL + 1ULL);
    return *state;
}

static int appendf(char* out, size_t* pos, size_t cap, const char* fmt, ...) {
    if (*pos >= cap) return 0;
    va_list ap;
    va_start(ap, fmt);
    int w = vsnprintf(out + *pos, cap - *pos, fmt, ap);
    va_end(ap);
    if (w < 0 || *pos + (size_t)w >= cap) {
        if (cap) out[cap - 1] = '\0';
        *pos = cap ? cap - 1 : 0;
        return 0;
    }
    *pos += (size_t)w;
    return 1;
}

static int hash_int_list(const unsigned long long* arr, int n, unsigned char digest[32]) {
    char buf[512];
    size_t pos = 0;
    for (int i = 0; i < n; ++i) {
        int w = snprintf(buf + pos, sizeof buf - pos, "%llu", arr[i]);
        if (w < 0 || pos + (size_t)w >= sizeof buf) return 0;
        pos += (size_t)w;
    }
    sha256((const unsigned char*)buf, pos, digest);
    return 1;
}

static void challenge_bits(const unsigned long long* commitments, char out[6]) {
    unsigned char digest[32];
    if (!hash_int_list(commitments, QUAD_ROUNDS, digest)) {
        memcpy(out, "00000", 6);
        return;
    }
    unsigned char last = digest[31];
    for (int i = 4; i >= 0; --i) out[4 - i] = (char)(((last >> i) & 1u) ? '1' : '0');
    out[5] = '\0';
}

static unsigned long long hash_to_modulus(const unsigned long long* arr, int n, unsigned long long mod) {
    if (mod == 0) return 0;
    unsigned char digest[32];
    if (!hash_int_list(arr, n, digest)) return 0;
    unsigned long long acc = 0ULL;
    for (int i = 0; i < 32; ++i) {
        acc = mulmod_u64(acc, 256ULL, mod);
        acc = (acc + (unsigned long long)(digest[i] % mod)) % mod;
    }
    return acc;
}

static unsigned long long find_discrete_log(unsigned long long g, unsigned long long y, unsigned long long p) {
    if (p <= 1) return ULLONG_MAX;
    unsigned long long q = p - 1ULL;
    unsigned long long cur = 1ULL % p;
    g %= p;
    y %= p;
    for (unsigned long long x = 0; x < q; ++x) {
        if (cur == y) return x;
        cur = mulmod_u64(cur, g, p);
    }
    return ULLONG_MAX;
}

static int append_array(char* out, size_t* pos, size_t cap, const unsigned long long* arr, int n) {
    if (!appendf(out, pos, cap, "[")) return 0;
    for (int i = 0; i < n; ++i) {
        if (!appendf(out, pos, cap, "%s%llu", (i == 0) ? "" : ", ", arr[i])) return 0;
    }
    return appendf(out, pos, cap, "]");
}

static int compute_quadratic(uint64_t* rng_state, unsigned long long c, unsigned long long p, unsigned long long outC[QUAD_ROUNDS], unsigned long long outP[QUAD_ROUNDS], char bits[6]) {
    if (p < 3 || (p & 3ULL) != 3ULL) return 0;
    unsigned long long u = modexp_u64(c % p, (p + 1ULL) / 4ULL, p);

    unsigned long long R[QUAD_ROUNDS], U[QUAD_ROUNDS];
    for (int i = 0; i < QUAD_ROUNDS; ++i) {
        unsigned long long r = (next_rand64(rng_state) % (p - 1ULL)) + 1ULL;
        R[i] = r;
        unsigned long long r2 = mulmod_u64(r, r, p);
        unsigned long long inv_r2 = modinv_u64(r2, p);
        unsigned long long inv_r = modinv_u64(r % p, p);
        if (inv_r2 == 0ULL || inv_r == 0ULL) return 0;
        outC[i] = mulmod_u64(c % p, inv_r2, p);
        U[i] = mulmod_u64(u, inv_r, p);
    }

    challenge_bits(outC, bits);
    for (int i = 0; i < QUAD_ROUNDS; ++i) outP[i] = (bits[i] == '1') ? U[i] : R[i];
    return 1;
}

static int compute_dlog(uint64_t* rng_state, unsigned long long g, unsigned long long y, unsigned long long p, unsigned long long* out_c, unsigned long long* out_r) {
    if (p <= 2) return 0;
    unsigned long long q = p - 1ULL;
    unsigned long long x = find_discrete_log(g, y, p);
    if (x == ULLONG_MAX) return 0;

    unsigned long long w = (next_rand64(rng_state) % q) + 1ULL;
    unsigned long long t = modexp_u64(g % p, w, p);
    unsigned long long pack[3] = { g % p, y % p, t };
    unsigned long long challenge = hash_to_modulus(pack, 3, q);
    unsigned long long cx = mulmod_u64(challenge % q, x % q, q);
    unsigned long long s = (w + q - cx) % q;

    if (out_c) *out_c = challenge;
    if (out_r) *out_r = s;
    return 1;
}

static uint64_t parse_seed(const char* frag, uint64_t fallback) {
    if (!frag) return fallback;
    const char* p = strstr(frag, "seed");
    if (!p) return fallback;
    const char* sep = strchr(p, ':');
    if (!sep) sep = strchr(p, '=');
    if (!sep) return fallback;
    uint64_t v = (uint64_t)strtoull(sep + 1, NULL, 10);
    return v ? v : fallback;
}

static uint64_t derive_seed(unsigned long long c, unsigned long long p, unsigned long long g, unsigned long long y, const char* frag, const char* encText) {
    unsigned char digest[32];
    unsigned long long arr[4] = {c, p, g, y};
    hash_int_list(arr, 4, digest);
    uint64_t s = 0;
    for (int i = 0; i < 8; ++i) {
        s = (s << 8) | digest[i];
    }
    if (frag && *frag) {
        /* fold frag bytes */
        for (const unsigned char* pch = (const unsigned char*)frag; *pch; ++pch) {
            s = s * 1315423911u + (uint64_t)(*pch);
        }
    }
    if (encText && *encText) {
        for (const unsigned char* pch = (const unsigned char*)encText; *pch; ++pch) {
            s = s * 2654435761u + (uint64_t)(*pch);
        }
    }
    if (s == 0) s = 1; /* avoid zero state */
    return s;
}

const char* zkpEntry(const char* alph, const char* encText, const char* frag) {
    (void)alph;
    static char out[2048];
    size_t pos = 0;
    out[0] = '\0';

    unsigned long long quad_c = 63723ULL, quad_p = 100003ULL;
    unsigned long long dlog_g = 2ULL, dlog_y = 10842ULL, dlog_p = 100003ULL;
    uint64_t seed = 0;

    if (frag && *frag) {
        int count = 0;
        unsigned long long* vals = parse_ull_array(frag, &count);
        if (vals) {
            if (count >= 2) { quad_c = vals[0]; quad_p = vals[1]; }
            if (count == 3) { dlog_g = vals[0]; dlog_y = vals[1]; dlog_p = vals[2]; }
            else if (count >= 5) { dlog_g = vals[2]; dlog_y = vals[3]; dlog_p = vals[4]; }
            free(vals);
        }
    }

    /* deterministic default seed derived from inputs unless user overrides */
    seed = parse_seed(frag, 0);
    if (seed == 0) seed = derive_seed(quad_c, quad_p, dlog_g, dlog_y, frag, encText);
    uint64_t rng_state = seed;

    unsigned long long Cvals[QUAD_ROUNDS], Pvals[QUAD_ROUNDS];
    char bits[6];
    if (!compute_quadratic(&rng_state, quad_c, quad_p, Cvals, Pvals, bits)) {
        appendf(out, &pos, sizeof out, "[zkp quad error]");
        return out;
    }

    unsigned long long chal = 0, resp = 0;
    if (!compute_dlog(&rng_state, dlog_g, dlog_y, dlog_p, &chal, &resp)) {
        appendf(out, &pos, sizeof out, "[zkp dlog error]");
        return out;
    }

    appendf(out, &pos, sizeof out, "1) P = ");
    append_array(out, &pos, sizeof out, Pvals, QUAD_ROUNDS);
    appendf(out, &pos, sizeof out, "\nC = ");
    append_array(out, &pos, sizeof out, Cvals, QUAD_ROUNDS);
    appendf(out, &pos, sizeof out, "\n\n2) c = %llu, r = %llu\n", chal, resp);
    return out;
}
