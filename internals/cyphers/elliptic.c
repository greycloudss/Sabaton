#include "elliptic.h"

#include "../../util/fragmentation.h"
#include "../../util/number.h"
#include "../../util/string.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>


static unsigned long long gcd_u64_simple(unsigned long long a, unsigned long long b){
    while (b){ unsigned long long t = a % b; a = b; b = t; }
    return a;
}

static unsigned long long mod_s64(long long a, unsigned long long m){
    if (m == 0) return 0ULL;
    long long r = a % (long long)m;
    if (r < 0) r += (long long)m;
    return (unsigned long long)r;
}

static int has_char(const char* s, char c){
    if (!s) return 0;
    while (*s){ if (*s == c) return 1; ++s; }
    return 0;
}


typedef struct { unsigned long long x, y; int inf; } ECPoint;
typedef struct { unsigned long long q, a_mod, b_mod; } ECCurve;

static ECPoint ec_inf(void){ ECPoint R; R.x=0; R.y=0; R.inf=1; return R; }
static ECPoint ec_xy(unsigned long long x, unsigned long long y){ ECPoint P; P.x=x; P.y=y; P.inf=0; return P; }

static ECPoint ec_add(const ECCurve* E, ECPoint P, ECPoint Q){
    if (P.inf) return Q;
    if (Q.inf) return P;

    unsigned long long q = E->q;

    if (P.x == Q.x){
        if (mod_s64((long long)P.y + (long long)Q.y, q) == 0ULL) return ec_inf();
        if (P.y != Q.y) return ec_inf();
    }

    unsigned long long lambda;

    if (P.x == Q.x && P.y == Q.y){
        if (P.y == 0ULL) return ec_inf();
        unsigned long long x2  = mulmod_u64(P.x, P.x, q);
        unsigned long long num = mod_s64((long long)mulmod_u64(3ULL, x2, q) + (long long)E->a_mod, q);
        unsigned long long den = mod_s64((long long)2ULL * (long long)P.y, q);
        unsigned long long inv = modinv_u64(den, q);
        if (!inv) return ec_inf();
        lambda = mulmod_u64(num, inv, q);
    } else {
        unsigned long long num = mod_s64((long long)Q.y - (long long)P.y, q);
        unsigned long long den = mod_s64((long long)Q.x - (long long)P.x, q);
        unsigned long long inv = modinv_u64(den, q);
        if (!inv) return ec_inf();
        lambda = mulmod_u64(num, inv, q);
    }

    unsigned long long x3 = mod_s64((long long)mulmod_u64(lambda, lambda, q) - (long long)P.x - (long long)Q.x, q);
    unsigned long long y3 = mod_s64((long long)mulmod_u64(lambda, mod_s64((long long)P.x - (long long)x3, q), q) - (long long)P.y, q);

    return ec_xy(x3, y3);
}

static ECPoint ec_mul(const ECCurve* E, unsigned long long k, ECPoint P){
    ECPoint R = ec_inf();
    ECPoint Q = P;
    while (k){
        if (k & 1ULL) R = ec_add(E, R, Q);
        Q = ec_add(E, Q, Q);
        k >>= 1ULL;
    }
    return R;
}


static char* mv_decrypt(const ECCurve* E, unsigned long long n, unsigned long long priv_r, const char* cipherText){
    int cnt = 0;
    unsigned long long* nums = parse_ull_array(cipherText, &cnt);
    if (!nums || cnt <= 0 || (cnt % 4) != 0){
        free(nums);
        return strdup("[bad ciphertext]");
    }

    int blocks = cnt / 4;
    int cpCount = blocks * 2;

    uint32_t* cps = (uint32_t*)malloc((size_t)cpCount * sizeof(uint32_t));
    if (!cps){ free(nums); return strdup("[alloc failed]"); }

    for (int bi = 0, i = 0; bi < blocks; ++bi, i += 4){
        ECPoint R = ec_xy(nums[i] % E->q, nums[i+1] % E->q);
        unsigned long long c1 = nums[i+2] % n;
        unsigned long long c2 = nums[i+3] % n;

        ECPoint S = ec_mul(E, priv_r, R);
        unsigned long long k1 = S.x % n;
        unsigned long long k2 = S.y % n;

        unsigned long long invk1 = modinv_u64(k1, n);
        unsigned long long invk2 = modinv_u64(k2, n);
        if (!invk1 || !invk2){
            free(nums); free(cps);
            return strdup("[bad block: no inverse mod n]");
        }

        unsigned long long m1 = mulmod_u64(c1, invk1, n);
        unsigned long long m2 = mulmod_u64(c2, invk2, n);

        cps[2*bi + 0] = (m1 <= 0x10FFFFULL) ? (uint32_t)m1 : (uint32_t)'?';
        cps[2*bi + 1] = (m2 <= 0x10FFFFULL) ? (uint32_t)m2 : (uint32_t)'?';
    }

    int outCap = cpCount * 4 + 1;
    char* out = (char*)malloc((size_t)outCap);
    if (!out){ free(nums); free(cps); return strdup("[alloc failed]"); }

    u32_to_utf8(cps, cpCount, out, outCap);

    free(nums);
    free(cps);
    return out;
}


static char* elgamal_sign(const ECCurve* E, unsigned long long n, ECPoint P,
                          unsigned long long priv_r, const char* msgText,
                          const FragMap* map)
{
    int msgCnt = 0;
    unsigned long long* msgArr = parse_ull_array(msgText, &msgCnt);
    if (!msgArr || msgCnt <= 0){ free(msgArr); return strdup("[bad message]"); }
    unsigned long long m = msgArr[0] % n;
    free(msgArr);

    ECPoint Q = ec_mul(E, priv_r, P);

    int okk = 0;
    long long k_ll = fragGetLongLong(map, "k", 0, &okk);

    unsigned long long k = 0ULL;
    if (okk){
        if (k_ll <= 0) return strdup("[bad k]");
        k = (unsigned long long)k_ll % n;
        if (k == 0ULL) return strdup("[bad k]");
        if (gcd_u64_simple(k, n) != 1ULL) return strdup("[k not invertible mod n]");
    } else {
        unsigned long long seed = (unsigned long long)time(NULL) ^ (unsigned long long)clock();
        srand((unsigned int)seed);
        for (int tries = 0; tries < 200000; ++tries){
            unsigned long long cand = 1ULL + (unsigned long long)(rand() % (int)(n - 1ULL));
            if (gcd_u64_simple(cand, n) != 1ULL) continue;
            k = cand;
            break;
        }
        if (!k) return strdup("[failed to pick k]");
    }

    ECPoint gamma = ec_mul(E, k, P);
    unsigned long long alpha = gamma.x % n;
    if (alpha == 0ULL) return strdup("[degenerate signature: f(gamma)=0]");

    unsigned long long kinv = modinv_u64(k, n);
    if (!kinv) return strdup("[no inverse for k mod n]");

    unsigned long long ralpha = mulmod_u64((priv_r % n), alpha, n);
    unsigned long long rhs    = mod_s64((long long)m - (long long)ralpha, n);
    unsigned long long delta  = mulmod_u64(kinv, rhs, n);

    char buf[512];
    snprintf(buf, sizeof(buf),
             "PASIRINKTAS k: %.llu \nVIEŠAS RAKTAS:[%llu,%llu]|PARAŠAS:[[%.llu,%.llu],%llu]",(unsigned long long) k,
             (unsigned long long)Q.x, (unsigned long long)Q.y,
             (unsigned long long)gamma.x, (unsigned long long)gamma.y,
             (unsigned long long)delta);

    return strdup(buf);
}


const char* ellipticEntry(const char* alph, const char* encText, const char* frag){
    (void)alph;
    if (!frag || !*frag) return strdup("[missing -frag]");
    if (!encText || !*encText) return strdup("[missing input]");

    FragMap map = fragmapParse(frag, '|', ':', '=');

    int ok = 0;

    long long q_ll = fragGetLongLong(&map, "q", 0, &ok);  if (!ok || q_ll <= 2){ fragmapFree(&map); return strdup("[bad/missing q]"); }
    long long a_ll = fragGetLongLong(&map, "a", 0, &ok);  if (!ok){ fragmapFree(&map); return strdup("[bad/missing a]"); }
    long long b_ll = fragGetLongLong(&map, "b", 0, &ok);  if (!ok){ fragmapFree(&map); return strdup("[bad/missing b]"); }

    long long n_ll = fragGetLongLong(&map, "n", 0, &ok);
    if (!ok) n_ll = fragGetLongLong(&map, "order", 0, &ok);
    if (!ok || n_ll <= 1){ fragmapFree(&map); return strdup("[bad/missing n]"); }

    long long r_ll = fragGetLongLong(&map, "r", 0, &ok);
    if (!ok) r_ll = fragGetLongLong(&map, "priv", 0, &ok);
    if (!ok || r_ll <= 0){ fragmapFree(&map); return strdup("[bad/missing r]"); }

    const char* Pstr = fragGetScalar(&map, "P");
    if (!Pstr) Pstr = fragGetScalar(&map, "base");
    if (!Pstr) Pstr = fragGetScalar(&map, "G");
    if (!Pstr){ fragmapFree(&map); return strdup("[bad/missing P]"); }

    int pcnt = 0;
    unsigned long long* parr = parse_ull_array(Pstr, &pcnt);
    if (!parr || pcnt < 2){ free(parr); fragmapFree(&map); return strdup("[bad P]"); }
    unsigned long long Px = parr[0];
    unsigned long long Py = parr[1];
    free(parr);

    ECCurve E;
    E.q     = (unsigned long long)q_ll;
    E.a_mod = mod_s64(a_ll, E.q);
    E.b_mod = mod_s64(b_ll, E.q);

    unsigned long long n = (unsigned long long)n_ll;
    unsigned long long priv_r = (unsigned long long)r_ll;
    ECPoint P = ec_xy(Px % E.q, Py % E.q);

    const char* mode = fragGetScalar(&map, "mode");
    int doDecrypt = 0;

    if (mode && mode[0]){
        if (mode[0] == 'm' || mode[0] == 'd') doDecrypt = 1;
        else doDecrypt = 0;
    } else {
        doDecrypt = has_char(encText, '[');
    }

    char* out = doDecrypt
        ? mv_decrypt(&E, n, priv_r, encText)
        : elgamal_sign(&E, n, P, priv_r, encText, &map);

    fragmapFree(&map);
    return out ? out : strdup("[alloc failed]");
}
