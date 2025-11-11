#include "merkle.h"






const char* merkleEntry(const char* alph, const char* encText, const char* frag){
    unsigned long long* v = NULL; int nBits = 0;
    unsigned long long p = 0, w1 = 0;
    extractKnapsackValues(frag, &v, &nBits, &p, &w1);
    if (!v || nBits <= 0 || p == 0 || w1 == 0) { if (v) free(v); return strdup("[bad fragment]"); }

    int cN = 0; int* cArr = parse_frag_array(encText, &cN);
    if (!cArr || cN <= 0) { if (v) free(v); if (cArr) free(cArr); return strdup("[bad ciphertext]"); }

    unsigned long long a = 0, a_inv = 0;
    unsigned long long v1 = v[0];
    unsigned long long g = (unsigned long long)gcd((int)(w1 % INT32_MAX), (int)(p % INT32_MAX));
    if (g == 0) g = 1;

    if (g == 1) {
        unsigned long long w1_inv = modinv_u64(w1, p);
        if (!w1_inv) { free(v); free(cArr); return strdup("[failed to invert w1 mod p]"); }
        a = mulmod_u64(v1, w1_inv, p);
    } else {
        if (v1 % g != 0) { free(v); free(cArr); return strdup("[inconsistent v1,w1,p]"); }
        unsigned long long p2  = p  / g;
        unsigned long long w12 = w1 / g;
        unsigned long long v12 = v1 / g;

        unsigned long long w12_inv = modinv_u64(w12, p2);
        if (!w12_inv) { free(v); free(cArr); return strdup("[no inverse of w1/g mod p/g]"); }

        unsigned long long a0 = mulmod_u64(v12, w12_inv, p2); // a ≡ a0 (mod p2)

        int found = 0;
        for (unsigned long long k = 0; k < g; ++k) {
            unsigned long long a_try = a0 + k * p2;
            if (gcd((int)(a_try % INT32_MAX), (int)(p % INT32_MAX)) != 1) continue;

            unsigned long long a_try_inv = modinv_u64(a_try, p);
            if (!a_try_inv) continue;

            unsigned long long sum = 0;
            int ok = 1;
            for (int i = 0; i < nBits; ++i) {
                unsigned long long wi = mulmod_u64(v[i], a_try_inv, p);
                if (wi <= sum) { ok = 0; break; }
                sum += wi;
            }
            if (ok) {
                a = a_try;
                a_inv = a_try_inv;
                found = 1;
                break;
            }
        }
        if (!found) { free(v); free(cArr); return strdup("[no valid lift for a]"); }
    }

    if (a_inv == 0) {
        a_inv = modinv_u64(a, p);
        if (!a_inv) { free(v); free(cArr); return strdup("[derived a not invertible mod p]"); }
    }

    unsigned long long* w = (unsigned long long*)malloc(sizeof(unsigned long long)*(size_t)nBits);
    if (!w) { free(v); free(cArr); return strdup("[alloc failed]"); }
    for (int i = 0; i < nBits; ++i) w[i] = mulmod_u64(v[i], a_inv, p);

    int* bytes = (int*)malloc(sizeof(int)*(size_t)cN);
    if (!bytes) { free(w); free(v); free(cArr); return strdup("[alloc failed]"); }
    for (int k = 0; k < cN; ++k) {
        unsigned long long Cprime = mulmod_u64((unsigned long long)(unsigned int)cArr[k], a_inv, p);
        int* bits = (int*)calloc((size_t)nBits, sizeof(int));
        if (!bits) { free(bytes); free(w); free(v); free(cArr); return strdup("[alloc failed]"); }
        unsigned long long rem = Cprime;
        for (int i = nBits - 1; i >= 0; --i) {
            if (w[i] <= rem) { bits[i] = 1; rem -= w[i]; }
        }
        bytes[k] = bits_to_byte_msb(bits, nBits);
        free(bits);
    }
    char* out = numbersToBytes(bytes, (size_t)cN);
    free(bytes); free(w); free(v); free(cArr);
    return out ? out : strdup("[alloc failed]");
}
