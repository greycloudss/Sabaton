#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../util/bigint.h"

/* Small helpers kept near Rabin to isolate CUDA-related code paths */
static inline void rabin_add_u32(BigInt* out, const BigInt* a, uint32_t v) {
    BigInt tmp;
    biFromU32(&tmp, v);
    biAdd(out, a, &tmp);
}

static inline void rabin_div_u32_exact(BigInt* out, const BigInt* a, uint32_t d) {
    BigInt q;
    uint32_t r;
    biDivmodSmall(&q, &r, a, d);
    if (r != 0) {
        fprintf(stderr, "Fatal: non-exact division in Rabin exponent\n");
    }
    biCopy(out, &q);
}

/* Number → text (2-digit codes) */
static inline char* rabin_number2text(const BigInt* M, const char* alph) {
    if (!alph || !*alph) return NULL;
    BigInt n;
    biCopy(&n, M);

    char tmp[1024];
    int pos = 0;

    while (!biIsZero(&n)) {
        BigInt q;
        uint32_t rem;

        biDivmodSmall(&q, &rem, &n, 100);
        int idx = (int)rem - 1;

        if (idx >= 0 && idx < (int)strlen(alph))
            tmp[pos++] = alph[idx];
        else
            tmp[pos++] = '?';

        biCopy(&n, &q);
    }

    char* out = (char*)malloc((size_t)pos + 1);
    if (!out) return NULL;
    for (int i = 0; i < pos; i++)
        out[i] = tmp[pos - 1 - i];
    out[pos] = '\0';

    return out;
}

/* CRT combine helper */
static inline void rabin_crt_combine(BigInt* out,
                        const BigInt* mp, const BigInt* mq,
                        const BigInt* p,  const BigInt* q) {
    BigInt n;
    biMul(&n, p, q);

    BigInt yp, yq;

    /* yp = p^-1 mod q */
    if (!biModInv(&yp, p, q)) {
        biZero(out);
        return;
    }

    /* yq = q^-1 mod p */
    if (!biModInv(&yq, q, p)) {
        biZero(out);
        return;
    }

    BigInt t1, t2, sum;

    /* t1 = mp * q * yq */
    biMul(&t1, mp, q);
    biMul(&t1, &t1, &yq);

    /* t2 = mq * p * yp */
    biMul(&t2, mq, p);
    biMul(&t2, &t2, &yp);

    biAdd(&sum, &t1, &t2);
    biMod(out, &sum, &n);
}

/* Rabin: compute 4 square roots */
static inline int rabin_decrypt_roots(const BigInt* c, const BigInt* p, const BigInt* q, BigInt roots[4]) {
    /* exp_p = (p + 1) / 4, exp_q = (q + 1) / 4 */
    BigInt exp_p, exp_q;
    rabin_add_u32(&exp_p, p, 1);
    rabin_div_u32_exact(&exp_p, &exp_p, 4);
    rabin_add_u32(&exp_q, q, 1);
    rabin_div_u32_exact(&exp_q, &exp_q, 4);

    /* square roots mod p, q */
    BigInt mp, mq;
    biPowmodTest(&mp, c, &exp_p, p);
    biPowmodTest(&mq, c, &exp_q, q);

    /* ensure reduced */
    biMod(&mp, &mp, p);
    biMod(&mq, &mq, q);

    int idx = 0;
    for (int sp = 0; sp < 2; ++sp) {
        BigInt ap;
        if (sp == 0) {
            biCopy(&ap, &mp);
        } else {
            BigInt tmp;
            biSub(&tmp, p, &mp);  /* (-mp) mod p */
            biCopy(&ap, &tmp);
        }

        for (int sq = 0; sq < 2; ++sq) {
            BigInt aq;
            if (sq == 0) {
                biCopy(&aq, &mq);
            } else {
                BigInt tmp;
                biSub(&tmp, q, &mq);  /* (-mq) mod q */
                biCopy(&aq, &tmp);
            }

            BigInt apq;
            rabin_crt_combine(&apq, &ap, &aq, p, q);
            biCopy(&roots[idx], &apq);
            idx++;
        }
    }
    return idx;
}
