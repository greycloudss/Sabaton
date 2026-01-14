#include "rabin.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Helpers to add/divide small ints exactly */
static void biAddU32(BigInt* out, const BigInt* a, uint32_t v) {
    BigInt tmp;
    biFromU32(&tmp, v);
    biAdd(out, a, &tmp);
}

static void biDivU32Exact(BigInt* out, const BigInt* a, uint32_t d) {
    BigInt q;
    uint32_t r;
    biDivmodSmall(&q, &r, a, d);
    if (r != 0) {
        fprintf(stderr, "Fatal: non-exact division in Rabin exponent\n");
    }
    biCopy(out, &q);
}

/* Lithuanian alphabet + space */
static const char* LIT_ALPH = "aąbcčdeęėfghiįyjklmnoprsštuųūvzž ";

/* ===============================
   CRT combine
   =============================== */
static void crt_combine(BigInt* out,
                        const BigInt* mp, const BigInt* mq,
                        const BigInt* p,  const BigInt* q) {
    BigInt n;
    biMul(&n, p, q);

    BigInt yp, yq;

    /* yp = p^-1 mod q */
    if (!biModInv(&yp, p, q)) {
        printf("CRT error: p inverse mod q does not exist\n");
        biZero(out);
        return;
    }

    /* yq = q^-1 mod p */
    if (!biModInv(&yq, q, p)) {
        printf("CRT error: q inverse mod p does not exist\n");
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

/* ===============================
   Number → text (2-digit codes)
   =============================== */
static char* number2text_alph(const BigInt* M, const char* alph) {
    const char* a = (alph && *alph) ? alph : LIT_ALPH;
    BigInt n;
    biCopy(&n, M);

    char tmp[1024];
    int pos = 0;

    while (!biIsZero(&n)) {
        BigInt q;
        uint32_t rem;

        biDivmodSmall(&q, &rem, &n, 100);
        int idx = (int)rem - 1;

        if (idx >= 0 && idx < (int)strlen(a))
            tmp[pos++] = a[idx];
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

/* ===============================
   Rabin: compute 4 square roots
   =============================== */
static int rabin_decrypt_roots(const BigInt* c, const BigInt* p, const BigInt* q, BigInt roots[4]) {
    /* exp_p = (p + 1) / 4, exp_q = (q + 1) / 4 */
    BigInt exp_p, exp_q;
    biAddU32(&exp_p, p, 1);
    biDivU32Exact(&exp_p, &exp_p, 4);
    biAddU32(&exp_q, q, 1);
    biDivU32Exact(&exp_q, &exp_q, 4);

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

            crt_combine(&roots[idx], &ap, &aq, p, q);
            idx++;
        }
    }
    return idx;
}

/* ===============================
   Entry (CPU path; gated by -gpu in main)
   =============================== */
const char* rabinEntry(const char* alph, const char* encText, const char* frag) {
    static char* out = NULL;
    if (out) { free(out); out = NULL; }

    if ((!frag || !*frag) && (!encText || !*encText))
        return "[frag error]";

    const char* a = (alph && *alph) ? alph : LIT_ALPH;

    const char* c_s = encText && *encText ? encText : NULL;
    const char* p_s = NULL;
    const char* q_s = NULL;

    char* copy = frag ? strdup(frag) : NULL;
    if (copy) {
        char* save = NULL;
        char* t1 = strtok_r(copy, "|", &save);
        char* t2 = strtok_r(NULL, "|", &save);
        char* t3 = strtok_r(NULL, "|", &save);

        if (c_s) {
            /* encText holds ciphertext; frag can be p|q or c|p|q */
            if (t1 && t2 && !t3) {
                p_s = t1; q_s = t2;
            } else if (t1 && t2 && t3) {
                c_s = t1; p_s = t2; q_s = t3;
            }
        } else {
            /* everything inside frag */
            if (t1 && t2 && t3) {
                c_s = t1; p_s = t2; q_s = t3;
            }
        }
    }

    if (!c_s || !p_s || !q_s) {
        if (copy) free(copy);
        return "[frag error]";
    }

    BigInt c, p, q;
    biFromDec(&c, c_s);
    biFromDec(&p, p_s);
    biFromDec(&q, q_s);

    BigInt roots[4];
    int rcount = rabin_decrypt_roots(&c, &p, &q, roots);
    if (rcount <= 0) {
        if (copy) free(copy);
        return "[decode failed]";
    }

    char* best = NULL;
    for (int i = 0; i < rcount; ++i) {
        char* txt = number2text_alph(&roots[i], a);
        if (!txt) continue;
        int ok = (strchr(txt, '?') == NULL);
        if (!best || ok) {
            if (best) free(best);
            best = txt;
            if (ok) break;
        } else {
            free(txt);
        }
    }

    if (!best) {
        if (copy) free(copy);
        return "[decode failed]";
    }

    char pbuf[256], qbuf[256], cbuf[256];
    biToDecString(&p, pbuf, sizeof(pbuf));
    biToDecString(&q, qbuf, sizeof(qbuf));
    biToDecString(&c, cbuf, sizeof(cbuf));

    size_t need = strlen(pbuf) + strlen(qbuf) + strlen(cbuf) + strlen(best) + 32;
    out = (char*)malloc(need);
    if (!out) {
        free(best);
        if (copy) free(copy);
        return "[oom]";
    }
    snprintf(out, need, "p=%s q=%s c=%s\n%s", pbuf, qbuf, cbuf, best);

    free(best);
    if (copy) free(copy);
    return out;
}
