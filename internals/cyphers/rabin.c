#include "rabin.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//./a.exe -decypher -rabin -frag "21197264541260668663598848260720037099|9334134961424238127|9334134961424238143"
//didnt fix this shits ass

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
static const char* LIT_ALPH =
    "aąbcčdeęėfghiįyjklmnoprsštuųūvzž ";

/* ===============================
   CRT combine
   =============================== */
void crt_combine(
    BigInt* out,
    const BigInt* mp, const BigInt* mq,
    const BigInt* p,  const BigInt* q
) {
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
   Number → text
   =============================== */
char* number2text(const BigInt* M) {
    BigInt n;
    biCopy(&n, M);

    char tmp[1024];
    int pos = 0;

    while (!biIsZero(&n)) {
        BigInt q;
        uint32_t rem;

        biDivmodSmall(&q, &rem, &n, 100);
        int idx = (int)rem - 1;

        if (idx >= 0 && idx < (int)strlen(LIT_ALPH))
            tmp[pos++] = LIT_ALPH[idx];
        else
            tmp[pos++] = '?';

        biCopy(&n, &q);
    }

    char* out = malloc(pos + 1);
    for (int i = 0; i < pos; i++)
        out[i] = tmp[pos - 1 - i];
    out[pos] = '\0';

    return out;
}


void rabin_decrypt(BigInt* c, BigInt* p, BigInt* q)
{
    BigInt n;
    biMul(&n, p, q);

    /* ---- compute exponents exactly ---- */
    BigInt exp_p, exp_q;

    /* exp_p = (p + 1) / 4 */
    biAddU32(&exp_p, p, 1);
    biDivU32Exact(&exp_p, &exp_p, 4);

    /* exp_q = (q + 1) / 4 */
    biAddU32(&exp_q, q, 1);
    biDivU32Exact(&exp_q, &exp_q, 4);

    /* ---- square roots mod p, q ---- */
BigInt mp, mq;
biPowmodTest(&mp, c, &exp_p, p);
biPowmodTest(&mq, c, &exp_q, q);

/* Reduce modulo p/q before negating */
biMod(&mp, &mp, p);
biMod(&mq, &mq, q);

/* ---- CRT inverses ---- */
BigInt yp, yq;
biModInv(&yp, p, q); /* p^-1 mod q */
biModInv(&yq, q, p); /* q^-1 mod p */

BigInt roots[4];
int idx = 0;

for (int sp = 0; sp < 2; ++sp) {
    BigInt ap;
    biCopy(&ap, &mp);
    if (sp) {
        BigInt tmp;
        biCopy(&tmp, &mp);
        biSub(&ap, p, &tmp);  // ap = (-mp) mod p
    }

    for (int sq = 0; sq < 2; ++sq) {
        BigInt aq;
        biCopy(&aq, &mq);
        if (sq) {
            BigInt tmp;
            biCopy(&tmp, &mq);
            biSub(&aq, q, &tmp);  // aq = (-mq) mod q
        }

        BigInt t1, t2, x;
        biMul(&t1, &ap, q);
        biMul(&t1, &t1, &yq);

        biMul(&t2, &aq, p);
        biMul(&t2, &t2, &yp);

        biAdd(&x, &t1, &t2);
        biMod(&roots[idx++], &x, &n);
    }
}


    /* ---- output ---- */
    for (int i = 0; i < 4; ++i) {
        char numbuf[256];
        biToDecString(&roots[i], numbuf, sizeof numbuf);

        char* txt = number2text(&roots[i]);

        printf("Root %d:\n", i + 1);
        printf("Number: %s\n", numbuf);
        printf("Text:   %s\n\n", txt);

        free(txt);
    }
}


/* ===============================
   Entry (framework-style)
   =============================== */
const char* rabinEntry(const char* alph,
                       const char* encText,
                       const char* frag)
{
    (void)alph;
    (void)encText;

    if (!frag || !*frag)
        return "[frag error]";

    /* frag format: c|p|q */
    char* copy = strdup(frag);
    char* c_s = strtok(copy, "|");
    char* p_s = strtok(NULL, "|");
    char* q_s = strtok(NULL, "|");

    if (!c_s || !p_s || !q_s) {
        free(copy);
        return "[frag error]";
    }

    BigInt c, p, q;
    biFromDec(&c, c_s);
    biFromDec(&p, p_s);
    biFromDec(&q, q_s);

    rabin_decrypt(&c, &p, &q);

    free(copy);
    return "[ok]";
}
