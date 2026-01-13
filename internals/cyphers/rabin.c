#include "rabin.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

//./a.exe -decypher -rabin -frag "21197264541260668663598848260720037099|9334134961424238127|9334134961424238143"
//kill me

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
        fprintf(stderr, "Fatal: non-exact division\n");
        exit(1);
    }
    biCopy(out, &q);
}

static void biNegMod(BigInt* out, const BigInt* x, const BigInt* m) {
    if (biIsZero(x)) {
        biZero(out);
    } else {
        biSub(out, m, x);
    }
}

static const char* LIT_ALPH =
    "aąbcčdeęėfghiįyjklmnoprsštuųūvzž ";


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

    BigInt exp_p, exp_q;
    biAddU32(&exp_p, p, 1);
    biDivU32Exact(&exp_p, &exp_p, 4);

    biAddU32(&exp_q, q, 1);
    biDivU32Exact(&exp_q, &exp_q, 4);

    BigInt mp, mq;
    biPowmod(&mp, c, &exp_p, p);
    biPowmod(&mq, c, &exp_q, q);

    BigInt yp, yq;
    if (!biModInv(&yp, p, q) || !biModInv(&yq, q, p)) {
        printf("Inverse error\n");
        return;
    }

    BigInt roots[4];
    int idx = 0;

    for (int sp = 0; sp < 2; ++sp) {
        BigInt ap;
        if (sp == 0)
            biCopy(&ap, &mp);
        else
            biNegMod(&ap, &mp, p);

        for (int sq = 0; sq < 2; ++sq) {
            BigInt aq;
            if (sq == 0)
                biCopy(&aq, &mq);
            else
                biNegMod(&aq, &mq, q);

            BigInt t1, t2, x;

            biMulMod(&t1, &ap, q, &n);
            biMulMod(&t1, &t1, &yq, &n);

            biMulMod(&t2, &aq, p, &n);
            biMulMod(&t2, &t2, &yp, &n);

            biAdd(&x, &t1, &t2);
            if (biCmp(&x, &n) >= 0) {
                biSub(&x, &x, &n);
            }

            biCopy(&roots[idx++], &x);
        }
    }

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


const char* rabinEntry(const char* alph,
                       const char* encText,
                       const char* frag)
{
    (void)alph;
    (void)encText;

    if (!frag || !*frag)
        return "[frag error]";

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
