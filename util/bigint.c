#include "bigint.h"



int biIsOne(const BigInt* x) {
    if (!x) return 0;
    if (x->len != 1) return 0;
    return x->part[0] == 1;
}

int biIsEven(const BigInt* x) {
    return !biIsOdd(x);
}


void biNormalize(BigInt* x) {
    while (x->len > 0 && x->part[x->len - 1] == 0) {
        x->len--;
    }
}

void biZero(BigInt* x) {
    if (!x) return;
    memset(x->part, 0, sizeof(x->part));
    x->len = 0;
}

void biFromU32(BigInt* x, uint32_t v) {
    biZero(x);
    if (v == 0) return;
    x->part[0] = v;
    x->len = 1;
}

int biIsZero(const BigInt* x) {
    return !x || x->len == 0;
}

int biIsOdd(const BigInt* x) {
    if (!x || x->len == 0) return 0;
    return (x->part[0] & 1u) != 0;
}

void biCopy(BigInt* dst, const BigInt* src) {
    if (!dst || !src) return;
    memcpy(dst->part, src->part, sizeof(uint32_t) * BIMAX_PARTS);
    dst->len = src->len;
}

int biCmp(const BigInt* a, const BigInt* b) {
    if (a->len < b->len) return -1;
    if (a->len > b->len) return  1;
    for (size_t i = a->len; i > 0; --i) {
        uint32_t ai = a->part[i - 1];
        uint32_t bi_ = b->part[i - 1];
        if (ai < bi_) return -1;
        if (ai > bi_) return  1;
    }
    return 0;
}

void biAdd(BigInt* out, const BigInt* a, const BigInt* b) {
    uint64_t carry = 0;
    size_t n = (a->len > b->len) ? a->len : b->len;
    if (n > BIMAX_PARTS) n = BIMAX_PARTS;

    for (size_t i = 0; i < n; ++i) {
        uint64_t av = (i < a->len) ? a->part[i] : 0ULL;
        uint64_t bv = (i < b->len) ? b->part[i] : 0ULL;
        uint64_t s = av + bv + carry;
        out->part[i] = (uint32_t)(s & 0xFFFFFFFFu);
        carry = s >> 32;
    }
    if (carry && n < BIMAX_PARTS) {
        out->part[n++] = (uint32_t)carry;
    }
    out->len = n;
    biNormalize(out);
}

int biSub(BigInt* out, const BigInt* a, const BigInt* b) {
    if (biCmp(a, b) < 0) {
        return 1; 
    }
    int64_t borrow = 0;
    size_t n = a->len;
    for (size_t i = 0; i < n; ++i) {
        int64_t av = a->part[i];
        int64_t bv = (i < b->len) ? b->part[i] : 0;
        int64_t d = av - bv - borrow;
        if (d < 0) {
            d += ((int64_t)1 << 32);
            borrow = 1;
        } else {
            borrow = 0;
        }
        out->part[i] = (uint32_t)(d & 0xFFFFFFFFu);
    }
    out->len = n;
    biNormalize(out);
    return 0;
}

void biShr1(BigInt* x) {
    uint32_t carry = 0;
    for (size_t i = x->len; i > 0; --i) {
        uint32_t part = x->part[i - 1];
        uint32_t newCarry = part & 1u;
        x->part[i - 1] = (part >> 1) | (carry << 31);
        carry = newCarry;
    }
    biNormalize(x);
}

void biAddMod(BigInt* out, const BigInt* a, const BigInt* b, const BigInt* mod) {
    BigInt tmp;
    biAdd(&tmp, a, b);
    if (biCmp(&tmp, mod) >= 0) {
        biSub(&tmp, &tmp, mod);
    }
    biCopy(out, &tmp);
}

void biMulMod(BigInt* out, const BigInt* a, const BigInt* b, const BigInt* mod) {
    BigInt res, x, y;
    biZero(&res);

    biCopy(&x, a);
    biCopy(&y, b);

    while (biCmp(&x, mod) >= 0) {
        biSub(&x, &x, mod);
    }
    while (biCmp(&y, mod) >= 0) {
        biSub(&y, &y, mod);
    }

    while (!biIsZero(&y)) {
        if (biIsOdd(&y)) {
            biAddMod(&res, &res, &x, mod);
        }
        biAddMod(&x, &x, &x, mod);
        biShr1(&y);
    }

    biCopy(out, &res);
}

void biPowmod(BigInt* out, const BigInt* base, const BigInt* exp, const BigInt* mod) {
    BigInt result, b, e;
    biFromU32(&result, 1); 
    biCopy(&b, base);
    biCopy(&e, exp);

    while (biCmp(&b, mod) >= 0) {
        biSub(&b, &b, mod);
    }

    while (!biIsZero(&e)) {
        if (biIsOdd(&e)) {
            biMulMod(&result, &result, &b, mod);
        }
        biMulMod(&b, &b, &b, mod);
        biShr1(&e);
    }

    biCopy(out, &result);
}


void biMulAddSmall(BigInt* out, const BigInt* a, uint32_t m, uint32_t add) {
    uint64_t carry = add;
    size_t n = a->len;
    for (size_t i = 0; i < n; ++i) {
        uint64_t p = (uint64_t)a->part[i] * (uint64_t)m + carry;
        out->part[i] = (uint32_t)(p & 0xFFFFFFFFu);
        carry = p >> 32;
    }
    if (carry && n < BIMAX_PARTS) {
        out->part[n++] = (uint32_t)carry;
    }
    out->len = n;
    biNormalize(out);
}

void biFromDec(BigInt* x, const char* s) {
    biZero(x);
    if (!s) return;

    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '+') s++;

    BigInt tmp;
    biZero(&tmp);

    for (; *s; ++s) {
        char c = *s;
        if (c < '0' || c > '9') break;
        uint32_t digit = (uint32_t)(c - '0');

        BigInt t2;
        biMulAddSmall(&t2, &tmp, 10u, digit);
        biCopy(&tmp, &t2);
    }
    biCopy(x, &tmp);
}

void biDivmodSmall(BigInt* q, uint32_t* r, const BigInt* a, uint32_t d) {
    if (q) biZero(q);
    uint64_t rem = 0;

    BigInt res;
    biZero(&res);

    if (a->len == 0 || d == 0) {
        if (r) *r = 0;
        if (q) biZero(q);
        return;
    }

    for (size_t i = a->len; i > 0; --i) {
        rem = (rem << 32) | a->part[i - 1];
        uint32_t qdigit = (uint32_t)(rem / d);
        rem = rem % d;
        if (qdigit != 0 || res.len != 0) {
            if (res.len < BIMAX_PARTS) {
                res.part[res.len++] = qdigit;
            }
        }
    }

    for (size_t i = 0; i < res.len / 2; ++i) {
        uint32_t tmp = res.part[i];
        res.part[i] = res.part[res.len - 1 - i];
        res.part[res.len - 1 - i] = tmp;
    }
    biNormalize(&res);
    if (q) biCopy(q, &res);
    if (r) *r = (uint32_t)rem;
}

char* biToAlphabet(const BigInt* m, const char* alph, int base) {
    if (!alph || base <= 1) return NULL;

    if (biIsZero(m)) {
        char* s = (char*)malloc(2);
        if (!s) return NULL;
        s[0] = alph[0];
        s[1] = '\0';
        return s;
    }

    BigInt tmp;
    biCopy(&tmp, m);

    uint32_t* digits = (uint32_t*)malloc(512 * sizeof(uint32_t));
    if (!digits) return NULL;
    size_t dcount = 0;

    while (!biIsZero(&tmp)) {
        BigInt q;
        uint32_t rem = 0;
        biDivmodSmall(&q, &rem, &tmp, (uint32_t)base);
        digits[dcount++] = rem;
        if (dcount >= 512) break;
        biCopy(&tmp, &q);
    }

    char* out = (char*)malloc(dcount + 1);
    if (!out) {
        free(digits);
        return NULL;
    }
    for (size_t i = 0; i < dcount; ++i) {
        uint32_t idx = digits[dcount - 1 - i];
        if ((int)idx >= base) idx = 0; 
        out[i] = alph[idx];
    }
    out[dcount] = '\0';

    free(digits);
    return out;
}

void biClear(BigInt* x) {
    if (!x) return;
    memset(x->part, 0, sizeof(x->part));
    x->len = 0;
}

// bit fucked used for phi shit cause wasnt working properly for the second task
int biModInv(BigInt* out, const BigInt* a, const BigInt* n) {
    if (!out || !a || !n) return 0;
    if (biIsZero(n)) return 0;


    if (!biIsOdd(n)) {
        if (a->len == 0) return 0;
        if (a->len > 1)  return 0;  
        uint32_t a_u = a->part[0];
        if (a_u == 0) return 0;

        BigInt r0;
        biCopy(&r0, n);
        uint32_t r1 = a_u;

        BigInt x0, x1;
        biZero(&x0);
        biFromU32(&x1, 1);

        while (r1 != 0) {
            BigInt q;
            uint32_t rem;
            biDivmodSmall(&q, &rem, &r0, r1);

            BigInt qx1;
            biMulMod(&qx1, &q, &x1, n);  

            BigInt x2;
            if (biCmp(&x0, &qx1) >= 0) {
                biSub(&x2, &x0, &qx1);
            } else {
                BigInt tmp;
                biSub(&tmp, n, &qx1);   
                biAdd(&x2, &x0, &tmp);  
            }

            BigInt new_r0;
            biFromU32(&new_r0, r1);
            biCopy(&r0, &new_r0);
            r1 = rem;

            BigInt tmpx;
            biCopy(&tmpx, &x1);
            biCopy(&x1, &x2);
            biCopy(&x0, &tmpx);
        }

        if (!biIsOne(&r0)) {
            return 0;
        }

        while (biCmp(&x0, n) >= 0) {
            BigInt tmp;
            biSub(&tmp, &x0, n);
            biCopy(&x0, &tmp);
        }
        biCopy(out, &x0);
        return 1;
    }


    BigInt u, v, x1, x2;

    biCopy(&u, a);
    while (biCmp(&u, n) >= 0) {
        biSub(&u, &u, n);
    }

    biCopy(&v, n);

    biZero(&x2);
    biFromU32(&x1, 1);

    if (biIsZero(&u)) {
        return 0;
    }
    if (biIsOne(&u)) {
        biCopy(out, &x1);
        return 1;
    }

    while (!biIsOne(&u) && !biIsOne(&v)) {
        while (!biIsZero(&u) && biIsEven(&u)) {
            biShr1(&u);
            if (biIsEven(&x1)) {
                biShr1(&x1);
            } else {
                biAdd(&x1, &x1, n);
                biShr1(&x1);
            }
        }

        while (!biIsZero(&v) && biIsEven(&v)) {
            biShr1(&v);
            if (biIsEven(&x2)) {
                biShr1(&x2);
            } else {
                biAdd(&x2, &x2, n);
                biShr1(&x2);
            }
        }

        if (biCmp(&u, &v) >= 0) {
            biSub(&u, &u, &v);

            if (biCmp(&x1, &x2) >= 0) {
                biSub(&x1, &x1, &x2);
            } else {
                biAdd(&x1, &x1, n);
                biSub(&x1, &x1, &x2);
            }
        } else {
            biSub(&v, &v, &u);

            if (biCmp(&x2, &x1) >= 0) {
                biSub(&x2, &x2, &x1);
            } else {
                biAdd(&x2, &x2, n);
                biSub(&x2, &x2, &x1);
            }
        }
    }

    if (biIsOne(&u)) {
        while (biCmp(&x1, n) >= 0) {
            biSub(&x1, &x1, n);
        }
        biCopy(out, &x1);
        return 1;
    } else if (biIsOne(&v)) {
        while (biCmp(&x2, n) >= 0) {
            biSub(&x2, &x2, n);
        }
        biCopy(out, &x2);
        return 1;
    }

    return 0;
}




void biMul(BigInt* out, const BigInt* a, const BigInt* b)
{
    BigInt res;
    biZero(&res);

    for (size_t i = 0; i < a->len; ++i) {
        uint64_t carry = 0;
        for (size_t j = 0; j < b->len && i + j < BIMAX_PARTS; ++j) {
            uint64_t cur = res.part[i + j];
            uint64_t prod = (uint64_t)a->part[i] * b->part[j] + cur + carry;

            res.part[i + j] = (uint32_t)(prod & 0xFFFFFFFFu);
            carry = prod >> 32;
        }

        if (carry && i + b->len < BIMAX_PARTS) {
            res.part[i + b->len] = (uint32_t)carry;
        }
    }

    res.len = a->len + b->len;
    if (res.len > BIMAX_PARTS) res.len = BIMAX_PARTS;
    biNormalize(&res);
    biCopy(out, &res);
}



uint32_t biModU32(const BigInt* a, uint32_t m)
{
    uint32_t r = 0;
    for (size_t i = a->len; i > 0; --i) {
        uint64_t cur = ((uint64_t)r << 32) | a->part[i - 1];
        r = (uint32_t)(cur % m);
    }
    return r;
}


void biDivU32(BigInt* q, const BigInt* a, uint32_t d)
{
    BigInt quotient;
    BigInt tmp = *a;
    uint32_t rem;

    biDivmodSmall(&quotient, &rem, &tmp, d);
    biCopy(q, &quotient);
}

