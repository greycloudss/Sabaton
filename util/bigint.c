#include "bigint.h"



static void bi_normalize(BigInt* x) {
    while (x->len > 0 && x->limb[x->len - 1] == 0) {
        x->len--;
    }
}

void bi_zero(BigInt* x) {
    if (!x) return;
    memset(x->limb, 0, sizeof(x->limb));
    x->len = 0;
}

void bi_from_u32(BigInt* x, uint32_t v) {
    bi_zero(x);
    if (v == 0) return;
    x->limb[0] = v;
    x->len = 1;
}

int bi_is_zero(const BigInt* x) {
    return !x || x->len == 0;
}

int bi_is_odd(const BigInt* x) {
    if (!x || x->len == 0) return 0;
    return (x->limb[0] & 1u) != 0;
}

void bi_copy(BigInt* dst, const BigInt* src) {
    if (!dst || !src) return;
    memcpy(dst->limb, src->limb, sizeof(uint32_t) * BI_MAX_LIMBS);
    dst->len = src->len;
}

int bi_cmp(const BigInt* a, const BigInt* b) {
    if (a->len < b->len) return -1;
    if (a->len > b->len) return  1;
    for (size_t i = a->len; i > 0; --i) {
        uint32_t ai = a->limb[i - 1];
        uint32_t bi_ = b->limb[i - 1];
        if (ai < bi_) return -1;
        if (ai > bi_) return  1;
    }
    return 0;
}

void bi_add(BigInt* out, const BigInt* a, const BigInt* b) {
    uint64_t carry = 0;
    size_t n = (a->len > b->len) ? a->len : b->len;
    if (n > BI_MAX_LIMBS) n = BI_MAX_LIMBS;

    for (size_t i = 0; i < n; ++i) {
        uint64_t av = (i < a->len) ? a->limb[i] : 0ULL;
        uint64_t bv = (i < b->len) ? b->limb[i] : 0ULL;
        uint64_t s = av + bv + carry;
        out->limb[i] = (uint32_t)(s & 0xFFFFFFFFu);
        carry = s >> 32;
    }
    if (carry && n < BI_MAX_LIMBS) {
        out->limb[n++] = (uint32_t)carry;
    }
    out->len = n;
    bi_normalize(out);
}

int bi_sub(BigInt* out, const BigInt* a, const BigInt* b) {
    if (bi_cmp(a, b) < 0) {
        return 1; 
    }
    int64_t borrow = 0;
    size_t n = a->len;
    for (size_t i = 0; i < n; ++i) {
        int64_t av = a->limb[i];
        int64_t bv = (i < b->len) ? b->limb[i] : 0;
        int64_t d = av - bv - borrow;
        if (d < 0) {
            d += ((int64_t)1 << 32);
            borrow = 1;
        } else {
            borrow = 0;
        }
        out->limb[i] = (uint32_t)(d & 0xFFFFFFFFu);
    }
    out->len = n;
    bi_normalize(out);
    return 0;
}

// x >>= 1
void bi_shr1(BigInt* x) {
    uint32_t carry = 0;
    for (size_t i = x->len; i > 0; --i) {
        uint32_t limb = x->limb[i - 1];
        uint32_t new_carry = limb & 1u;
        x->limb[i - 1] = (limb >> 1) | (carry << 31);
        carry = new_carry;
    }
    bi_normalize(x);
}

void bi_add_mod(BigInt* out, const BigInt* a, const BigInt* b, const BigInt* mod) {
    BigInt tmp;
    bi_add(&tmp, a, b);
    if (bi_cmp(&tmp, mod) >= 0) {
        bi_sub(&tmp, &tmp, mod);
    }
    bi_copy(out, &tmp);
}

void bi_mul_mod(BigInt* out, const BigInt* a, const BigInt* b, const BigInt* mod) {
    BigInt res, x, y;
    bi_zero(&res);

    bi_copy(&x, a);
    bi_copy(&y, b);

    while (bi_cmp(&x, mod) >= 0) {
        bi_sub(&x, &x, mod);
    }
    while (bi_cmp(&y, mod) >= 0) {
        bi_sub(&y, &y, mod);
    }

    while (!bi_is_zero(&y)) {
        if (bi_is_odd(&y)) {
            bi_add_mod(&res, &res, &x, mod);
        }
        bi_add_mod(&x, &x, &x, mod);
        bi_shr1(&y);
    }

    bi_copy(out, &res);
}

void bi_powmod(BigInt* out, const BigInt* base, const BigInt* exp, const BigInt* mod) {
    BigInt result, b, e;
    bi_from_u32(&result, 1); 
    bi_copy(&b, base);
    bi_copy(&e, exp);

    while (bi_cmp(&b, mod) >= 0) {
        bi_sub(&b, &b, mod);
    }

    while (!bi_is_zero(&e)) {
        if (bi_is_odd(&e)) {
            bi_mul_mod(&result, &result, &b, mod);
        }
        bi_mul_mod(&b, &b, &b, mod);
        bi_shr1(&e);
    }

    bi_copy(out, &result);
}

static void bi_mul_small(BigInt* out, const BigInt* a, uint32_t m) {
    uint64_t carry = 0;
    size_t n = a->len;
    for (size_t i = 0; i < n; ++i) {
        uint64_t p = (uint64_t)a->limb[i] * (uint64_t)m + carry;
        out->limb[i] = (uint32_t)(p & 0xFFFFFFFFu);
        carry = p >> 32;
    }
    if (carry && n < BI_MAX_LIMBS) {
        out->limb[n++] = (uint32_t)carry;
    }
    out->len = n;
    bi_normalize(out);
}

static void bi_mul_add_small(BigInt* out, const BigInt* a, uint32_t m, uint32_t add) {
    uint64_t carry = add;
    size_t n = a->len;
    for (size_t i = 0; i < n; ++i) {
        uint64_t p = (uint64_t)a->limb[i] * (uint64_t)m + carry;
        out->limb[i] = (uint32_t)(p & 0xFFFFFFFFu);
        carry = p >> 32;
    }
    if (carry && n < BI_MAX_LIMBS) {
        out->limb[n++] = (uint32_t)carry;
    }
    out->len = n;
    bi_normalize(out);
}

void bi_from_dec(BigInt* x, const char* s) {
    bi_zero(x);
    if (!s) return;

    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '+') s++;

    BigInt tmp;
    bi_zero(&tmp);

    for (; *s; ++s) {
        char c = *s;
        if (c < '0' || c > '9') break;
        uint32_t digit = (uint32_t)(c - '0');

        BigInt t2;
        bi_mul_add_small(&t2, &tmp, 10u, digit);
        bi_copy(&tmp, &t2);
    }
    bi_copy(x, &tmp);
}

void bi_divmod_small(BigInt* q, uint32_t* r, const BigInt* a, uint32_t d) {
    if (q) bi_zero(q);
    uint64_t rem = 0;

    BigInt res;
    bi_zero(&res);

    if (a->len == 0 || d == 0) {
        if (r) *r = 0;
        if (q) bi_zero(q);
        return;
    }

    for (size_t i = a->len; i > 0; --i) {
        rem = (rem << 32) | a->limb[i - 1];
        uint32_t qdigit = (uint32_t)(rem / d);
        rem = rem % d;
        if (qdigit != 0 || res.len != 0) {
            if (res.len < BI_MAX_LIMBS) {
                res.limb[res.len++] = qdigit;
            }
        }
    }

    for (size_t i = 0; i < res.len / 2; ++i) {
        uint32_t tmp = res.limb[i];
        res.limb[i] = res.limb[res.len - 1 - i];
        res.limb[res.len - 1 - i] = tmp;
    }
    bi_normalize(&res);
    if (q) bi_copy(q, &res);
    if (r) *r = (uint32_t)rem;
}

char* bi_to_alphabet(const BigInt* m, const char* alph, int base) {
    if (!alph || base <= 1) return NULL;

    if (bi_is_zero(m)) {
        char* s = (char*)malloc(2);
        if (!s) return NULL;
        s[0] = alph[0];
        s[1] = '\0';
        return s;
    }

    BigInt tmp;
    bi_copy(&tmp, m);

    uint32_t* digits = (uint32_t*)malloc(512 * sizeof(uint32_t));
    if (!digits) return NULL;
    size_t dcount = 0;

    while (!bi_is_zero(&tmp)) {
        BigInt q;
        uint32_t rem = 0;
        bi_divmod_small(&q, &rem, &tmp, (uint32_t)base);
        digits[dcount++] = rem;
        if (dcount >= 512) break;
        bi_copy(&tmp, &q);
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

void bi_clear(BigInt* x) {
    if (!x) return;
    memset(x->limb, 0, sizeof(x->limb));
    x->len = 0;
}
