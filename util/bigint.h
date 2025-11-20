#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

#define BI_MAX_LIMBS 8 


typedef struct {
    uint32_t limb[BI_MAX_LIMBS]; 
    size_t   len;                
} BigInt;

void bi_zero(BigInt* x);
void bi_from_u32(BigInt* x, uint32_t v);
int  bi_is_zero(const BigInt* x);
int  bi_is_odd(const BigInt* x);
void bi_copy(BigInt* dst, const BigInt* src);
int  bi_cmp(const BigInt* a, const BigInt* b);
void bi_add(BigInt* out, const BigInt* a, const BigInt* b);
int  bi_sub(BigInt* out, const BigInt* a, const BigInt* b);
void bi_shr1(BigInt* x);
void bi_add_mod(BigInt* out, const BigInt* a, const BigInt* b, const BigInt* mod);
void bi_mul_mod(BigInt* out, const BigInt* a, const BigInt* b, const BigInt* mod);
void bi_powmod(BigInt* out, const BigInt* base, const BigInt* exp, const BigInt* mod);
void bi_from_dec(BigInt* x, const char* s);
void bi_divmod_small(BigInt* q, uint32_t* r, const BigInt* a, uint32_t d);
char* bi_to_alphabet(const BigInt* m, const char* alph, int base);
void bi_clear(BigInt* x);