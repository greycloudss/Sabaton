#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

#define BIMAX_PARTS 8 


typedef struct {
    uint32_t part[BIMAX_PARTS]; 
    size_t   len;                
} BigInt;

void biZero(BigInt* x);
int biIsOne(const BigInt* x);
void biFromU32(BigInt* x, uint32_t v);
int  biIsZero(const BigInt* x);
int  biIsOdd(const BigInt* x);
int biIsEven(const BigInt* x);
void biCopy(BigInt* dst, const BigInt* src);
int  biCmp(const BigInt* a, const BigInt* b);
void biAdd(BigInt* out, const BigInt* a, const BigInt* b);
int  biSub(BigInt* out, const BigInt* a, const BigInt* b);
void biShr1(BigInt* x);
void biAddMod(BigInt* out, const BigInt* a, const BigInt* b, const BigInt* mod);
void biMulMod(BigInt* out, const BigInt* a, const BigInt* b, const BigInt* mod);
void biPowmod(BigInt* out, const BigInt* base, const BigInt* exp, const BigInt* mod);
void biFromDec(BigInt* x, const char* s);
void biDivmodSmall(BigInt* q, uint32_t* r, const BigInt* a, uint32_t d);
char* biToAlphabet(const BigInt* m, const char* alph, int base);
void biClear(BigInt* x);
void biNormalize(BigInt* x);

int biModInv(BigInt* out, const BigInt* a, const BigInt* n);

void biMul(BigInt* out, const BigInt* a, const BigInt* b);

uint32_t biModU32(const BigInt* a, uint32_t m);

void biDivU32(BigInt* q, const BigInt* a, uint32_t d);