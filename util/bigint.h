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
void biOne(BigInt* x);
void biShl1(BigInt* x);
int  biGetBit(const BigInt* x, size_t pos);
void biSetBit(BigInt* x, size_t pos);
void biAddMod(BigInt* out, const BigInt* a, const BigInt* b, const BigInt* mod);
void biMulMod(BigInt* out, const BigInt* a, const BigInt* b, const BigInt* mod);
void biPowMod(BigInt* out, const BigInt* base, const BigInt* exp, const BigInt* mod);
void biFromDec(BigInt* x, const char* s);
void biDivmodSmall(BigInt* q, uint32_t* r, const BigInt* a, uint32_t d);
char* biToAlphabet(const BigInt* m, const char* alph, int base);
void biClear(BigInt* x);
void biNormalize(BigInt* x);

int biModInv(BigInt* out, const BigInt* a, const BigInt* n);

void biMul(BigInt* out, const BigInt* a, const BigInt* b);

uint32_t biModU32(const BigInt* a, uint32_t m);

void biDivU32(BigInt* q, const BigInt* a, uint32_t d);

void biDivMod(const BigInt* dividend, const BigInt* divisor, BigInt* quotient, BigInt* remainder);

void biMod(BigInt* r, const BigInt* a, const BigInt* b);
void biDiv(BigInt* q, const BigInt* a, const BigInt* b);
void biMulMod1(BigInt* out, const BigInt* a, const BigInt* b, const BigInt* mod);

void biToDecString(const BigInt* x, char* out, size_t out_len);


void biPowMod(BigInt* res, const BigInt* base, const BigInt* exp, const BigInt* mod);
void biAddUInt(BigInt* out, const BigInt* a, uint32_t b);
void biDivUInt(BigInt* q, const BigInt* a, uint32_t b);
void biFromDecString(const BigInt* x, const char* s);


void biMulModTest(BigInt* out, const BigInt* a, const BigInt* b, const BigInt* mod);
void biPowmodTest(BigInt* res, const BigInt* base, const BigInt* exp, const BigInt* mod);