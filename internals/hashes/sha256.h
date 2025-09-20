#pragma once
#include <stdlib.h>

typedef unsigned int u32;
typedef unsigned long long u64;

typedef struct {
    u32 state[8];
    u64 bitlen;
    unsigned int datalen;
    unsigned char data[64];
} sha256_ctx;

void sha256_init(sha256_ctx* ctx);
void sha256_update(sha256_ctx* ctx, const unsigned char* data, size_t len);
void sha256_final(sha256_ctx* ctx, unsigned char out[32]);
void sha256(const unsigned char* data, size_t len, unsigned char out[32]);
