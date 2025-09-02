#pragma once
#include <stdlib.h>

typedef unsigned int u32;
typedef unsigned long long u64;

typedef struct {
    u32 state[5];
    u64 bitlen;
    unsigned int datalen;
    unsigned char data[64];
} sha1_ctx;

void sha1_init(sha1_ctx* ctx);
void sha1_update(sha1_ctx* ctx, const unsigned char* data, size_t len);
void sha1_final(sha1_ctx* ctx, unsigned char out[20]);
void sha1(const unsigned char* data, size_t len, unsigned char out[20]);
