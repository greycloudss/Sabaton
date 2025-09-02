#include "sha256.h"

static u32 ror32(u32 x, u32 r) {
    return (x >> r) | (x << (32 - r));
}

static u32 ch(u32 x, u32 y, u32 z) {
    return (x & y) ^ (~x & z);
}

static u32 maj(u32 x, u32 y, u32 z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

static u32 e0(u32 x) {
    return ror32(x, 2) ^ ror32(x, 13) ^ ror32(x, 22);
}

static u32 e1(u32 x) {
    return ror32(x, 6) ^ ror32(x, 11) ^ ror32(x, 25);
}

static u32 s0(u32 x) {
    return ror32(x, 7) ^ ror32(x, 18) ^ (x >> 3);
}

static u32 s1(u32 x) {
    return ror32(x, 17) ^ ror32(x, 19) ^ (x >> 10);
}

static const u32 k256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static void sha256_transform(sha256_ctx* ctx, const unsigned char block[64]) {
    u32 w[64];
    for (unsigned int i = 0; i < 16; ++i) {
        u32 v = ((u32) block[i * 4 + 0] << 24) |
                ((u32) block[i * 4 + 1] << 16) |
                ((u32) block[i * 4 + 2] << 8) |
                (u32) block[i * 4 + 3];
        w[i] = v;
    }
    for (unsigned int i = 16; i < 64; ++i) {
        w[i] = s1(w[i - 2]) + w[i - 7] + s0(w[i - 15]) + w[i - 16];
    }

    u32 a = ctx->state[0];
    u32 b = ctx->state[1];
    u32 c = ctx->state[2];
    u32 d = ctx->state[3];
    u32 e = ctx->state[4];
    u32 f = ctx->state[5];
    u32 g = ctx->state[6];
    u32 h = ctx->state[7];

    for (unsigned int i = 0; i < 64; ++i) {
        u32 t1 = h + e1(e) + ch(e, f, g) + k256[i] + w[i];
        u32 t2 = e0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

void sha256_init(sha256_ctx* ctx) {
    ctx->state[0] = 0x6a09e667u;
    ctx->state[1] = 0xbb67ae85u;
    ctx->state[2] = 0x3c6ef372u;
    ctx->state[3] = 0xa54ff53au;
    ctx->state[4] = 0x510e527fu;
    ctx->state[5] = 0x9b05688cu;
    ctx->state[6] = 0x1f83d9abu;
    ctx->state[7] = 0x5be0cd19u;
    ctx->bitlen = 0;
    ctx->datalen = 0;
}

void sha256_update(sha256_ctx* ctx, const unsigned char* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->datalen++] = data[i];
        if (ctx->datalen == 64) {
            sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

void sha256_final(sha256_ctx* ctx, unsigned char out[32]) {
    u64 bits = ctx->bitlen + (u64) ctx->datalen * 8u;
    ctx->data[ctx->datalen++] = 0x80u;

    if (ctx->datalen > 56) {
        while (ctx->datalen < 64) {
            ctx->data[ctx->datalen++] = 0;
        }
        sha256_transform(ctx, ctx->data);
        ctx->datalen = 0;
    }

    while (ctx->datalen < 56) {
        ctx->data[ctx->datalen++] = 0;
    }

    unsigned char lenb[8];
    for (int i = 7; i >= 0; --i) {
        lenb[i] = (unsigned char) (bits & 0xFFu);
        bits >>= 8;
    }
    for (int i = 0; i < 8; ++i) {
        ctx->data[56 + i] = lenb[i];
    }

    sha256_transform(ctx, ctx->data);

    for (int i = 0; i < 8; ++i) {
        u32 v = ctx->state[i];
        out[i * 4 + 0] = (unsigned char) (v >> 24);
        out[i * 4 + 1] = (unsigned char) (v >> 16);
        out[i * 4 + 2] = (unsigned char) (v >> 8);
        out[i * 4 + 3] = (unsigned char) (v);
    }
}

void sha256(const unsigned char* data, size_t len, unsigned char out[32]) {
    sha256_ctx c;
    sha256_init(&c);
    sha256_update(&c, data, len);
    sha256_final(&c, out);
}
