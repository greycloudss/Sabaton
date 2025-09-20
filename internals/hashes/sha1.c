#include "sha1.h"

static u32 rol32(u32 x, u32 r) {
    return (x << r) | (x >> (32 - r));
}

static void sha1_transform(sha1_ctx* ctx, const unsigned char block[64]) {
    u32 w[80];
    for (unsigned int i = 0; i < 16; ++i) {
        u32 v = ((u32) block[i * 4 + 0] << 24) |
                ((u32) block[i * 4 + 1] << 16) |
                ((u32) block[i * 4 + 2] << 8) |
                (u32) block[i * 4 + 3];
        w[i] = v;
    }
    for (unsigned int i = 16; i < 80; ++i) {
        w[i] = rol32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }

    u32 a = ctx->state[0];
    u32 b = ctx->state[1];
    u32 c = ctx->state[2];
    u32 d = ctx->state[3];
    u32 e = ctx->state[4];

    for (unsigned int i = 0; i < 80; ++i) {
        u32 f, k;
        if (i < 20) {
            f = (b & c) | ((~b) & d);
            k = 0x5A827999u;
        } else if (i < 40) {
            f = b ^ c ^ d;
            k = 0x6ED9EBA1u;
        } else if (i < 60) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8F1BBCDCu;
        } else {
            f = b ^ c ^ d;
            k = 0xCA62C1D6u;
        }
        u32 t = rol32(a, 5) + f + e + k + w[i];
        e = d;
        d = c;
        c = rol32(b, 30);
        b = a;
        a = t;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
}

void sha1_init(sha1_ctx* ctx) {
    ctx->state[0] = 0x67452301u;
    ctx->state[1] = 0xEFCDAB89u;
    ctx->state[2] = 0x98BADCFEu;
    ctx->state[3] = 0x10325476u;
    ctx->state[4] = 0xC3D2E1F0u;
    ctx->bitlen = 0;
    ctx->datalen = 0;
}

void sha1_update(sha1_ctx* ctx, const unsigned char* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->datalen++] = data[i];
        if (ctx->datalen == 64) {
            sha1_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

void sha1_final(sha1_ctx* ctx, unsigned char out[20]) {
    u64 bits = ctx->bitlen + (u64) ctx->datalen * 8u;
    ctx->data[ctx->datalen++] = 0x80u;

    if (ctx->datalen > 56) {
        while (ctx->datalen < 64) {
            ctx->data[ctx->datalen++] = 0;
        }
        sha1_transform(ctx, ctx->data);
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

    sha1_transform(ctx, ctx->data);

    for (int i = 0; i < 5; ++i) {
        u32 v = ctx->state[i];
        out[i * 4 + 0] = (unsigned char) (v >> 24);
        out[i * 4 + 1] = (unsigned char) (v >> 16);
        out[i * 4 + 2] = (unsigned char) (v >> 8);
        out[i * 4 + 3] = (unsigned char) (v);
    }
}

void sha1(const unsigned char* data, size_t len, unsigned char out[20]) {
    sha1_ctx c;
    sha1_init(&c);
    sha1_update(&c, data, len);
    sha1_final(&c, out);
}
