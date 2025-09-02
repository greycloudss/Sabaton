#include "murmur3.h"

static u32 rotl32(u32 x, int r) {
    return (x << r) | (x >> (32 - r));
}

u32 murmur3_32(const unsigned char* data, size_t len, u32 seed) {
    const u32 c1 = 0xCC9E2D51u;
    const u32 c2 = 0x1B873593u;
    u32 h = seed;
    size_t nblocks = len / 4;
    const u32* blocks = (const u32*) (const void*) data;
    for (size_t i = 0; i < nblocks; ++i) {
        u32 k = blocks[i];
        k *= c1;
        k = rotl32(k, 15);
        k *= c2;
        h ^= k;
        h = rotl32(h, 13);
        h = h * 5u + 0xE6546B64u;
    }
    const unsigned char* tail = data + (nblocks * 4);
    u32 k1 = 0;
    switch (len & 3u) {
        case 3: k1 ^= (u32) tail[2] << 16;
        case 2: k1 ^= (u32) tail[1] << 8;
        case 1:
            k1 ^= (u32) tail[0];
            k1 *= c1;
            k1 = rotl32(k1, 15);
            k1 *= c2;
            h ^= k1;
    }
    h ^= (u32) len;
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}
