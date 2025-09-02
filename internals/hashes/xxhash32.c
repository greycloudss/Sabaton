#include "xxhash32.h"

static u32 rotl32(u32 x, int r) {
    return (x << r) | (x >> (32 - r));
}

u32 xxhash32(const unsigned char* data, size_t len, u32 seed) {
    const u32 P1 = 0x9E3779B1u;
    const u32 P2 = 0x85EBCA77u;
    const u32 P3 = 0xC2B2AE3Du;
    const u32 P4 = 0x27D4EB2Fu;
    const u32 P5 = 0x165667B1u;

    const unsigned char* p = data;
    const unsigned char* bEnd = p + len;
    u32 h;

    if (len >= 16) {
        u32 v1 = seed + P1 + P2;
        u32 v2 = seed + P2;
        u32 v3 = seed + 0;
        u32 v4 = seed - P1;
        const unsigned char* limit = bEnd - 16;
        do {
            v1 += (*(const u32*) (const void*) (p)) * P2;
            v1 = rotl32(v1, 13);
            v1 *= P1;
            p += 4;

            v2 += (*(const u32*) (const void*) (p)) * P2;
            v2 = rotl32(v2, 13);
            v2 *= P1;
            p += 4;

            v3 += (*(const u32*) (const void*) (p)) * P2;
            v3 = rotl32(v3, 13);
            v3 *= P1;
            p += 4;

            v4 += (*(const u32*) (const void*) (p)) * P2;
            v4 = rotl32(v4, 13);
            v4 *= P1;
            p += 4;
        } while (p <= limit);
        h = rotl32(v1, 1) + rotl32(v2, 7) + rotl32(v3, 12) + rotl32(v4, 18);
    } else {
        h = seed + P5;
    }

    h += (u32) len;

    while ((p + 4) <= bEnd) {
        u32 k1 = (*(const u32*) (const void*) p) * P3;
        k1 = rotl32(k1, 17) * P1;
        h ^= k1;
        h = rotl32(h, 17) * P4;
        p += 4;
    }

    while (p < bEnd) {
        h ^= (*p) * P5;
        h = rotl32(h, 11) * P1;
        ++p;
    }

    h ^= h >> 15;
    h *= P2;
    h ^= h >> 13;
    h *= P3;
    h ^= h >> 16;
    return h;
}