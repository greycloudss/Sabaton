#pragma once
#include <stdlib.h>

typedef unsigned int u32;

u32 murmur3_32(const unsigned char* data, size_t len, u32 seed);
