#pragma once
#include <stdlib.h>

typedef unsigned int u32;

u32 xxhash32(const unsigned char* data, size_t len, u32 seed);
