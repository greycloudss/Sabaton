#pragma once
#include <stdlib.h>

typedef unsigned int u32;

u32 crc32_reflected(const unsigned char* data, size_t len);
