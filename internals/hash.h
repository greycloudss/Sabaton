#pragma once
#include "../util/number.h"

#include "hashes/xxhash32.h"
#include "hashes/sha256.h"
#include "hashes/sha1.h"
#include "hashes/murmur3.h"
#include "hashes/crc32.h"

void freeWordCombinations(const char** list, size_t count);
const char** createWordCombinations(int wordLength, size_t* outCount);