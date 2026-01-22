#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "../../util/number.h"
#include "../hashes/sha256.h"

/* Entry point that mirrors other cipher modules. */
const char* zkpEntry(const char* alph, const char* encText, const char* frag);
