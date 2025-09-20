#pragma once
#include <stdlib.h>
#include "../../util/string.h"
#include "../../util/number.h"
#include "../../util/misc.h"

const char* pieceAffineCaesar(const char* alph, const char* encText, const char* knownFrag);
const char* bruteAffineCaesar(const char* alph, const char* encText);
const char* affineCaesarEntry(const char* alph, const char* encText, const char* knownFrag);