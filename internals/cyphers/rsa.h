#ifndef RSA_H
#define RSA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../../util/string.h"
#include "../../util/number.h"
#include "../../util/fragmentation.h"



const char* rsaEntry(const char* alph, const char* encText, const char* frag);

#endif