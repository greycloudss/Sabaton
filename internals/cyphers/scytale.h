#pragma once
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "../../util/string.h"
#include "../../util/number.h"
#include "../../util/misc.h"

const char* scytaleEntry(const char* alph, const char* encText, const char* frag);


#ifdef USE_CUDA
    #ifdef __cplusplus
    extern "C" {
    #endif

    const char* scytaleBrute(const char* alph, const char* encText, const char* frag);

    #ifdef __cplusplus
    }
    #endif
#endif