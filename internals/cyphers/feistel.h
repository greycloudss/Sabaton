#pragma once
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include "../../util/string.h"
#include "../../util/number.h"
#include "../../util/misc.h"


const char* feistelEntry(const char* encText, const char* frag, char flag);


#ifdef USE_CUDA
    #ifdef __cplusplus
    extern "C" {
    #endif


    extern char g_funcFlag;


    const char* feistelBrute(const char* alph, const char* encText, const char* frag);

    #ifdef __cplusplus
    }
    #endif
#endif