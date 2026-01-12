#pragma once
#include "../../../arguments.h"
#include "../../cyphers/feistel.h"
#include "../../cyphers/stream.h"
#include "../../cyphers/rsa.h"
#include "../../cyphers/rabin.h"
#include "../../cyphers/knapsack.h"


#ifdef USE_CUDA
    #ifdef __cplusplus
    extern "C" {
    #endif

    void entryCudaEnhancement(Arguments* Args);
    const char* rsaBruteCuda(const char* alph, const char* encText, const char* frag);
    const char* rabinBruteCuda(const char* alph, const char* encText, const char* frag);
    const char* merkleBruteCuda(const char* alph, const char* encText, const char* frag);

    #ifdef __cplusplus
    }
    #endif
#endif
