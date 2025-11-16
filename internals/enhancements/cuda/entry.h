#pragma once
#include "../../../arguments.h"
#include "../../cyphers/feistel.h"
#include "../../cyphers/scytale.h"


#ifdef USE_CUDA
    #ifdef __cplusplus
    extern "C" {
    #endif

    void entryCudaEnhancement(Arguments* Args);

    #ifdef __cplusplus
    }
    #endif
#endif