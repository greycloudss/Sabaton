#pragma once
#include "../../../arguments.h"
#include "../../cyphers/feistel.h"
#include "../../cyphers/stream.h"
#include "../../cyphers/block.h"
#include "../../cyphers/a5.h"
#include "../../cyphers/rsa.h"
#include "../../cyphers/rabin.h"
#include "../../cyphers/knapsack.h"
#include "../../cyphers/aes.h"
#include "../../cyphers/shamir.h"
#include "../../cyphers/asmuth.h"
#include "../../cyphers/elliptic.h"
#include "../../cyphers/elgamal.h"


#ifdef USE_CUDA
    #ifdef __cplusplus
    extern "C" {
    #endif

    void entryCudaEnhancement(Arguments* Args);
    const char* rsaBruteCuda(const char* alph, const char* encText, const char* frag);
    const char* rsaModexpCuda(const char* alph, const char* encText, const char* frag);
    const char* rabinBruteCuda(const char* alph, const char* encText, const char* frag);
    const char* merkleBruteCuda(const char* alph, const char* encText, const char* frag);
    const char* aesBruteCuda(const char* alph, const char* encText, const char* frag);
    const char* shamirCuda(const char* alph, const char* encText, const char* frag);
    const char* asmuthCuda(const char* alph, const char* encText, const char* frag);
    const char* ellipticCuda(const char* alph, const char* encText, const char* frag);
    const char* elgamalCuda(const char* alph, const char* encText, const char* frag);
    const char* blumGoldwasserCuda(const char* alph, const char* encText, const char* frag);
    const char* a5Cuda(const char* alph, const char* encText, const char* frag);
    const char* blockCuda(const char* encText, const char* frag, char flag);

    #ifdef __cplusplus
    }
    #endif
#endif
