#include "../../cyphers/scytale.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>




__global__ void scytaleKernel() {
 
}

extern "C" const char* scytaleBrute(const char* alph, const char* encText, const char* frag) {
    return NULL;
}

#endif