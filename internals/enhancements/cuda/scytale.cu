#include "../../cyphers/scytale.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


__device__ void scytale_block_decrypt_dev(const int* encInt, int idx, const int* keyBytes, int rounds, char funcFlag, unsigned char* outL, unsigned char* outR) {

}

__global__ void scytaleKernel(const int* encInt, int bigN, int rounds, char funcFlag, unsigned char* outPlain, int* found, int* foundKey) {
 
}

extern "C" const char* scytaleBrute(const char* alph, const char* encText, const char* frag) {
    return NULL;
}

#endif