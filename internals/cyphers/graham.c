#include "graham.h"



static unsigned long long gcd_u64(unsigned long long a, unsigned long long b){
    while (b){ unsigned long long t = a % b; a = b; b = t; }
    return a;
}

static int is_printable8(int x){
    return (x >= 32 && x <= 126); 
}

const char* grahamEntry(const char* alph, const char* encText, const char* frag){
    unsigned long long* publicWeights = NULL;  
    int keyLen = 0;
    unsigned long long modulusM = 0;          
    unsigned long long firstPrivateWeight = 0; 

    extractKnapsackValues(frag, &publicWeights, &keyLen, &modulusM, &firstPrivateWeight);
    if (!publicWeights || keyLen <= 0 || modulusM == 0 || firstPrivateWeight == 0){
        if (publicWeights) free(publicWeights);
        return strdup("[bad fragment]");
    }

    int ctCount = 0;
    unsigned long long* ctValues = parse_ull_array(encText, &ctCount);
    if (!ctValues || ctCount <= 0){
        if (publicWeights) free(publicWeights);
        if (ctValues) free(ctValues);
        return strdup("[bad ciphertext]");
    }

    unsigned long long multiplierT = 0, invMultiplierT = 0;
    unsigned long long v1 = publicWeights[0];
    unsigned long long g = gcd_u64(firstPrivateWeight, modulusM);
    if (g == 0) g = 1;

    if (g == 1) {
        unsigned long long invW1 = modinv_u64(firstPrivateWeight, modulusM);
        if (!invW1){ free(publicWeights); free(ctValues); return strdup("[failed to invert w1 mod M]"); }
        multiplierT = mulmod_u64(v1, invW1, modulusM);
        invMultiplierT = modinv_u64(multiplierT, modulusM);
        if (!invMultiplierT){ free(publicWeights); free(ctValues); return strdup("[no inverse for t mod M]"); }
    } else {
        if (v1 % g != 0) { free(publicWeights); free(ctValues); return strdup("[inconsistent v1,w1,M]"); }

        unsigned long long M2  = modulusM / g;
        unsigned long long w12 = firstPrivateWeight / g;
        unsigned long long v12 = v1 / g;

        unsigned long long w12_inv = modinv_u64(w12, M2);
        if (!w12_inv) { free(publicWeights); free(ctValues); return strdup("[no inverse of w1/g mod M/g]"); }

        unsigned long long t0 = mulmod_u64(v12, w12_inv, M2);

        int bestScore = -1;
        unsigned long long bestT = 0, bestInv = 0;

        for (unsigned long long k = 0; k < g; ++k) {
            unsigned long long tTry = t0 + k * M2;
            unsigned long long invTry = modinv_u64(tTry, modulusM);
            if (!invTry) continue;

            int score = 0;
            for (int i = 0; i < ctCount; ++i){
                unsigned long long cStar = mulmod_u64(ctValues[i], invTry, modulusM);

                int value = 0;
                for (int bit = 19; bit >= 12; --bit){
                    int bitVal = (int)((cStar >> (bit - 1)) & 1ULL);
                    value = (value << 1) | bitVal;
                }
                if (is_printable8(value & 0xFF)) ++score;
            }

            if (score > bestScore){
                bestScore = score; bestT = tTry; bestInv = invTry;
            }
        }

        if (bestScore < 0){ free(publicWeights); free(ctValues); return strdup("[no valid lift for t]"); }
        multiplierT    = bestT;
        invMultiplierT = bestInv;
    }

    int* plainBytes = (int*)malloc(sizeof(int)*(size_t)ctCount);
    if (!plainBytes){ free(publicWeights); free(ctValues); return strdup("[alloc failed]"); }

    for (int i = 0; i < ctCount; ++i){
        unsigned long long cStar = mulmod_u64(ctValues[i], invMultiplierT, modulusM);

        int value = 0;
        for (int bit = 19; bit >= 12; --bit){
            int bitVal = (int)((cStar >> (bit - 1)) & 1ULL);
            value = (value << 1) | bitVal;
        }
        plainBytes[i] = value & 0xFF;
    }

    char* result = numbersToBytes(plainBytes, (size_t)ctCount);

    free(plainBytes);
    free(publicWeights);
    free(ctValues);

    return result ? result : strdup("[alloc failed]");
}