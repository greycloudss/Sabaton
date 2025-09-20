#include "hash.h"

#define MIN_ASCII 21
#define MAX_ASCII 127
#define MAX_LENGTH 128

#define RANGE (MAX_ASCII - MIN_ASCII)

const char** createWordCombinations(int wordLength, size_t* outCount) {
    if (wordLength <= 0 || wordLength > MAX_LENGTH || outCount == NULL) return NULL;
    size_t count = (size_t)iPow(RANGE, wordLength);
    char** words = (char**)malloc(count * sizeof(char*));
    if (!words) return NULL;
    for (size_t i = 0; i < count; ++i) {
        words[i] = (char*)malloc((size_t)wordLength + 1);
        if (!words[i]) {
            for (size_t j = 0; j < i; ++j) free(words[j]);
            free(words);
            return NULL;
        }
    }
    for (size_t idx = 0; idx < count; ++idx) {
        size_t n = idx;
        for (int pos = wordLength - 1; pos >= 0; --pos) {
            words[idx][pos] = (char)(MIN_ASCII + (n % RANGE));
            n /= RANGE;
        }
        words[idx][wordLength] = '\0';
    }
    *outCount = count;
    return (const char**)words;
}

void freeWordCombinations(const char** list, size_t count) {
    if (!list) return;
    char** words = (char**)list;
    for (size_t i = 0; i < count; ++i) free(words[i]);
    free(words);
}