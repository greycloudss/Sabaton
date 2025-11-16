#include "enigma.h"


char* runSimpleEnigma(const char* text, const char* alph,
                      const int* rotor1, const int* rotor2,
                      int pos1, int pos2, int decrypt) {
    size_t N = strlen(alph);
    int start1 = pos1;
    size_t len = strlen(text);
    char* result = malloc(len + 1);

    for (size_t k = 0; k < len; k++) {
        char c = text[k];
        const char* p = strchr(alph, c);
        if (!p) {
            result[k] = c;
            continue;
        }
        int index = (int)(p - alph);

        if (!decrypt) {
            int step1 = (rotor1[(index + pos1) % N] - pos1 + N) % N;
            int step2 = (rotor2[(step1 + pos2) % N] - pos2 + N) % N;
            result[k] = alph[step2];
        } else {
            int step1 = (rotor2[(index + pos2) % N] - pos2 + N) % N;
            int step2 = (rotor1[(step1 + pos1) % N] - pos1 + N) % N;
            result[k] = alph[step2];
        }

        pos1 = (pos1 + 1) % N;
        if (pos1 == start1) pos2 = (pos2 + 1) % N;
    }
    result[len] = '\0';
    return result;
}



char* runReflectorEnigma(const char* text, const char* alph,
                         const int* rotor1, const int* rotor2,
                         int pos1, int pos2, const int* reflector) {
    size_t N = strlen(alph);
    int inv1[256], inv2[256];
    invertVector(rotor1, inv1, (int)N);
    invertVector(rotor2, inv2, (int)N);

    int start1 = pos1;
    size_t len = strlen(text);
    char* result = malloc(len + 1);

    for (size_t k = 0; k < len; k++) {
        char c = text[k];
        const char* p = strchr(alph, c);
        if (!p) {
            result[k] = c;
            continue;
        }
        int index = (int)(p - alph);

        int step1 = (rotor1[(index + pos1) % N] - pos1 + N) % N;
        int step2 = (rotor2[(step1 + pos2) % N] - pos2 + N) % N;
        int step3 = reflector[step2];
        int step4 = (inv2[(step3 + pos2) % N] - pos2 + N) % N;
        int step5 = (inv1[(step4 + pos1) % N] - pos1 + N) % N;
        result[k] = alph[step5];

        pos1 = (pos1 + 1) % N;
        if (pos1 == start1) pos2 = (pos2 + 1) % N;
    }
    result[len] = '\0';
    return result;
}



const char* enigmaEntry(const char* alph, const char* encText, const char* frag) {
    static char* output = NULL;
    if (output) {
        free(output);
        output = NULL;
    }

    int rotor1[256], rotor2[256], reflector[256];
    int r1Count = 0, r2Count = 0, refCount = 0;
    int key[2] = { -1, -1 };
    char* plainFrag = NULL;

    char* copy = strdup(frag);
    char* part = strtok(copy, "|");
    while (part) {
        if (strncmp(part, "R1:", 3) == 0) {
            parseCSV(part + 3, rotor1, &r1Count);
        } else if (strncmp(part, "R2:", 3) == 0) {
            parseCSV(part + 3, rotor2, &r2Count);
        } else if (strncmp(part, "REF:", 4) == 0) {
            parseCSV(part + 4, reflector, &refCount);
        } else if (strncmp(part, "KEY:", 4) == 0) {
            int tmp[8] = {0}, count = 0;
            parseCSV(part + 4, tmp, &count);
            if (count > 0) key[0] = tmp[0];
            if (count > 1) key[1] = tmp[1];
        } else if (strncmp(part, "PLAIN:", 6) == 0) {
            plainFrag = strdup(part + 6);
        }
        part = strtok(NULL, "|");
    }
    free(copy);

    size_t N = strlen(alph);
    if (N == 0) {
        if (plainFrag) free(plainFrag);
        return strdup("[no alphabet]");
    }

    int inv1[256], inv2[256];
    invertVector(rotor1, inv1, (int)N);
    invertVector(rotor2, inv2, (int)N);

    int p1Start = (key[0] == -1 ? 0 : key[0]);
    int p1End   = (key[0] == -1 ? (int)N : key[0] + 1);
    int p2Start = (key[1] == -1 ? 0 : key[1]);
    int p2End   = (key[1] == -1 ? (int)N : key[1] + 1);


    if (key[0] != -1 && key[1] != -1) {
        char* cand;
        if (refCount>0)
            cand = runReflectorEnigma(encText, alph, rotor1, rotor2, key[0], key[1], reflector);
        else
            cand = runSimpleEnigma(encText, alph, inv1, inv2, key[0], key[1], 1);

        printf("[key %d,%d] %s\n", key[0], key[1], cand);
        if (!output) output = strdup(cand);
        free(cand);
    }
    else {
        for (int p1 = 0; p1 < (key[0]==-1 ? (int)N : 1); p1++) {
            for (int p2 = 0; p2 < (key[1]==-1 ? (int)N : 1); p2++) {
                int pos1 = (key[0]==-1 ? p1 : key[0]);
                int pos2 = (key[1]==-1 ? p2 : key[1]);
                char* cand;
                if (refCount>0)
                    cand = runReflectorEnigma(encText, alph, rotor1, rotor2, pos1, pos2, reflector);
                else
                    cand = runSimpleEnigma(encText, alph, inv1, inv2, pos1, pos2, 1);

                int ok = 1;
                if (plainFrag) {
                    if (strlen(plainFrag)==1) {
                        char first=0;
                        for (size_t i=0; cand[i]; i++) {
                            if (strchr(alph, cand[i])) {
                                first = cand[i];
                                break;
                            }
                        }
                        ok = (first == plainFrag[0]);
                    } else {
                        ok = (strstr(cand, plainFrag)!=NULL);
                    }
                }

                if (ok) {
                    printf("[key %d,%d] %s\n", pos1, pos2, cand);
                    if (!output) output = strdup(cand);
                }
                free(cand);
            }
        }
    }

    if (!output) output = strdup("[no match]");
    if (plainFrag) free(plainFrag);
    return output;
}

