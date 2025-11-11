#include "aes.h"




static size_t next_pow2(size_t n){
    size_t p=1; while(p < n) p <<= 1; return p;
}


static HashTable *hashCreate(size_t n_entries_est) {
    HashTable *h = malloc(sizeof(*h));
    size_t size = next_pow2(n_entries_est * 2);
    h->table = calloc(size, sizeof(HashEntry));
    h->size = size;
    h->mask = size - 1;
    return h;
}

static void hashFree(HashTable *h) {
    if (!h) return;
    free(h->table);
    free(h);
}

static inline size_t hash_u64(uint64_t x, size_t mask) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (size_t)x & mask;
}

static inline uint64_t pack_mid(const int mid[4]) {
    uint64_t h = 0xcbf29ce484222325ULL; 
    for (int i = 0; i < 4; ++i) {
        uint64_t v = (uint64_t)(mid[i] & 0xFFFFFFFFULL);
        h ^= v;
        h *= 0x100000001b3ULL; 
    }
    return h;
}

static int hashInsert(HashTable *h, uint64_t key, int k1val) {
    if (key == 0) key = 1;
    size_t idx = hash_u64(key, h->mask);
    for (;;) {
        if (h->table[idx].key == 0) {
            h->table[idx].key = key;
            h->table[idx].k1_val = k1val;
            return 1;
        }
        idx = (idx + 1) & h->mask;
    }
    return 0;
}

// lookup returns -1 if not found; otherwise returns stored k1val
static int hashLookup(HashTable *h, uint64_t key) {
    if (key == 0) key = 1;
    size_t idx = hash_u64(key, h->mask);
    for (;;) {
        uint64_t k = h->table[idx].key;
        if (k == 0) return -1;
        if (k == key) return h->table[idx].k1_val;
        idx = (idx + 1) & h->mask;
    }
}

static int getSubkey(const int key[4], int p, int a, int b, int nk[4]) {
    int k22 = mod(key[3], p);
    int t;

    if (k22 == 0) {
        t = mod(b, p);
    } else {
        int inv_k22;
        if (!modinv(k22, p, &inv_k22)) return 0;
        long long tmp = (long long)a * inv_k22 + b;
        t = mod((int)tmp, p);
    }

    nk[0] = mod(key[0] + t, p);          
    nk[1] = mod(key[1] + nk[0], p);      
    nk[2] = mod(key[2] + nk[1], p);      
    nk[3] = mod(key[3] + nk[2], p);      

    return 1;
}




static int generateKeys(const int Kstart[4], int rounds, int p, int a, int b, int (*outKeys)[4]) {
    if (rounds <= 0) return 0;
    memcpy(outKeys[0], Kstart, 4 * sizeof(int));
    for (int r = 1; r < rounds; ++r) {
        if (!getSubkey(outKeys[r-1], p, a, b, outKeys[r])) return 0;
    }
    return 1;
}




static int encryptRound(const int* block, const int* key, const int* T,
                         int a, int b, int p, int* out)
{
    if (!block || !key || !T || !out) return 0;

    // Layer 1 
    for (int i = 0; i < 4; ++i) {
        int m = mod(block[i], p);
        if (m == 0) {
            out[i] = mod(b, p);              
        } else {
            int invVal;
            if (!modinv(m, p, &invVal)) return 0;
            long long t = (long long)a * invVal + (long long)b;
            out[i] = mod((int)(t % p), p);
        }
    }

    // Layer 2 
    { int tmp = out[2]; out[2] = out[3]; out[3] = tmp; }

    // Layer 3 
    for (int i = 0; i < 2; ++i) {
        int v0 = out[i];
        int v1 = out[2 + i];
        long long n0 = (long long)T[0] * v0 + (long long)T[1] * v1;
        long long n1 = (long long)T[2] * v0 + (long long)T[3] * v1;
        out[i]     = mod((int)(n0 % p), p);
        out[2 + i] = mod((int)(n1 % p), p);
    }

    // Layer 4
    for (int i = 0; i < 4; ++i)
        out[i] = mod(out[i] + key[i], p);

    return 1;
}



static int decryptRound(const int* block, const int* key, const int* tInv,
                          int a, int b, int p, int* out)
{
    //layer 4 all the way down to layer 1
    if (!block || !key || !tInv || !out) return 0;

    // layer 4:
    for (int i = 0; i < 4; ++i) {
        out[i] = mod(block[i] - key[i], p);
    }

    // layer 3: use provided tInv
    for (int i = 0; i < 2; ++i) {
        int v0 = out[i];
        int v1 = out[2 + i];

        int new0 = mod(tInv[0]*v0 + tInv[1]*v1, p);
        int new1 = mod(tInv[2]*v0 + tInv[3]*v1, p);

        out[i]     = new0;
        out[2 + i] = new1;
    }

    // layer 2:
    {
        int tmp = out[2];
        out[2] = out[3];
        out[3] = tmp;
    }

    // layer 1:
    int aInv;
    if (!modinv(a, p, &aInv)) return 0;

    for (int i = 0; i < 4; ++i) {
        if (out[i] == b) {
            out[i] = 0;
            continue;
        }

        int val = mod((int)((long)aInv * mod(out[i] - b, p)), p);
        int invVal;
        if (!modinv(val, p, &invVal)) return 0;
        out[i] = invVal;
    }

    return 1;
}



static int encryptFull(const int in[4], const int Kstart[4],
                        int rounds, int p, int a, int b,
                        const int T[4], int out[4])
{
    if (rounds <= 0) return 0;
    int (*keys)[4] = malloc(rounds * 4 * sizeof(int));
    if (!keys) return 0;
    if (!generateKeys(Kstart, rounds, p, a, b, keys)) { free(keys); return 0; }

    int curIn[4], tmp[4];
    memcpy(curIn, in, 4 * sizeof(int));

    for (int r = 0; r < rounds; ++r) {
        if (!encryptRound(curIn, keys[r], T, a, b, p, tmp)) { free(keys); return 0; }
        memcpy(curIn, tmp, 4 * sizeof(int));
    }
    memcpy(out, curIn, 4 * sizeof(int));
    free(keys);
    return 1;
}




static int decryptFull(const int in[4], const int Kstart[4],
                        int rounds, int p, int a, int b,
                        const int T[4], int out[4])
{
    if (rounds <= 0) return 0;
    int (*keys)[4] = malloc(rounds * 4 * sizeof(int));
    if (!keys) return 0;
    if (!generateKeys(Kstart, rounds, p, a, b, keys)) { free(keys); return 0; }

    int tInv[4];
    if (!inv2x2mod(T, p, tInv)) { free(keys); return 0; }

    int curIn[4], tmp[4];
    memcpy(curIn, in, 4 * sizeof(int));

    for (int r = rounds - 1; r >= 0; --r) {
        if (!decryptRound(curIn, keys[r], tInv, a, b, p, tmp)) { free(keys); return 0; }
        memcpy(curIn, tmp, 4 * sizeof(int));
    }

    memcpy(out, curIn, 4 * sizeof(int));
    free(keys);
    return 1;
}




char* decryptAESV(const int* cipher, int nBlocks, int p, int a, int b,
                  const int T[4], const int K1[4], int rounds)
{
    if (!cipher || nBlocks <= 0 || rounds <= 0) 
        return NULL;

    size_t total = (size_t)nBlocks * 4;
    int *out = malloc(total * sizeof(int));
    if (!out) 
        return NULL;

    for (int i = 0; i < nBlocks; ++i) {
        const int *in_block = &cipher[i * 4];
        int *out_block = &out[i * 4];
        if (!decryptFull(in_block,K1, rounds, p, a, b, T, out_block)) {
            free(out);
            return NULL;
        }
    }

    for (size_t i = 0; i < total; ++i) {
    out[i] = mod(out[i], p); 
    if (out[i] >= 65 && out[i] <= 90) continue;
    out[i] = 65 + (out[i] % 26);
}


    char *text = numbersToBytes(out, total);
    free(out);


    return text;
}





const char* mitmHash(const int key1_template[4], const int key2_template[4],
                      const int message[4], const int cipher[4],
                      int p, int a, int b, const int T[4],
                      int K1max, int K2max, int rounds)
{
    HashTable *ht = hashCreate((size_t)K1max);

    int k1_unknown_count = 0, k2_unknown_count = 0;
    for (int i = 0; i < 4; ++i) {
        if (key1_template[i] == -1) k1_unknown_count++;
        if (key2_template[i] == -1) k2_unknown_count++;
    }
    if (k1_unknown_count != 1 || k2_unknown_count != 1) {
        hashFree(ht);
        return "[mitmHash requires exactly one unknown index per key]";
    }

    int midF[4], midB[4];
    int K1[4];
    memcpy(K1, key1_template, sizeof(int)*4);

    int k1_unknown = -1, k2_unknown = -1;
    for (int i = 0; i < 4; ++i) {
        if (key1_template[i] == -1) k1_unknown = i;
        if (key2_template[i] == -1) k2_unknown = i;
    }

    if (k1_unknown < 0 || k2_unknown < 0) {
        hashFree(ht);
        return "[no unknown key index detected]";
    }

    for (int k1x = 0; k1x < K1max; ++k1x) {
        K1[k1_unknown] = k1x;
        if (!encryptFull(message, K1, rounds, p, a, b, T, midF)) continue;
        uint64_t pk = pack_mid(midF);
        hashInsert(ht, pk, k1x);
    }

    int K2[4];
    memcpy(K2, key2_template, sizeof(int)*4);

    static char resultBuf[256];
    for (int k2x = 0; k2x < K2max; ++k2x) {
        if (k2x % 32 == 0) {
            printf("  scanning K2[%d]=%d...\n", k2_unknown, k2x);
            fflush(stdout);
        }
        K2[k2_unknown] = k2x;
        if (!decryptFull(cipher, K2, rounds, p, a, b, T, midB)) continue;

        uint64_t pk = pack_mid(midB);
        int foundK1 = hashLookup(ht, pk);
        if (foundK1 >= 0) {
            snprintf(resultBuf, sizeof(resultBuf),
                    "FOUND match!\n"
                    "p=%d a=%d b=%d\n"
                    "T=[%d,%d,%d,%d]\n"
                    "rounds=%d\n"
                    "K1[%d]=%d\n"
                    "K2[%d]=%d\n"
                    "Meeting block=[%d,%d,%d,%d]",
                    p, a, b,
                    T[0], T[1], T[2], T[3],
                    rounds,
                    k1_unknown, foundK1,
                    k2_unknown, k2x,
                    midB[0], midB[1], midB[2], midB[3]);
                
            hashFree(ht);
            return resultBuf;
        }
    }
    

    hashFree(ht);
    return "[no match found]";
}


const char* mitmHashSingle(
    const int key1_template[4],
    const int key2_template[4],
    const int message[4],
    const int cipher[4],
    int p, int a, int b,
    const int T[4],
    int rounds)
{
    int k1Space = p;
    int k2Space = p;
    return mitmHash(key1_template, key2_template, message, cipher,
                    p, a, b, T, k1Space, k2Space, rounds);
}




const char* aesEntry(const char* alphabet, const char* cipherText, const char* fragment) {
    static char* output = NULL;
    if (output) { free(output); output = NULL; }

    int p = 0, a = 0, b = 0;
    int T[4] = {0}, Tcount = 0;
    int Tmax = 0;

    int rounds = 3;

    int key[4] = {0}, keyCount = 0;
    int key1[4] = {0}, key1Count = 0;
    int key2[4] = {0}, key2Count = 0;
    int message[4] = {0}, messageCount = 0;
    int *cipher = NULL;
    int cipherCount = 0;

    if (fragment && fragment[0]) {
        char* copy = strdup(fragment);
        char* part = strtok(copy, "|");
        while (part) {
            while (*part == ' ' || *part == '\t') ++part;

            if (strncmp(part, "p:", 2) == 0) p = stoi(part + 2);
            else if (strncmp(part, "a:", 2) == 0) a = stoi(part + 2);
            else if (strncmp(part, "b:", 2) == 0) b = stoi(part + 2);
            else if (strncmp(part, "T:", 2) == 0) parseCSV(part + 2, T, &Tcount);
            else if (strncmp(part, "K:", 2) == 0) parseCSV(part + 2, key, &keyCount);
            else if (strncmp(part, "K1:", 3) == 0) parseCSV(part + 3, key1, &key1Count);
            else if (strncmp(part, "K2:", 3) == 0) parseCSV(part + 3, key2, &key2Count);
            else if (strncmp(part, "M:", 2) == 0) parseCSV(part + 2, message, &messageCount);
            else if (strncmp(part, "R:", 2) == 0) rounds = stoi(part + 2);

            part = strtok(NULL, "|");
        }
        free(copy);
    }

    if (!cipher && cipherText && cipherText[0]) {
        int n = 0;
        int *tmp = parse_frag_array(cipherText, &n);
        if (tmp && n >= 4 && (n % 4) == 0) {
            cipher = tmp;
            cipherCount = n;
        } else {
            if (tmp) free(tmp);
            cipher = NULL;
            cipherCount = 0;
        }
    }

    if (p <= 0 || a == 0 || b == 0 || Tcount != 4) {
        return strdup("[missing or invalid parameters p, a, b, or T]");
    }


    int hasSingleKey = (keyCount == 4);
    int hasDoubleKey = (key1Count == 4 && key2Count == 4);
    int hasBlocks = (messageCount == 4 && cipherCount == 4);

    int useMITM = 0;
    if (hasDoubleKey) {
        useMITM = 1;
    }
    else if (hasBlocks && (key1Count == 4 || key2Count == 4)) {
        useMITM = 1;
    }
    else if (hasSingleKey && hasBlocks) {
        for (int i = 0; i < 4; ++i) {
            if (key[i] == -1) { useMITM = 1; break; }
        }
    }

    if (!useMITM) {
        if (!hasSingleKey) return strdup("[missing key for AES-V decryption]");
        if (!cipher || cipherCount == 0) return strdup("[missing ciphertext input]");

        const char* result = decryptAESV(cipher, cipherCount / 4, p, a, b, T, key, rounds);
        if (result) {
            output = strdup(result);
            free((void*)result);
            free(cipher);
        } else {
            output = strdup("[decryptAESV failed]");
        }
    }
    else {
        if (!hasBlocks) return strdup("[missing message or cipher blocks for MITM attack]");

        int k1[4] = {-1, -1, -1, -1};
        int k2[4] = {-1, -1, -1, -1};
        for (int i = 0; i < 4; i++) {
            if (key1Count == 4) k1[i] = key1[i];
            if (key2Count == 4) k2[i] = key2[i];
        }
        if (!key1Count && hasSingleKey) {
            for (int i = 0; i < 4; i++) k1[i] = key[i];
        }



        const char* result = mitmHashSingle(k1, k2, message, cipher, p,a,b,
                                             T, rounds);
        if (result) output = strdup(result);
        else output = strdup("[mitmAESV failed]");
    }

    return output;
}
