#include "knapsack.h"



static unsigned long long parseAfterPrefix(const char* s, size_t skip) {
    const char* p = s + skip;
    while (*p && isspace((unsigned char)*p)) ++p;
    while (*p && !isdigit((unsigned char)*p)) ++p;
    return strtoull(p, NULL, 10);
}

void extractKnapsackValues(
    const char* frag,
    unsigned long long** keys_out, int* keyCount_out,
    unsigned long long* p_out,
    unsigned long long* w1_out)
{
    *keys_out = NULL;
    *keyCount_out = 0;
    *p_out = 0;
    *w1_out = 0;

    if (!frag || !frag[0]) return;

    char* copy = strdup(frag);
    if (!copy) return;

    char *saveOuter = NULL;
    char* part = strtok_r(copy, "|", &saveOuter);
    while (part) {
        while (*part && isspace((unsigned char)*part)) ++part;

        if (strncmp(part, "key:", 4) == 0) {
            const char* list = part + 4;

            char* t2 = strdup(list);
            if (t2) {
                for (char* c = t2; *c; ++c) {
                    if (*c == '[' || *c == ']') *c = ' ';
                    else if (*c == ',') *c = ' ';
                }

                int count = 0;
                {
                    char* tmp = strdup(t2);
                    if (tmp) {
                        char *save1 = NULL;
                        char* tok = strtok_r(tmp, " \t\r\n", &save1);
                        while (tok) { ++count; tok = strtok_r(NULL, " \t\r\n", &save1); }
                        free(tmp);
                    }
                }

                if (count > 0) {
                    unsigned long long* arr = (unsigned long long*)malloc(sizeof(unsigned long long) * (size_t)count);
                    if (arr) {
                        int i = 0;
                        char *save2 = NULL;
                        char* tok = strtok_r(t2, " \t\r\n", &save2);
                        while (tok && i < count) {
                            arr[i++] = strtoull(tok, NULL, 10);
                            tok = strtok_r(NULL, " \t\r\n", &save2);
                        }
                        *keys_out = arr;
                        *keyCount_out = i;
                    }
                }
                free(t2);
            }
        }
        else if (strncmp(part, "p:", 2) == 0) {
            *p_out = parseAfterPrefix(part, 2);
        }
        else if (strncmp(part, "w1:", 3) == 0) {
            *w1_out = parseAfterPrefix(part, 3);
        }

        part = strtok_r(NULL, "|", &saveOuter);
    }

    free(copy);
}