#include "main.h"
#include <stdio.h>
#include "util/string.h"
#include <locale.h>


volatile char killswitch = 0;
#ifdef _WIN32
    #include <windows.h>
    #include <shellapi.h>
    #include <wchar.h>
    void printASCII(void) {
        HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
        if (h == INVALID_HANDLE_VALUE) return;

        static const wchar_t* art =
            L"  \u2588\u2588\u2588\u2588\u2588\u2588  \u256d\u2550\u2550\u2550       \u256d\u256c\u256c\u2550    \u256d\u2550\u2550     \u256d\u2550\u2550\u2588\u2588\u2588\u2588\u2588\u2557 \u2502\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2584    \u2588 \n"
            L"\u255a\u2588\u2588    \u2554 \u2502\u2588\u2588\u2588\u2588\u2564    \u2566\u2588\u2588\u2588\u2588\u2557 \u2502\u2588\u2588\u2588\u2588\u2584   \u2566  \u2588\u2588\u255f \u2552\u2502\u2502\u2588\u2588\u2500\u2500\u2588\u2588\u255f \u2588\u2588\u2502\u2009\u2518\u2588\u2588   \u2588 \n"
            L"\u2591 \u2592\u2588\u2588\u255b   \u2502\u2588\u2588  \u2570\u2588\u2584  \u2502\u2588\u2588\u2502 \u2570\u256e\u256e\u2502\u2588\u2588  \u2570\u2588\u2584 \u2502 \u2562\u2588\u2588\u2502 \u2570\u2502\u2502\u2588\u2588  \u2588\u2588\u255f\u2502\u2588\u2588  \u2570\u2588 \u2588\u2588\u255f\n"
            L"  \u2502   \u2588\u2588\u2554\u2502\u2591\u2588\u2588\u2588\u2588\u2588\u2588\u255b \u2502\u2588\u2588\u2579\u2550  \u2591\u2588\u2588\u2588\u2588\u2588\u2588\u255b\u2502 \u2562\u2588\u2588\u250f \u2502 \u2588\u2588   \u2588\u2588\u2591\u2502\u2588\u2588\u255f  \u2502\u2571\u2588\u2588\u255f\n"
            L"\u255a\u2588\u2588\u2588\u2588\u2588\u2554\u2502\u2502 \u2566   \u2562\u2588\u2588\u255f\u2591\u2502\u2591\u2588\u2588  \u2570\u2588\u2588\u2502 \u2502\u2588\u2588\u255f \u2591 \u2591 \u2588\u2588\u2588\u2588\u2588\u2552\u2502\u2588\u2588\u2591   \u2562\u2588\u2588\u2591\n"
            L"\u255a \u2502\u2570\u2502 \u2502 \u2502\u2502   \u2562\u2502\u2588\u2591\u2502\u2591\u2562\u2588\u2588\u2588\u257b\u2554\u2502 \u2502\u2502   \u2562\u2502\u2588\u255f \u2502 \u2502 \u2502\u2591\u2502\u2591 \u2591\u2502\u2588\u2588\u2591   \u2502 \u2502 \n"
            L"\u2591 \u2591\u2502  \u2591 \u2591  \u2502   \u2502\u2502 \u2502\u2591  \u2502   \u2502\u2502 \u2502\u2591    \u2591 \u2502 \u2502\u2591 \u2591 \u2591  \u2591 \u2591  \u2591  \n"
            L"\u2591  \u2591  \u2591    \u2591   \u2502    \u2591   \u2591   \u2502    \u2591      \u2591 \u2502 \u2591    \u2591     \u2591   \u2591 \n"
            L"      \u2591        \u2591  \u2591 \u2591            \u2591  \u2591            \u2591 \u2591           \u2591 \n"
            L"                         \u2591                                       \n\n";

        DWORD w;
        WriteConsoleW(h, art, (DWORD)wcslen(art), &w, NULL);
    }
#else

void printASCII()
{
    // cuz why not be cool and hip like the youngins
    printf("  ██████  ▄▄▄       ▄▄▄▄    ▄▄▄     ▄▄▄█████▓ ▒█████   ███▄    █ \n");
    printf("▒██    ▒ ▒████▄    ▓█████▄ ▒████▄   ▓  ██▒ ▓▒▒██▒  ██▒ ██ ▀█   █ \n");
    printf("░ ▓██▄   ▒██  ▀█▄  ▒██▒ ▄██▒██  ▀█▄ ▒ ▓██░ ▒░▒██░  ██▒▓██  ▀█ ██▒\n");
    printf("  ▒   ██▒░██▄▄▄▄██ ▒██░█▀  ░██▄▄▄▄██░ ▓██▓ ░ ▒██   ██░▓██▒  ▐▌██▒\n");
    printf("▒██████▒▒ ▓█   ▓██▒░▓█  ▀█▓ ▓█   ▓██▒ ▒██▒ ░ ░ ████▓▒░▒██░   ▓██░\n");
    printf("▒ ▒▓▒ ▒ ░ ▒▒   ▓▒█░░▒▓███▀▒ ▒▒   ▓▒█░ ▒ ░░   ░ ▒░▒░▒░ ░ ▒░   ▒ ▒ \n");
    printf("░ ░▒  ░ ░  ▒   ▒▒ ░▒░▒   ░   ▒   ▒▒ ░   ░      ░ ▒ ▒░ ░ ░░   ░ ▒░\n");
    printf("░  ░  ░    ░   ▒    ░    ░   ░   ▒    ░      ░ ░ ░ ▒     ░   ░ ░ \n");
    printf("      ░        ░  ░ ░            ░  ░            ░ ░           ░ \n");
    printf("                         ░                                       \n\n");
}
#endif

void printHelp()
{
    printASCII();
    printf("Usage:\n");
    printf("  <prog> -decypher -affineCaesar -alph <alphabet> [-frag <fragment> | -brute]\n");
    printf("\n");
    printf("-decypher           Enable cipher-decoding mode.\n");
    printf("-scytale            Select the Scytale module.\n");
    printf("-transposition      Select the Transposition module.\n");
    printf("-fleissner          Select the Fleissner module.\n");
    printf("                    Use -frag \"N;H\" to specify the mask and decrypt.\n");
    printf("                    N: size of the mask(4 = 4x4)\n");
    printf("                    H: holes (if holes are at [0, 2], then the H will look something like 101...) \n");
    printf("-bifid              Select the Bifid cipher module.\n");
    printf("                    Use -frag \"KEY;5\" to specify the key and column number.\n");
    printf("-affineCaesar       Select the Affine Caesar cipher module.\n");
    printf("-hill               Select the Hill cipher module.\n");
    printf("-vigenere           Select the Vigenere cipher module.\n");
    printf("                    Use -frag \"crack:min-max\" to guess key length in [min,max] and decrypt.\n");
    printf("                    For autokey Vigenere prefix the fragment with \"auto:\" (e.g. -frag \"auto:VYRAS\").\n");
    printf("-enigma             Select the Enigma cipher module.\n");
    printf("                    For Enigma, pack rotors, reflector and key into -frag as\n");
    printf("                    \"R1:<...>|R2:<...>|KEY:<...>\".\n");
    printf("-feistel            Select the Feistel cipher module.\n");
    printf("                    For Feistel, pack function and keys into -frag as \"f=<0..3>;k=[...]\".\n");
    printf("                    Example: -frag \"f=0;k=[108,59,164]\" or just -frag \"[108,59,164]\".\n");
    printf("-aes                Select the AES variation cipher module.\n");
    printf("                    For AES-V, specify parameters inside -frag as\n");
    printf("                    \"p:<prime>|a:<num>|b:<num>|T:<4 vals>|K:<4 vals>|R:<rounds>\".\n");
    printf("                    Example: -frag \"p:317|a:13|b:15|T:1,11,31,4|K:132,293,146,119|R:3\".\n");
    printf("                    Ciphertext is provided as a 2x2 block list, e.g.:\n");
    printf("                    \"[[116,301,44,15],[139,239,295,293],...]\".\n");
    printf("                    For meet-in-the-middle (MITM) searches, use K1: and K2: templates\n");
    printf("                    with '?' for unknown bytes to scan:\n");
    printf("                    \"p:317|a:10|b:10|T:11,12,13,14|K1:?,128,272,271|K2:212,?,117,102|M:101,177,263,224|R:3\".\n");
    printf("-merkle             Select the Merkle–Hellman knapsack module.\n");
    printf("                    Use -frag \"key:<v1,...,vn>|p:<modulus>|w1:<first_private_weight>\".\n");
    printf("                    Example:\n");
    printf("                    -frag \"key:39342119,111996362,9301087,85349912,114247265,79246980,68969224,40465975|p:114537401|w1:811451\"\n");
    printf("                    Ciphertext is a list of block sums, e.g.:\n");
    printf("                    \"[206647361,326360316,424226893,314791694,241010404,9301087,...]\".\n");
    printf("                    Notes: v is the public weight vector; p is the modulus; w1 is the first private weight.\n");
    printf("-graham             Select the Graham–Shamir knapsack module.\n");
    printf("                    Use -frag \"key:<v1,...,vn>|p:<modulus>|w1:<first_private_weight>\".\n");
    printf("                    Example:\n");
    printf("                    -frag \"key:158474690964,197857591142,123672933023,16130333379,142979253486,151965468545,30534200386,127450405592|p:211936606955|w1:41985769523\"\n");
    printf("                    Ciphertext is a list of large integers, e.g.:\n");
    printf("                    \"[495043978037,617076731681,774459852174,...]\".\n");
    printf("                    Bit slice: message bits are 12–19 from the right (1-based), packed MSB=bit19.\n");
    printf("-block              Select the Feistel *block cipher* module (ECB/CBC/CFB/CRT).\n");
    printf("                    Syntax: -frag '<MODE>:[k1,k2,k3];f=<0..3>' '<[[L,R], [L,R], ...]]'\n");
    printf("                    MODE: ECB (also accepts EBC), CBC, CFB, CRT (also accepts CTR).\n");
    printf("                    Keys: exactly three round keys [k1,k2,k3] (bytes 0..255).\n");
    printf("                    f=<id> selects F(r,k) variant used in labs:\n");
    printf("                       f=0 : (r|k)^((r//16)&k)\n");
    printf("                       f=1 : (r^k)&((k//16)|r)\n");
    printf("                       f=2 : (r|k)^((k//16)&r)\n");
    printf("                       f=3 : (r^k)&((r//16)|k)\n");
    printf("                    Blocks are byte pairs [L,R]. 3 Feistel rounds, final swap undone.\n");
    printf("                    CBC/CFB: supply IV as the *first* pair in the ciphertext array.\n");
    printf("                    CRT/CTR: keystream from a=F(i,k1); encrypt [a,a], XOR with C_i.\n");
    printf("-alph <alphabet>    Alphabet string to operate on; its length m defines modulo m.\n");
    printf("                    Characters not in this string pass through unchanged.\n");
    printf("-frag <fragment>    Known plaintext snippet (e.g., prefix). Uses it to solve keys (a,b)\n");
    printf("                    directly and decrypt once; fastest if the snippet is correct.\n");
    printf("-brute              Try all valid keys (a coprime with m; b in [0..m-1]) and output each\n");
    printf("-enhancedBrute      ONLY FOR LITHUANIAN; Try all valid keys but also sort by phonetic coherency \n");
    printf("                    candidate plaintext with its keys.\n");
    printf("\n");
    printf("Notes:\n");
    printf("  > Provide -alph with the exact ordering you expect (case/diacritics included).\n");
    printf("  > Prefer -frag when you know a snippet; use -brute when you don't.\n");
    printf("  > If both -frag and -brute are given, the tool will attempt -frag first and may fall back to -brute.\n");
    printf("\n");
    printf("Examples:\n");
    printf("  <prog> -decypher -affineCaesar -alph \"ABCDEFGHIJKLMNOPQRSTUVWXYZ\" -frag \"THE\"\n");
    printf("  <prog> -decypher -affineCaesar -alph \"AĄBCČDEĘĖFGHIY...Ž\" -brute\n");
    printf("  <prog> -decypher -feistel -frag \"f=1;k=[?,30]" "[[92, 6], [91, 4], [74, 11], [78, 9], ... ]\"\n");
    printf("  <prog> -decypher -block -frag \"CRT:[210, ...];f=0' '[[238, 113], [252, 109], ... ]'\n");
    printf(" <prog> -decypher -fleissner -frag \"4;1010000100000000\" \"JAEIFWFEWF...\"\n");
    printf("  <prog> -decypher -bifid -alph \"ABCDEFGHIKLMNOPQRSTUVWXYZ\" -frag \"KEY;5\" \"TAFRQOS...\"\n");
}

void parseArgs(Arguments *args, const int argv, const char **argc)
{
    memset(args->flags, 0, sizeof(args->flags));
    args->decrypt = 0;
    args->decypher = 0;
    args->enhancedBrute = 0;
    args->brute = 0;
    args->frag = NULL;
    args->alph = NULL;
    args->wordlist = NULL;
    args->encText = NULL;
    args->out = NULL;

    args->affineCaesar = 0;
    args->hill = 0;
    args->vigenere = 0;
    args->enigma = 0;
    args->feistel = 0;
    args->block = 0;
    args->stat = 0;

    args->scytale = 0;
    args->transposition = 0;
    args->fleissner = 0;
    args->bifid = 0;
    args->stream = 0;


    args->enigma = 0;
    args->aes = 0;
    args->merkle = 0;
    args->graham = 0;

    args->banner = 0;

    for (int i = 1; i < argv; ++i){
        const char *a = argc[i];

        if (strcmp(a, "-h") == 0){
            printHelp();
            return;
        }

        if (strcmp(a, "-decrypt") == 0){
            args->decrypt = 1;
            continue;
        }

        if (strcmp(a, "-decypher") == 0){
            args->decypher = 1;
            continue;
        }

        if (strcmp(a, "-enhancedBrute") == 0){
            args->brute = 1;
            args->enhancedBrute = 1;
            continue;
        }

        if (args->decrypt){
            if (strcmp(a, "-w") == 0){
                if (i + 1 < argv)
                {
                    args->flags[0] = 1;
                    args->wordlist = argc[++i];
                }
                continue;
            }
            if (strcmp(a, "-ml") == 0){
                if (i + 1 < argv)
                {
                    args->flags[1] = 1;
                    args->minLength = (unsigned char)stoi(argc[++i]);
                }
                continue;
            }
            if (strcmp(a, "-xl") == 0){
                if (i + 1 < argv)
                {
                    args->flags[2] = 1;
                    args->maxLength = (unsigned char)stoi(argc[++i]);
                }
                continue;
            }
            if (strcmp(a, "-s") == 0){
                if (i + 1 < argv)
                {
                    args->flags[3] = 1;
                    args->specialRegex = argc[++i];
                }
                continue;
            }
        }

        if (args->decypher){
            if (strcmp(a, "-affineCaesar") == 0){
                args->affineCaesar = 1;
                continue;
            }

            if (strcmp(a, "-hill") == 0){
                args->hill = 1;
                continue;
            }

            if (strcmp(a, "-scytale") == 0){
                args->scytale = 1;
                continue;
            }
            if (strcmp(a, "-fleissner") == 0){
                args->fleissner = 1;
                continue;
            }

            if (strcmp(a, "-stream") == 0){
                args->stream = 1;
                continue;
            }
            if (strcmp(a, "-stat") == 0){
                args->stat = 1;
                continue;
            }
            

            if (strcmp(a, "-vigenere") == 0 || strcmp(a, "-vig") == 0){
                args->vigenere = 1;
                continue;
            }

            if (strcmp(a, "-enigma") == 0){
                args->enigma = 1;
                continue;
            }

            if (strcmp(a, "-block") == 0) {
                args->block = 1;
                continue;
            }
            if (strcmp(a, "-feistel") == 0){
                args->feistel = 1;
                continue;
            }

            if (strcmp(a, "-aes") == 0){
                args->aes = 1;
                continue;
            }
            if (strcmp(a, "-graham") == 0){
                args->graham = 1;
                continue;
            }
            if (strcmp(a, "-merkle") == 0){
                args->merkle = 1;
                continue;
            }
            if (strcmp(a, "-transposition") == 0) {
                args->transposition = 1;
                continue;
            }
            if (strcmp(a, "-bifid") == 0) {
                args->bifid = 1;
                continue;
            }
            if (strcmp(a, "-brute") == 0){
                args->brute = 1;
                continue;
            }
            if (strcmp(a, "-frag") == 0){
                if (i + 1 < argv){
                    args->frag = argc[++i];
                }
                continue;
            }
            if (strcmp(a, "-alph") == 0){
                if (i + 1 < argv){
                    args->alph = argc[++i];
                }
                continue;
            }
        }

        if (strcmp(a, "-banner") == 0){
            args->banner = 1;
            continue;
        }

        if (a[0] != '-' && !args->encText){
            args->encText = a;
            continue;
        }
    }
}

void decypher(Arguments *args)
{
    if (!args || !args->decypher)
        return;

    if (args->banner)
        printASCII();

    if (args->feistel){
        if (args->brute || !args->frag){
            const char *res = feistelEntry(args->encText, NULL, 0);
            args->out = res;
            return;
        }
        else{
            char flag = 0;
            char *keys = NULL;
            feistel_extract(args->frag, &flag, &keys);
            const char *res = feistelEntry(args->encText, keys, flag);
            if (keys)
                free(keys);
            args->out = res;
            return;
        }
    }

    if (args->block) {
        if (args->brute || !args->frag) {
            const char* res = blockEntry(args->encText, NULL, 0);
            args->out = res;
            return;
        } else {
            char flag = 0;
            char* keys = NULL;
            feistel_extract(args->frag, &flag, &keys);
            const char* res = blockEntry(args->encText, keys, flag);
            if (keys) free(keys);
            args->out = res;
            return;
        }
    }

    if (args->scytale) {
        const char* frag = args->brute ? NULL : args->frag;
        const char* res = scytaleEntry(args->alph, args->encText, frag);
        args->out = res;
        return;
    }
    if (args->bifid) {
        const char* frag = args->brute ? NULL : args->frag;
        const char* res = bifidEntry(args->alph, args->encText, frag);
        args->out = res;
        return;
    }
    if (args->stream) {
        const char* frag = args->brute ? NULL : args->frag;
        const char* res = streamEntry(args->alph, args->encText, frag);
        args->out = res;
        return;
    }
    if (args->stat) {
        const char* res = statEntry(args->alph, args->encText);
        args->out = res;
        return;
    }
    if (args->transposition) {
        const char* frag = args->brute ? NULL : args->frag;
        const char* res = transpositionEntry(args->alph, args->encText, frag);
        args->out = res;
        return;
    }
    if (args->fleissner) {
        const char* frag = args->brute ? NULL : args->frag;
        const char* res = fleissnerEntry(args->alph, args->encText, frag);
        args->out = res;
        return;
    }

    if (args->hill){
        const char *frag = args->brute ? NULL : args->frag;
        const char *res = hillEntry(args->alph, args->encText, frag);
        args->out = res;
        return;
    }

    if (args->affineCaesar){
        const char *frag = args->brute ? NULL : args->frag;
        const char *res = affineCaesarEntry(args->alph, args->encText, frag);
        args->out = res;
        return;
    }

    if (args->enigma){
        const char *frag = args->brute ? NULL : args->frag;
        const char *res = enigmaEntry(args->alph, args->encText, frag);
        args->out = res;
        return;
    }

    if (args->aes){
        const char *frag = args->brute ? NULL : args->frag;
        const char *res = aesEntry(args->alph, args->encText, frag);
        args->out = res;
        return;
    }
    if (args->vigenere) {
        const char* frag = args->brute ? NULL : args->frag;
        const char* res = vigenereEntry(args->alph, args->encText, frag);
        args->out = res;
        return;
    }
    if (args->graham) {
        const char* frag = args->brute ? NULL : args->frag;
        const char* res = grahamEntry(args->alph, args->encText, frag);
        args->out = res;
        return;
    }
    if (args->merkle) {
        const char* frag = args->brute ? NULL : args->frag;
        const char* res = merkleEntry(args->alph, args->encText, frag);
        args->out = res;
        return;
    }


}

int main(int argc, const char **argv)
{
    setlocale(LC_ALL, "");
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    int wargc = 0;
    LPWSTR *wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (!wargv)
        return 1;
    char **a = (char **)malloc(sizeof(char *) * wargc);
    if (!a){
        LocalFree(wargv);
        return 1;
    }
    for (int i = 0; i < wargc; ++i){
        int need = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, NULL, 0, NULL, NULL);
        if (need <= 0){
            a[i] = (char *)malloc(1);
            if (a[i])
                a[i][0] = '\0';
            continue;
        }
        a[i] = (char *)malloc((size_t)need);
        if (!a[i]){
            a[i] = (char *)malloc(1);
            if (a[i])
                a[i][0] = '\0';
            continue;
        }
        WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, a[i], need, NULL, NULL);
    }
    Arguments args = (Arguments){ .minLength = 0, .maxLength = 244, .specialRegex = "[!\"#$%&'()*+,-./:;<=>?@[\\\\\\]^_`{|}~]" };
    parseArgs(&args, wargc, (const char**)a);
    //Default alphabet
    if (!args.alph) args.alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    decypher(&args);
    if (args.out && args.out[0]){
        if (strcmp(getExtension(args.out), "txt") == 0 && args.enhancedBrute && !args.feistel)
            recognEntry(args.out);
        print(args.out);
    }
    for (int i = 0; i < wargc; ++i)
        free(a[i]);
    free(a);
    LocalFree(wargv);
    return 0;
#else
    Arguments args = (Arguments){.minLength = 0, .maxLength = 244, .specialRegex = "[!\"#$%&'()*+,-./:;<=>?@[\\\\\\]^_`{|}~]"};
    parseArgs(&args, argc, argv);
    if (!args.alph) args.alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    decypher(&args);
    if (args.out && args.out[0]){
        if (strcmp(getExtension(args.out), "txt") == 0 && args.enhancedBrute && !args.feistel)
            recognEntry(args.out);
        print(args.out);
    }
    return 0;
#endif
}
