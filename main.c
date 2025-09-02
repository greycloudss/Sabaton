#include "main.h"
#include <stdio.h>

volatile char killswitch = 0;




void printHelp() {
    printf("Usage: a.out [options]\n\n");
    printf("Options:\n");
    printf("  -w <file>       Path to wordlist file\n");
    printf("  -ml <number>    Minimum password length\n");
    printf("  -xl <number>    Maximum password length\n");
    printf("  -s <pattern>    Special character regex\n");
    printf("  -h              Show this help message\n\n");
    printf("Examples:\n");
    printf("  a.out -w words.txt -ml 4 -xl 8 -s !@#^%\n");
}

void parseArgs(Arguments* args, const int argv, const char** argc) {
	memset(args->flags, 0, sizeof(args->flags));

	for (int i = 1; i < argv; ++i) {
		if (strcmp(argc[i], "-w") == 0) {
			args->flags[0] = 1;
			args->wordlist = argc[++i];
			printf("%s\n", args->wordlist);
			continue;
		}
	
		if (strcmp(argc[i], "-ml") == 0) {
			args->flags[1] = 1;
			args->minLength = stoi(argc[++i]);
			printf("%d\n", args->minLength);
			continue;
		}

		if (strcmp(argc[i], "-xl") == 0) {
			args->flags[2] = 1;
			args->maxLength = stoi(argc[++i]);
			printf("%d\n", args->maxLength);
			continue;
		}

		if (strcmp(argc[i], "-s") == 0) {
			args->flags[3] = 1;
			args->specialRegex = argc[++i];
			printf("%s\n", args->specialRegex);
			continue;
		}

		if (strcmp(argc[i], "-h") == 0) {
			printHelp();
			return;
		}
	}
}

int main(int argv, const char** argc) {
	Arguments args = { .minLength = 0, .maxLength = 244, .specialRegex = "[!\"#$%&'()*+,-./:;<=>?@[\\\\\\]^_`{|}~]" };	
	parseArgs(&args, argv, argc);

	return 0;
}

