#pragma once
#include <string.h>

#define FLAG_COUNT 4

extern volatile char killswitch;


typedef struct {
	char flags[FLAG_COUNT];
	const char* wordlist;
	unsigned char minLength;
	unsigned char maxLength;
	const char* specialRegex;
} Arguments;
