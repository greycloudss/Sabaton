#pragma once
#include <stdio.h>
#include <unistd.h>
#include <string.h>
	
typedef struct {
	void* startPtr;
	void* endPtr;

	int regionSize;
} Wordlist;

int getCount(Wordlist wlist) {
	if (!wlist.startPtr || !wlist.endPtr) return -1; 
	int count = 0;
	char* curPtr_ch = (char*) wlist.startPtr;
	char* endPtr_ch = (char*)  wlist.endPtr;
	while (curPtr_ch != endPtr_ch) {
		if (*curPtr_ch == '\0') count++;
			curPtr_ch++;
	}
}

// will return a char* to the new array
void resize(int count, Wordlist wlist) {
	if (count == 0) return;

	if (count > 0) {
		memset(wlist.endPtr, 0, count);
		wlist.regionSize = (char*)wlist.endPtr - (char*)wlist.startPtr + 2;
		memset(wlist.endPtr, '\0', wlist.regionSize + 1);
	
		return;
	}
		
	// if count < 0, since its already negative im adding it 
	
	memset((void*)((char*)wlist.endPtr + count - 1), 0, count);
	memset(wlist.endPtr, '\0', 1);
	
	return;
}
	
	// dont know if this is mega brain or mega dumb
char setAddress(void* startPtr, void* endPtr, char startEnd_fl, void* address) {
	if (startEnd_fl == 0) startPtr = address;
	if (startEnd_fl == 1) endPtr = address;
	
	return startEnd_fl < 0 || startEnd_fl > 1 ? 0 : 1;
}