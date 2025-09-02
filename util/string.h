#pragma once

int stoi(const char* string) {
	int sign = (*string=='-') ? -1 : 1;
	long n = 0;
  
	string += (*string == '+' || *string == '-') ? 1 : 0;

	while (*string >= '0' && *string <= '9') n = n * 10 + (*string++ - '0');
  
	return (int)(sign * n);
}