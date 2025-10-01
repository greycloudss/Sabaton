#include <stdio.h>
#include <string.h>

char encrypted_text[];
int num;

void decrypt_brute(int rows, int k) {
    int len = strlen(encrypted_text);
    int cols = (len + rows - 1) / rows;
    int short_cols = cols * rows - len;

    printf("k = %d, ", k);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int col_len = (j < cols - short_cols) ? rows : rows - 1;
            if (i < col_len) {
                int pos = 0;
                for (int x = 0; x < j; x++)
                    pos += (x < cols - short_cols) ? rows : rows - 1;
                pos += i;
                putchar(encrypted_text[pos]);
            }
        }
    }
    putchar('\n');
}

void decrypt_k(){
    int k = num;
    decrypt_brute(k + 1, k);
}

int main() {
    printf("k = 0, %s\n", encrypted_text);

    if(num == 0){
        for (int k = 1; k < 30; k++) {
            decrypt_brute(k + 1, k);
        }
    } else {
        decrypt_k();
    }
    return 0;
}
