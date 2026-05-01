/* The classic. Combines while, nested if/else, modulo, and printf with both
   string and int format args. */

#include <stdio.h>

int main(void) {
    int i = 1;
    while (i <= 15) {
        if (i % 15 == 0) {
            printf("FizzBuzz\n");
        } else if (i % 3 == 0) {
            printf("Fizz\n");
        } else if (i % 5 == 0) {
            printf("Buzz\n");
        } else {
            printf("%d\n", i);
        }
        i = i + 1;
    }
    return 0;
}
