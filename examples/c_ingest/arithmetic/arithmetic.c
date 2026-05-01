/* Function-to-function calls plus printf with a %d argument.
   Demonstrates: int params, arithmetic, calling a user function from main. */

#include <stdio.h>

int square(int x) {
    return x * x;
}

int sum_squares(int a, int b) {
    return square(a) + square(b);
}

int main(void) {
    printf("3^2 + 4^2 = %d\n", sum_squares(3, 4));
    return 0;
}
