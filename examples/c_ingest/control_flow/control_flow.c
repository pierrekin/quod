/* Branching with if/else, including nested-else (else if) chains.
   Demonstrates: comparison operators, short-circuit boolean operators. */

#include <stdio.h>

int classify(int x) {
    if (x < 0) {
        return -1;
    } else if (x == 0) {
        return 0;
    } else {
        return 1;
    }
}

int in_unit_range(int x) {
    return x >= 0 && x <= 100;
}

int main(void) {
    printf("classify(-5) = %d\n", classify(-5));
    printf("classify(0)  = %d\n", classify(0));
    printf("classify(42) = %d\n", classify(42));
    printf("in_unit_range(50) = %d\n", in_unit_range(50));
    printf("in_unit_range(150) = %d\n", in_unit_range(150));
    return 0;
}
