/* Float arithmetic via the C ingester. Exercises fadd/fsub/fmul/fdiv
   and confirms the values match what strict IEEE 754 would produce. */

#include <stdio.h>

int main(void) {
    double a = 1.5;
    double b = 2.25;
    double sum  = a + b;
    double diff = a - b;
    double prod = a * b;
    double quot = b / a;
    printf("sum  = %g\n", sum);
    printf("diff = %g\n", diff);
    printf("prod = %g\n", prod);
    printf("quot = %g\n", quot);
    return 0;
}
