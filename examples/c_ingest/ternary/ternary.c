/* Ternary `cond ? a : b` lifts to a layer-A c.ternary that pairs
   with a layer-B IfExpr (core node lowered via branch + phi). */

#include <stdio.h>

int abs_val(int x) {
    /* Comparison-shaped condition — lifts directly. */
    return x < 0 ? -x : x;
}

int max3(int a, int b, int c) {
    /* Nested ternary in a return value. */
    return a > b ? (a > c ? a : c) : (b > c ? b : c);
}

int sign_or_zero(int x) {
    /* Integer condition — `x` (not a comparison) gets the C
       "nonzero ⇒ true" widening to `x != 0` on the layer-B side. */
    return x ? (x > 0 ? 1 : -1) : 0;
}

int main(void) {
    printf("abs_val(-7)         = %d\n", abs_val(-7));
    printf("abs_val(5)          = %d\n", abs_val(5));
    printf("max3(2, 9, 4)       = %d\n", max3(2, 9, 4));
    printf("max3(2, 4, 9)       = %d\n", max3(2, 4, 9));
    printf("sign_or_zero(0)     = %d\n", sign_or_zero(0));
    printf("sign_or_zero(-3)    = %d\n", sign_or_zero(-3));
    printf("sign_or_zero(42)    = %d\n", sign_or_zero(42));
    return 0;
}
