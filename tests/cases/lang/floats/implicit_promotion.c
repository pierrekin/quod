/* Implicit int→double promotion via the ingester. C's
   usual-arithmetic-conversion rules insert IMPLICIT_CAST_EXPR around
   the int operand of `int + double`; the layer-B translator
   materializes it as a Cast(F64Type) wrapping the int operand. */

#include <stdio.h>

double add_int_double(int n, double x) {
    return n + x;
}

double from_int(int n) {
    return n;
}

int main(void) {
    printf("add_int_double(3, 0.25) = %g\n", add_int_double(3, 0.25));
    printf("from_int(7) = %g\n", from_int(7));
    return 0;
}
