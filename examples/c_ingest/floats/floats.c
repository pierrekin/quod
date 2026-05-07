/* Float arithmetic, comparison, casts.
   Demonstrates: float and double params/returns, float arithmetic
   (fadd/fsub/fmul/fdiv/frem), float comparison (flt/feq), explicit
   cast `(int)x` (fptosi.sat), implicit promotion `int → double`
   (sitofp), float-widening `(double)f32` (fpext), and one literal of
   each width (`1.5f`, `2.5`). */

#include <stdio.h>

double scale(double x, double k) {
    return x * k;
}

int below(double a, double b) {
    return a < b;
}

int truncate(double x) {
    return (int)x;
}

double promote(int n) {
    return n;
}

double widen32(float f) {
    return f;
}

int main(void) {
    double y = scale(1.5, 2.5);
    printf("scale(1.5, 2.5) = %g\n", y);
    printf("below(1.5, 2.5) = %d\n", below(1.5, 2.5));
    printf("truncate(3.7) = %d\n", truncate(3.7));
    printf("promote(5) = %g\n", promote(5));
    printf("widen32(1.5f) = %g\n", (double)widen32(1.5f));
    return 0;
}
