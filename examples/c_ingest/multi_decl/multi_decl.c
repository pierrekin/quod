/* Multi-declarator declarations: `int a, b, c;` introducing several
   locals in one statement. Layer A preserves the source-form grouping
   as a c.multi_var_decl; the lift expands it to N consecutive Lets on
   layer B, and the lift-checker pairs them 1:N. */

#include <stdio.h>

int sum3(int x, int y, int z) {
    int a = x, b = y, c = z;
    return a + b + c;
}

int linear_combo(int x) {
    /* Initializers can refer to earlier declarators in the same statement
       (left-to-right evaluation order is preserved across the lift). */
    int a = x + 1, b = a * 2, c = b - 1;
    return c;
}

int main(void) {
    printf("sum3(1, 2, 3)        = %d\n", sum3(1, 2, 3));
    printf("linear_combo(10)     = %d\n", linear_combo(10));
    return 0;
}
