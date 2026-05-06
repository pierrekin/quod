/* Void-returning functions and bare `return;`. The C ingester maps
   `void f(...)` to a quod Function with return_type=VoidType and emits
   a bare `Return()` for `return;` statements. Falling off the end of a
   void body is treated as an implicit `return;`. */

#include <stdio.h>

void greet(int n) {
    /* Early exit via bare `return;`. */
    if (n <= 0) {
        return;
    }
    printf("hello %d\n", n);
}

void no_explicit_return(int n) {
    /* Falls through without an explicit `return;`. The ingester
       synthesizes a `Return()` terminator at lift time. */
    printf("count = %d\n", n);
}

int main(void) {
    greet(0);     /* prints nothing */
    greet(1);     /* prints "hello 1" */
    greet(2);     /* prints "hello 2" */
    no_explicit_return(42);
    return 0;
}
