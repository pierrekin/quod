/* Variable-offset pointer arithmetic: cursor walks a string by index.
   The ingester translates `&buf[i]` (with `i` an int local) by widening
   `i` to i64 via quod.widen and feeding it into quod.ptr_offset. */

#include <stdio.h>

int main(void) {
    char *s = "abcdef";
    int i = 0;
    while (i < 6) {
        printf("%s\n", &s[i]);
        i = i + 2;
    }
    return 0;
}
