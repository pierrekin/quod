/* Float comparison via the C ingester. `<` on float operands lifts
   to BinOp(flt) and returns the C-typical 0/1 int result. */

#include <stdio.h>

int main(void) {
    printf("1.5 <  2.0 = %d\n", 1.5 < 2.0);
    printf("2.0 == 2.0 = %d\n", 2.0 == 2.0);
    printf("2.5 >  1.5 = %d\n", 2.5 > 1.5);
    printf("3.0 != 3.0 = %d\n", 3.0 != 3.0);
    return 0;
}
