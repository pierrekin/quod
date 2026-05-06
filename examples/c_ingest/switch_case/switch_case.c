/* Switch / case lowered to an if-else-if chain at layer B. The
   supported subset: every case ends in `break;` or `return ...;`,
   and shared-empty-case stacking (`case 1: case 2: shared;`) is
   allowed. Implicit fallthrough (case body without break/return)
   refuses at ingest. */

#include <stdio.h>

int day_length(int day) {
    /* Standard switch with break. Shared cases for 30-day months. */
    switch (day) {
        case 1:
            return 31;
        case 4:
        case 6:
        case 9:
        case 11:
            return 30;
        case 2:
            return 28;
        default:
            return 31;
    }
}

int classify(int x) {
    /* Switch yielding values via assignment + trailing break. */
    int label = 0;
    switch (x) {
        case 0:
            label = 100;
            break;
        case 1:
        case 2:
        case 3:
            label = 200;
            break;
        default:
            label = 999;
            break;
    }
    return label;
}

int main(void) {
    printf("day_length(1)  = %d\n", day_length(1));
    printf("day_length(2)  = %d\n", day_length(2));
    printf("day_length(4)  = %d\n", day_length(4));
    printf("day_length(11) = %d\n", day_length(11));
    printf("day_length(7)  = %d\n", day_length(7));
    printf("classify(0)    = %d\n", classify(0));
    printf("classify(2)    = %d\n", classify(2));
    printf("classify(99)   = %d\n", classify(99));
    return 0;
}
