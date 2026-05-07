/* `#pragma STDC FP_CONTRACT OFF` explicitly grants strict IEEE
   behavior — exactly what quod codegen produces. The pragma walker
   accepts OFF (only ON and DEFAULT are refused) so this file ingests
   and runs cleanly. */

#include <stdio.h>

#pragma STDC FP_CONTRACT OFF

int main(void) {
    printf("hello\n");
    return 0;
}
