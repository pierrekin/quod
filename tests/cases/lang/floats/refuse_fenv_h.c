/* `<fenv.h>` is the standard C header for the floating-point
   environment (fesetround, feenableexcept, feclearexcept, etc.).
   quod doesn't model the fenv environment; refusing the include is
   the cleanest single point of refusal for the whole API. */

#include <fenv.h>

int main(void) {
    return 0;
}
