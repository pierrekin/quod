/* `#pragma STDC FENV_ACCESS ON` declares that the program reads or
   writes the floating-point environment (rounding mode, exception
   flags). quod doesn't model the fenv environment, so refuse. */

#pragma STDC FENV_ACCESS ON

int main(void) {
    return 0;
}
