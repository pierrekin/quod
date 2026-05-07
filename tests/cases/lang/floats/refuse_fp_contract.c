/* `#pragma STDC FP_CONTRACT ON` requests FMA contraction; quod
   refuses because its codegen doesn't insert FMA contraction (strict
   IEEE 754) and silently allowing the pragma would mislead the
   programmer about which bits the compiler produces. */

#pragma STDC FP_CONTRACT ON

int main(void) {
    return 0;
}
