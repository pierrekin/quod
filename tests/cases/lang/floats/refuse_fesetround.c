/* Belt-and-braces: a user could declare a fenv API directly without
   including <fenv.h>. The CALL_EXPR-time blocklist refuses the call
   even though the include refusal didn't fire. */

extern int fesetround(int);

int main(void) {
    fesetround(0);
    return 0;
}
