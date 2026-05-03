/* power: raise base to n-th power; n >= 0; recursive version */
int power(int base, int n)
{
    if (n == 0)
        return 1;
    else
        return base * power(base, n-1);
}
