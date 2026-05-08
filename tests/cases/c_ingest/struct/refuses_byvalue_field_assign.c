struct Point { int x; int y; };
int set_local(int v) {
    struct Point p = {0, 0};
    p.x = v;
    return p.x;
}
