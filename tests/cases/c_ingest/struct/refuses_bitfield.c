struct B { int flag : 1; int rest : 31; };
int read_flag(struct B b) { return b.flag; }
