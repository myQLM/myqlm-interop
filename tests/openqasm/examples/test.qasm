OPENQASM 2.0;

qreg q[4];

gate vc a,b,c {
    x a;
    cx b,c;
    ccx a, b, c;
    h a;
    y a;
    z a;
}
gate rp(p) a1, a2, a3 {
pH(-p) a1;
rx(-p) a1;
ry(-p) a1;
rz(-p) a1;
}
rp(-pi/2) q[0], q[1], q[2];
vc q[0], q[1], q[2];
cx q[0], q[1];
