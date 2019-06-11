// QFT and measure, version 2
OPENQASM 2.0;
qreg q[4];
creg c0[1];
creg c1[1];
creg c2[1];
creg c3[1];
// -- Included qelib, because parser doesn't deal with inclusion -- //
gate ch a,b {
h b; sdg b;
cx a,b;
h b; t b;
cx a,b;
t b; h b; s b; x b; s a;
}
gate crz(lambda) a,b
{
  u1(lambda/2) b;
  cx a,b;
  u1(-lambda/2) b;
  cx a,b;
}
gate cu1(lambda) a,b
{
  u1(lambda/2) a;
  cx a,b;
  u1(-lambda/2) b;
  cx a,b;
  u1(lambda/2) b;
}
gate cu3(theta,phi,lambda) c, t
{
  u1((lambda-phi)/2) t;
  cx c,t;
  u3(-theta/2,0,-(phi+lambda)/2) t;
  cx c,t;
  u3(theta/2,phi,0) t;
}
// QFT version 3
h q;
barrier q;
h q[0];
// measure q[0] -> c0[0];
if(c0==1) u1(pi/2) q[1];
h q[1];
// measure q[1] -> c1[0];
if(c0==1) u1(pi/4) q[2];
if(c1==1) u1(pi/2) q[2];
h q[2];
// measure q[2] -> c2[0];
if(c0==1) u1(pi/8) q[3];
if(c1==1) u1(pi/4) q[3];
if(c2==1) u1(pi/2) q[3];
h q[3];
// measure q[3] -> c3[0];
