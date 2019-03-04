#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
@brief 

@namespace ...
@authors Reda Drissi <mohamed-reda.drissi@atos.net>
@copyright 2019  Bull S.A.S.  -  All rights reserved.
           This is not Free or Open Source software.
           Please contact Bull SAS for details about its license.
           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois


Description ...

Overview
=========


"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qat.lang.AQASM import *
from math import pi
from pprint import pprint
import numpy as np
def get_qindex(circ, name, index):
    ret = 0
    for reg in circ.qregs:
        if name != reg.name:
            ret += reg.size
        else:
            return ret + index
def get_cindex(circ, name, index):
    ret = 0
    for reg in circ.cregs:
        if name != reg.name:
            ret += reg.size
        else:
            return ret + index
# Example oqasm circuit #

qreg1 = QuantumRegister(2)
qreg2 = QuantumRegister(3)
creg1 = ClassicalRegister(2)

ocirc = QuantumCircuit(qreg1, qreg2, creg1)

ocirc.sdg(qreg1[0])
ocirc.ch(qreg1[0], qreg1[1])
ocirc.u_base(pi, pi, pi, qreg1[0])
ocirc.iden(qreg1[0])
ocirc.u0(pi, qreg1[0])
ocirc.ccx(qreg1[0], qreg1[1], qreg2[0])
ocirc.rzz(pi, qreg1[0], qreg1[1])
ocirc.measure(qreg1, creg1)
ops = ocirc.data

# Let's add the U gate, u1/2/3 would be dealt with through setting
# the appropriate params to 0
def gen_U(theta, phi, lamda):
    m11 = (np.e ** (1j*(phi + lamda)/2)) * np.cos(theta/2)
    m12 = (-1) * (np.e ** (1j*(phi - lamda)/2)) * np.sin(theta/2)
    m21 = (np.e ** (1j*(phi - lamda)/2)) * np.sin(theta/2)
    m22 = (np.e ** (1j*(phi + lamda)/2)) * np.cos(theta/2)
    return np.array([[m11, m12],[m21, m22]], dtype = np.complex128)

def gen_RZZ(theta):
    return np.diag([1, np.exp(1j*theta), np.exp(1j*theta), 1])

U = AbstractGate("U", [float]*3, arity=1,
                 matrix_generator=gen_U)
RZZ = AbstractGate("RZZ", [float], arity=2,
                   matrix_generator=gen_RZZ)
# get qbits

def process_U(params):
    return U(params[0], params[1], params[2])

def process_U2(params):
    return U(0, params[0], params[1])

gate_dic={'h': H, 'x': X, 'y': Y, 'z': Z, 'xbase': X, 'swap': SWAP,
          'id': I, 's': S, 'sdg': S.dag(), 't': T, 'tdg': T.dag(),
          'rx': RX, 'ry': RY, 'rz': RZ, 'rzz': RZZ, 'u0': I, 'u1': RY,
          'u2': process_U2, 'u3': process_U, 'U': process_U}

def get_gate(gate, params):
    """ generates pyAQASM corresponding object """
    if gate =="u0":
        return I
    elif gate[0] == 'c':
        return get_gate(gate[1:], params).ctrl()
    elif len(params)==0:
        return gate_dic[gate]
    elif len(params)==1:
        return gate_dic[gate](params[0])
    else:
        return gate_dic[gate](params)

def to_qlm_circ(qiskit_circuit):
    """ translates a qiskit circuit into a qlm circuit"""
    prog = Program()
    qbits_num = 0

    for reg in ocirc.qregs:
        qbits_num += qbits_num + reg.size
    qbits = prog.qalloc(qbits_num)

    cbits_num = 0
    for reg in ocirc.cregs:
        cbits_num += cbits_num + reg.size
    cbits = prog.calloc(cbits_num)
    for op in ops:
        qb = [] # qbits arguments
        cb = [] # cbits arguments
        prms = [] # gate parameters
        # Get qbit arguments
        for reg in op.qargs:
            qb.append(qbits[get_qindex(ocirc, reg[0].name, reg[1])])

        # Get cbit arguments
        for reg in op.cargs:
            cb.append(qbits[get_cindex(ocirc, reg[0].name, reg[1])])

        # Get parameters
        for p in op.param:
            prms.append(float(p))
        # Apply measure #
        if op.name == 'measure':
            prog.measure(qb, cb)
        else:
            # Apply gates #
            prog.apply(get_gate(op.name, prms), qb)

    return prog.to_circ()
#print(ocirc)
acirc = to_qlm_circ(ocirc)
# Simulation #
#from qat.mps import get_qpu_server
#from qat.core.task import Task

#task = Task(acirc, get_qpu_server())
#outp = task.execute()
#print(outp.probability, outp.state)
#for res in task.states():
#    print(res.state, res.amplitude)

