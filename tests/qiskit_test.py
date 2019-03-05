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


Description This is a test suite for qiskit circuit converter

Overview
=========


"""
import unittest
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qat.interop.ocirc2acirc import to_qlm_circ
from qat.lang.AQASM import *
from qat.core.circ import extract_syntax
from qat.comm.datamodel.ttypes import OpType
import numpy as np

def gen_gates(ocirc):
    gates_1qb_0prm = [ocirc.x, ocirc.y, ocirc.z, ocirc.iden,
                      ocirc.s, ocirc.t, ocirc.h, ocirc.sdg, ocirc.tdg]
    gates_2qb_0prm = [ocirc.ch, ocirc.cx, ocirc.swap]
    gates_1qb_1prm = [ocirc.rx, ocirc.ry, ocirc.rz, ocirc.u0, ocirc.u1]
    gates_2qb_1prm = [ocirc.crz, ocirc.cu1, ocirc.rzz]
    gates_3qb_0prm = [ocirc.cswap, ocirc.ccx]
    return gates_1qb_0prm, gates_1qb_1prm, gates_2qb_0prm,\
            gates_2qb_1prm, gates_3qb_0prm

#-- Generating qlm circuit to compare --#
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

pygates_1qb = [X, Y, Z, I, S, T, H, S.dag(), T.dag(),
               RX(3.14), RY(3.14), RZ(3.14), I, RY(3.14)]
pygates_2qb = [H.ctrl(), CNOT, SWAP, RZ(3.14).ctrl(),
               RY(3.14).ctrl(), RZZ(3.14)]
class TestQiskit2QLMConversion(unittest.TestCase):
    """ Tests the function converting qiskit circuit
        to qlm circuit"""

    def test_default_gates_and_qbit_reorder(self):
        qreg1 = QuantumRegister(2)
        qreg2 = QuantumRegister(1)
        qreg3 = QuantumRegister(2)
        creg = ClassicalRegister(5)
        ocirc = QuantumCircuit(qreg1, qreg2, qreg3, creg)
        gates_1qb_0prm, gates_1qb_1prm, gates_2qb_0prm,\
                gates_2qb_1prm, gates_3qb_0prm = gen_gates(ocirc)
        for op in gates_1qb_0prm:
            op(qreg2[0])
        for op in gates_1qb_1prm:
            op(3.14, qreg2[0])
        for op in gates_2qb_0prm:
            op(qreg3[0], qreg1[1])
        for op in gates_2qb_1prm:
            op(3.14, qreg3[0], qreg1[1])
        for op in gates_3qb_0prm:
            op(qreg2[0], qreg3[1], qreg1[1])
        ocirc.u2(3.14, 3.14, qreg3[0])
        ocirc.u_base(3.14, 3.14, 3.14, qreg3[0])
        ocirc.measure(qreg1[0], creg[4])
        ocirc.measure(qreg1[1], creg[3])
        ocirc.measure(qreg2[0], creg[2])
        ocirc.measure(qreg3[0], creg[1])
        ocirc.measure(qreg3[1], creg[0])
        result= to_qlm_circ(ocirc)
        prog = Program()
        qubits = prog.qalloc(5)
        cbits = prog.calloc(5)
        for op in pygates_1qb:
            prog.apply(op, qubits[2])
        for op in pygates_2qb:
            prog.apply(op, qubits[3], qubits[1])

        prog.apply(SWAP.ctrl(), qubits[2], qubits[4], qubits[1])
        prog.apply(CCNOT, qubits[2], qubits[4], qubits[1])
        prog.apply(U(0, 3.14, 3.14), qubits[3])
        prog.apply(U(3.14, 3.14, 3.14), qubits[3])
        for i in range(5):
            prog.measure(qubits[i], cbits[4-i])
        expected = prog.to_circ()
        self.assertEqual(len(result.ops), len(expected.ops))
        for i in range(len(result.ops)):
            res_op = result.ops[i]
            exp_op = expected.ops[i]
            if res_op.type == OpType.MEASURE:
                self.assertEqual(res_op, exp_op)
                continue
            result_gate_name, result_gate_params = \
                    extract_syntax(result.gateDic[res_op.gate],
                                   result.gateDic)
            #print("got gate {} with params {} on qbits {}"
            #      .format(result_gate_name, result_gate_params,
            #              res_op.qbits))
            expected_gate_name, expected_gate_params = \
                    extract_syntax(expected.gateDic[exp_op.gate],
                                   expected.gateDic)
            #print("expected gate {} with params {} on qbits {}"
            #      .format(expected_gate_name, expected_gate_params,
            #              exp_op.qbits))
            self.assertEqual(expected_gate_name, result_gate_name)
            self.assertEqual(expected_gate_params, result_gate_params)
            self.assertEqual(exp_op.qbits, res_op.qbits)
    def test_invalid_gate(self):
        qreg = QuantumRegister(5)
        ocirc = QuantumCircuit(qreg)
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
