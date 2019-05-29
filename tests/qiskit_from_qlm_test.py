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


Description This is a test suite for qlm to qiskit circuit converter

Overview
=========


"""
import unittest
from qat.lang.AQASM.gates import *
from qat.lang.AQASM import Program
from qat.interop.qiskit.converters import to_qiskit_circuit
from qiskit import QuantumCircuit, QuantumRegister, QuantumCircuit, ClassicalRegister
from qat.core.util import extract_syntax
import numpy as np


pygates_1qb = [X, Y, Z, I, S, T, S.dag(), T.dag(), H,
               RX(3.14), RY(3.14), RZ(3.14)]
pygates_2qb = [SWAP, CNOT, Y.ctrl(), Z.ctrl(), H.ctrl(), RZ(3.14).ctrl()]


gate_names = {'x': 'X', 'y': 'Y', 'z':'Z', 'h':'H', 'rx': 'RX', 'id': 'I',
              'ry':'RY', 'rz':'RZ', 's': 'S', 't': 'T', 'sdg': 'D-S',
              'tdg': 'D-T', 'swap': 'SWAP', 'cx': 'CNOT', 'ccx': 'CCNOT',
              'cy': 'C-Y', 'cz': 'CSIGN', 'ch': 'C-H', 'crz': 'C-RZ',
              'cswap': 'C-SWAP', 'measure': 'measure'}
def extract_qiskit(op):
    name = op[0].name
    params = op[0]._params
    qubits = [op[1][i][1] for i in range(len(op[1]))]
    cbits = [op[2][i][1] for i in range(len(op[2]))]
    return gate_names[name], params, qubits, cbits

def print_qiskit(circuit):
    result = ""
    for entry in circuit.data:
        name, params, qubits, cbits = extract_qiskit(entry)
        result += "Gate {} with params {} on qubits {} and cbits {}\n".format(
            name, params, qubits, cbits
        )
    print(result)
    return result


def qiskit_1qb(qc):
    return [qc.x, qc.y, qc.z, qc.iden, qc.s, qc.t, qc.sdg, qc.tdg, qc.h]

def qiskit_1prm(qc):
    return [qc.rx, qc.ry, qc.rz]

def qiskit_2qb(qc):
    return [qc.swap, qc.cx, qc.cy, qc.cz, qc.ch]

def qiskit_2qb_1prm(qc):
    return [qc.crz]

class TestQiskit2QLMConversion(unittest.TestCase):
    """ Tests the function converting qiskit circuit
        to qlm circuit"""

    def test_default_gates(self):
        prog = Program()
        qreg = prog.qalloc(3)

        for op in pygates_1qb:
            prog.apply(op, qreg[0])

        for op in pygates_2qb:
            prog.apply(op, qreg[0], qreg[1])

        prog.apply(CCNOT, qreg[0], qreg[1], qreg[2])
        prog.apply(SWAP.ctrl(), qreg[0], qreg[1], qreg[2])


        qlm_circuit = prog.to_circ()
        result = to_qiskit_circuit(qlm_circuit)


        qiskit_qreg = QuantumRegister(3)
        expected = QuantumCircuit(qiskit_qreg)
        for op in qiskit_1qb(expected):
            op(qiskit_qreg[0])
        for op in qiskit_1prm(expected):
            op(3.14, qiskit_qreg[0])
        for op in qiskit_2qb(expected):
            op(qiskit_qreg[0], qiskit_qreg[1])
        for op in qiskit_2qb_1prm(expected):
            op(3.14, qiskit_qreg[0], qiskit_qreg[1])

        expected.ccx(*qiskit_qreg)
        expected.cswap(*qiskit_qreg)

        expected_str = print_qiskit(expected)
        result_str = print_qiskit(result)
        self.assertEqual(result_str, expected_str)

    def test_measures(self):
        prog = Program()
        qreg = prog.qalloc(3)
        creg = prog.calloc(3)

        prog.apply(H, qreg[0])
        prog.apply(H, qreg[1])
        prog.apply(H, qreg[2])

        prog.measure(qreg, creg)

        result = to_qiskit_circuit(prog.to_circ())

        qiskit_qreg = QuantumRegister(3)
        qiskit_creg = ClassicalRegister(3)
        expected = QuantumCircuit(qiskit_qreg, qiskit_creg)
        expected.h(qiskit_qreg[0])
        expected.h(qiskit_qreg[1])
        expected.h(qiskit_qreg[2])
        expected.measure(qiskit_qreg, qiskit_creg)


        result_str=print_qiskit(result)
        expected_str=print_qiskit(expected)
        self.assertEqual(result_str, expected_str)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
