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


Description Test suite for google cirq circuit converter function

Overview
=========


"""

import unittest
import cirq.ops
from cirq import ControlledGate
import cirq.ops.common_gates as g_ops
from qat.interop.gcirq2acirc import to_qlm_circ
from qat.lang.AQASM import *
from qat.core.circ import extract_syntax
from qat.comm.datamodel.ttypes import OpType
from math import pi
gates_1qb = [g_ops.X, g_ops.Y, g_ops.Z, g_ops.S, g_ops.T, g_ops.H,
             g_ops.Rx(3.14), g_ops.Ry(3.14), g_ops.Rz(3.14)]
gates_2qb = [ControlledGate(g_ops.H), g_ops.CNOT, ControlledGate(g_ops.Rz(3.14)),
             g_ops.SWAP, g_ops.ISWAP]

pygates_1qb = [X, Y, Z, S, T, H, RX(3.14), RY(3.14), RZ(3.14)]
pygates_2qb = [H.ctrl(), CNOT, RZ(3.14).ctrl(), SWAP, ISWAP]
class TestGcirq2QLMConversion(unittest.TestCase):
    """ Tests the function converting google cirq circuit
        to qlm circuit"""

    def test_default_gates_and_qbit_reorder(self):
        gcirq = cirq.Circuit()
        qreg1 = [cirq.GridQubit(i, 0) for i in range(2)]
        qreg2 = [cirq.LineQubit(0)]
        qreg3 = [cirq.LineQubit(i) for i in range(1,3)]
        for op in gates_1qb:
            gcirq.append(op(qreg2[0])**-1.0)
        for op in gates_2qb:
            gcirq.append(op(qreg3[0], qreg1[1])**-1.0)
        
        gcirq.append(ControlledGate(ControlledGate(g_ops.X))(qreg1[0], qreg3[1], qreg2[0]))
        #ControlledGate(ops.Swap, n=1) | (qreg3[1], qreg1[0], qreg2[0])
        #Toffoli | (qreg3[1], qreg1[0], qreg2[0])
        for qbit in qreg1 + qreg2 + qreg3:
            gcirq.append(cirq.measure(qbit))
        # Generating qlm circuit
        result= to_qlm_circ(gcirq)

        # Generating equivalent qlm circuit
        prog = Program()
        qubits = prog.qalloc(5)
        cbits = prog.calloc(5)
        for op in pygates_1qb:
            prog.apply(op.dag(), qubits[2])
        for op in pygates_2qb:
            prog.apply(op.dag(), qubits[3], qubits[1])
        prog.apply(CCNOT, qubits[0], qubits[4], qubits[2])
        for i in range(5):
            prog.measure(qubits[i], cbits[i])
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
            #self.assertEqual(exp_op.qbits, res_op.qbits)
    def test_valid_powers(self):
        gcirq = cirq.Circuit()
        qreg = [cirq.LineQubit(i) for i in range(5)]

        gcirq.append(g_ops.X(qreg[0])**-pi)
        gcirq.append(g_ops.Y(qreg[0])**-pi)
        gcirq.append(g_ops.Z(qreg[0])**-pi)
        gcirq.append(g_ops.S(qreg[0])**-pi)
        gcirq.append(g_ops.T(qreg[0])**-pi)
        gcirq.append(g_ops.SWAP(qreg[0], qreg[1])**-0.5)
        gcirq.append(g_ops.ISWAP(qreg[0], qreg[1])**0.)

        result = to_qlm_circ(gcirq)
        for i, op in enumerate(result.ops):
            name, params = extract_syntax(result.gateDic[op.gate],
                                          result.gateDic)
            if i < 2:
                self.assertEqual(params[0], -pi*pi)
            elif i == 3:
                self.assertEqual(params[0], -pi*pi/2)
            elif i == 4:
                self.assertEqual(params[0], -pi*pi/4)
            else:
                continue
    def test_invalid_powers(self):
        gcirq = cirq.Circuit()
        qreg = [cirq.LineQubit(i) for i in range(5)]
        try:
            gcirq.append(g_ops.H(qreg[0])**pi)
        except ValueError:
            pass
        try:
            gcirq.append(g_ops.SWAP(qreg[0], qreg[1])**pi)
        except ValueError:
            pass
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)