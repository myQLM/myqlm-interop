#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

"""
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
"""

import unittest
import importlib.abc
import cirq.ops
import cirq
from cirq import ControlledGate
import cirq.ops.common_gates as g_ops
from qat.interop.cirq.converters import cirq_to_qlm
from qat.lang.AQASM import *
try:
    from qat.core.util import extract_syntax
except ImportError:
    from qat.core.circ import extract_syntax
from numpy import array, cos, sin, complex128, pi, sqrt
from qat.comm.datamodel.ttypes import OpType

# Adding parity gates and their rotations:
def gen_XX():
    return array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=complex128,
    )


def gen_YY():
    return array(
        [
            [0.0 + 0.0j, 0.0 - 0.0j, 0.0 - 0.0j, -1.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 - 0.0j, 0.0 - 0.0j],
            [0.0 + 0.0j, 1.0 - 0.0j, 0.0 + 0.0j, 0.0 - 0.0j],
            [-1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=complex128,
    )


def gen_ZZ():
    return array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.0j, -0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j, -0.0 + 0.0j],
            [0.0 + 0.0j, -0.0 + 0.0j, -0.0 + 0.0j, 1.0 - 0.0j],
        ],
        dtype=complex128,
    )


def gen_RXX(phi):
    return array(
        [
            [cos(phi / 2), 0, 0, -sin(phi / 2) * 1.0j],
            [0, cos(phi / 2), -sin(phi / 2) * 1.0j, 0],
            [0, -sin(phi / 2) * 1.0j, cos(phi / 2), 0],
            [-sin(phi / 2) * 1.0j, 0, 0, cos(phi / 2)],
        ],
        dtype=complex128,
    )


def gen_RYY(phi):
    return np.array(
        [
            [cos(phi / 2), 0, 0, sin(phi / 2) * 1.0j],
            [0, cos(phi / 2), -sin(phi / 2) * 1.0j, 0],
            [0, -sin(phi / 2) * 1.0j, cos(phi / 2), 0],
            [sin(phi / 2) * 1.0j, 0, 0, cos(phi / 2)],
        ],
        dtype=complex128,
    )


def gen_RZZ(phi):
    return np.array(
        [
            [cos(phi / 2) - sin(phi / 2) * 1.0j, 0, 0, 0],
            [0, cos(phi / 2) + sin(phi / 2) * 1.0j, 0, 0],
            [0, 0, cos(phi / 2) + sin(phi / 2) * 1.0j, 0],
            [0, 0, 0, cos(phi / 2) - sin(phi / 2) * 1.0j],
        ],
        dtype=complex128,
    )


XX = AbstractGate("XX", [], arity=2, matrix_generator=gen_XX)
YY = AbstractGate("YY", [], arity=2, matrix_generator=gen_YY)
ZZ = AbstractGate("ZZ", [], arity=2, matrix_generator=gen_ZZ)

RXX = AbstractGate("RXX", [float], arity=2, matrix_generator=gen_RXX)
RYY = AbstractGate("RYY", [float], arity=2, matrix_generator=gen_RYY)
RZZ = AbstractGate("RZZ", [float], arity=2, matrix_generator=gen_RZZ)


gates_1qb = [
    cirq.X,
    cirq.Y,
    cirq.Z,
    cirq.S,
    cirq.T,
    cirq.H,
    cirq.rx(3.14),
    cirq.ry(3.14),
    cirq.rz(3.14),
]
gates_2qb = [
    ControlledGate(cirq.H),
    cirq.CNOT,
    ControlledGate(cirq.rz(3.14)),
    cirq.SWAP,
    cirq.ISWAP,
    cirq.XX,
    cirq.YY,
    cirq.ZZ,
    cirq.CZ,
]

pygates_1qb = [X, Y, Z, S, T, H, RX(3.14), RY(3.14), RZ(3.14)]
pygates_2qb = [H.ctrl(), CNOT, RZ(3.14).ctrl(), SWAP, ISWAP, XX(), YY(), ZZ(), Z.ctrl()]


class TestGcirq2QLMConversion(unittest.TestCase):
    """ Tests the function converting google cirq circuit
        to qlm circuit"""

    def test_default_gates_and_qbit_reorder(self):
        gcirq = cirq.Circuit()
        qreg1 = [cirq.LineQubit(i) for i in range(3)]
        for op in gates_1qb:
            gcirq.append(op(qreg1[0]) ** -1.0)
        for op in gates_2qb:
            gcirq.append(op(qreg1[0], qreg1[1]) ** -1.0)

        gcirq.append(cirq.CCX(qreg1[0], qreg1[1], qreg1[2]))
        gcirq.append(cirq.CSWAP(qreg1[0], qreg1[1], qreg1[2]))
        gcirq.append(cirq.CCZ(qreg1[0], qreg1[1], qreg1[2]))
        # Toffoli | (qreg3[1], qreg1[0], qreg2[0])
        for qbit in qreg1:
            gcirq.append(cirq.measure(qbit))
        # Generating qlm circuit
        result = cirq_to_qlm(gcirq)

        # Generating equivalent qlm circuit
        prog = Program()
        qubits = prog.qalloc(3)
        cbits = prog.calloc(3)

        for op in pygates_1qb:
            prog.apply(op.dag(), qubits[0])

        for op in pygates_2qb:
            prog.apply(op.dag(), qubits[0], qubits[1])

        prog.apply(X.ctrl().ctrl(), qubits[0], qubits[1], qubits[2])
        prog.apply(SWAP.ctrl(), qubits[0], qubits[1], qubits[2])
        prog.apply(Z.ctrl().ctrl(), qubits[0], qubits[1], qubits[2])

        for i in range(3):
            prog.measure(qubits[i], cbits[i])
        expected = prog.to_circ()
        self.assertEqual(len(result.ops), len(expected.ops))
        for i in range(len(result.ops)):
            res_op = result.ops[i]
            exp_op = expected.ops[i]
            if res_op.type == OpType.MEASURE:
                self.assertEqual(res_op, exp_op)
                continue
            result_gate_name, result_gate_params = extract_syntax(
                result.gateDic[res_op.gate], result.gateDic
            )
            expected_gate_name, expected_gate_params = extract_syntax(
                expected.gateDic[exp_op.gate], expected.gateDic
            )
            self.assertEqual(expected_gate_name, result_gate_name)
            self.assertEqual(expected_gate_params, result_gate_params)

    def test_valid_powers(self):
        gcirq = cirq.Circuit()
        qreg = [cirq.LineQubit(i) for i in range(5)]

        gcirq.append(cirq.X(qreg[0]) ** -3.67)
        gcirq.append(cirq.Y(qreg[0]) ** 7.9)
        gcirq.append(cirq.Z(qreg[0]) ** sqrt(5))
        gcirq.append(cirq.S(qreg[0]) ** -pi)
        gcirq.append(cirq.T(qreg[0]) ** (sqrt(7)-pi))
        gcirq.append(cirq.SWAP(qreg[0], qreg[1]) ** -0.5)
        gcirq.append(cirq.ISWAP(qreg[0], qreg[1]) ** 16.0)

        result = cirq_to_qlm(gcirq)
        for i, op in enumerate(result.ops):
            name, params = extract_syntax(result.gateDic[op.gate], result.gateDic)
            if i == 0:
                self.assertEqual(params[0], -3.67 * pi)
            elif i == 1:
                self.assertEqual(params[0], 7.9 * pi)
            elif i == 2:
                self.assertEqual(params[0], sqrt(5) * pi)
            elif i == 3:
                self.assertEqual(params[0], -pi * pi / 2)
            elif i == 4:
                self.assertEqual(params[0], (sqrt(7) - pi) * pi / 4)
            else:
                continue

    def test_invalid_powers(self):
        gcirq = cirq.Circuit()
        qreg = [cirq.LineQubit(i) for i in range(5)]
        cirq_to_qlm(gcirq)
        try:
            gcirq.append(cirq.H(qreg[0]) ** pi)
        except ValueError:
            pass
        try:
            gcirq.append(cirq.SWAP(qreg[0], qreg[1]) ** pi)
        except ValueError:
            pass


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
