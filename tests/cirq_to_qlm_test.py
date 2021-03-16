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
import cirq
from qat.interop.cirq.converters import qlm_to_cirq
from qat.lang.AQASM import *
from numpy import pi

import sympy

class TestQLM2GcirqConversion(unittest.TestCase):
    """ Tests the function converting qlm circuit to google cirq circuit
    """

    def test(self):
        nb_qubits = 5
        datapoints = [1, 0, 1, 0, 0]

        expected = cirq.Circuit()
        qreg = [cirq.LineQubit(i+1) for i in range(nb_qubits)]
        for qb, xi in zip(qreg, datapoints):
            if xi:
                expected.append(cirq.X(qb))
            else:
                expected.append(cirq.rx(pi/2)(qb))

        for qbit in qreg:
            expected.append(cirq.measure(qbit))

        prog = Program()
        qbits = prog.qalloc(nb_qubits)
        for qb, xi in zip(qbits, datapoints):
            if xi:
                prog.apply(X, qb)
            else:
                prog.apply(RX(pi/2), qb)

        circuit = prog.to_circ()
        result = qlm_to_cirq(circuit)

        self.assertEqual(len(result), len(expected))
        for op_e, op_r in zip(expected, result):
            self.assertEqual(op_e, op_r)

    def test_parametrized_circuit(self):
        nb_qubits = 3
        symb='a'

        expected = cirq.Circuit()
        qreg1 = [cirq.LineQubit(i+1) for i in range(nb_qubits)]
        var_cirq = sympy.symbols(symb)
        for qb in qreg1:
            expected.append(cirq.ry(var_cirq)(qb))
            expected.append(cirq.rz(4*var_cirq**2)(qb))

        for qbit in qreg1:
            expected.append(cirq.measure(qbit))

        prog = Program()
        qbits = prog.qalloc(nb_qubits)

        variable = prog.new_var(float, symb)
        for qbit in qbits:
            prog.apply(RY(variable),qbit)
            prog.apply(RZ(4*variable**2),qbit)

        circ = prog.to_circ()
        result = qlm_to_cirq(circ)

        self.assertEqual(len(result), len(expected))
        for op_e, op_r in zip(expected, result):
            self.assertEqual(op_e, op_r)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
