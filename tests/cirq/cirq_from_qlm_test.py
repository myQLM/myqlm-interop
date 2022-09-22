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

from random import uniform
from math import pi

import unittest
import importlib.abc
import cirq
from qat.interop.cirq.converters import qlm_to_cirq, cirq_to_qlm
from qat.lang.AQASM import *
from numpy import pi

from qat.core.variables import cos

import sympy

class TestQLM2GcirqConversion(unittest.TestCase):
    """ Tests the function converting qlm circuit to google cirq circuit
    """

    def test_simple_circuit_conversion(self):
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

    def test_parametrized_circuit_with_symbolic_variables(self):
        nb_qubits = 3
        symb='a'

        expected = cirq.Circuit()
        qreg1 = [cirq.LineQubit(i+1) for i in range(nb_qubits)]
        var_cirq = sympy.symbols(symb)
        for qb in qreg1:
            expected.append(cirq.ry(var_cirq)(qb))
            expected.append(cirq.rz(4*sympy.cos(var_cirq)**2)(qb))

        for qbit in qreg1:
            expected.append(cirq.measure(qbit))

        prog = Program()
        qbits = prog.qalloc(nb_qubits)

        variable = prog.new_var(float, symb)
        for qbit in qbits:
            prog.apply(RY(variable),qbit)
            prog.apply(RZ(4*cos(variable)**2),qbit)

        circ = prog.to_circ()
        result = qlm_to_cirq(circ)

        self.assertEqual(len(result), len(expected))
        for op_e, op_r in zip(expected, result):
            self.assertEqual(op_e, op_r)


class TestGateSet():
    """
    Ensure that all the main gates could be translated
    from myQLM to cirq
    """
    def test_no_param(self):
        """
        Ensure that not parametrized gate can be translated
        from myQLM to cirq
        """
        # Init program
        prog = Program()
        qbits = prog.qalloc(3)

        # Add non parametrized gate
        for gate in [H, X, Y, Z, S, T, SWAP, ISWAP, CNOT, CSIGN, CCNOT]:
            prog.apply(gate, qbits[0:gate.arity])

        qlm_circ = prog.to_circ()

        # Translate the circuit twice
        cirq_circ = qlm_to_cirq(qlm_circ)
        qlm_circ_cp, _ = cirq_to_qlm(cirq_circ, sep_measures=True)

        # Check output
        assert len(qlm_circ_cp) == len(qlm_circ)

        for result, expected in zip(qlm_circ_cp.iterate_simple(), qlm_circ.iterate_simple()):
            # Fix result if the name is invalid
            if result[0] == "C-C-X":
                result = ("CCNOT", result[1], result[2])

            assert result == expected

    def test_param(self):
        """
        Ensure that parametrized gate can be translated
        from myQLM to cirq
        """
        # Init program
        prog = Program()
        qbits = prog.qalloc(1)

        # Apply gates
        for gate in [RX, RY, RZ, PH]:
            prog.apply(gate(uniform(0, 2 * pi)), qbits)

        # Generate the circuit and translate it twice
        qlm_circ = prog.to_circ()
        cirq_circ = qlm_to_cirq(qlm_circ)
        qlm_circ_cp, _ = cirq_to_qlm(cirq_circ, sep_measures=True)

        # Check output
        assert len(qlm_circ_cp) == len(qlm_circ)

        for result, expected in zip(qlm_circ_cp.iterate_simple(), qlm_circ.iterate_simple()):
            # Check names
            assert (result[0] == expected[0]) or sorted([result[0], expected[0]]) == ["PH", "RZ"]

            # Check values
            assert result[1:] == expected[1:]


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
