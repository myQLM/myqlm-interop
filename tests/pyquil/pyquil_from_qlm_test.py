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

import sys
import pytest
import unittest
from qat.lang.AQASM.gates import *
from qat.lang.AQASM.program import Program
import numpy as np


if sys.version_info < (3, 7):
    pytest.skip("python version < 3.7: skipping pyquil_binder tests", allow_module_level=True)


from qat.interop.pyquil.converters import qlm_to_pyquil
from pyquil import Program as Prg
from pyquil import gates as pg


pygates_1qb = [X, Y, Z, I, S, T, H, RX(3.14), RY(3.14), RZ(3.14), PH(3.14)]
pygates_2qb = [SWAP, CNOT, H.ctrl(), RZ(3.14).ctrl(), RY(3.14).ctrl()]

quil_1qb = [pg.X, pg.Y, pg.Z, pg.I, pg.S, pg.T, pg.H]
quil_params = [pg.RX, pg.RY, pg.RZ, pg.PHASE]
quil_ctrl = [pg.H]
quil_ctrl_prm = [pg.RZ, pg.RY]


def extract_syntax(circuit):
    result = []
    supername = None
    for op in circuit.ops:
        gate = circuit.gateDic[op.gate]
        ctrl = gate.nbctrls
        dag = gate.is_dag
        qubits = op.qbits
        i = 0
        while gate.subgate:
            if gate.name[0] != "_" and supername is None:
                supername = gate.name
            gate = circuit.gateDic[gate.subgate]
            if gate.is_dag:
                i += 1

        if ctrl is None or supername is not None:
            ctrl = 0
        name = "C-" * ctrl

        if i % 2 == 0:
            if dag:
                name += "D-"
        else:
            if not dag:
                name += "D-"
        if supername is not None:
            name += supername
        else:
            name += gate.syntax.name
        params = [param.double_p for param in gate.syntax.parameters]
        result.append({"name": name, "params": params, "qubits": qubits})
        supername = None
    return result


def print_aq(circuit):
    data = extract_syntax(circuit)
    result = ""
    for entry in data:
        result += "Gate {} with params {} on qubits {}\n".format(
            entry["name"], entry["params"], entry["qubits"]
        )
        supername = None
    print(result)


class TestQLM2PyquilConversion(unittest.TestCase):
    """
    Tests the function converting qlm circuit to pyquil circuit
    We have to manually add measure in pyquil circuit because qlm does it
    automatically
    """

    def test_default_gates(self):
        # Create qlm program
        prog = Program()
        qreg = prog.qalloc(3)

        for op in pygates_1qb:
            prog.apply(op, qreg[0])

        for op in pygates_2qb:
            prog.apply(op, qreg[0], qreg[1])

        prog.apply(CCNOT, qreg[0], qreg[1], qreg[2])

        qlm_circuit = prog.to_circ()
        result = qlm_to_pyquil(qlm_circuit)

        # Create pyquil program
        expected = Prg()
        expected_creg = expected.declare("ro", "BIT", 3)
        for op in quil_1qb:
            expected += op(0)
        for op in quil_params:
            expected += op(3.14, 0)

        expected += pg.SWAP(0, 1)
        expected += pg.CNOT(0, 1)
        for op in quil_ctrl:
            expected += op(1).controlled(0)
        for op in quil_ctrl_prm:
            expected += op(3.14, 1).controlled(0)
        expected += pg.CCNOT(0, 1, 2)
        expected += pg.MEASURE(0, expected_creg[0])
        expected += pg.MEASURE(1, expected_creg[1])
        expected += pg.MEASURE(2, expected_creg[2])

        self.assertEqual(str(result), str(expected))

    def test_recursive_ctrl_and_dagger(self):
        # Create qlm program
        prog = Program()
        qreg = prog.qalloc(5)
        prog.apply(
            Y.ctrl().ctrl().ctrl().ctrl().dag().dag().dag(),
            *qreg
        )
        qlm_circuit = prog.to_circ()
        result = qlm_to_pyquil(qlm_circuit)

        # Create pyquil program
        expected = Prg()
        expected_creg = expected.declare("ro", "BIT", 5)
        expected += (
            pg.Y(4).controlled(0).controlled(1).controlled(2).controlled(3).dagger()
        )
        for qbit, cbit in enumerate(expected_creg):
            expected += pg.MEASURE(qbit, cbit)

        self.assertEqual(str(result), str(expected))

    def test_measures(self):
        # Create qlm program
        prog = Program()
        qreg = prog.qalloc(3)

        prog.apply(H, qreg[0])
        prog.apply(H, qreg[1])
        prog.apply(H, qreg[2])

        result = qlm_to_pyquil(prog.to_circ())

        # Create pyquil program
        expected = Prg()
        cbs = expected.declare("ro", "BIT", 3)
        expected += pg.H(0)
        expected += pg.H(1)
        expected += pg.H(2)
        expected += pg.MEASURE(0, cbs[0])
        expected += pg.MEASURE(1, cbs[1])
        expected += pg.MEASURE(2, cbs[2])

        self.assertEqual(str(result), str(expected))


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
