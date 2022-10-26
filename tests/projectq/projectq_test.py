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
from projectq.cengines import MainEngine
from projectq.ops import (
    SGate,
    XGate,
    YGate,
    ZGate,
    TGate,
    HGate,
    SwapGate,
    R,
    Rz,
    Rx,
    Ry,
    Toffoli,
    Sdag,
    Tdag,
    All,
    Measure,
    ControlledGate,
    CX,
)
from projectq import ops
from qat.interop.projectq.converters import AqasmPrinter, AqasmEngine
from qat.lang.AQASM import Program
from qat.lang.AQASM.gates import *
try:
    from qat.core.util import extract_syntax
except ImportError:
    from qat.core.circ import extract_syntax
from qat.comm.datamodel.ttypes import OpType
from qat.interop.openqasm.qasm_parser import ImplementationError


gates_1qb = [
    ops.X,
    ops.Y,
    ops.Z,
    ops.S,
    ops.T,
    ops.H,
    Sdag,
    Tdag,
    Rx(3.14),
    Ry(3.14),
    Rz(3.14),
    R(3.14),
]
gates_2qb = [ControlledGate(ops.H), CX, ControlledGate(Rz(3.14)), ops.Swap]

pygates_1qb = [
    X,
    Y,
    Z,
    S,
    T,
    H,
    S.dag(),
    T.dag(),
    RX(3.14),
    RY(3.14),
    RZ(3.14),
    PH(3.14),
]
pygates_2qb = [H.ctrl(), CNOT, RZ(3.14).ctrl()]


class TestProjectq2QLMConversion(unittest.TestCase):
    """ Tests the function converting projectq circuit
        to qlm circuit"""

    def test_default_gates_and_qbit_reorder(self):
        aq = AqasmPrinter(MainEngine)
        eng = AqasmEngine(aq, engine_list=[aq])
        qreg1 = eng.allocate_qureg(2)
        qreg2 = eng.allocate_qureg(1)
        qreg3 = eng.allocate_qureg(2)
        for op in gates_1qb:
            op | qreg2[0]
        for op in gates_2qb:
            op | (qreg3[0], qreg1[1])

        ControlledGate(ops.Swap, n=1) | (qreg3[1], qreg1[0], qreg2[0])
        Toffoli | (qreg3[1], qreg1[0], qreg2[0])
        All(Measure) | qreg1
        All(Measure) | qreg2
        All(Measure) | qreg3
        # Generating qlm circuit
        result = eng.projectq_to_qlm()

        # Generating equivalent qlm circuit
        prog = Program()
        qubits = prog.qalloc(5)
        cbits = prog.calloc(5)
        for op in pygates_1qb:
            prog.apply(op, qubits[2])
        for op in pygates_2qb:
            prog.apply(op, qubits[3], qubits[1])
        prog.apply(SWAP, qubits[1], qubits[3])
        prog.apply(SWAP.ctrl(), qubits[4], qubits[0], qubits[2])
        prog.apply(X.ctrl().ctrl(), qubits[0], qubits[4], qubits[2])
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
            result_gate_name, result_gate_params = extract_syntax(
                result.gateDic[res_op.gate], result.gateDic
            )
            # print("got gate {} with params {} on qbits {}"
            #      .format(result_gate_name, result_gate_params,
            #              res_op.qbits))
            expected_gate_name, expected_gate_params = extract_syntax(
                expected.gateDic[exp_op.gate], expected.gateDic
            )
            # print("expected gate {} with params {} on qbits {}"
            #      .format(expected_gate_name, expected_gate_params,
            #              exp_op.qbits))
            self.assertEqual(expected_gate_name, result_gate_name)
            self.assertEqual(expected_gate_params, result_gate_params)
            self.assertEqual(exp_op.qbits, res_op.qbits)

    def test_dynamic_measures(self):
        aq = AqasmPrinter(MainEngine)
        eng = AqasmEngine(aq, engine_list=[aq])
        qreg = eng.allocate_qureg(2)
        ops.Swap | (qreg[0], qreg[1])
        All(Measure) | qreg
        try:
            int(qreg[0])
        except ImplementationError:
            pass
        circ=eng.projectq_to_qlm()
        print(circ.ops)

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
