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
import logging
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter
from qat.interop.qiskit import qiskit_to_qlm, RXX, RZZ, MS, \
        U2, U3, R, BackendToQPU
from qat.lang.AQASM import Program
from qat.lang.AQASM.gates import H, X, Y, Z, SWAP, I, S, \
        T, RX, RY, RZ, CNOT

from qat.core.util import extract_syntax
from qat.comm.datamodel.ttypes import OpType

LOGGER = logging.getLogger()
# Set level to logging.DEBUG in order to see more information
LOGGER.setLevel(logging.WARNING)

# redirects log writing to terminal
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.DEBUG)
LOGGER.addHandler(STREAM_HANDLER)


def gen_gates(ocirc):
    """
    Generates a tuple object of five lists each containing gate methods
    applied to the qiskit circuit passed as a parameter.

    Args:
        ocirc: QuantumCircuit object
    """
    gates_1qb_0prm = [
        ocirc.x,
        ocirc.y,
        ocirc.z,
        ocirc.id,
        ocirc.s,
        ocirc.t,
        ocirc.h,
        ocirc.sdg,
        ocirc.tdg,
    ]
    gates_2qb_0prm = [ocirc.ch, ocirc.cx, ocirc.swap]
    gates_1qb_1prm = [ocirc.rx, ocirc.ry, ocirc.rz, ocirc.p]
    gates_2qb_1prm = [ocirc.crz, ocirc.rxx, ocirc.rzz]
    gates_3qb_0prm = [ocirc.cswap, ocirc.ccx]
    return (
        gates_1qb_0prm,
        gates_1qb_1prm,
        gates_2qb_0prm,
        gates_2qb_1prm,
        gates_3qb_0prm,
    )


PYGATES_1QB = [
    X,
    Y,
    Z,
    I,
    S,
    T,
    H,
    S.dag(),
    T.dag(),
    RX(3.14),
    RY(3.14),
    RZ(3.14),
    RZ(3.14)
]

PYGATES_2QB = [
    H.ctrl(),
    CNOT,
    SWAP,
    RZ(3.14).ctrl(),
    RXX(3.14),
    RZZ(3.14)
]


class TestQiskit2QLMConversion(unittest.TestCase):
    """
    Tests the function converting a qiskit circuit
    to a qlm circuit.
    """

    def test0_default_gates_and_qbit_reorder(self):
        """
        Tries out every default gate and check that they match
        once the Qiskit circuit is translated into a QLM circuit.
        """
        qreg1 = QuantumRegister(2)
        qreg2 = QuantumRegister(1)
        qreg3 = QuantumRegister(2)
        creg = ClassicalRegister(5)
        ocirc = QuantumCircuit(qreg1, qreg2, qreg3, creg)

        gates_1qb_0prm, gates_1qb_1prm, gates_2qb_0prm, \
            gates_2qb_1prm, gates_3qb_0prm = gen_gates(ocirc)

        for gate_op in gates_1qb_0prm:
            gate_op(qreg2[0])
        for gate_op in gates_1qb_1prm:
            gate_op(3.14, qreg2[0])
        for gate_op in gates_2qb_0prm:
            gate_op(qreg3[0], qreg1[1])
        for gate_op in gates_2qb_1prm:
            gate_op(3.14, qreg3[0], qreg1[1])
        for gate_op in gates_3qb_0prm:
            gate_op(qreg2[0], qreg3[1], qreg1[1])

        ocirc.u(3.14, 3.14, 3.14, qreg3[0])
        ocirc.r(3.14, 3.14, qreg3[0])
        ocirc.ms(3.14, [qreg1[1], qreg2[0], qreg3[0]])

        ocirc.measure(qreg1[0], creg[4])
        ocirc.measure(qreg1[1], creg[3])
        ocirc.measure(qreg2[0], creg[2])
        ocirc.measure(qreg3[0], creg[1])
        ocirc.measure(qreg3[1], creg[0])

        result = qiskit_to_qlm(ocirc)

        prog = Program()
        qubits = prog.qalloc(5)
        cbits = prog.calloc(5)
        for gate_op in PYGATES_1QB:
            prog.apply(gate_op, qubits[2])
        for gate_op in PYGATES_2QB:
            prog.apply(gate_op, qubits[3], qubits[1])

        prog.apply(SWAP.ctrl(), qubits[2], qubits[4], qubits[1])
        prog.apply(X.ctrl().ctrl(), qubits[2], qubits[4], qubits[1])
        prog.apply(U3(3.14, 3.14, 3.14), qubits[3])
        prog.apply(R(3.14, 3.14), qubits[3])
        prog.apply(MS(3.14, 3), qubits[1], qubits[2], qubits[3])

        for i in range(5):
            prog.measure(qubits[i], cbits[4 - i])

        expected = prog.to_circ()

        self.assertEqual(len(result.ops), len(expected.ops))
        for res_op, exp_op in zip(result.ops, expected.ops):
            if res_op.type == OpType.MEASURE:
                self.assertEqual(res_op, exp_op)
                continue
            result_gate_name, result_gate_params = extract_syntax(
                result.gateDic[res_op.gate], result.gateDic
            )
            LOGGER.debug("got gate {} with params {} on qbits {}".format(
                result_gate_name, result_gate_params, res_op.qbits))

            expected_gate_name, expected_gate_params = extract_syntax(
                expected.gateDic[exp_op.gate], expected.gateDic
            )
            LOGGER.debug("expected gate {} with params {} on qbits {}"
                         .format(expected_gate_name, expected_gate_params,
                                 exp_op.qbits))

            self.assertEqual(expected_gate_name, result_gate_name)
            self.assertEqual(expected_gate_params, result_gate_params)
            self.assertEqual(exp_op.qbits, res_op.qbits)

        LOGGER.debug("\nResults obtained:")
        qpu = BackendToQPU()
        result_job = result.to_job(nbshots=1024)
        qiskit_result = qpu.submit(result_job)
        for entry in qiskit_result.raw_data:
            LOGGER.debug("State: {}\t probability: {}".format(
                entry.state, entry.probability))

        LOGGER.debug("\nResults expected:")
        expected_job = expected.to_job(nbshots=1024)
        qlm_result = qpu.submit(expected_job)
        for entry in qlm_result.raw_data:
            LOGGER.debug("State: {}\t probability: {}".format(
                entry.state, entry.probability))

        self.assertEqual(len(qiskit_result.raw_data), len(qlm_result.raw_data))
        states_expected = [str(entry.state) for entry in qlm_result.raw_data]
        for entry in qiskit_result.raw_data:
            self.assertTrue(str(entry.state) in states_expected)

    def test1_abstract_variables(self):
        """
        Tests the conversion of Parameter objects from Qiskit into
        abstract Variable objects in QLM via qiskit_to_qlm.
        """
        qreg = QuantumRegister(1)
        circ = QuantumCircuit(qreg)
        param0 = Parameter("param0")
        param1 = Parameter("param1")
        param2 = Parameter("param2")
        param3 = Parameter("param3")
        param0.expr = 1
        param1.expr = 3.14
        param4 = param0 + param1 + param2 - param3
        param5 = param0 * param1 * (param2 + 4.54) * param3
        param6 = param5 * param4
        param7 = param4 / (param2 - 7)
        circ.rx(param0, 0)
        circ.rx(param1, 0)
        circ.rx(param2, 0)
        circ.rx(param3, 0)
        circ.rx(param4, 0)
        circ.rx(param5, 0)
        circ.rx(param6, 0)
        circ.rx(param7, 0)
        qlm_circ = qiskit_to_qlm(circ)
        i = 0
        for _, params, _ in qlm_circ.iterate_simple():
            for param in params:
                LOGGER.debug(param.to_thrift())
                if i == 0:
                    self.assertEqual(param.to_thrift(), "param0")
                if i == 1:
                    self.assertEqual(param.to_thrift(), "param1")
                if i == 2:
                    self.assertEqual(param.to_thrift(), "param2")
                if i == 3:
                    self.assertEqual(param.to_thrift(), "param3")
                if i == 4:
                    self.assertEqual(param.to_thrift(),
                                     "+ + + * -1.0 param3 param2 param0 param1")
                if i == 5:
                    self.assertEqual(param.to_thrift(),
                                     "* * * + 4.54 param2 param0 param1 param3")
                if i == 6:
                    self.assertEqual(param.to_thrift(),
                                     "* * * * + + + * -1.0 param3 param2 param0 param1 "
                                     + "+ 4.54 param2 param0 param1 param3")
                if i == 7:
                    self.assertEqual(param.to_thrift(),
                                     "* + + + * -1.0 param3 param2 param0 param1"
                                     + " ** + -7.0 param2 -1.0")
            i += 1

        prog = Program()
        qubits = prog.qalloc(1)
        var0 = prog.new_var(float, "param0")
        var1 = prog.new_var(float, "param1")
        var2 = prog.new_var(float, "param2")
        var3 = prog.new_var(float, "param3")
        var4 = var0 + var1 + var2 - var3
        var5 = var0 * var1 * (var2 + 4.54) * var3
        var6 = var5 * var4
        var7 = var4 / (var2 - 7)
        prog.apply(RX(var0), qubits[0])
        prog.apply(RX(var1), qubits[0])
        prog.apply(RX(var2), qubits[0])
        prog.apply(RX(var3), qubits[0])
        prog.apply(RX(var4), qubits[0])
        prog.apply(RX(var5), qubits[0])
        prog.apply(RX(var6), qubits[0])
        prog.apply(RX(var7), qubits[0])
        qlm_circ_expected = prog.to_circ()
        qlm_circ_expected(var0=1)
        qlm_circ_expected(var1=3.14)
        for _, params, _ in qlm_circ_expected.iterate_simple():
            for param in params:
                LOGGER.debug(param.to_thrift())


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
