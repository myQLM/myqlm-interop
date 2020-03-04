#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

"""
@brief

@namespace ...
@authors Reda Drissi <mohamed-reda.drissi@atos.net>
         Cyprien Lambert <cyprien.lambert@atos.net>
@copyright 2019-2020 Bull S.A.S.  -  All rights reserved.
           This is not Free or Open Source software.
           Please contact Bull SAS for details about its license.
           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois


Description This is a test suite for qlm to qiskit circuit converter

Overview
=========


"""

import unittest
import logging
from qat.lang.AQASM import Program, QRoutine
from qat.lang.AQASM.gates import H, X, Y, Z, SWAP, I, S, \
        T, RX, RY, RZ, CNOT, CCNOT
from qat.interop.qiskit import qlm_to_qiskit, U2, U3, RXX, RZZ, R, MS
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter

LOGGER = logging.getLogger()
# Set level to logging.DEBUG in order to see more information
LOGGER.setLevel(logging.WARNING)

# redirects log writing to terminal
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.DEBUG)
LOGGER.addHandler(STREAM_HANDLER)

PYGATES_1QB = [X, Y, Z, I, S, T, S.dag(), T.dag(), H,
               RX(3.14), RY(3.14), RZ(3.14), U2(3.14, 3.14), R(3.14, 3.14),
               U3(3.14, 3.14, 3.14)]
PYGATES_2QB = [SWAP, CNOT, Y.ctrl(), Z.ctrl(), H.ctrl(), RZ(3.14).ctrl(),
               RXX(3.14), RZZ(3.14)]


GATE_NAMES = {'x': 'X', 'y': 'Y', 'z': 'Z', 'h': 'H', 'rx': 'RX', 'id': 'I',
              'ry': 'RY', 'rz': 'RZ', 's': 'S', 't': 'T', 'sdg': 'D-S',
              'tdg': 'D-T', 'swap': 'SWAP', 'cx': 'CNOT', 'ccx': 'CCNOT',
              'cy': 'C-Y', 'cz': 'CSIGN', 'ch': 'C-H', 'crz': 'C-RZ',
              'cswap': 'C-SWAP', 'u2': 'U2', 'u3': 'U3', 'rxx': 'RXX',
              'rzz': 'RZZ', 'r': 'R', 'ms': 'MS', 'measure': 'measure'}


def extract_qiskit(gate_op):
    """
    Extracts relevent informations from a gate.

    Args:
        gate_op: Tuple object from the list QuantumCircuit.data

    Returns:
        A tuple objects with: name of the gate, parameters of the gate,
        qubits affected by the gate, cbits affected by the gate
    """
    name = gate_op[0].name
    params = gate_op[0]._params
    qubits = gate_op[1]
    cbits = gate_op[2]
    return GATE_NAMES[name], params, qubits, cbits


def print_qiskit(circuit):
    """
    Print each gate of a qiskit circuit.

    Args:
        circuit: QuantumCircuit object
    """
    result = ""
    for entry in circuit.data:
        name, params, qubits, cbits = extract_qiskit(entry)
        result += "Gate {} with params {} on qubits {} and cbits {}\n".format(
            name, params, qubits, cbits
        )
    LOGGER.debug(result)
    return result


def qiskit_1qb(qc):
    """
    Returns the list of the qiskit gate methods that affect 1 qubit and
    take no parameter.
    """
    return [qc.x, qc.y, qc.z, qc.iden, qc.s, qc.t, qc.sdg, qc.tdg, qc.h]


def qiskit_1qb_1prm(qc):
    """
    Returns the list of the qiskit gate methods that affect 1 qubit and
    take 1 parameter.
    """
    return [qc.rx, qc.ry, qc.rz]


def qiskit_1qb_2prm(qc):
    """
    Returns the list of the qiskit gate methods that affect 1 qubit and
    take 2 parameter.
    """
    return [qc.u2, qc.r]


def qiskit_1qb_3prm(qc):
    """
    Returns the list of the qiskit gate methods that affect 1 qubit and
    take 3 parameter.
    """
    return [qc.u3]


def qiskit_2qb(qc):
    """
    Returns the list of the qiskit gate methods that affect 2 qubit and
    take no parameter.
    """
    return [qc.swap, qc.cx, qc.cy, qc.cz, qc.ch]


def qiskit_2qb_1prm(qc):
    """
    Returns the list of the qiskit gate methods that affect 2 qubit and
    take 1 parameter.
    """
    return [qc.crz, qc.rxx, qc.rzz]


class TestQLM2QiskitConversion(unittest.TestCase):
    """
    Tests the function converting a qlm circuit
    to a qiskit circuit.
    """

    def test0_default_gates(self):
        """
        Tests all the default QLM gates which conversion in Qiskit
        are supported.
        """
        prog = Program()
        qreg = prog.qalloc(5)

        for gate_op in PYGATES_1QB:
            prog.apply(gate_op, qreg[0])

        for gate_op in PYGATES_2QB:
            prog.apply(gate_op, qreg[0], qreg[1])

        prog.apply(CCNOT, qreg[0], qreg[1], qreg[2])
        prog.apply(SWAP.ctrl(), qreg[0], qreg[1], qreg[2])

        prog.apply(MS(3.14, 3), qreg[1], qreg[2], qreg[4])

        qlm_circuit = prog.to_circ()
        result = qlm_to_qiskit(qlm_circuit)

        qiskit_qreg = QuantumRegister(5)
        qiskit_creg = ClassicalRegister(5)
        expected = QuantumCircuit(qiskit_qreg, qiskit_creg)
        for gate_op in qiskit_1qb(expected):
            gate_op(qiskit_qreg[0])
        for gate_op in qiskit_1qb_1prm(expected):
            gate_op(3.14, qiskit_qreg[0])
        for gate_op in qiskit_1qb_2prm(expected):
            gate_op(3.14, 3.14, qiskit_qreg[0])
        for gate_op in qiskit_1qb_3prm(expected):
            gate_op(3.14, 3.14, 3.14, qiskit_qreg[0])
        for gate_op in qiskit_2qb(expected):
            gate_op(qiskit_qreg[0], qiskit_qreg[1])
        for gate_op in qiskit_2qb_1prm(expected):
            gate_op(3.14, qiskit_qreg[0], qiskit_qreg[1])

        expected.ccx(*qiskit_qreg[:3])
        expected.cswap(*qiskit_qreg[:3])

        # for the MS gate test
        for i in [1, 2, 4]:
            for j in [1, 2, 4]:
                if j > i:
                    expected.rxx(3.14, qiskit_qreg[i], qiskit_qreg[j])

        expected.measure(qiskit_qreg, qiskit_creg)

        LOGGER.debug("qlm_to_qiskit test with standard circuit:")
        expected_str = print_qiskit(expected)
        result_str = print_qiskit(result)
        self.assertEqual(len(result_str), len(expected_str))

        for i in range(len(result.data)):
            r_name, r_params = extract_qiskit(result.data[i])[0:2]
            e_name, e_params = extract_qiskit(expected.data[i])[0:2]
            self.assertEqual(r_name, e_name)
            self.assertEqual(r_params, e_params)

    def test1_abstract_gate(self):
        """
        Tests an AbstractGate translation to Qiskit.
        Only abstract gates defined via a circuit are supported.
        """
        prog = Program()
        qreg = prog.qalloc(3)
        routine = QRoutine()

        for gate_op in PYGATES_1QB:
            routine.apply(gate_op, [0])

        for gate_op in PYGATES_2QB:
            routine.apply(gate_op, [0, 1])

        routine.apply(CCNOT, [0, 1, 2])
        routine.apply(SWAP.ctrl(), [0, 1, 2])

        prog.apply(routine.box("custom_gate"), qreg)
        qlm_circuit = prog.to_circ()
        result = qlm_to_qiskit(qlm_circuit)

        qiskit_qreg = QuantumRegister(3)
        qiskit_creg = ClassicalRegister(3)
        expected = QuantumCircuit(qiskit_qreg, qiskit_creg)
        for gate_op in qiskit_1qb(expected):
            gate_op(qiskit_qreg[0])
        for gate_op in qiskit_1qb_1prm(expected):
            gate_op(3.14, qiskit_qreg[0])
        for gate_op in qiskit_1qb_2prm(expected):
            gate_op(3.14, 3.14, qiskit_qreg[0])
        for gate_op in qiskit_1qb_3prm(expected):
            gate_op(3.14, 3.14, 3.14, qiskit_qreg[0])
        for gate_op in qiskit_2qb(expected):
            gate_op(qiskit_qreg[0], qiskit_qreg[1])
        for gate_op in qiskit_2qb_1prm(expected):
            gate_op(3.14, qiskit_qreg[0], qiskit_qreg[1])

        expected.ccx(*qiskit_qreg)
        expected.cswap(*qiskit_qreg)

        expected.measure(qiskit_qreg, qiskit_creg)

        LOGGER.debug("qlm_to_qiskit test with a QRoutine:")
        expected_str = print_qiskit(expected)
        result_str = print_qiskit(result)
        self.assertEqual(len(result_str), len(expected_str))

        for i in range(len(result.data)):
            r_name, r_params = extract_qiskit(result.data[i])[0:2]
            e_name, e_params = extract_qiskit(expected.data[i])[0:2]
            self.assertEqual(r_name, e_name)
            self.assertEqual(r_params, e_params)

    def test2_abstract_variables(self):
        """
        Tests the translation of abstract variables and
        ArithExpression into Qiskit via qlm_to_qiskit()
        """
        prog = Program()
        qubits = prog.qalloc(1)
        var0 = prog.new_var(float, "param0")
        var1 = prog.new_var(float, "param1")
        var2 = prog.new_var(float, "param2")
        var3 = prog.new_var(float, "param3")
        var0.set(1)
        var1.set(3.14)
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
        qlm_circ = prog.to_circ()
        qiskit_circ = qlm_to_qiskit(qlm_circ)
        LOGGER.debug("Parameters gotten:")
        for gate_op in qiskit_circ.data:
            for param in gate_op[0]._params:
                LOGGER.debug(param)

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
        param7 = param4 / (param2 - 7.0)
        circ.rx(param0, 0)
        circ.rx(param1, 0)
        circ.rx(param2, 0)
        circ.rx(param3, 0)
        circ.rx(param4, 0)
        circ.rx(param5, 0)
        circ.rx(param6, 0)
        circ.rx(param7, 0)
        LOGGER.debug("Parameters expected:")
        for gate_op in circ.data:
            for param in gate_op[0]._params:
                LOGGER.debug(param)

        for gotten, expected in zip(qiskit_circ.data, circ.data):
            self.assertEqual(str(gotten[0]._params[0]),
                             str(expected[0]._params[0]))


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
