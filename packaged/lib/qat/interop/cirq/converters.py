#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@brief 

#@file qat/interop/cirq/converters.py
#@namespace qat.interop.cirq.converters
#@authors Reda Drissi <mohamed-reda.drissi@atos.net>
#@copyright 2019  Bull S.A.S.  -  All rights reserved.
#           This is not Free or Open Source software.
#           Please contact Bull SAS for details about its license.
#           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois

"""
Converts a Google cirq circuit object into a qlm circuit
object, you can directly use :
qlm_circ=to_qlm_circ(your_google_circ)
This is a placeholder, names and packaging might change
to keep consistency


Note:
    when mixing LineQubit and GridQubit, all grid\
    qubits will be allocated first, then all line qubits.\
    The order will follow coordinates.

"""
from qat.lang.AQASM import *
from math import pi
from numpy import array, complex128, cos, sin, diag
from typing import cast
import cirq

ops = cirq.ops
from cirq.ops import common_gates, controlled_gate


# Adding parity gates
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


def gen_iSWAP3():
    return diag([ 1, -1, -1, 1]).astype(complex128)

XX = AbstractGate("XX", [], arity=2, matrix_generator=gen_XX)
YY = AbstractGate("YY", [], arity=2, matrix_generator=gen_YY)
ZZ = AbstractGate("ZZ", [], arity=2, matrix_generator=gen_ZZ)

RXX = AbstractGate("RXX", [float], arity=2, matrix_generator=gen_RXX)
RYY = AbstractGate("RYY", [float], arity=2, matrix_generator=gen_RYY)
RZZ = AbstractGate("RZZ", [float], arity=2, matrix_generator=gen_RZZ)

iSWAP3 = AbstractGate("iSWAP3", [], arity=2, matrix_generator=gen_iSWAP3)

# handy functions to avoid unreadable long if else blocks


def process_XX(exp):
    if exp == 1.0:
        return XX()
    elif exp == -1.0:
        return XX().dag()
    else:
        return RXX(exp)


def process_YY(exp):
    if exp == 1.0:
        return YY()
    elif exp == -1.0:
        return YY().dag()
    else:
        return RYY(exp)


def process_ZZ(exp):
    if exp == 1.0:
        return ZZ()
    elif exp == -1.0:
        return ZZ().dag()
    else:
        return RZZ(exp)


def process_H(exp):
    if abs(exp) == 1.0:
        return H
    if exp != int(exp):
        raise ValueError(
            "H gate doesn't support arbitrary powers\n"
            + "Only integer values are supported"
        )
    if abs(exp) % 2:
        return H
    else:
        return "none"

def process_X(exp):
    if abs(exp) == 1.0:
        return X
    else:
        return RX(pi * exp)


def process_Y(exp):
    if exp == 1.0:
        return Y
    elif exp == -1.0:
        return Y.dag()
    else:
        return RY(pi * exp)


def process_Z(exp):
    if abs(exp) == 1.0:
        return Z
    elif abs(exp) == 0.5:
        return process_S(exp * 2)
    elif abs(exp) == 0.25:
        return process_T(exp * 4)
    else:
        return RZ(pi * exp)


def process_S(exp):
    if exp == 1.0:
        return S
    elif exp == -1.0:
        return S.dag()
    else:
        return RZ(pi * exp / 2)


def process_T(exp):
    if exp == 1.0:
        return T
    elif exp == -1.0:
        return T.dag()
    else:
        return RZ(pi * exp / 4)


def process_RX(exp):
    return RX(pi * exp)


def process_RY(exp):
    return RY(pi * exp)


def process_RZ(exp):
    return RZ(pi * exp)


def process_SWAP(exp):
    if abs(exp) == 1.0:
        return SWAP
    elif abs(exp) == 0.5:
        return SQRTSWAP
    elif exp == int(exp):
        if exp % 2:
            return SWAP
        else:
            return "none"
    else:
        raise ValueError(
            "SWAP gate doesn't support arbitrary powers\n"
            + "Only integer values and +/- 0.5 are supported"
        )


def process_ISWAP(exp):
    """ Processes iSWAP powers in Z:
        1: iSWAP
        2: diag(1, -1, -1, 1)
        3: SWAP
        4: I
    """
    if exp != int(exp):
        raise ValueError("Non integer powers aren't supported \
                         for iSWAP gate")
    if abs(exp) % 4 == 1:
        if exp < 0:
            return ISWAP.dag()
        else:
            return ISWAP
    elif abs(exp) % 4 == 2:
        if exp < 0:
            return iSWAP3().dag()
        else:
            return iSWAP3()
    elif abs(exp) % 4 == 3:
        if exp < 0:
            return SWAP.dag()
        else:
            return SWAP
    else:
        return "none"


def process_CX(exp):
    return process_X(exp).ctrl()


def process_CCX(exp):
    return process_X(exp).ctrl().ctrl()


def process_CZ(exp):
    return process_Z(exp).ctrl()


def process_CZZ(exp):
    return process_Z(exp).ctrl().ctrl()


def process_CSWAP(exp):
    return process_SWAP(exp).ctrl()


# dictionary linking cirq gate types and corresponding pyaqasm gates
gate_dic = {
    common_gates.HPowGate: process_H,
    common_gates.XPowGate: process_X,
    common_gates.YPowGate: process_Y,
    common_gates.ZPowGate: process_Z,
    common_gates.S: process_S,
    common_gates.T: process_T,
    common_gates.SwapPowGate: process_SWAP,
    common_gates.ISwapPowGate: process_ISWAP,
    common_gates.CNotPowGate: process_CX,
    common_gates.CZPowGate: process_CZ,
    cirq.ops.three_qubit_gates.CSwapGate: process_CSWAP,
    cirq.ops.three_qubit_gates.CCXPowGate: process_CCX,
    cirq.ops.three_qubit_gates.CCZPowGate: process_CZZ,
    cirq.ops.parity_gates.XXPowGate: process_XX,
    cirq.ops.parity_gates.YYPowGate: process_YY,
    cirq.ops.parity_gates.ZZPowGate: process_ZZ,
}

# gets a cirq gate object and outputs corresponding pyaqasm gate
def _get_gate(gate):
    """ Takes a cirq gate object and returns corresponding
        QLM gate

    Args:
        gate: cirq.ops gate to build from

    Returns:
        Corresponding QLM gate
    """
    try:
        if controlled_gate.ControlledGate == type(gate):
            return _get_gate(gate.sub_gate).ctrl()
        elif gate.exponent == 0.0:
            return "none"
        else:
            return gate_dic[type(gate)](gate.exponent)
    except AttributeError:
        return gate_dic[type(gate)](1.0)


# master function converting cirq object to pyaqasm circuit object
def to_qlm_circ(cirq, sep_measure=False):
    """ Converts a google cirq circuit to a qlm circuit

    Args:
        cirq: the cirq circuit to convert
        sep_measure: if set to True measures won't be included in the\
        resulting circuits, qubits to be measured will be put in a list,\
        the resulting measureless circuit and this list will be returned\
        in a tuple: (resulting_circuit, list_qubits).\
        If set to False, measures will be converted normally

    Returns:
        if sep_measure is True a tuple of two elements will be returned,
        first one is the QLM resulting circuit with no measures, and the
        second element of the returned tuple is a list of all qubits that
        should be measured.
        if sep_measure is False, the QLM resulting circuit is returned
        directly
    """
    # building a qubit map to use correct qubits
    qubits = ops.QubitOrder.as_qubit_order(ops.QubitOrder.DEFAULT).order_for(
        cirq.all_qubits()
    )
    qmap = {qbit: i for i, qbit in enumerate(qubits)}

    # extracting operations
    operations = tuple(ops.flatten_op_tree(cirq.all_operations()))

    # pyaqasm initialization
    prog = Program()
    qreg = prog.qalloc(0)
    qreg.qbits.extend(prog.qalloc(len(qubits)))
    to_measure = []
    # building operations
    for op in operations:
        qbs = []
        for qb in op.qubits:
            if sep_measure:
                to_measure.append(qmap[qb])
            qbs.append(qreg[qmap[qb]])
        if ops.MeasurementGate.is_measurement(cast(ops.GateOperation, op)):
            if not sep_measure:
                prog.measure(qbs, qbs)
        elif _get_gate(op.gate) == "none":
            continue
        else:
            prog.apply(_get_gate(op.gate), qbs)
    if sep_measure:
        return prog.to_circ(), to_measure
    else:
        return prog.to_circ()
