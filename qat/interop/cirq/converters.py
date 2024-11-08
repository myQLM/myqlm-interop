#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.. License

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

myQLM provides binders to translate quantum circuit from Google Cirq
to myQLM and vice-versa throught functions :func:`~qat.interop.cirq.cirq_to_qlm`
and :func:`~qat.interop.cirq.qlm_to_cirq`

.. code-block:: python

    from qat.interop.cirq import cirq_to_qlm

    qlm_circ = cirq_to_qlm(your_google_circ)

Or

.. code-block:: python

    from qat.interop.cirq import qlm_to_cirq

    google_circ = qlm_to_cirq(your_qlm_circ)

.. note::
    when mixing LineQubit and GridQubit, all grid
    qubits will be allocated first, then all line qubits.
    The order will follow coordinates.
"""

import warnings
from math import pi
from typing import cast
from numpy import array, complex128, cos, sin, diag, ufunc, isclose
import sympy

import cirq
from cirq.ops import common_gates, controlled_gate

from qat.core.util import extract_syntax
from qat.core.variables import ArithExpression, Variable
from qat.lang.AQASM import Program, AbstractGate, H, X, Y, Z, S, T, RX, RY, RZ, \
    SWAP, ISWAP, SQRTSWAP

ops = cirq.ops


# Adding parity gates
def gen_XX():
    """ Generates the matrix of an XX gate """
    # pylint: disable=invalid-name
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
    """ Generates the matrix of an YY gate """
    # pylint: disable=invalid-name
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
    """ Generates the matrix of a ZZ gate """
    # pylint: disable=invalid-name
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
    """ Generates the matrix of a RXX gate """
    # pylint: disable=invalid-name
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
    """ Generates the matrix of a RYY gate """
    # pylint: disable=invalid-name
    return array(
        [
            [cos(phi / 2), 0, 0, sin(phi / 2) * 1.0j],
            [0, cos(phi / 2), -sin(phi / 2) * 1.0j, 0],
            [0, -sin(phi / 2) * 1.0j, cos(phi / 2), 0],
            [sin(phi / 2) * 1.0j, 0, 0, cos(phi / 2)],
        ],
        dtype=complex128,
    )


def gen_RZZ(phi):
    """ Generates the matrix of RZZ gate """
    # pylint: disable=invalid-name
    return array(
        [
            [cos(phi / 2) - sin(phi / 2) * 1.0j, 0, 0, 0],
            [0, cos(phi / 2) + sin(phi / 2) * 1.0j, 0, 0],
            [0, 0, cos(phi / 2) + sin(phi / 2) * 1.0j, 0],
            [0, 0, 0, cos(phi / 2) - sin(phi / 2) * 1.0j],
        ],
        dtype=complex128,
    )


def gen_iSWAP3():
    """ Generates the matrix of a iSWAP """
    # pylint: disable=invalid-name
    return diag([1, -1, -1, 1]).astype(complex128)


XX = AbstractGate("XX", [], arity=2, matrix_generator=gen_XX)
YY = AbstractGate("YY", [], arity=2, matrix_generator=gen_YY)
ZZ = AbstractGate("ZZ", [], arity=2, matrix_generator=gen_ZZ)

RXX = AbstractGate("RXX", [float], arity=2, matrix_generator=gen_RXX)
RYY = AbstractGate("RYY", [float], arity=2, matrix_generator=gen_RYY)
RZZ = AbstractGate("RZZ", [float], arity=2, matrix_generator=gen_RZZ)

iSWAP3 = AbstractGate("iSWAP3", [], arity=2, matrix_generator=gen_iSWAP3)

# handy functions to avoid unreadable long if else blocks


def process_XX(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix XX ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    if isinstance(exp, (int, float)) and isclose(exp, 1.0):
        return XX()
    if isinstance(exp, (int, float)) and isclose(exp, -1.0):
        return XX().dag()
    return RXX(exp)


def process_YY(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix YY ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    if isinstance(exp, (int, float)) and isclose(exp, 1.0):
        return YY()
    if isinstance(exp, (int, float)) and isclose(exp, -1.0):
        return YY().dag()
    return RYY(exp)


def process_ZZ(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix ZZ ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    if isinstance(exp, (int, float)) and isclose(exp, 1.0):
        return ZZ()
    if isinstance(exp, (int, float)) and isclose(exp, -1.0):
        return ZZ().dag()
    return RZZ(exp)


def process_H(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix H ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    if isinstance(exp, (int, float)) and isclose(abs(exp), 1.0):
        return H
    if exp != int(exp):
        raise ValueError(
            "H gate doesn't support arbitrary powers\n"
            + "Only integer values are supported"
        )
    if isinstance(exp, (int, float)) and abs(exp) % 2:
        return H
    return "none"


def process_X(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix X ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    if isinstance(exp, (int, float)) and isclose(abs(exp), 1.0):
        return X
    return RX(pi * exp)


def process_Y(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix Y ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    if isinstance(exp, (int, float)) and isclose(exp, 1.0):
        return Y
    if isinstance(exp, (int, float)) and isclose(exp, -1.0):
        return Y.dag()
    return RY(pi * exp)


def process_Z(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix Z ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    if isinstance(exp, (int, float)) and isclose(abs(exp), 1.0):
        return Z
    if isinstance(exp, (int, float)) and isclose(abs(exp), 0.5):
        return process_S(exp * 2)
    if isinstance(exp, (int, float)) and isclose(abs(exp), 0.25):
        return process_T(exp * 4)
    return RZ(pi * exp)


def process_S(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix S ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    if isinstance(exp, (int, float)) and isclose(exp, 1.0):
        return S
    if isinstance(exp, (int, float)) and isclose(exp, -1.0):
        return S.dag()
    return RZ(pi * exp / 2)


def process_T(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix T ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    if isinstance(exp, (int, float)) and isclose(exp, 1.0):
        return T
    if isinstance(exp, (int, float)) and isclose(exp, -1.0):
        return T.dag()
    return RZ(pi * exp / 4)


def process_RX(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix RX ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    return RX(pi * exp)


def process_RY(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix RY ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    return RY(pi * exp)


def process_RZ(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix RZ ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    return RZ(pi * exp)


def process_SWAP(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix SWAP ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    if isinstance(exp, (int, float)) and isclose(abs(exp), 1.0):
        return SWAP
    if isinstance(exp, (int, float)) and isclose(abs(exp), 0.5):
        return SQRTSWAP
    if exp == int(exp):
        if exp % 2:
            return SWAP
        return "none"
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
    # pylint: disable=invalid-name
    if exp != int(exp):
        raise ValueError("Non integer powers aren't supported \
                         for iSWAP gate")

    if isinstance(exp, (int, float)) and abs(exp) % 4 == 1:
        if exp < 0:
            return ISWAP.dag()
        return ISWAP

    if isinstance(exp, (int, float)) and abs(exp) % 4 == 2:
        if exp < 0:
            return iSWAP3().dag()
        return iSWAP3()

    if isinstance(exp, (int, float)) and abs(exp) % 4 == 3:
        if exp < 0:
            return SWAP.dag()
        return SWAP

    return "none"


def process_CX(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix CNOT ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    return process_X(exp).ctrl()


def process_CCX(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix CCNOT ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    return process_X(exp).ctrl().ctrl()


def process_CZ(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix CZ ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    return process_Z(exp).ctrl()


def process_CCZ(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix CCZ ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    return process_Z(exp).ctrl().ctrl()


def process_CSWAP(exp):
    """
    Generates the myQLM gates corresponding to
    the matrix CSWAP ** exp

    Args:
        exp (float): exposant

    Returns:
        Gate
    """
    # pylint: disable=invalid-name
    return process_SWAP(exp).ctrl()


# dictionary linking cirq gate types and corresponding pyaqasm gates
gate_dic = {
    common_gates.HPowGate: process_H,
    common_gates.XPowGate: process_X,
    common_gates.YPowGate: process_Y,
    common_gates.ZPowGate: process_Z,
    common_gates.S: process_S,
    common_gates.T: process_T,
    type(cirq.X): process_X,
    type(cirq.Y): process_Y,
    type(cirq.Z): process_Z,
    type(cirq.S): process_Z,
    type(cirq.T): process_Z,
    common_gates.SwapPowGate: process_SWAP,
    common_gates.ISwapPowGate: process_ISWAP,
    common_gates.CNotPowGate: process_CX,
    common_gates.CZPowGate: process_CZ,
    cirq.ops.three_qubit_gates.CSwapGate: process_CSWAP,
    cirq.ops.three_qubit_gates.CCXPowGate: process_CCX,
    cirq.ops.three_qubit_gates.CCZPowGate: process_CCZ,
    cirq.ops.parity_gates.XXPowGate: process_XX,
    cirq.ops.parity_gates.YYPowGate: process_YY,
    cirq.ops.parity_gates.ZZPowGate: process_ZZ,
    cirq.contrib.acquaintance.permutation.SwapPermutationGate: process_SWAP,
}


try:
    cirq_latest_version_gates = {
        common_gates.Rx: process_RX,
        common_gates.Ry: process_RY,
        common_gates.Rz: process_RZ,
    }
except AttributeError:
    cirq_latest_version_gates = {}


gate_dic.update(cirq_latest_version_gates)


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

        if isinstance(gate.exponent, (int, float)) and isclose(gate.exponent, 0.0):
            return "none"

        return gate_dic[type(gate)](gate.exponent)

    except AttributeError:
        return gate_dic[type(gate)](1.0)


# master function converting cirq object to pyaqasm circuit object
def cirq_to_qlm(circ, sep_measures=False, **kwargs):
    """ Converts a google cirq circuit to a qlm circuit

    Args:
        cirq: the cirq circuit to convert
        sep_measures: Separates measures from the
            circuit:

             - if set to :code:`True` measures won't be included in the resulting circuits,
               qubits to be measured will be put in a list, the resulting measureless
               circuit and this list will be returned in a tuple : (resulting_circuit, list_qubits)
             - if set to :code:`False`, measures will be converted normally (Default set to False)

        kwargs: these are the options that you would use on a regular
            to_circ function, to generate a QLM circuit from a PyAQASM program
            these are added for more flexibility, for advanced users


    Returns:
        :code:`tuple` or :class:`~qat.core.Circuit`: If :code:`sep_measures` is set
        to:

         - :code:`True`: the result is a tuple composed of a
           :class:`~qat.core.Circuit` and a list of qubits that should be
           measured
         - :code:`False`: the result is a :class:`~qat.core.Circuit`
    """
    # building a qubit map to use correct qubits
    qubits = ops.QubitOrder.as_qubit_order(ops.QubitOrder.DEFAULT).order_for(
        circ.all_qubits()
    )
    qmap = {qbit: i for i, qbit in enumerate(qubits)}

    # extracting operations
    operations = tuple(ops.flatten_op_tree(circ.all_operations()))

    # pyaqasm initialization
    prog = Program()
    qreg = prog.qalloc(len(qubits))
    to_measure = []
    # building operations
    for op in operations:
        qbs = []
        for qb in op.qubits:
            qbs.append(qreg[qmap[qb]])
            if (cirq.is_measurement(cast(ops.GateOperation, op))
                    and sep_measures):
                to_measure.append(qmap[qb])
        if cirq.is_measurement(cast(ops.GateOperation, op)):
            if not sep_measures:
                prog.measure(qbs, qbs)
        elif isinstance(_get_gate(op.gate), str) and _get_gate(op.gate) == "none":
            continue
        else:
            prog.apply(_get_gate(op.gate), qbs)

    if sep_measures:
        return prog.to_circ(**kwargs), list(set(to_measure))

    return prog.to_circ(**kwargs)


QLM_GATE_DIC = {
    'H': cirq.H,
    'X': cirq.X,
    'Y': cirq.Y,
    'Z': cirq.Z,
    'RX': cirq.rx,
    'RY': cirq.ry,
    'RZ': cirq.rz,
    'S': common_gates.S,
    'T': common_gates.T,
    'SWAP': common_gates.SWAP,
    'ISWAP': common_gates.ISWAP,
    'CNOT': common_gates.CNOT,
    'CSIGN': common_gates.CZ,
    'CCNOT': ops.three_qubit_gates.CCX,
    'PH': common_gates.ZPowGate
}


def _qat2sympy(ae):
    """
    Transforms qat Arithmetic expressions into Sympy

    Args:
        ae (qat.core.variables.ArithExpression): Arithmetic expression
    Returns:
        Sympy expression
    """
    if isinstance(ae, ArithExpression):
        if isinstance(ae.symbol.evaluator, ufunc):
            return sympy.sympify(ae.symbol.evaluator.__name__)(*[_qat2sympy(c) for c in ae.children])
        return ae.symbol.evaluator(*[_qat2sympy(c) for c in ae.children])

    if isinstance(ae, Variable):
        return sympy.symbols(ae.name)
    return ae


def qlm_to_cirq(qlm_circuit):
    """ Converts a QLM circuit to a cirq circuit.

    Args:
        qlm_circuit: the input QLM circuit to convert

    Returns:
        A cirq Circuit object resulting from the conversion
    """
    cirq_circ = cirq.Circuit()
    qreg = [cirq.LineQubit(i + 1) for i in range(qlm_circuit.nbqbits)]

    for op in qlm_circuit.ops:
        if op.type == 0:
            name, params = extract_syntax(qlm_circuit.gateDic[op.gate],
                                          qlm_circuit.gateDic)
            nbctrls = name.count('C-')
            dag = name.count('D-')
            if name == "I":
                continue
            gate = QLM_GATE_DIC[name.rsplit('-', 1)[-1]]
            if len(params) > 0:
                for i, param in enumerate(params):
                    if isinstance(param, Variable):
                        params[i] = sympy.symbols(param.name)
                    elif isinstance(param, ArithExpression):
                        params[i] = _qat2sympy(param)

                if name.rsplit('-', 1)[-1] == 'PH':
                    gate = gate(exponent=params[0] / pi)
                else:
                    gate = gate(*params)

            if dag % 2 == 1:
                gate = cirq.inverse(gate)

            if nbctrls > 0:
                for _ in range(nbctrls):
                    gate = ops.ControlledGate(gate)
            cirq_circ.append(gate.on(*[qreg[i] for i in op.qbits]))

        elif op.type == 1:
            for qb in op.qbits:
                cirq_circ.append(cirq.measure(qreg[qb]))

    # to unify the interface adding measures here
    for qbit in qreg:
        cirq_circ.append(cirq.measure(qbit))
    return cirq_circ


def to_qlm_circ(cirq_circ, sep_measures=False, **kwargs):
    """
    Deprecated function. Please use cirq_to_qlm instead
    """
    warnings.warn(
        "to_qlm_circ is deprecated, please use cirq_to_qlm",
        FutureWarning,
    )
    return cirq_to_qlm(cirq_circ, sep_measures, **kwargs)


def to_cirq_circ(qlm_circuit):
    """
    Deprecated function. Please use qlm_to_cirq instead
    """
    warnings.warn(
        "to_cirq_circ is deprecated, please use qlm_to_cirq",
        FutureWarning,
    )
    return qlm_to_cirq(qlm_circuit)
