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

myQLM provides binders to translate quantum circuit from PyQuil to myQLM
and vice-verse throught functions :func:`~qat.interop.pyquil.pyquil_to_qlm`
and :func:`~qat.interop.pyquil.qlm_to_pyquil`

.. code-block:: python

    from qat.interop.pyquil import pyquil_to_qlm

    qlm_circuit = pyquil_to_qlm(your_pyquil_circuit)

Or

.. code-block:: python

    from qat.interop.pyquil import qlm_to_pyquil

    pyquil_circuit = qlm_to_pyquil(your_qlm_circuit)
"""

import warnings
import pyquil.quilatom
import pyquil.gates as pg
from pyquil import Program
from pyquil.quilbase import Measurement, Declare, Gate

import qat.lang.AQASM.gates as aq
from qat.lang.AQASM import Program as QlmProgram
from qat.comm.gate_set.ttypes import ParamType

try:
    from qat.core.util import extract_syntax
except ImportError:
    from qat.core.circ import extract_syntax


QLM_GATE_DIC = {
    "I": aq.I,
    "H": aq.H,
    "X": aq.X,
    "Y": aq.Y,
    "Z": aq.Z,
    "RX": aq.RX,
    "RY": aq.RY,
    "RZ": aq.RZ,
    "CNOT": aq.CNOT,
    "S": aq.S,
    "T": aq.T,
    "CCNOT": aq.CCNOT,
    "SWAP": aq.SWAP,
    "PHASE": aq.PH,
    "PH": aq.PH,
    "ISWAP": aq.ISWAP,
}


def _get_parameter(param):
    """
    Extract the float number wrapped in the gate parameter.
    Only non-abstract float number can be extracted, otherwise, a TypeError is raised

    Args:
        param: parameter

    Returns:
        float: number
    """
    if param.type != ParamType.DOUBLE or param.is_abstract:
        raise TypeError("Could not translate the QLM circuit to PyQuil, the circuit is composed of gate parametrized "
                        "with untranslatable arguments (i.e., non-floating arguments or abstract variables)")

    return param.double_p


def build_qbits(qbits):
    """ Builds a list of pyquil atoms from a list of integers
    Args:
        qbits: list of integers designing qubits indexes
    Returns:
        list of pyquil atoms
    """
    res = []
    for qb in qbits:
        res.append(pyquil.quilatom.Qubit(qb))
    return res


def build_gate(dic, ident, qubits):
    """ Builds a pyquil operation from a QLM circuit's operation

    Args:
        dic: QLM circuit's GateDictionary
        ident: string identifying the gate used in this operation
        qubits: qubits on which to apply
    Returns:
        A pyquil gate operation
    """
    qlm_gate = dic[ident]
    name = extract_syntax(dic[qlm_gate.name], dic)[0]
    basename = name.rsplit("C-", 1)[-1].rsplit("D-", 1)[-1]

    nbctrls = name.count("C-")
    dag = name.count("D-")

    if basename == "PH":
        basename = "PHASE"

    if nbctrls > 0:
        # build control and targets
        targets = []
        arity = len(qubits) - nbctrls
        targets = qubits[-arity:]
        controls = list(qubits[:nbctrls])
        # base gate
        try:
            params = [
                _get_parameter(param) for param in dic[qlm_gate.subgate].syntax.parameters
            ]
        except AttributeError:
            params = []

        quil_gate = pyquil.quilbase.Gate(basename, params, targets)
        # applying controls (order isn't important)
        for ctrl in controls:
            quil_gate = quil_gate.controlled(ctrl)
        if dag:
            quil_gate = quil_gate.dagger()
        return quil_gate

    if dag:
        params = [_get_parameter(param) for param in qlm_gate.syntax.parameters]
        # if it's a pair numbr of times, then it goes back to normal
        return pyquil.quilbase.Gate(basename, params, qubits).dagger()

    params = [_get_parameter(param) for param in qlm_gate.syntax.parameters]
    if None in params:
        raise TypeError("Unsupported parameter type")
    return pyquil.quilbase.Gate(basename, params, qubits)


def qlm_to_pyquil(qlm_circuit, program_pragma=None):
    """ Converts a QLM circuit to a pyquil circuit

    .. warning::

        Only circuit parametrized with float numbers can be translated to PyQuil.
        Thus, variational circuits cannot be translated to PyQuil

    Args:
        qlm_circuit: QLM circuit to convert
    Returns:
        Pyquil circuit
    """
    if program_pragma is not None:
        program = Program(program_pragma)
    else:
        program = Program()
    creg = program.declare("ro", "BIT", qlm_circuit.nbcbits)

    for op in qlm_circuit.ops:
        if op.type == 0:
            qubits = build_qbits(op.qbits)
            program += build_gate(qlm_circuit.gateDic, op.gate, qubits)
        elif op.type == 1:
            for qb, cb in zip(op.qbits, op.cbits):
                program += pg.MEASURE(qb, creg[cb])
    # Adding measures to unify interface
    for qb, cbit in enumerate(creg):
        program += pg.MEASURE(qb, cbit)
    return program


def build_cregs(prog, pyquil_prog):
    """
    Allocates a classical register to a QLM program according to a pyquil program

    Returns: tuple containing the QLM program and the list of pyquil
             classical registers
    """
    creg_size = 0
    pq_cregs = []
    for op in pyquil_prog.instructions:
        if not isinstance(op, Declare):
            continue
        # (name, offset)
        pq_cregs.append((op.name, creg_size))
        creg_size += op.memory_size
    return (prog.calloc(creg_size), pq_cregs)


def _sanitize_param(param):
    if isinstance(param, complex) and param.imag == 0:
        return param.real
    return param


def pyquil_to_qlm(pyquil_prog, sep_measures=False, **kwargs):
    """ Converts a pyquil circuit into a qlm circuit

    Args:
        pyquil_prog: the pyquil circuit to convert
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
    prog = QlmProgram()
    qreg = prog.qalloc(len(pyquil_prog.get_qubits()))
    creg = None
    quil_regs = None
    to_measure = []
    if not sep_measures:
        creg, quil_regs = build_cregs(prog, pyquil_prog)
    else:
        to_measure = []
    for op in pyquil_prog.instructions:
        if isinstance(op, Gate):
            ctrls = 0
            if len(op.params) > 0:
                if op.name == "CPHASE":
                    gate = aq.PH(*op.params)
                    ctrls += 1
                elif op.name in QLM_GATE_DIC:
                    gate = QLM_GATE_DIC[op.name](*map(_sanitize_param, op.params))
                elif op.name.replace("C", "") in QLM_GATE_DIC:
                    gate = QLM_GATE_DIC[op.name.replace("C", "")](*map(_sanitize_param, op.params))
                    ctrls += len(op.name) - len(op.name.replace("C", ""))
                else:
                    raise ValueError(f"Gate {op.name} is not supported")
            else:
                if op.name in QLM_GATE_DIC:
                    gate = QLM_GATE_DIC[op.name]
                elif op.name.replace("C", "") in QLM_GATE_DIC:
                    gate = QLM_GATE_DIC[op.name.replace("C", "")]
                    ctrls += len(op.name) - len(op.name.replace("C", ""))
                else:
                    raise ValueError(f"Gate {op.name} is not supported")
            if op.modifiers.count('DAGGER') % 2 == 1:
                gate = gate.dag()
            ctrls += op.modifiers.count('CONTROLLED')
            qubits = op.qubits
            if ctrls > 0:
                for _ in range(ctrls):
                    gate = gate.ctrl()
                qubits = list(reversed(op.qubits[:ctrls]))
                qubits.extend(op.qubits[ctrls:])
            qubits = [qreg[qbit.index] for qbit in qubits]
            prog.apply(gate, *qubits)
        elif isinstance(op, Measurement):
            if not sep_measures:
                pq_reg = op.classical_reg.name
                real_offset = 0
                for entry in quil_regs:
                    if entry[0] == pq_reg:
                        real_offset = entry[1]
                real_offset += op.classical_reg.offset
                prog.measure(qreg[op.qubit.index], creg[real_offset])
            else:
                to_measure.append(op.qubit.index)
    if sep_measures:
        return prog.to_circ(**kwargs), list(set(to_measure))
    return prog.to_circ(**kwargs)


def job_to_pyquil(qlm_job):
    """ Converts a QLM job's circuit to a pyquil circuit
    Args:
        qlm_job: the QLM job which circuit we want to convert
    Returns:
        A Pyquil circuit
    """
    return qlm_to_pyquil(qlm_job.circuit)


def to_pyquil_circ(qlm_circuit):
    """ Deprecated """
    warnings.warn(
        "to_pyquil_circ is deprecated, please use qlm_to_pyquil",
        FutureWarning,
    )
    return qlm_to_pyquil(qlm_circuit)


def to_qlm_circ(pyquil_prog, sep_measures=False, **kwargs):
    """ Deprecated """
    warnings.warn(
        "to_qlm_circ is deprecated, please use pyquil_to_qlm",
        FutureWarning,
    )
    return pyquil_to_qlm(pyquil_prog, sep_measures, **kwargs)
