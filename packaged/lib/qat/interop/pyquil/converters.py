#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@brief
#
#@file qat/interop/pyquil/converters.py
#@namespace qat.interop.pyquil.converters
#@authors Reda Drissi <mohamed-reda.drissi@atos.net>
#@copyright 2019  Bull S.A.S.  -  All rights reserved.
#           This is not Free or Open Source software.
#           Please contact Bull SAS for details about its license.
#           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois

"""
Circuit conversion functions for pyquil
"""

import pyquil.quilatom
import pyquil.gates as pg
from pyquil import Program
from pyquil.quilbase import Measurement, Declare, Gate
import qat.lang.AQASM.gates as aq
try:
    from qat.core.util import extract_syntax
except ImportError:
    from qat.core.circ import extract_syntax
import numpy as np


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
    "PH": aq.PH,
    "ISWAP": aq.ISWAP,
}

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
        arity = dic[qlm_gate.subgate].arity
        targets = qubits[-arity:]
        controls = qubits[0 : nbctrls]
        # base gate
        try:
            params = [
                param.double_p for param in dic[qlm_gate.subgate].syntax.parameters
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
    elif dag:
        params = [param.double_p for param in qlm_gate.syntax.parameters]
        # if it's a pair numbr of times, then it goes back to normal
        return pyquil.quilbase.Gate(basename, params, qubits).dagger()
    else:
        params = [param.double_p for param in qlm_gate.syntax.parameters]
        return pyquil.quilbase.Gate(basename, params, qubits)


def to_pyquil_circ(qlm_circuit):
    """ Converts a QLM circuit to a pyquil circuit

    Args:
        qlm_circuit: QLM circuit to convert
    Returns:
        Pyquil circuit
    """
    p = Program()
    creg = p.declare("ro", "BIT", qlm_circuit.nbcbits)

    for op in qlm_circuit.ops:
        if op.type == 0:
            qubits = build_qbits(op.qbits)
            p += build_gate(qlm_circuit.gateDic, op.gate, qubits)
        elif op.type == 1:
            for qb, cb in zip(op.qbits, op.cbits):
                p += pg.MEASURE(qb, creg[cb])
    return p

def build_cregs(prog, pyquil_prog):
    creg_size = 0
    pq_cregs = []
    for op in pyquil_prog.instructions:
        if not isinstance(op, Declare):
            continue
        # (name, offset)
        pq_cregs.append((op.name, creg_size))
        creg_size += op.memory_size
    return (prog.calloc(creg_size), pq_cregs)


def to_qlm_circ(pyquil_prog, sep_measures=False, **kwargs):
        """ Converts a pyquil circuit into a qlm circuit\
 this function uses either new or old architecture,\
 depending on the qiskit version currently in use

    Args:
        pyquil_prog: the pyquil circuit to convert
        sep_measure: if set to True measures won't be included in the resulting circuits, qubits to be measured will be put in a list, the resulting measureless circuit and this list will be returned in a tuple : (resulting_circuit, list_qubits). If set to False, measures will be converted normally
        kwargs: these are the options that you would use on a regular \
        to_circ function, these are added for more flexibility, for\
        advanced users


    Returns:
        if sep_measure is True a tuple of two elements will be returned,
        first one is the QLM resulting circuit with no measures, and the
        second element of the returned tuple is a list of all qubits that
        should be measured.
        if sep_measure is False, the QLM resulting circuit is returned
        directly
    """
    from qat.lang.AQASM import Program as QlmProgram
    import qat.lang.AQASM.gates as aq
    prog = QlmProgram()
    qreg = prog.qalloc(len(pyquil_prog.get_qubits()))
    if not sep_measures:
        creg, quil_regs = build_cregs(prog, pyquil_prog)
    else:
        to_measure = []
    for op in pyquil_prog.instructions:
        if isinstance(op, Gate):
            if len(op.params) > 0:
                gate = QLM_GATE_DIC[op.name](*params)
            else:
                gate = QLM_GATE_DIC[op.name]
            if op.modifiers.count('DAGGER')%2 == 1:
                gate = gate.dag()
            ctrls = op.modifiers.count('CONTROLLED')
            qubits = op.qubits
            if ctrls > 0:
                for _ in range(ctrls):
                    gate = gate.ctrl()
                qubits = op.qubits[ctrls:]
                qubits.extend(reversed(op.qubits[:ctrls]))
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
        return prog.to_circ(**kwargs), to_measure
       return prog.to_circ(**kwargs)

def job_to_pyquil(qlm_job):
    """ Converts a QLM job's circuit to a pyquil circuit
    Args:
        qlm_job: the QLM job which circuit we want to convert
    Returns:
        A Pyquil circuit
    """
    return to_pyquil_circ(qlm_job.circuit)
