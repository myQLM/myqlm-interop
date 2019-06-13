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
import qat.lang.AQASM.gates as aq
from qat.core.util import extract_syntax
import numpy as np


GATE_DIC = [
    "I",
    "H",
    "X",
    "Y",
    "Z",
    "RX",
    "RY",
    "RZ",
    "CNOT",
    "S",
    "T",
    "CNOT",
    "CCNOT",
    "SWAP",
    "PH",
    "ISWAP",
]


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


def job_to_pyquil(qlm_job):
    """ Converts a QLM job's circuit to a pyquil circuit
    Args:
        qlm_job: the QLM job which circuit we want to convert
    Returns:
        A Pyquil circuit
    """
    return to_pyquil_circ(qlm_job.circuit)
