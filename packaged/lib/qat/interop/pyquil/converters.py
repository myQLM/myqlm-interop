#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@brief

@namespace ...
@authors Reda Drissi <mohamed-reda.drissi@atos.net>
@copyright 2019  Bull S.A.S.  -  All rights reserved.
           This is not Free or Open Source software.
           Please contact Bull SAS for details about its license.
           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois


Description ...

Overview
=========


"""

import pyquil.quilatom
import pyquil.gates as pg
from pyquil import Program
import qat.lang.AQASM.gates as aq

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
    res = []
    for qb in qbits:
        res.append(pyquil.quilatom.Qubit(qb))
    return res


def build_gate(dic, ident, qubits):
    qlm_gate = dic[ident]
    if qlm_gate.nbctrls is not None and qlm_gate.nbctrls > 0:
        # build control and targets
        targets = []
        arity = dic[qlm_gate.subgate].arity
        targets = qubits[-arity:]
        controls = qubits[0 : qlm_gate.nbctrls]
        # base gate
        try:
            params = [
                param.double_p for param in dic[qlm_gate.subgate].syntax.parameters
            ]
        except AttributeError:
            params = []
        i = 0
        qlm_gate = dic[qlm_gate.subgate]
        name = dic[qlm_gate.name].syntax
        if name is None:
            if dic[qlm_gate.name].subgate[0] == "_":
                while dic[qlm_gate.name].is_dag:
                    qlm_gate = dic[qlm_gate.subgate]
                    i += 1
                name = dic[qlm_gate.name].name
                if name[0] == "_":
                    try:
                        name = dic[qlm_gate.name].syntax.name
                    except:
                        print(dic[qlm_gate.name])
            else:
                name = name.subgate
        else:
            name = dic[qlm_gate.name].syntax.name
        if name == "PH":
            name = "PHASE"
        quil_gate = pyquil.quilbase.Gate(name, params, targets)
        # applying controls (order isn't important)
        for ctrl in controls:
            quil_gate = quil_gate.controlled(ctrl)
        if i % 2 == 1:
            quil_gate = quil_gate.dagger()
        return quil_gate
    elif qlm_gate.is_dag:
        i = 0
        # find how many times this gate has been daggered
        while qlm_gate.is_dag:
            qlm_gate = dic[qlm_gate.subgate]
            i += 1
        name = qlm_gate.syntax.name
        params = [param.double_p for param in qlm_gate.syntax.parameters]
        # if it's a pair numbr of times, then it goes back to normal
        if i % 2 == 0:
            if name == "PH":
                name = "PHASE"
            return pyquil.quilbase.Gate(name, params, qubits)
        # else it's daggered once
        else:
            return pyquil.quilbase.Gate(name, params, qubits).dagger()
    else:
        name = qlm_gate.syntax.name
        if name == "PH":
            name = "PHASE"
        params = [param.double_p for param in qlm_gate.syntax.parameters]
        return pyquil.quilbase.Gate(name, params, qubits)


def to_pyquil(qlm_circuit, nbshots=1):
    p = Program()
    creg = p.declare("creg", "BIT", qlm_circuit.nbcbits)

    for op in qlm_circuit.ops:
        if op.type == 0:
            qubits = build_qbits(op.qbits)
            p += build_gate(qlm_circuit.gateDic, op.gate, qubits)
        elif op.type == 1:
            for qb in op.qbits:
                p += pg.MEASURE(qb, creg[op.cbits[qb]])
    p.wrap_in_numshots_loop(nbshots)
    return p


def job_to_pyquil(qlm_job):
    return to_pyquil(qlm_job.circuit, qlm_job.nbshots)
