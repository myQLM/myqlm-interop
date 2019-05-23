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

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qat.comm.shared.ttypes import Job, Batch, Result
from qat.comm.datamodel.ttypes import Circuit
from qat.lang.AQASM.gates import H, X, Y, Z, S, T, RX, RY, RZ, I, SWAP


def gen_qiskit_gateset(qc):
    return {H: qc.h, X: qc.x, Y: qc.y, Z: qc.z, SWAP: qc.swap,
            I: qc.id, S: qc.s, S.dag(): qc.sdg, T: qc.t,
            T.dag(): qc.tdg, RX: qc.rx, RY: qc.ry, RZ: qc.rz}

def to_qiskit_circuit(qlm_circuit):
    qreg = QuantumRegister(qlm_circuit.nbqbits)
    creg = ClassicalRegister(qlm_circuit.nbcbits)
    qc = QuantumCircuit(qreg, creg)
    dic = gen_qiskit_gateset(qc)
    for op in qlm_circuit.ops:
        if op.type == 0:
            try:
                dic[op.gate](op.gate.parameters + [qreg[i.index] for i in op.qbits])
            except KeyError:
                raise ValueError("Gate {} not supported by qiskit API".format(op.gate.name))
        elif op.type == 1:
            for index in range(len(op.qbits)):
                qc.measure(op.qbits[index], op.cbits[index])
    return qc

def job_to_qiskit_circuit(qlm_job):
    return to_qiskit_circuit(qlm_job.circuit)
