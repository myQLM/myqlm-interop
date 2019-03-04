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
from qat.lang.AQASM import *
from math import pi
from typing import cast
import cirq
ops = cirq.ops
from cirq.ops import common_gates, controlled_gate

def process_H(exp):
    if abs(exp) == 1.0:
        return H
    else:
        raise ValueError("H gate doesn't support arbitrary powers\n"
                         +"Only -1, 0, 1 values are supported")

def process_X(exp):
    if abs(exp) == 1.0:
        return X
    else:
        return RX(pi*exp)

def process_Y(exp):
    if exp == 1.0:
        return Y
    elif exp == -1.0:
        return Y.dag()
    else:
        return RY(pi*exp)

def process_Z(exp):
    if abs(exp) == 1.0:
        return Z
    elif abs(exp) == 0.5:
        return process_S(exp*2)
    elif abs(exp) == 0.25:
        return process_T(exp*4)
    else:
        return RZ(pi*exp)

def process_S(exp):
    if exp == 1.0:
        return S
    elif exp == -1.0:
        return S.dag()
    else:
        return RZ(pi*exp/2)

def process_T(exp):
    if exp == 1.0:
        return T
    elif exp == -1.0:
        return T.dag()
    else:
        return RZ(pi*exp/4)

def process_RX(exp):
    return RX(pi*exp)

def process_RY(exp):
    return RY(pi*exp)

def process_RZ(exp):
    return RZ(pi*exp)

def process_SWAP(exp):
    if abs(exp) == 1.0:
        return SWAP
    elif abs(exp) == 0.5:
        return SQRTSWAP
    elif exp == int(exp):
        if exp%2:
            return SWAP
        else:
            return "none"
    else:
        raise ValueError("SWAP gate doesn't support arbitrary powers\n"
                         +"Only -1, -0.5, 0, 0.5, 1 values are supported")
def process_ISWAP(exp):
    if exp == 1.0:
        return ISWAP
    elif exp == -1.0:
        return ISWAP.dag()
    else:
        raise ValueError("ISWAP gate doesn't support arbitrary powers\n"
                         +"Only -1, 0, 1 values are supported")


gate_dic = {common_gates.HPowGate: process_H,
            common_gates.XPowGate: process_X,
            common_gates.YPowGate: process_Y,
            common_gates.ZPowGate: process_Z,
            common_gates.S: process_S,
            common_gates.T: process_T,
            common_gates.SwapPowGate: process_SWAP,
            common_gates.ISwapPowGate: process_ISWAP}

def _get_gate(gate):
    if controlled_gate.ControlledGate == type(gate):
        return _get_gate(gate.sub_gate).ctrl()
    elif gate.exponent == 0.0:
        return "none"
    else:
        return gate_dic[type(gate)](gate.exponent)

class Gcirc2acirc:
    """ Convert Google circuit into qlm circuit"""

    def __init__(self, gcirc, qubit_order=ops.QubitOrder.DEFAULT):
        # pyaqasm
        self.prog = Program()
        self.qreg = self.prog.qalloc(0)
        # extracting qbits from google circuit
        self.qubits = ops.QubitOrder.as_qubit_order(
            qubit_order).order_for(gcirc.all_qubits())
        # mapping Gcirq qubits coords into qlm qbit register
        self.qmap = { qbit:i for i, qbit in enumerate(self.qubits)}
        self.qreg.qbits.extend(self.prog.qalloc(len(self.qubits)))
        # extracting operations
        self.operations = tuple(ops.flatten_op_tree(
            gcirc.all_operations()))
        # getting measurements
        self.measurements = tuple(
            cast(ops.GateOperation, op)
            for op in self.operations
                if ops.MeasurementGate.is_measurement(
                    cast(ops.GateOperation, op)))


    def to_qlm_circ(self):
        for op in self.operations:
            #print(type(op.gate))
            qbs = []
            for qb in op.qubits:
                qbs.append(self.qreg[self.qmap[qb]])
            if ops.MeasurementGate.is_measurement(
                cast(ops.GateOperation, op)):
                self.prog.measure(qbs, qbs)
            gate= _get_gate(op.gate)
            if gate == "none":
                continue
            else:
                self.prog.apply(gate, qbs)
        self.prog.to_circ()


if __name__=="__main__":

    q = cirq.GridQubit(3, 4)
    q2 = cirq.GridQubit(2, 1)
    circuit = cirq.Circuit.from_ops(
        cirq.ControlledGate(common_gates.X)(q, q2)**-1,
        common_gates.SWAP(q, q2)**0.5,
        common_gates.ISWAP(q, q2)**-1.0,
        common_gates.T(q)**-1.0,
        common_gates.Rz(pi*2)(q)**0.0,
        common_gates.SWAP(q, q2)**-1.0
        )
    circ = Gcirc2acirc(circuit)
    circ.to_qlm_circ()
