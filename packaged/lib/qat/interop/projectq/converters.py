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


Description converts a projectq circuit into a qlm circuit,
            this one is tricky, since projectq handles gates dynamically
            we created a new class inheriting from their MainEngine,
            so you should use this engine instead, then code as you would 
            code your projectq circuit : Example :
            aq = AqasmPrinter(MainEngine)
            eng = AqasmEngine(aq, engine_list=[aq])
            q = eng.allocate_qureg(2)
            X | q[0]
            H | q[0]
            CNOT | (q[0], q[1])
            # then recover your generated qlm circuit with
            circ=eng.to_qlm_circ()

Overview
=========


"""
from math import pi
import projectq
from projectq.cengines import LastEngineException, MainEngine
from projectq.ops import AllocateQubitGate, DeallocateQubitGate, AllocateDirtyQubitGate
from projectq.ops import (
    SGate,
    XGate,
    YGate,
    ZGate,
    TGate,
    HGate,
    SwapGate,
    R,
    Rz,
    Rx,
    Ry,
    Ph,
    MeasureGate,
    FlushGate,
    BasicGate,
)
from projectq import ops
import qat.lang.AQASM as aqsm
from qat.lang.parser.qasm_parser import ImplementationError
from qat.mps import get_qpu_server
from qat.core.task import Task

# TODO Gates to add : SqrtX(should be Rx(pi/2)
# TODO and SqrtSwap (not found in this version),
# TODO Gates we have : I, ISIGN, SQRTSWAP
gate_dic = {
    XGate: aqsm.X,
    YGate: aqsm.Y,
    ZGate: aqsm.Z,
    HGate: aqsm.H,
    TGate: aqsm.T,
    SGate: aqsm.S,
    R: aqsm.PH,
    Rx: aqsm.RX,
    Ry: aqsm.RY,
    Rz: aqsm.RZ,
    SwapGate: aqsm.SWAP,
    Ph: aqsm.PH,
}


def QFT(n):
    qft_routine = aqsm.QRoutine()
    if n == 1:
        qft_routine.apply(aqsm.H, 0)
        return qft_routine

    qft_routine.apply(QFT(n - 1), list(range(n - 1)))

    for i in range(n - 1):
        qft_routine.apply(aqsm.PH(pi / pow(2.0, n - i - 1)).ctrl(), n - 1, i)

    qft_routine.apply(aqsm.H, n - 1)
    return qft_routine


def _get_pyqasm_gate(gate, targets=None, controls=0):
    """
        Returns the corresponding pyaqasm gate
    """
    if isinstance(gate, ops.DaggeredGate):
        return _get_pyqasm_gate(gate._gate, targets, controls).dag()
    if controls > 0:
        return _get_pyqasm_gate(gate, targets, controls - 1).ctrl()
    else:
        try:
            gate._angle  # the angle needs to be verified before
            return gate_dic[type(gate)](gate._angle)
        except AttributeError:
            if isinstance(gate, ops._qftgate.QFTGate):
                return QFT(targets)
            else:
                return gate_dic[type(gate)]
        except KeyError:
            print("Error " + str(gate))


# Overloading measurements


def _newbool(self):
    raise ImplementationError(
        "To measure a qubit you need to execute"
        + " the circuit, dynamic measures aren't "
        + "implemented yet"
    )


projectq.types._qubit.Qubit.__bool__ = _newbool


class AqasmEngine(MainEngine):
    """
    A compiler engine which can print and export commands in AQASM format.
    """

    def __init__(self, aq, engine_list=[], verbose=False):
        MainEngine.__init__(self, engine_list=engine_list)
        self.prog = aq.prog
        self.qb = aq.qb
        self.nbqb = aq.nbqb
        self.verbose = verbose

    def allocate_qubit(self, dirty=False):
        self.nbqb += 1
        self.qb.qbits.extend(self.prog.qalloc(1))
        return MainEngine.allocate_qubit(self, dirty)

    def to_qlm_circ(self):
        return self.prog.to_circ()


class AqasmPrinter(MainEngine):
    def __init__(self, engine=MainEngine):
        engine.__init__(self)
        self.prog = aqsm.Program()
        self.nbqb = 0
        self.qb = self.prog.qalloc(0)

    def _out_cmd(self, cmd):
        if (
            isinstance(cmd.gate, AllocateQubitGate)
            or isinstance(cmd.gate, DeallocateQubitGate)
            or isinstance(cmd.gate, AllocateDirtyQubitGate)
        ):
            return
        if isinstance(cmd.gate, MeasureGate):
            inp_qb = []
            for reg in cmd.qubits:
                for qbit in reg:
                    inp_qb.append(self.qb[int(str(qbit))])
            self.prog.measure(inp_qb, inp_qb)

        elif isinstance(cmd.gate, BasicGate):
            controls = cmd.all_qubits[0]
            inp_qb = []
            for reg in cmd.all_qubits:
                for qbit in reg:
                    inp_qb.append(self.qb[int(str(qbit))])
            self.prog.apply(
                _get_pyqasm_gate(
                    cmd.gate, targets=len(cmd._qubits), controls=len(controls)
                ),
                inp_qb,
            )

    def is_available(self, cmd):
        try:
            return MainEngine.is_available(self, cmd)
        except LastEngineException:
            return True

    def receive(self, command_list):
        for cmd in command_list:
            if not cmd.gate == FlushGate():
                self._out_cmd(cmd)
