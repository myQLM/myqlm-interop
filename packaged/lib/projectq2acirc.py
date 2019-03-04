#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@cond INTERN


@copyright 2017  Bull S.A.S.  -  All rights reserved.
           This is not Free or Open Source software.
           Please contact Bull SAS for details about its license.
           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois


"""
from projectq.cengines import LastEngineException, MainEngine
from projectq.ops import AllocateQubitGate, DeallocateQubitGate, AllocateDirtyQubitGate
#from projectq.ops._basics import *
from projectq.ops import SGate, XGate, YGate, ZGate, TGate, HGate\
                                , SwapGate, R, Rz, Rx, Ry, Toffoli\
                                , MeasureGate, FlushGate, BasicGate
from projectq import ops
import qat.lang.AQASM as aqsm
from pprint import pprint
from math import pi

from qat.mps import get_qpu_server
from qat.core.task import Task
#TODO Gates to add : SqrtX(should be Rx(pi/2)
#TODO and SqrtSwap (not found in this version),
#TODO Gates we have : I, ISIGN, SQRTSWAP
gate_dic = { XGate : aqsm.X , YGate: aqsm.Y, ZGate: aqsm.Z, HGate: aqsm.H,
            TGate: aqsm.T, SGate: aqsm.S, R: aqsm.PH, Rx: aqsm.RX,
            Ry: aqsm.RY, Rz: aqsm.RZ, SwapGate: aqsm.SWAP}

def QFT(n):
    qft_routine = aqsm.QRoutine()
    if(n == 1):
        qft_routine.apply(aqsm.H, 0)
        return qft_routine

    qft_routine.apply(QFT(n-1), list(range(n-1)))

    for i in range(n-1):
        qft_routine.apply(aqsm.PH(pi/pow(2.,n-i-1)).ctrl(), n-1, i)

    qft_routine.apply(aqsm.H, n-1)
    return qft_routine

def _get_pyqasm_gate(gate, targets=None, controls=0):
    '''
        Returns the corresponding pyaqasm gate
    '''
    if isinstance(gate, ops.DaggeredGate):
        return _get_pyqasm_gate(gate._gate, targets, controls).dag()
    if controls > 0:
            return _get_pyqasm_gate(gate, targets, controls-1).ctrl()
    else:
        try:
            gate._angle # the angle needs to be verified before
            return gate_dic[type(gate)](gate._angle)
        except AttributeError:
            if isinstance(gate, ops._qftgate.QFTGate):
                return QFT(targets)
            else:
                return gate_dic[type(gate)]
        except KeyError:
            print("Error "+str(gate))
    #elif isinstance(gate, ops.SqrtSwapGate):
    #    return SSWAP
    #elif isinstance(gate, ops.SqrtXGate):
    #    return SqrtSWAP


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

    def allocate_qubit(self):
        self.nbqb+=1
        self.qb.qbits.extend(self.prog.qalloc(1))
        return MainEngine.allocate_qubit(self)
    def to_qlm_circ(self):
        return self.prog.to_circ()

class AqasmPrinter(MainEngine):
    def __init__(self, engine=MainEngine):
        engine.__init__(self)
        self.prog = aqsm.Program()
        self.nbqb = 0
        self.qb = self.prog.qalloc(0)

    def _out_cmd(self, cmd):
        if isinstance(cmd.gate, AllocateQubitGate) or\
                isinstance(cmd.gate, DeallocateQubitGate) or\
                isinstance(cmd.gate, AllocateDirtyQubitGate):
            return
        res_s = ""

        if isinstance(cmd.gate, MeasureGate):
            inp_qb = []
            for reg in cmd.qubits:
                for qbit in reg:
                    inp_qb.append(self.qb[int(str(qbit))])
            self.prog.measure(inp_qb, inp_qb)

        elif isinstance(cmd.gate, BasicGate):
            #res_s += str(_gate_to_string(cmd.gate))
            controls = cmd.all_qubits[0]
            targets = cmd.all_qubits[1]
            res_s = 'CTRL(' * len(controls) + res_s + ')' * len(controls) + " "
            print(cmd.all_qubits)
            print(cmd.gate)
            pprint(vars(cmd))
            for qbit in controls + targets:
                res_s += 'q['+str(qbit)+'],'
            res_s = res_s[:-1]
            inp_qb = []
            for reg in cmd.all_qubits:
                for qbit in reg:
                    inp_qb.append(self.qb[int(str(qbit))])
            self.prog.apply(
                _get_pyqasm_gate(
                            cmd.gate,
                            targets=len(cmd._qubits),
                            controls=len(controls)),
                inp_qb)
        #if self.verbose:
            #print(res_s)

    def is_available(self, cmd):
        try:
            return MainEngine.is_available(self, cmd)
        except LastEngineException:
            return True

    def receive(self, command_list):
        for cmd in command_list:
            if not cmd.gate == FlushGate():
                self._out_cmd(cmd)

if __name__=="__main__":
    aq = AqasmPrinter(MainEngine)
    eng = AqasmEngine(aq, engine_list=[aq])
    q = eng.allocate_qureg(2)
    p = eng.allocate_qubit()
    c = eng.allocate_qubit()
    #ops.X | p
    Toffoli | (q[0], q[1], p)
    ops.ControlledGate(R(3.14), n=2) | (q[0], p, c)
    ops.Sdag | q[0]
    ops.QFT | (p, c)
    ops.All(ops.Measure) | q
    print("--------------------------------------------------------------")
    acirc = eng.to_qlm_circ()

    task = Task(acirc, get_qpu_server())
    outp = task.execute()
    print(outp.probability, outp.state)
    for res in task.states():
        print(res.state, res.amplitude)
