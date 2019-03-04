#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@cond INTERN


@copyright 2017  Bull S.A.S.  -  All rights reserved.
           This is not Free or Open Source software.
           Please contact Bull SAS for details about its license.
           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois


"""
from projectq.cengines import LastEngineException, MainEngine, BasicEngine
from projectq.ops import AllocateQubitGate, DeallocateQubitGate, AllocateDirtyQubitGate
#from projectq.ops._basics import *
from projectq.ops._gates import XGate, TGate, HGate, SwapGate, R, Rz, Rx, Ry, MeasureGate, FlushGate, BasicGate
from projectq import ops
from projectq.meta import get_control_count
from qat.lang.AQASM import *
from pprint import pprint
def _gate_to_string(gate):
    '''
    Gate of type projectq.ops.gate .
    '''
    pprint(vars(gate))
    if isinstance(gate, SwapGate):
        return SWAP
    if isinstance(gate, R):
        print(gate._angle)
        return PH(gate._angle) # gate._angle  # noqa
    if isinstance(gate, Rx):
        return RX(gate._angle)  # noqa
    if isinstance(gate, Ry):
        return RY(gate._angle)  # noqa
    if isinstance(gate, Rz):
        return RZ(gate._angle) # noqa
    if isinstance(gate, HGate):
        return H
    if isinstance(gate, TGate):
        return T
    if isinstance(gate, XGate):
        return X
    elif isinstance(gate, ops.YGate):
        return Y
    elif isinstance(gate, ops.ZGate):
        return Z
    elif isinstance(gate, ops.SGate):
        return S
    elif isinstance(gate, ops.DaggeredGate):
        return _gate_to_string(gate._gate).dag()
    elif gate == ops.NOT:
        return gate_to_string(gate._gate).ctrl()
    #elif isinstance(gate, ops.SqrtSwapGate):
    #    return SSWAP
    #elif isinstance(gate, ops.SqrtXGate):
    #    return SqrtSWAP
    else:
        return str(gate)


class AqasmEngine(MainEngine):
    """
    A compiler engine which can print and export commands in AQASM format.
    """
    def __init__(self, engine_list, verbose=False):
        MainEngine.__init__(self, engine_list=engine_list)
        self.prog = Program()
        self.qb = self.prog.qalloc(0)
        self.cb = self.prog.calloc(0)
        self.nbqb = 0
        self.verbose = verbose

    def allocate_qubit(self):
        self.nbqb+=1
        self.qb.qbits.extend(self.prog.qalloc(1))
        print("Engine !")
        return MainEngine.allocate_qubit(self)

    def _out_cmd(self, cmd):
        if isinstance(cmd.gate, AllocateQubitGate):
            print("this is an allocation")
            pprint(vars(cmd))
        elif isinstance(cmd.gate, DeallocateQubitGate) or\
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
            for reg in cmd.qubits:
                for qbit in reg:
                    inp_qb.append(self.qb[int(str(qbit))])
            self.prog.apply(_gate_to_string(cmd.gate), inp_qb)
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

class AqasmPrinter(MainEngine):
    """
    A compiler engine which can print and export commands in AQASM format.
    """
    def __init__(self, verbose=False):
        MainEngine.__init__(self)
        self.prog = Program()
        self.qb = self.prog.qalloc(0)
        self.cb = self.prog.calloc(0)
        self.nbqb = 0
        self.verbose = verbose

    def out(self, out_file):
        """
            Exports the circuit in a .aqasm file.


        Args:
            self: the engine
            out_file: a file name
        """
    
        if out_file != None:
            fil = open(out_file, 'w')
            fil.write("BEGIN\nqubits ")
            fil.write(str(self.nbqb))
            fil.write("\n")
            fil.write(self.out_buffer)
            fil.write("END\n")
            fil.close()
    def allocate_qubit(self):
        self.nbqb+=1
        self.qb.qbits.extend(self.prog.qalloc(1))
        print("hello !")
        return MainEngine.allocate_qubit(self)

if __name__=="__main__":
    aq = AqasmPrinter(BasicEngine)
    eng = AqasmEngine(engine_list=[aq])
    #q = eng.allocate_qureg(2)
    p = eng.allocate_qubit()
    c = eng.allocate_qubit()
    #ops.X | p
    ops.ControlledGate(ops.X, n=1) | (p,c)
    #ops.All(ops.Measure) | q
    print("--------------------------------------------------------------")
    circ = eng.prog.to_circ()
