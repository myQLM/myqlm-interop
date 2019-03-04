#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@cond INTERN


@copyright 2017  Bull S.A.S.  -  All rights reserved.
           This is not Free or Open Source software.
           Please contact Bull SAS for details about its license.
           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois


"""
from projectq.cengines import LastEngineException, BasicEngine, MainEngine
from projectq.ops import AllocateQubitGate, DeallocateQubitGate, AllocateDirtyQubitGate, CX
#from projectq.ops._basics import *
from projectq.ops._gates import H, SwapGate, R, Rz, Rx, Ry, MeasureGate, FlushGate, BasicGate

def _gate_to_string(gate):
    '''
    Gate of type projectq.ops.gate .
    '''
    if isinstance(gate, SwapGate):
        return "SWAP"
    if isinstance(gate, R):
        return "PH[" + str(gate._angle) + "]"  # noqa
    if isinstance(gate, Rx):
        return "RX[" + str(gate._angle) + "]"  # noqa
    if isinstance(gate, Ry):
        return "RY[" + str(gate._angle) + "]"  # noqa
    if isinstance(gate, Rz):
        return "RZ[" + str(gate._angle) + "]"  # noqa
    return str(gate)


class AqasmPrinter(MainEngine):
    """
    A compiler engine which can print and export commands in AQASM format.
    """
    

    def __init__(self, verbose=False):
        MainEngine.__init__(self)
        self.out_buffer = ""
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

    def _out_cmd(self, cmd):
        if isinstance(cmd.gate, AllocateQubitGate) or\
                isinstance(cmd.gate, DeallocateQubitGate) or\
                isinstance(cmd.gate, AllocateDirtyQubitGate):
            return
        res_s = ""

        if isinstance(cmd.gate, MeasureGate):
            res_s += "MEAS "
            for qbit in cmd.qubits[0]:
                res_s += "q["+str(qbit)+"],"
            res_s = res_s[:-1]

        elif isinstance(cmd.gate, BasicGate):
            res_s += _gate_to_string(cmd.gate)
            controls = cmd.all_qubits[0]
            targets = cmd.all_qubits[1]
            res_s = 'CTRL(' * len(controls) + res_s + ')' * len(controls) + " "
            for qbit in controls + targets:
                res_s += 'q['+str(qbit)+'],'
            res_s = res_s[:-1]
            for qbit in controls + targets:
                if int(str(qbit)) > self.nbqb - 1:
                    self.nbqb = int(str(qbit)) + 1

        self.out_buffer += res_s
        if self.verbose:
            print(res_s)
        self.out_buffer += "\n"

    def is_available(self, cmd):
        try:
            return BasicEngine.is_available(self, cmd)
        except LastEngineException:
            return True

    def receive(self, command_list):
        for cmd in command_list:
            if not cmd.gate == FlushGate():
                self._out_cmd(cmd)

if __name__=="__main__":
    aq = AqasmPrinter()
    eng = MainEngine(engine_list=[aq])
    q = eng.allocate_qubit()
    p = eng.allocate_qubit()
    H | q
    R(3.14) | p
    CX | (p,q)
    print(eng.out_buffer)
    eng.flush()
