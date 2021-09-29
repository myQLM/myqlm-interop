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

myQLM provides binders to translate ProjectQ circuits into
myQLM circuits through classes :class:`~qat.interop.projectq.AqasmPrinter`
and :class:`~qat.interop.projectq.AqasmEngine`

.. run-block:: python

    from projectq.ops import X, H, CNOT
    from projectq.cengines import MainEngine
    from qat.interop.projectq import AqasmPrinter, AqasmEngine

    # Define AQASM engine
    aq = AqasmPrinter(MainEngine)
    eng = AqasmEngine(aq, engine_list=[aq])

    # Create a ProjectQ circuit
    q = eng.allocate_qureg(2)
    X | q[0]
    H | q[0]
    CNOT | (q[0], q[1])

    # Then recover your generated qlm circuit with
    circ = eng.projectq_to_qlm()
    print("The circuit is composed of the gates:",
          list(circ.iterate_simple()))
"""

import warnings
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
import qat.lang.AQASM.gates as aq_gates
import qat.lang.AQASM as aqsm
from qat.interop.openqasm.qasm_parser import ImplementationError

# TODO Gates to add : SqrtX(should be Rx(pi/2)
# TODO and SqrtSwap (not found in this version),
# TODO Gates we have : I, ISIGN, SQRTSWAP
gate_dic = {
    XGate: aq_gates.X,
    YGate: aq_gates.Y,
    ZGate: aq_gates.Z,
    HGate: aq_gates.H,
    TGate: aq_gates.T,
    SGate: aq_gates.S,
    R: aq_gates.PH,
    Rx: aq_gates.RX,
    Ry: aq_gates.RY,
    Rz: aq_gates.RZ,
    SwapGate: aq_gates.SWAP,
    Ph: aq_gates.PH,
}
param_list = [aq_gates.PH, aq_gates.RX, aq_gates.RY, aq_gates.RZ]


def QFT(n):  # pylint: disable=invalid-name
    """ Creates and returns an AQASM QFT QRoutine of size n """
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

    try:
        gate.angle  # the angle needs to be verified before
        # in version 0.4 this changed from "_angle" to "angle"
        return gate_dic[type(gate)](gate.angle)
    except AttributeError as err:
        if gate_dic[type(gate)] in param_list:
            print(vars(gate))
            raise ValueError(f"Gate {gate} needs a param") from err
        if isinstance(gate, ops._qftgate.QFTGate):
            return QFT(targets)
        return gate_dic[type(gate)]
    except KeyError:
        print("Error " + str(gate))
        return None


# Overloading measurements


def _newbool(self):  # pylint: disable=missing-function-docstring
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

    def __init__(self, aq, engine_list=None, verbose=False):
        if engine_list is None:
            engine_list = []
        MainEngine.__init__(self, engine_list=engine_list)
        self.prog = aq.prog
        self.qb = aq.qb
        self.nbqb = aq.nbqb
        self.to_measure = aq.to_measure
        self.verbose = verbose

    def allocate_qubit(self, dirty=False):
        self.nbqb += 1
        self.qb.append(self.prog.qalloc(1))
        return MainEngine.allocate_qubit(self, dirty)

    def projectq_to_qlm(self, sep_measure=False, **kwargs):
        """
        Generates the QLM circuit corresponding to all projectq
        commands we sent to the engine

        Args:
            sep_measures: Separates measures from the
                circuit:

                 - if set to :code:`True` measures won't be included in the resulting circuits,
                   qubits to be measured will be put in a list, the resulting measureless
                   circuit and this list will be returned in a tuple : (resulting_circuit, list_qubits)
                 - if set to :code:`False`, measures will be converted normally (Default set to False)

            kwargs: these are the options that you would use on a regular
                to_circ function, these are added for more flexibility, for
                advanced users

        Returns:
            :code:`tuple` or :class:`~qat.core.Circuit`: If :code:`sep_measures` is set
            to:

             - :code:`True`: the result is a tuple composed of a
               :class:`~qat.core.Circuit` and a list of qubits that should be
               measured
             - :code:`False`: the result is a :class:`~qat.core.Circuit`
        """
        # this is only for backwards compatibility with old arch
        if sep_measure:
            circuit = self.prog.to_circ(**kwargs)
            for qreg in circuit.qregs:
                if qreg.length == 0:
                    del qreg
            return circuit, self.to_measure

        for qbit, cbit in zip(self.to_measure, self.prog.calloc(len(self.to_measure))):
            self.prog.measure(qbit, [cbit])
        circuit = self.prog.to_circ(**kwargs)
        try:
            for qreg in circuit.qregs:
                if qreg.length == 0:
                    del qreg
        except AttributeError:
            pass
        return circuit

    def to_qlm_circ(self, sep_measure=False, **kwargs):
        """ Deprecated """
        warnings.warn(
            "to_qlm_circ is deprecated, please use projectq_to_qlm",
            FutureWarning,
        )
        return self.projectq_to_qlm(sep_measure, **kwargs)


class AqasmPrinter(MainEngine):
    """
    A compiler engine which can retrieve all data sent to the stream
    then send it over to the :class:`~.AqasmEngine`
    """
    def __init__(self, engine=MainEngine, **kwargs):
        engine.__init__(self)
        self.prog = aqsm.Program(**kwargs)
        self.nbqb = 0
        self.qb = []
        self.to_measure = []

    def _out_cmd(self, cmd):
        """ Send a command to the engine """
        if isinstance(
            cmd.gate,
            (AllocateQubitGate, DeallocateQubitGate, AllocateDirtyQubitGate)
        ):
            return
        if isinstance(cmd.gate, MeasureGate):
            inp_qb = []
            for reg in cmd.qubits:
                for qbit in reg:
                    inp_qb.append(self.qb[int(str(qbit))][0])
            self.to_measure.append(inp_qb)
            # self.prog.measure(inp_qb, inp_qb)

        elif isinstance(cmd.gate, BasicGate):
            controls = cmd.all_qubits[0]
            inp_qb = []
            for reg in cmd.all_qubits:
                for qbit in reg:
                    inp_qb.append(self.qb[int(str(qbit))][0])
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
