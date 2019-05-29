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
from pyquil import get_qc

from qat.interop.pyquil.converters import to_pyquil
from qat.core.qpu.qpu import QPUHandler
from qat.core.wrappers.result import State
from qat.comm.shared.ttypes import Result as QlmRes
from qat.comm.shared.ttypes import Sample as ThriftSample

from collections import Counter


def generate_qlm_result(pyquil_result):
    """ Converts pyquil result to QLM Result """

    # Pyquil encodes measures in a matrix, where line i is the measures
    # for trial i, and column j contains the measurements for qubit j

    # Build a list of states

    nbshots = len(pyquil_result)
    measurements = [
        sum([b << i for i, b in enumerate(entry)]) for entry in pyquil_result
    ]

    counts = Counter(measurements)
    qlm_result = QlmRes()
    qlm_result.raw_data = [
        ThriftSample(state=State(state, qregs={}), probability=freq / nbshots)
        for state, freq in counts.items()
    ]
    return qlm_result


class PyquilQPU(QPUHandler):
    def __init__(self, qpu=None, plugins=None, compiler=True):
        super(QPUHandler, self).__init__(plugins)
        self.qpu = qpu
        self.compiler = True

    def set_qpu(self, qpu):
        self.qpu = qpu

    def submit_job(self, qlm_job):
        pyquil_circuit = to_pyquil(qlm_job)
        if self.compiler:
            executable = self.qpu.compile(pyquil_circuit)
        else:
            executable = pyquil_circuit
        # qc.run_and_measure(pyquil_circuit, trials=1)
        return generate_qlm_result(self.qpu.run(executable))
