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


class PyquilQPU(QPUHandler):
    def __init__(self, qpu=None, plugins=None, compiler=True):
        super(QPUHandler, self).__init__(plugins)
        self.qpu = qpu
        self.compiler = True

    def set_qpu(self, qpu):
        self.qpu = qpu

    def submit_job(self, qlm_job):
        pyquil_circuit = to_pyquil(qlm_job)
        qc = get_qc(self.qpu)
        if compiler:
            executable = qc.compile(pyquil_circuit)
        else:
            executable = pyquil_circuit
        # qc.run_and_measure(pyquil_circuit, trials=1)
        return qc.run(executable)
