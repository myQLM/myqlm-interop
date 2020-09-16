#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
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
"""

import cirq
from qat.interop.cirq.converters import qlm_to_cirq

from qat.core.qpu.qpu import QPUHandler
from qat.core.wrappers.result import Result as QlmRes, State, Sample
from qat.comm.shared.ttypes import Job

from collections import Counter
import numpy as np


def generate_qlm_result(cirq_result):
    """ Converts cirq result to QLM Result

    Args:
        cirq_result: The result object generated by cirq
    Returns:
        A QLM Result object built from cirq_result
    """
    # Cirq encodes measures in a dictionary where the key is the coordinate of
    # the qubit, and the value is a list where the ith element is the result
    # of the measure of the qubit during the ith trial.

    nbshots = len(cirq_result.measurements[next(iter(cirq_result.measurements))])
    measurements = ["" for _ in range(nbshots)]
    for entry in cirq_result.measurements.values():
        for shot in range(len(entry)):
            measurements[shot] += str(int(entry[shot][0]))

    measurements = [ int(_, 2) for _ in measurements]
    counts = Counter(measurements)
    qlm_result = QlmRes()
    qlm_result.raw_data = [
        Sample(state=state,
               probability=freq / nbshots,
               err=np.sqrt(freq/nbshots *(1.-freq/nbshots)/(nbshots-1))
               if nbshots > 1 else None
              )
        for state, freq in counts.items()
    ]
    return qlm_result


class CirqQPU(QPUHandler):
    def __init__(self, qpu=cirq.Simulator(), plugins=None):
        super(QPUHandler, self).__init__(plugins)
        self.qpu = qpu

    def set_qpu(self, qpu):
        self.qpu = qpu

    def submit_job(self, qlm_job):
        qlm_circuit = qlm_job.circuit
        nbshots = qlm_job.nbshots
        cirq_circuit = qlm_to_cirq(qlm_circuit)
        result = generate_qlm_result(self.qpu.run(cirq_circuit,
                                                  repetitions=nbshots))
        return result
