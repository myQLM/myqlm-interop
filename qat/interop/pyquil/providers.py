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

myQLM can be used to connect to a PyQuil QVM. This module is composed of a main
class :class:`~qat.interop.pyquil.PyquilQPU` used to wrap a QVM into a myQLM QPU.

In this section, we assume a QVM is running locally and that this QVM is listenning
to the port 15011. The following code defines a myQLM QPU wrapping the PyQuil QVM:

.. code-block:: python

    from pyquil.api import QVMConnection
    from qat.interop.pyquil import PyquilQPU

    # Define port and ip
    IP   = "127.0.0.1"
    PORT = "15011"

    # Define a QPU
    qvm = QVMConnection(endpoint="http://{ip}:{port}".format(ip=IP, port=PORT))
    qpu = PyquilQPU(qvm)

    # Submit a job to the QVM
    result = qpu.submit(qlm_job)
"""

from pyquil import get_qc

from qat.interop.pyquil.converters import qlm_to_pyquil
from qat.core.qpu.qpu import QPUHandler
from qat.core.wrappers.result import State
from qat.comm.shared.ttypes import Result as QlmRes
from qat.comm.shared.ttypes import Sample as ThriftSample
from qat.comm.shared.ttypes import Job

from collections import Counter
import numpy as np

def generate_qlm_result(pyquil_result):
    """ Converts pyquil result to QLM Result

    Args:
        pyquil_result: The result object generated by pyquil
    Returns:
        A QLM Result object built from pyquil_result
    """

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
        ThriftSample(state=state,
                     probability=freq / nbshots,
                     err=np.sqrt(freq / nbshots*(1.-freq/nbshots)(nbshots-1))
                     if nbshots > 1 else None
                    )
        for state, freq in counts.items()
    ]
    return qlm_result


class PyquilQPU(QPUHandler):
    """
    QPU wrapper over pyquil, to run a QLM circuit on a pyquil
    simulator or rigetti's quantum chip

    Args:
        qpu: the instance of pyquil's simulator/connection to real
               quantum chip or simulator
        plugins: plugins to use
        compiler: if set to True(default value) the circuit will be
                    compiled by pyquil, otherwise the user compiles
                    the circuit manually and tells the pyquil qpu to
                    skip compilation
    """
    def __init__(self, qpu=None, plugins=None, compiler=True):
        super(QPUHandler, self).__init__(plugins)
        self.qpu = qpu
        self.compiler = True

    def set_qpu(self, qpu):
        self.qpu = qpu

    def submit_job(self, qlm_job):
        qlm_circuit = qlm_job.circuit
        pyquil_circuit = qlm_to_pyquil(qlm_circuit)
        if self.compiler:
            try:
                executable = self.qpu.compile(pyquil_circuit)
            except AttributeError:
                executable = pyquil_circuit
        else:
            executable = pyquil_circuit
        # qc.run_and_measure(pyquil_circuit, trials=1)
        result = generate_qlm_result(self.qpu.run(executable))
        return result

    def __submit(self, qlm_batch):
        if isinstance(qlm_batch, Job):
            return self.submit_job(qlm_batch)
        else:
            results = []

            for job in qlm_batch.jobs:
                results.append(self.submit_job(job))
            return results
