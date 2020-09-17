#!/usr/bind/env python3.6
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


myQLM can be used to connect to a Qiskit Backend. This module is composed of
three main classes:

 - :class:`~qat.interop.qiskit.BackendToQPU`: Synchronous QPU,
   capable of running in a Qiskit backend

   .. code-block:: python

       from qat.interop.qiskit import BackendToQPU

       # Declare your IBM token
       MY_IBM_TOKEN = "..."

       # Wrap a Qiskit backend in a QPU
       qpu = BackendToQPU(token=MY_IBM_TOKEN, ibmq_backend="ibmq_armonk")

       # Submit a job to IBMQ
       result = qpu.submit(job)

 - :class:`~qat.interop.qiskit.AsyncBackendToQPU`: Asynchronous QPU,
   capable of running in a Qiskit Backend. This QPU returns instances
   of :class:`~qat.interop.qiskit.QiskitJob`

   .. code-block:: python

       import time
       from qat.interop.qiskit import AsyncBackendToQPU

       # Declare your IBM token
       MY_IBM_TOKEN = "..."

       # Wrap a Qiskit backend in a QPU
       async_qpu = AsyncBackendToQPU(token=MY_IBM_TOKEN, ibmq_backend="ibmq_armonk")

       # Submit a job to IBMQ
       async_result = async_qpu.submit(job)

       # Wait for the result
       while not async_result.result():
           time.sleep(1)

       # Get result
       result = async_result.result()

 - :class:`~qat.interop.qiskit.QPUToBackend`: Qiskit backend,
   capable of running in a QLM QPU

   .. code-block:: python

       from qat.qpus import PyLinalg
       from qat.interop.qiskit import QPUToBackend
       from qiskit import execute

       # Creates a Qiskit Backend
       qpu = PyLinalg()
       backend = QPUToBackend(qpu)

       # Returns a qiskit result
       qiskit_result = execute(qiskit_circuit, backend, shots=15).result()
"""

import os
from collections import Counter
import warnings
import numpy as np

from qiskit.providers import BaseBackend, BaseJob
from qiskit.providers.models.backendconfiguration import (
    BackendConfiguration,
    GateConfig,
)
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.assembler import disassemble
from qiskit.validation.base import Obj
from qiskit import execute, Aer, IBMQ

# QLM imports
from qat.interop.qiskit.converters import qiskit_to_qlm
from qat.interop.qiskit.converters import job_to_qiskit_circuit
from qat.comm.datamodel.ttypes import QRegister
from qat.comm.shared.ttypes import Job
from qat.comm.shared.ttypes import Result as QlmRes
from qat.core.qpu.qpu import QPUHandler, get_registers
from qat.core import Batch
from qat.core.wrappers.result import aggregate_data
from qat.core.wrappers.result import Result as WResult, BatchResult, Sample


def to_string(state, nbqbits):
    """
    Converts a state into a string.

    Args:
        state: Int representing the quantum state written in decimal
        nbqbits: Number of qubits of the quantum state

    Returns:
        String of the quantum state in binary form
    """
    state_str = bin(state)[2:]
    state_str = "0" * (nbqbits - len(state_str)) + state_str
    return state_str


def generate_qlm_result(qiskit_result):
    """
    Generates a QLM Result from a Qiskit result.

    Args:
        qiskit_result: The qiskit Result object to convert

    Returns:
        A QLM Result object built from the data in qiskit_result
    """

    nbshots = qiskit_result.results[0].shots
    try:
        counts = [vars(result.data.counts) for result in qiskit_result.results]
    except AttributeError:
        print("No measures, so the result is empty")
        return QlmRes(raw_data=[])
    counts = [{int(k, 16): v for k, v in count.items()} for count in counts]
    ret = QlmRes(raw_data=[])
    for state, freq in counts[0].items():
        if not isinstance(state, int):
            print("State is {}".format(type(state)))
        ret.raw_data.append(
            Sample(state=state,
                   probability=freq / nbshots,
                   err=np.sqrt(
                       freq / nbshots*(1.-freq/nbshots)/(nbshots-1))
                   if nbshots > 1 else None)
        )
    return ret


def generate_qlm_list_results(qiskit_result):
    """
    Generates a QLM Result from a qiskit result.

    Args:
        qiskit_result: The qiskit.Result object to convert

    Returns:
        A QLM Result object built from the data in qiskit_result
    """

    nbshots = qiskit_result.results[0].shots
    try:
        counts = [vars(result.data.counts) for result in qiskit_result.results]
    except AttributeError:
        print("No measures, so the result is empty")
        return QlmRes(raw_data=[])
    counts = [{int(k, 16): v for k, v in count.items()} for count in counts]
    ret_list = []
    for count in counts:
        ret = QlmRes(raw_data=[])
        for state, freq in count.items():
            if not isinstance(state, int):
                print("State is {}".format(type(state)))
            ret.raw_data.append(
                Sample(state=state,
                       probability=freq / nbshots,
                       err=np.sqrt(
                           freq / nbshots*(1.-freq/nbshots)/(nbshots-1))
                       if nbshots > 1 else None)
            )
        ret_list.append(ret)
    return ret_list


def _generate_experiment_result(qlm_result, head):
    """
    Generates a Qiskit experiment result.

    Args:
        qlm_result: qat.core.wrappers.Result object which data is aggregated
        head: Header of the experiment

    Returns:
        An ExperimentResult structure.
    """
    samples = [hex(s.state.state) for s in qlm_result.raw_data]
    counts = dict(Counter(samples))
    data = ExperimentResultData.from_dict({"counts": counts})
    return ExperimentResult(
        shots=len(qlm_result.raw_data),
        success=True,
        data=data,
        header=Obj.from_dict(head),
    )


def _qlm_to_qiskit_result(
        backend_name,
        backend_version,
        qobj_id,
        job_id,
        success,
        qlm_results,
        headers
        ):
    """
    Tranform a QLM result into a Qiskit result structure.

    Args:
        backend_name:
        backend_version:
        qobj_id:
        job_id:
        success:
        qlm_results: List of qat.core.wrappers.Result objects
        headers: List of the experiments' headers

    Returns:
        A qiskit Result structure.
    """
    return Result(
        backend_name=backend_name,
        backend_version=backend_version,
        qobj_id=qobj_id,
        job_id=job_id,
        success=success,
        results=[
            _generate_experiment_result(result, head)
            for result, head in zip(qlm_results, headers)
        ],
    )


class QLMJob(BaseJob):
    """
    QLM Job implement the required BaseJob interface of Qiskit with a
    small twist: everything is computed synchronously (meaning that the
    job is stored at submit and computed at result).
    """

    def set_results(self, qlm_result, qobj_id, headers):
        """
        Sets the results of the Job.

        Args:
            qlm_result: :class:`~qat.core.wrappers.result.Result` object
            qobj_id: Identifier of the initial Qobj structure
            headers: List of the experiments' headers, gotten from
                    the initial Qobj structure's experiments
        """
        self._results = _qlm_to_qiskit_result(
            self._backend._configuration.backend_name,
            self._backend._configuration.backend_version,
            qobj_id,
            self._job_id,
            True,
            qlm_result,
            headers,
        )

    def status(self):
        pass

    def cancel(self):
        pass

    def submit(self):
        pass

    def result(self):
        return self._results


_QLM_GATE_NAMES = [
    "id",
    "iden",
    "u0",
    "u1",
    "u2",
    "u3",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "cx",
    "cy",
    "cz",
    "ch",
    "crz",
    "cu1",
    "cu3",
    "swap",
    "ccx",
    "cswap",
    "r",
]

_QLM_GATES = [GateConfig(name="FOO", parameters=[], qasm_def="BAR")]

_QLM_PARAMS = {
    "backend_name": "QiskitConnector",  # Name of the back end
    "backend_version": "0.0.1",  # Version of the back end
    "n_qubits": 100,  # Nb qbits
    "basis_gates": _QLM_GATE_NAMES,  # We accept all the gates of Qiskit
    "gates": _QLM_GATES,  # They don't even use it for their simulators, so...
    "local": True,  # Its a local backend
    "simulator": True,  # Its a simulator. Is it though?
    "conditional": True,  # We support conditionals
    "open_pulse": False,  # We do not support open Pulse
    "memory": False,  # We do not support Memory (wth?)
    "max_shots": 4096,
    "coupling_map": None,
}  # Max shots is 4096 (required :/)


class NoQpuAttached(Exception):
    """
    Exception raised in QPUToBackend.run() when there is not qpu attached to it
    """


_QLM_BACKEND = BackendConfiguration(**_QLM_PARAMS)


class QPUToBackend(BaseBackend):
    """
    Basic connector implementing a Qiskit Backend, plugable on a QLM QPU.

    Parameters:
        qpu: :class:`~qat.core.qpu.QPUHandler` object
        configuration: BackendConfiguration object, leave default value for
                standard uses
        provider: Provider responsible for this backend
    """

    def __init__(self, qpu=None, configuration=_QLM_BACKEND, provider=None):
        """
        Args:
            qpu: :class:`~qat.core.qpu.QPUHandler` object
            configuration: BackendConfiguration object, leave default value for
                    standard uses
            provider: Provider responsible for this backend
        """
        super(QPUToBackend, self).__init__(configuration, provider)
        self.id_counter = 0
        self._qpu = qpu

    def set_qpu(self, qpu):
        """
        Sets the QLM QPU that this backend is supposed to use.

        Args:
            qpu: QLM QPU object
        """
        self._qpu = qpu

    def run(self, qobj):
        """ Convert all the circuits inside qobj into a Batch of
            QLM jobs before sending them into a QLM qpu.

        Args:
            qobj: Qiskit batch of circuits to run

        Returns:
            Returns a :class:`~qat.interop.qiskit.QLMJob` object containing
            the results of the QLM qpu execution after being converted into
            Qiskit results
        """
        if self._qpu is None:
            raise NoQpuAttached("No qpu attached to the QLM connector.")
        headers = [exp.header.to_dict() for exp in qobj.experiments]
        circuits = disassemble(qobj)[0]
        nbshots = qobj.config.shots
        qlm_task = Batch(jobs=[])
        for circuit in circuits:
            qlm_circuit = qiskit_to_qlm(circuit)
            job = qlm_circuit.to_job(aggregate_data=False)
            job.nbshots = nbshots
            job.qubits = list(range(0, qlm_circuit.nbqbits))
            qlm_task.jobs.append(job)

        results = self._qpu.submit(qlm_task)
        for res in results:
            for sample in res.raw_data:
                sample.intermediate_measures = None
            res = aggregate_data(res)

        # Creating a job that will contain the results
        job = QLMJob(self, str(self.id_counter))
        self.id_counter += 1
        job.set_results(results, qobj.qobj_id, headers)
        return job


class BackendToQPU(QPUHandler):
    """
    Wrapper around any Qiskit simulator / quantum chip connection.
    Despite the asynchronous nature of Qiskit's backends, this class
    defines a synchronous QPU. If you need an asynchronous, please use
    :class:`~qat.interop.qiskit.AsyncBackendToQPU`.

    This QPU can be instantiated
    using:

     - a Qiskit backend: please use the keyword argument :code:`backend`
     - an IBM token and the name of the backend: please the keyword arguments
       :code:`token` and :code:`ibmq_backend` (the default backend is
       :code:`"ibmq_qasm_simulator"`)
     - *no argument*: the :code:`"qasm_simulator"` is used if no argment is specified

    Args:
        backend: The Backend Qiskit object that is supposed to execute
            the circuit.
        plugins (list): linked plugins
        token (str): Qiskit IBMQ login token. If not supplied, loaded from the environment
            variable :code:`QISKIT_TOKEN`
        ibmq_backend (str, optional): name of the backend. Defaults to 'ibmq_qasm_simulator'.
    """
    def __init__(self, backend=None, plugins=None, token=None,
                 ibmq_backend='ibmq_qasm_simulator'):
        """
        Args:
            backend: The Backend Qiskit object to be wrapped
            plugins: Any plugins to be added (c.f qat.core documentation)
            token: Qiskit IBMQ login token. If not supplied, loaded from env
                    variable QISKIT_TOKEN. Only used if backend is None.
            ibmq_backend: Name of the IBM Quantum Experience backend, default
                    value is 'ibmq_qasm_simulator', which goes up to 32qubits
        """
        super().__init__(plugins)
        self.set_backend(backend, token, ibmq_backend)

    def set_backend(self, backend=None, token=None,
                    ibmq_backend='ibmq_qasm_simulator'):
        """
        Sets the backend that will execute circuits.

        Args:
            backend: The Backend Qiskit object to be wrapped
            plugins: Any plugins to be added (c.f qat.core documentation)
            token: Qiskit IBMQ login token. If not supplied, loaded from env
                    variable QISKIT_TOKEN. Only used if backend is None.
            ibmq_backend: Name of the IBM Quantum Experience backend, default
                    value is 'ibmq_qasm_simulator', which goes up to 32qubits
        """
        if backend is None:
            if token is None:
                token = os.getenv("QISKIT_TOKEN")
            if token is not None:
                if 'token' not in IBMQ.stored_account().keys() or \
                        IBMQ.stored_account()['token'] != token:
                    IBMQ.save_account(token, overwrite=True)

                provider = IBMQ.load_account()
                self.backend = provider.get_backend(ibmq_backend)
            else:
                self.backend = Aer.get_backend("qasm_simulator")
        else:
            self.backend = backend

    def _submit_batch(self, qlm_batch):
        """
        Submits a Batch object to execute on a Qiskit backend.

        Args:
            qlm_batch:

        Returns:
            A QLM BatchResult object
        """
        if self.backend is None:
            raise ValueError("Backend cannot be None")

        if isinstance(qlm_batch, Job):
            qlm_batch = Batch(jobs=[qlm_batch])
        qiskit_circuits = []
        for qlm_job in qlm_batch.jobs:
            qiskit_circuit = job_to_qiskit_circuit(qlm_job)
            qiskit_circuits.append(qiskit_circuit)
        qiskit_result = execute(
            qiskit_circuits, self.backend,
            shots=qlm_batch.jobs[0].nbshots or self.backend.configuration().max_shots,
            coupling_map=None).result()
        results = generate_qlm_list_results(qiskit_result)
        new_results = []
        for result in results:
            new_results.append(WResult.from_thrift(result))
        return _wrap_results(qlm_batch, new_results)

    def submit_job(self, qlm_job):
        """
        Submits a Job to execute on a Qiskit backend.

        Args:
            qlm_job: :class:`~qat.core.Job` object

        Returns:
            :class:`~qat.core.Result` object
        """
        if self.backend is None:
            raise ValueError("Backend cannot be None")

        qiskit_circuit = job_to_qiskit_circuit(qlm_job)
        qiskit_result = execute(
            qiskit_circuit, self.backend,
            shots=qlm_job.nbshots or self.backend.configuration().max_shots,
            coupling_map=None).result()
        result = generate_qlm_result(qiskit_result)
        return result


def _wrap_results(qlm_batch, results):
    """
    Wrap a Result structure using the corresponding Job's information
    This is mainly to provide a cleaner/higher level interface for the user

    Args:
        qlm_batch: QLM Batch which results are to be wrapped
        results: list of Result object to be wrapped

    Returns:
        :class:`~qat.core.Result` or :class:`~qat.core.BatchResult` object if the batch submitted
        contains several jobs
    """
    for i in range(len(qlm_batch.jobs)):
        qlm_job = qlm_batch.jobs[i]
        result = results[i]
        qreg_list = None
        if qlm_job.circuit is not None:
            qreg_list = get_registers(qlm_job.circuit.qregs, qlm_job.qubits)
        if qreg_list is not None:
            result.wrap_samples(qreg_list)
        else:
            length = 0
            if qlm_job.qubits is not None:
                length = len(qlm_job.qubits)
            result.wrap_samples([QRegister(start=0, length=length, type=1)])

    return BatchResult(results=results, meta_data=qlm_batch.meta_data)


class QiskitJob:
    """
    Wrapper around Qiskit's asynchronous jobs.
    """
    def __init__(self, qlm_batch, async_job):
        """
        Args:
            qlm_batch: :class:`~qat.core.Batch` or :class:`~qat.core.Job` object.
                    If a QLM Job object is given, it will be converted in a QLM
                    Batch object
            async_job: Qiskit job instance derived from BaseJob.
                    Result of a previous asynchronous execution of qlm_batch
        """
        self._job_id = async_job.job_id()
        self._handler = async_job
        if isinstance(qlm_batch, Job):
            self._qlm_batch = Batch(jobs=[qlm_batch])
        else:
            self._qlm_batch = qlm_batch

    def job_id(self):
        """ Returns the job's ID. """
        return self._job_id

    def status(self):
        """ Returns the job status. """
        return self._handler.status()._name_

    def result(self):
        """
        Returns the result if available.

        Returns:
            :class:`~qat.core.Result` object or
            :class:`~qat.core.BatchResult` object
            if the batch submitted contains several jobs
        """
        if self.status() == 'DONE':
            results = generate_qlm_list_results(self._handler.result())
            new_results = []
            for result in results:
                new_results.append(WResult.from_thrift(result))
            batch_result = _wrap_results(self._qlm_batch, new_results)
            if not batch_result.results or len(batch_result.results) == 1:
                return batch_result.results[0]
            return batch_result

        return None

    def cancel(self):
        """
        Attempts to cancel the job.

        Returns:
            Boolean indicating whether the attempt was successful or not
        """
        ret = self._handler.cancel()

        if ret:
            print("job successefully cancelled")
            return True

        print("Unable to cancel job")
        return False

    def dump(self, file_name):
        """
        Dumps the :class:`~qat.core.Batch` object used for creating the job into a
        binary file. This file should later be used with AsyncBackendToQPU's
        :func:`~qat.interop.qiskit.AsyncBackendToQPU.retrieve_job`.

        Args:
            file_name: Name of the binary file to create
        """
        if isinstance(self._qlm_batch.meta_data, dict):
            self._qlm_batch.meta_data['job_id'] = self._job_id
        else:
            self._qlm_batch.meta_data = {'job_id': self._job_id}

        self._qlm_batch.dump(file_name)


class AsyncBackendToQPU(QPUHandler):
    """
    Wrapper around any Qiskit simulator / quantum chip connection.
    This class defines an asynchronous QPU. If you need a synchronous QPU, please use
    :class:`~qat.interop.qiskit.BackendToQPU`.

    This asynchronous QPU can be instantiated using:

     - a Qiskit backend: please use the keyword argument :code:`backend`
     - an IBM token and the name of the backend: please the keyword arguments
       :code:`token` and :code:`ibmq_backend` (the default backend is
       :code:`"ibmq_qasm_simulator"`)
     - *no argument*: the :code:`"qasm_simulator"` is used if no argment is specified

    .. warning::

        Since this QPU is asynchronous, plugins can't be piped to this QPU

    Args:
        backend: The Backend Qiskit object that is supposed to execute
            the circuit.
        token (str): Qiskit IBMQ login token. If not supplied, loaded from the environment
            variable :code:`QISKIT_TOKEN`
        ibmq_backend (str): name of the backend. Defaults to 'ibmq_qasm_simulator'.
    """
    def __init__(self, backend=None, token=None,
                 ibmq_backend='ibmq_qasm_simulator'):
        """
        Args:
            backend: The Backend Qiskit object to be wrapped
            token: Qiskit IBMQ login token. If not supplied, loaded from env
                    variable QISKIT_TOKEN. Only used if backend is None
            ibmq_backend: Name of the IBM Quantum Experience backend, default
                    value is 'ibmq_qasm_simulator', which goes up to 32qubits
        """
        super().__init__()
        self.set_backend(backend, token, ibmq_backend)

    def set_backend(self, backend=None, token=None,
                    ibmq_backend='ibmq_qasm_simulator'):
        """
        Sets the backend that will execute circuits.
        If no backend and no token are specified, the backend  will be
        a simulator.

        Args:
            backend: The Backend Qiskit object to be wrapped
            token: Qiskit IBMQ login token. If not supplied, loaded from env
                    variable QISKIT_TOKEN. Only used if backend is None
            ibmq_backend: Name of the IBM Quantum Experience backend, default
                    value is 'ibmq_qasm_simulator', which goes up to 32qubits
        """
        if backend is None:
            if token is None:
                token = os.getenv("QISKIT_TOKEN")
            if token is not None:
                if 'token' not in IBMQ.stored_account().keys() or \
                        IBMQ.stored_account()['token'] != token:
                    IBMQ.save_account(token, overwrite=True)

                provider = IBMQ.load_account()
                self.backend = provider.get_backend(ibmq_backend)
            else:
                self.backend = Aer.get_backend("qasm_simulator")
        else:
            self.backend = backend

    def submit_job(self, qlm_job):
        """
        Submits a QLM job to be executed on the previously
        selected backend, if no backends are chosen an exception is raised.

        Args:
            qlm_job: :class:`~qat.core.Job` object to be executed

        Returns:
            A :class:`~qat.interop.qiskit.QiskitJob` object with the same
            interface as a job derived from BaseJob for the user to have
            information on their job execution
        """
        if self.backend is None:
            raise ValueError("Backend cannot be None")

        qiskit_circuit = job_to_qiskit_circuit(qlm_job)
        async_job = execute(
            qiskit_circuit, self.backend,
            shots=qlm_job.nbshots or self.backend.configuration().max_shots,
            coupling_map=None)
        return QiskitJob(qlm_job, async_job)

    def submit(self, qlm_batch):
        """
        Submits a QLM batch of jobs and returns the corresponding QiskitJob.

        Args:
            qlm_batch: :class:`~qat.core.Batch` or :class:`~qat.core.Job`.
                    If a single job is provided, a batch is created
                    from this job.
        Returns:
            :class:`~qat.interop.qiskit.QiskitJob` object with the same
            interface as a job derived from BaseJob for the user to have
            information on their job execution
        """
        if self.backend is None:
            raise ValueError("Backend cannot be None")

        if isinstance(qlm_batch, Job):
            qlm_batch = Batch(jobs=[qlm_batch])
        qiskit_circuits = []
        for qlm_job in qlm_batch.jobs:
            qiskit_circuit = job_to_qiskit_circuit(qlm_job)
            qiskit_circuits.append(qiskit_circuit)
        async_job = execute(
            qiskit_circuits, self.backend,
            shots=qlm_batch.jobs[0].nbshots or self.backend.configuration().max_shots,
            coupling_map=None)
        return QiskitJob(qlm_batch, async_job)

    def retrieve_job(self, file_name):
        """
        Retrieves a QiskitJob from a binary file in which the QLM Batch object
        - from which the QiskitJob has been created - has been dumped.

        Args:
            file_name: Name of the binary file

        Returns:
            :class:`~qat.interop.qiskit.QiskitJob` object
        """
        qlm_batch = Batch.load(file_name)
        async_job = self.backend.retrieve_job(qlm_batch.meta_data['job_id'])
        return QiskitJob(qlm_batch, async_job)


class QiskitConnector:
    """
    Component connecting a QPU to Qiskit by returning a QPUToBackend.
    """
    def __or__(self, qpu):
        """
        Converts a QPU to a QPUToBackend object. The syntax is similar to
        plugins' but the type is changed by the operation.

        Args:
            qpu:

        Return:
            QPUToBackend object
        """
        backend = QPUToBackend(_QLM_BACKEND)
        backend.set_qpu(qpu)
        return backend


class Qiskitjob(QiskitJob):
    """ Deprecated, use QiskitJob."""
    def __init__(self, qlm_job, qobj):
        warnings.warn(
            "Qiskitjob is deprecated, please use QiskitJob.",
            FutureWarning,
        )
        super(Qiskitjob, self).__init__(qlm_job, qobj)


class QLMBackend(QPUToBackend):
    """
    Deprecated, use QPUToBackend.
    """
    def __init__(self, qpu=None, configuration=_QLM_BACKEND, provider=None):
        warnings.warn(
            "QLMBackend is deprecated, please use QPUToBackend.",
            FutureWarning,
        )
        super(QLMBackend, self).__init__(
            qpu=qpu, configuration=configuration, provider=provider)


class QiskitQPU(BackendToQPU):
    """
    Deprecated, use BackendToQPU.
    """
    def __init__(self, backend=None, plugins=None, token=None, url=None,
                 ibmq_backend='ibmq_qasm_simulator'):
        warnings.warn(
            "QiskitQPU(backend=None, plugins=None, token=None, url=None) "
            + "is deprecated, please use BackendToQPU(backend=None, "
            + "plugins=None, token=None, ibmq_backend='ibmq_qasm_simulator')",
            FutureWarning,
        )
        del url
        super(QiskitQPU, self).__init__(backend=backend, plugins=plugins,
                                        token=token, ibmq_backend=ibmq_backend)


class AsyncQiskitQPU(AsyncBackendToQPU):
    """
    Deprecated, use AsyncBackendToQPU.
    """
    def __init__(self, backend=None, plugins=None, token=None, url=None,
                 ibmq_backend='ibmq_qasm_simulator'):
        warnings.warn(
            "AsyncQiskitQPU(backend=None, plugins=None, token=None, url=None) "
            + "is deprecated, please use AsyncBackendToQPU(backend=None, "
            + "token=None, ibmq_backend='ibmq_qasm_simulator')",
            FutureWarning,
        )
        del plugins, url
        super(AsyncQiskitQPU, self).__init__(backend=backend, token=token,
                                             ibmq_backend=ibmq_backend)


class QLMConnector(QiskitConnector):
    """
    Deprecated, use QiskitConnector.
    """
    def __or__(self, qpu):
        warnings.warn(
            "QLMConnector is deprecated, please use QiskitConnector()",
            FutureWarning,
        )
        return super().__or__(qpu)


QLMConnector = QLMConnector()
