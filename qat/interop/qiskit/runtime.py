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

Qiskit Runtime QPUs can be used within myQLM by using the
:class:`~qat.interop.qiskit.runtime.QiskitRuntimeQPU` class. This class wraps both
the "Sampler" and the "Estimator" primitives, which means that this QPU can measure
either:

 - a list of qubits (i.e. sampling mode)
 - an observable (i.e. an observable)

.. code-block:: python

    from qat.interop.qiskit.runtime import QiskitRuntimeQPU

    # Wraps Qiskit QPU within a myQLM QPU
    qpu = QiskitRuntimeQPU(backend="ibmq_qasm_simulator")

    # Submit a job to this QPU
    result = qpu.submit(job)

By default, :class:`~qat.interop.qiskit.runtime.QiskitRuntimeQPU` uses the
:code:`QiskitRuntimeService` with no parameter. To execute the previous code,
your credentials must be stored on your computer

.. dropdown:: Saving Qiskit Runtime credentials
    :icon: code

    Function :code:`QiskitRuntimeService.save_account` can be used to store credentials
    on your computer. Please refer to the Qiskit documentation to get more information on
    this function

    .. code-block:: python

        from qiskit_ibm_runtime import QiskitRuntimeService

        # Define your IBM Token
        MY_IBM_TOKEN = ...

        # Save your credentials
        QiskitRuntimeService.save_account(channel="ibm_quantum", token=MY_IBM_TOKEN)
"""

# Standard import
import logging
from math import sqrt

# Qiskit imports
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Sampler

# myQLM imports
from qat.core import Result, BatchResult, Observable, Term
from qat.core.qpu import QPUHandler
from qat.comm.shared.ttypes import ProcessingType

from .converters import job_to_qiskit_circuit

MAX_SHOTS = 8096
_LOGGER = logging.getLogger("qat.interop.qiskit.runtime")


def _parse_qiskit_sampling_result(qiskit_result: dict, list_nbqbits: list):
    """
    Parse Qiskit result (returned by Sampler). This function takes in argument
    the number of qubits composing each circuit

    Args:
        qiskit_result (dict): Qiskit result
        list_nbqbits (list[int]): List of integer telling the
            number of qubits for each sample

    Returns:
        list[qat.core.Result]: list of Qiskit result
    """
    # Log Qiskit result
    _LOGGER.debug("Parsing Qiskit result: %s", repr(qiskit_result))

    # Loop over results
    for samples, metadata, nbqbits in zip(qiskit_result.quasi_dists, qiskit_result.metadata, list_nbqbits):
        # Create result
        result = Result(lsb_first=True, nbqbits=nbqbits)
        result.meta_data = {"nbshots": str(metadata["shots"])}

        # Add sample
        for state, probability in samples.items():
            result.add_sample(int(state, 2), probability=probability)

        # Yield result
        yield result


def _parse_qiskit_observable_result(qiskit_result: dict):
    """
    Parse Qiskit result (returned by Executor)

    Args:
        qiskit_result (dict): Qiskit result

    Returns:
        list[qat.core.Result]: list of Qiskit result
    """
    # Log Qiskit result
    _LOGGER.debug("Parsing Qiskit result: %s", repr(qiskit_result))

    # Put each Qiskit result in the list of myQLM results
    for value, metadata in zip(qiskit_result.values, qiskit_result.metadata):
        # Compute standard error
        error = metadata.get("variance")

        if error is not None:
            error = sqrt(error)  # Standard error = sqrt(variance)

        # Yield result
        yield Result(
            value=value, meta_data={key: str(val) for key, val in metadata.items()},
            error=error
        )


def _term_to_str(term: Term, nbqbits: int) -> str:
    """
    Cast the pauli representation as a string. For instance, a term
    composed of a X on qubit 3 will be translated into "IIXII...I"

    Args:
        term (Term): myQLM term
        nbqbits (int): number of qubits

    Returns:
        str: string representation
    """
    # Init string
    string = ["I"] * nbqbits

    # Update string
    for char, qbit in zip(term.op, term.qbits):
        string[qbit] = char

    # Return string
    return "".join(string)


def _observable_to_qiskit(observable: Observable):
    """
    Translate a myQLM observable into a Qiskit observable

    Args:
        observable (Observable): myQLM observable
    """
    # Create list of terms
    qiskit_terms = [("I" * observable.nbqbits, observable.constant_coeff)]
    qiskit_terms.extend(
        [(_term_to_str(term, observable.nbqbits), term.coeff)
         for term in observable.terms]
    )

    # Return Qiskit observable
    return SparsePauliOp.from_list(qiskit_terms)


class QiskitRuntimeQPU(QPUHandler):
    """
    IBM Q-Experience QPU. This QPU uses IBM runtime to execute
    a quantum job. This QPU wraps both the :code:`Sampler` and
    :code:`Estimator` primitives, which means that this QPU can measure
    both:

        - a list of qubits (i.e. sampling mode)
        - an observable (i.e. observable mode)

    .. warnings::

        If a batch is composed of both sampling jobs and observable jobs,
        two requests will be done to the Runtime server

    Args:
        backend_name (str): Name of the IBM backend used to execute
            submitted jobs (e.g. :code:`"ibmq_qasm_simulator"`)
        skip_transpilation (bool, optional): Skip transpilation - if set to True,
            Qiskit runtime will not transpile circuits, otherwise, Qiskit runtime
            will transpile circuits
            Default: False (transpilation done by Qiskit runtime)
        service (Runtime service, optional): Service used to connect to IBM
            runtime
            Default: QiskitRuntimeService()
    """
    def __init__(self, backend: str, skip_transpilation: bool = False, service=None):
        # Call parent constructor
        super().__init__()

        # Store backend
        self.backend = backend

        # Store transpilation flag
        self.skip_transpilation = skip_transpilation

        # Store service
        self.service = service or QiskitRuntimeService()

    def _submit_multiple_jobs(self, jobs_list: list):
        """
        Submit a list of Atos' jobs to Qiskit runtime
        These jobs will be submitted simulteanously to Qiskit Runtime

        Args:
            jobs_list (list[qat.core.Job]): list of Atos jobs

        Returns:
            list[qat.core.Result]: list of result
        """
        # Get number of shots
        nbshots = max((job.nbshots or MAX_SHOTS) for job in jobs_list)

        # Get circuits (SAMPLE and OBSERVABLE types are treated separatly) and observables
        samp_circuits = [(job_to_qiskit_circuit(job), index) for index, job in enumerate(jobs_list)
                         if job.type == ProcessingType.SAMPLE]
        nbqbits = [job.circuit.nbqbits for job in jobs_list
                   if job.type == ProcessingType.SAMPLE]

        obs_circuits = [(job_to_qiskit_circuit(job), index) for index, job in enumerate(jobs_list)
                        if job.type == ProcessingType.OBSERVABLE]
        observables = [_observable_to_qiskit(job.observable) for job in jobs_list
                       if job.type == ProcessingType.OBSERVABLE]

        # Create result list
        myqlm_results = [None] * len(jobs_list)

        # Treat SAMPLE jobs
        if samp_circuits:
            # Get circuits and indexes
            circuits, indexes = zip(*samp_circuits)

            # Create sampler
            with Sampler(circuits=circuits, skip_transpilation=self.skip_transpilation,
                         service=self.service, options={"backend": self.backend}) as qpu:
                # Send request to QPU
                indices = [idx for idx, _ in enumerate(circuits)]
                raw_result = qpu(circuit_indices=indices, shots=nbshots)

            # Put each Qiskit result in the list of myQLM results
            for index, parsed_result in zip(indexes, _parse_qiskit_sampling_result(raw_result, nbqbits)):
                myqlm_results[index] = parsed_result

        # Treat OBSERVABLE jobs
        if obs_circuits:
            # Get circuits and indexes
            circuits, indexes = zip(*obs_circuits)

            # Create estimator
            with Estimator(circuits=circuits, observables=observables, skip_transpilation=self.skip_transpilation,
                           service=self.service, options={"backend": self.backend}) as qpu:
                # Send request to QPU
                indices = [idx for idx, _ in enumerate(circuits)]
                raw_result = qpu(circuit_indices=indices, observable_indices=indices, shots=nbshots)

            # Put each Qiskit result in the list of myQLM results
            for index, parsed_result in zip(indexes, _parse_qiskit_observable_result(raw_result)):
                myqlm_results[index] = parsed_result

        # Return list of myQLM results
        return myqlm_results

    def _submit_batch(self, batch):
        """
        Submits a batch to Qiskit runtime. This function use internally
        method submit_multiple_jobs

        Args:
            batch (qat.core.Batch): QLM batch

        Returns:
            qat.core.BatchResult: Batch result
        """
        # Get list of results
        results = self._submit_multiple_jobs(batch.jobs)

        # Return wrapped result
        return BatchResult(results=results, meta_data=batch.meta_data)

    def submit_job(self, job):
        """
        Submit a single job to Qiskit runtime

        Args:
            job (qat.core.Job): QLM job

        Returns:
            qat.core.Result: QLM result
        """
        # Get list of results
        results = self._submit_multiple_jobs([job])

        # Return first item of the list
        return results[0]
