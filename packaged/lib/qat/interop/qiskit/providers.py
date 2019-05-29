from qiskit.providers import BaseBackend, BaseJob
from qiskit.providers.models.backendconfiguration import (
    BackendConfiguration,
    GateConfig,
)
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.converters import qobj_to_circuits
from qiskit.validation.base import Obj
from qiskit import execute

# QLM imports
from qat.interop.qiskit.converters import qlm_circ_sep_meas
from qat.interop.qiskit.converters import job_to_qiskit_circuit
from qat.comm.shared.ttypes import Job, Batch
from qat.comm.shared.ttypes import Result as QlmRes
from qat.core.qpu.qpu import QPUHandler
from qat.async.asyncqpu.Qiskitjob import Qiskitjob
from qat.core.wrappers.result import State
from qat.comm.shared.ttypes import Sample as ThriftSample
from collections import Counter


def to_string(state, nbqbits):
    st = bin(state)[2:]
    st = "0" * (nbqbits - len(st)) + st
    return st


def generate_qlm_result(qiskit_result):
    """
    Generates a QLM Result from a qiskit result
    """

    nbshots = qiskit_result.results[0].shots
    counts = [vars(result.data.counts) for result in qiskit_result.results]
    counts = [{int(k, 16): v for k, v in count.items()} for count in counts]

    ret = QlmRes()
    ret.raw_data = []
    for state, freq in counts[0].items():
        ret.raw_data.append(
            ThriftSample(state=State(state, qregs={}), probability=freq / nbshots)
        )
    return ret


def generate_experiment_result(qlm_result, nbqbits, head):
    """
    Generates an experiment result.
    Returns a qiskit.ExperimentResult structure.
    """
    counts = dict()
    samples = [hex(s.state) for s in qlm_result.raw_data]
    counts = dict(Counter(samples))
    data = ExperimentResultData.from_dict({"counts": counts})
    return ExperimentResult(
        shots=len(qlm_result.raw_data),
        success=True,
        data=data,
        header=Obj.from_dict(head),
    )


def generate_result(
    backend_name,
    backend_version,
    qobj_id,
    job_id,
    success,
    qlm_results,
    n_list,
    headers,
):
    """
    Tranform a QLM result into a qiskit result structure.
    Returns a qiskit.Result structure.
    """
    return Result(
        backend_name=backend_name,
        backend_version=backend_version,
        qobj_id=qobj_id,
        job_id=job_id,
        success=success,
        results=[
            generate_experiment_result(result, n, head)
            for result, n, head in zip(qlm_results, n_list, headers)
        ],
    )


class QLMJob(BaseJob):
    """
    QLM Job.
    QLM Job implement the required BaseJob interface of Qiskit with a small twist:
    everything is computed synchronously (meaning that the job is stored at submit and
    computed at result).
    """

    def add_results(self, qlm_result, qobj_id, n_list, headers):
        self._results = generate_result(
            self._backend._configuration.backend_name,
            self._backend._configuration.backend_version,
            qobj_id,
            self._job_id,
            True,
            qlm_result,
            n_list,
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


## TODO :  can be improved by publishing any pyAQASM abstract gate via its circuit implementation.
## For now lets stick with the QASM usual gate set

_QLM_GATE_NAMES = [
    "id",
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
]

_QLM_GATES = [GateConfig(name="FOO", parameters=[], qasm_def="BAR")]

_QLM_PARAMS = {
    "backend_name": "QLMConnector",  # Name of the back end
    "backend_version": "0.0.1",  # Version of the back end
    "n_qubits": 100,  # Nb qbits
    "basis_gates": _QLM_GATE_NAMES,  # We accept all the gates of Qiskit (and more, but hey..)
    "gates": _QLM_GATES,  # They don't even use it for their simulators, so...
    "local": True,  # Its a local backend
    "simulator": True,  # Its a simulator. Is it though?
    "conditional": True,  # We support conditionals
    "open_pulse": False,  # We do not support open Pulse
    "memory": False,  # We do not support Memory (wth?)
    "max_shots": 4096,
}  # Max shots is 4096 (required :/)


class NoQpuAttached(Exception):
    pass


_QLM_BACKEND = BackendConfiguration(**_QLM_PARAMS)


class QLMBackend(BaseBackend):
    """
    Basic connector implementing a Qiskit Backend, plugable on a QLM QPUHandler.
    """

    def __init__(self, configuration, provider=None):
        super(QLMBackend, self).__init__(configuration, provider)
        self.id_counter = 0
        self._qpu = None

    def set_qpu(self, qpu):
        self._qpu = qpu

    def run(self, qobj):
        if self._qpu is None:
            raise NoQpuAttached("No qpu attached to the QLM connector.")
        headers = [exp.header.as_dict() for exp in qobj.experiments]
        circuits = qobj_to_circuits(qobj)
        nbshots = qobj.config.shots
        qlm_task = Batch(jobs=[])
        n_list = []
        for circuit in circuits:
            qlm_circuit, to_measure = qlm_circ_sep_meas(circuit)
            job = qlm_circuit.to_job()
            job.nbshots = nbshots
            job.qubits = [q for q, c in to_measure]
            n_list.append(len(to_measure))
            qlm_task.jobs.append(job)

        results = self._qpu.submit(qlm_task)
        # Creating a job that will contain the results
        job = QLMJob(self, str(self.id_counter))
        self.id_counter += 1
        job.add_results(results, qobj.qobj_id, n_list, headers)
        return job


class QiskitQPU(QPUHandler):
    def __init__(self, backend=None, plugins=None):
        super(QPUHandler, self).__init__(plugins)
        self.backend = backend

    def set_backend(self, backend):
        self.backend = backend

    def submit_job(self, qlm_job):
        qiskit_circuit = job_to_qiskit_circuit(qlm_job)
        qiskit_result = execute(qiskit_circuit, self.backend).result()
        return generate_qlm_result(qiskit_result)


class AsyncQiskitQPU(QPUHandler):
    def __init__(self, backend=None, plugins=None):
        super(QPUHandler, self).__init__(plugins)
        self.backend = backend

    def set_backend(self, backend):
        self.backend = backend

    def submit_job(self, qlm_job):
        qiskit_circuit = job_to_qiskit_circuit(qlm_job)
        async_job = execute(qiskit_circuit, self.backend)
        return Qiskitjob(async_job)


class QLMConnector:
    def __or__(self, qpu):
        backend = QLMBackend(_QLM_BACKEND)
        backend.set_qpu(qpu)
        return backend


QLMConnector = QLMConnector()
