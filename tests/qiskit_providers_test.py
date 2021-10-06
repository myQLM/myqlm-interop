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

"""
Tests for qat.interop.qiskit.providers.py
"""

import time
import unittest
import logging
import os
from qiskit import QuantumRegister, QuantumCircuit
from qiskit import ClassicalRegister, execute, Aer
from qat.lang.AQASM import Program, H, CNOT
from qat.pylinalg import PyLinalg
from qat.core.wrappers import Batch
from qat.interop.qiskit import BackendToQPU, AsyncBackendToQPU, \
        QiskitConnector, QPUToBackend

LOGGER = logging.getLogger()
# Set level to logging.DEBUG in order to see more information
LOGGER.setLevel(logging.WARNING)

# redirects log writing to terminal
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.DEBUG)
LOGGER.addHandler(STREAM_HANDLER)


class Test0BackendToQPU(unittest.TestCase):
    """
    Creates a default BackendToQPU.
    It is going to use qiskit's "quasm_simulator" if it cannot connect
    to IBMQ using QISKIT_TOKEN environment variable.
    Runs a QLM circuit and checks for relevent results.
    """

    def test0_qiskit_qpu_2states(self):
        """
        In the case of a H and a CNOT gate, 2 states are expected in output.
        """
        nbqubits = 2
        prog = Program()

        qreg = prog.qalloc(nbqubits)
        creg = prog.calloc(nbqubits)

        prog.apply(H, qreg[0])
        prog.apply(CNOT, qreg[0], qreg[1])

        prog.measure(qreg, creg)
        qlm_circuit = prog.to_circ()

        qlm_job1 = qlm_circuit.to_job(nbshots=1024)
        qlm_job2 = qlm_circuit.to_job(nbshots=1024)
        batch = Batch(jobs=[qlm_job1, qlm_job2])

        # a backend is specified
        backend = Aer.get_backend('aer_simulator')
        qpu = BackendToQPU(backend)

        results = qpu.submit(batch)
        string = "\nBackendToQPU test with a Hadamard and a CNOT " \
            + "(expects two different measured states):"
        LOGGER.debug(string)
        for result in results.results:
            for entry in result.raw_data:
                LOGGER.debug("State: %s\t probability: %s",
                             entry.state, entry.probability)
            self.assertEqual(2, len(result.raw_data))
            self.assertTrue("|00>" in
                            [str(result.raw_data[i].state) for i in range(2)])
            self.assertTrue("|11>" in
                            [str(result.raw_data[i].state) for i in range(2)])

    def test1_qiskit_qpu_4states(self):
        """
        In the case of two H gates, 4 states are expected in output.
        """
        nbqubits = 2
        prog = Program()

        qreg = prog.qalloc(nbqubits)
        creg = prog.calloc(nbqubits)

        prog.apply(H, qreg[0])
        prog.apply(H, qreg[1])

        prog.measure(qreg, creg)
        qlm_circuit = prog.to_circ()

        qlm_job = qlm_circuit.to_job(nbshots=1024)

        # no backend is specified
        qpu = BackendToQPU()

        result = qpu.submit(qlm_job)
        string = "\nBackendToQPU with a Hadamard on each qubit " \
            + "(expects four different measured states):"
        LOGGER.debug(string)
        for entry in result.raw_data:
            LOGGER.debug("State: %s\t probability: %s",
                         entry.state, entry.probability)

        self.assertEqual(4, len(result.raw_data))


class Test1AsyncBackendToQPU(unittest.TestCase):
    """
    Creates a default AsyncBackendToQPU.
    It is going to use qiskit's "quasm_simulator" if it cannot connect
    to IBMQ using QISKIT_TOKEN environment variable.
    Runs a QLM circuit and checks for relevent results.
    """

    def test0_asyncqiskit_qpu_2states(self):
        """
        In the case of a H and a CNOT gate, 2 states are expected in output.
        """
        nbqubits = 2

        prog = Program()

        qreg = prog.qalloc(nbqubits)
        creg = prog.calloc(nbqubits)

        prog.apply(H, qreg[0])
        prog.apply(CNOT, qreg[0], qreg[1])

        prog.measure(qreg, creg)
        qlm_circuit = prog.to_circ()

        qlm_job1 = qlm_circuit.to_job(nbshots=1024)
        qlm_job2 = qlm_circuit.to_job(nbshots=1024)
        batch = Batch(jobs=[qlm_job1, qlm_job2])

        # a backend is specified
        backend = Aer.get_backend('aer_simulator')
        qpu = AsyncBackendToQPU(backend)

        job = qpu.submit(batch)
        string = "\nAsyncBackendToQPU test with a Hadamard and a CNOT " \
            + "(expects two different measured states):"
        LOGGER.debug(string)
        LOGGER.debug("ID: %s\t status : %s", job.job_id(), job.status())
        time.sleep(0.01)
        LOGGER.debug("ID: %s\t status : %s", job.job_id(), job.status())
        time.sleep(0.2)
        LOGGER.debug("ID: %s\t status : %s", job.job_id(), job.status())

        while job.result() is None:
            time.sleep(5)

        results = job.result()

        for result in results:
            for entry in result.raw_data:
                LOGGER.debug("State: %s\t probability: %s",
                             entry.state, entry.probability)
                self.assertEqual(2, len(result.raw_data))
                self.assertTrue("|00>" in [str(result.raw_data[i].state)
                                           for i in range(2)])
                self.assertTrue("|11>" in [str(result.raw_data[i].state)
                                           for i in range(2)])

    def test1_asyncqiskit_qpu_4states(self):
        """
        In the case of two H gates, 4 states are expected in output.
        """
        nbqubits = 2

        prog = Program()

        qreg = prog.qalloc(nbqubits)
        creg = prog.calloc(nbqubits)

        prog.apply(H, qreg[0])
        prog.apply(H, qreg[1])

        prog.measure(qreg, creg)
        qlm_circuit = prog.to_circ()

        qlm_job = qlm_circuit.to_job(nbshots=1024)

        # no backend is specified
        qpu = AsyncBackendToQPU()

        async_job = qpu.submit_job(qlm_job)

        string = "\nAsyncBackendToQPU test with a Hadamard on each qubit " \
            + "(expects four different measured states):"
        LOGGER.debug(string)
        LOGGER.debug("ID: %s\t status : %s",
                     async_job.job_id(), async_job.status())
        time.sleep(0.01)
        LOGGER.debug("ID: %s\t status : %s",
                     async_job.job_id(), async_job.status())
        time.sleep(0.2)
        LOGGER.debug("ID: %s\t status : %s",
                     async_job.job_id(), async_job.status())

        loop_nb = 0
        while async_job.result() is None:
            time.sleep(5)
            loop_nb += 1
            if loop_nb > 4:
                return

        result = async_job.result()
        for entry in result.raw_data:
            LOGGER.debug("State: %s\t probability: %s",
                         entry.state, entry.probability)

        self.assertEqual(4, len(result.raw_data))

    def test2_asyncqiskit_qpu_ibmq_experience(self):
        """
        Same as test0 of the same class, but using ibmq token if it is
        provided in the file tests/ibmq_token. Only create this file if
        you do want to test the ibmq service. If it is not provided,
        it will simply run a test identical to test0.
        """
        nbqubits = 2

        prog = Program()

        qreg = prog.qalloc(nbqubits)
        creg = prog.calloc(nbqubits)

        prog.apply(H, qreg[0])
        prog.apply(CNOT, qreg[0], qreg[1])

        prog.measure(qreg, creg)
        qlm_circuit = prog.to_circ()

        qlm_job1 = qlm_circuit.to_job(nbshots=1024)
        qlm_job2 = qlm_circuit.to_job(nbshots=1024)
        batch = Batch(jobs=[qlm_job1, qlm_job2])

        # a backend is specified
        backend = Aer.get_backend('aer_simulator')
        qpu = AsyncBackendToQPU(backend)
        path_to_file = os.path.join(os.path.dirname(__file__), './ibmq_token')

        LOGGER.warning("This may take a few seconds...")

        if os.path.exists(path_to_file) and os.path.getsize(path_to_file) > 0:
            with open(path_to_file, 'r') as token:
                qpu.set_backend(token=token.read(),
                                ibmq_backend='ibmq_qasm_simulator')

        job = qpu.submit(batch)
        string = "\nAsyncBackendToQPU test with a Hadamard and a CNOT " \
            + "(expects two different measured states):"
        LOGGER.debug(string)
        LOGGER.debug("ID: %s\t status : %s", job.job_id(), job.status())
        time.sleep(0.01)
        LOGGER.debug("ID: %s\t status : %s", job.job_id(), job.status())
        time.sleep(0.2)
        LOGGER.debug("ID: %s\t status : %s", job.job_id(), job.status())

        while job.result() is None:
            time.sleep(5)

        results = job.result()

        for result in results:
            for entry in result.raw_data:
                LOGGER.debug("State: %s\t probability: %s",
                             entry.state, entry.probability)
                self.assertEqual(2, len(result.raw_data))
                self.assertTrue("|00>" in [str(result.raw_data[i].state)
                                           for i in range(2)])
                self.assertTrue("|11>" in [str(result.raw_data[i].state)
                                           for i in range(2)])


class Test2QPUToBackend(unittest.TestCase):
    """
    Creates a Qiskit circuit and uses a QPUToBackend in order to
    simulate it with another simulator (PyLinalg) inside
    Qiskit's ecosystem.
    Also tests the run() function of Backend.
    Only checks if the results' size is correct.
    """

    def test0_qlm_backend(self):
        """
        Simple test for QPUToBackend object generation and basic use.
        """
        nbqubits = 2

        qreg = QuantumRegister(nbqubits)
        creg = ClassicalRegister(nbqubits)

        qiskit_circuit = QuantumCircuit(qreg, creg)

        qiskit_circuit.h(qreg[0])
        qiskit_circuit.cx(qreg[0], qreg[1])
        qiskit_circuit.measure(qreg, creg)

        qpu = PyLinalg()
        backend = QPUToBackend(qpu)

        result = execute(qiskit_circuit, backend, shots=15).result()

        LOGGER.debug("\nQPUToBackend test:")
        LOGGER.debug(result.results)
        self.assertEqual(1, len(result.results))

    def test1_qlm_backend_connector(self):
        """
        The QPUToBackend object is here generated by using the QiskitConnector.
        """
        nbqubits = 2

        qreg = QuantumRegister(nbqubits)
        creg = ClassicalRegister(nbqubits)

        qiskit_circuit = QuantumCircuit(qreg, creg)

        qiskit_circuit.h(qreg[0])
        qiskit_circuit.cx(qreg[0], qreg[1])
        qiskit_circuit.measure(qreg, creg)

        backend = QiskitConnector() | PyLinalg()

        result = execute(qiskit_circuit, backend, shots=15).result()

        LOGGER.debug("\nQPUToBackend test via QiskitConnector:")
        LOGGER.debug(result.results)
        self.assertEqual(1, len(result.results))

    def test2_qlm_backend_run_1_circuit(self):
        """
        Here a circuit runs into a QLM QPU by using the QLMBacked object.
        """
        nbqubits = 2
        qreg = QuantumRegister(nbqubits)
        creg = ClassicalRegister(nbqubits)

        qiskit_circuit = QuantumCircuit(qreg, creg)

        qiskit_circuit.h(qreg[0])
        qiskit_circuit.cx(qreg[0], qreg[1])
        qiskit_circuit.measure(qreg, creg)

        backend = QiskitConnector() | PyLinalg()
        result = backend.run(qiskit_circuit).result()

        LOGGER.debug("\nQPUToBackend test with a QLM job sent into a QLM qpu:")
        LOGGER.debug(result.results)
        self.assertEqual(1, len(result.results))

    def test3_qlm_backend_run_2_circuit(self):
        """
        Here two circuits run into a QLM QPU by using the QLMBacked object.
        """
        nbqubits = 2
        qreg = QuantumRegister(nbqubits)
        creg = ClassicalRegister(nbqubits)

        qiskit_circuit_1 = QuantumCircuit(qreg, creg)
        qiskit_circuit_1.h(qreg[0])
        qiskit_circuit_1.cx(qreg[0], qreg[1])
        qiskit_circuit_1.measure(qreg, creg)

        qiskit_circuit_2 = QuantumCircuit(qreg, creg)
        qiskit_circuit_2.h(qreg[0])
        qiskit_circuit_2.h(qreg[1])
        qiskit_circuit_2.measure(qreg, creg)

        backend = QiskitConnector() | PyLinalg()
        qiskit_circuits = []
        qiskit_circuits.append(qiskit_circuit_1)
        qiskit_circuits.append(qiskit_circuit_2)

        result = backend.run(qiskit_circuits).result()

        LOGGER.debug(
            "\nQPUToBackend test with a list of QLM jobs sent into a QLM qpu:")
        LOGGER.debug(result.results)
        self.assertEqual(2, len(result.results))


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
