#!/usr/bin/env python
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

import math
import itertools
from collections import Counter
from functools import reduce
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Pauli
from qiskit import Aer
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import Shor, Grover, QAOA
from qiskit.aqua.components.oracles import LogicalExpressionOracle as SAT
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.components.initial_states.custom import Custom

from qat.pylinalg import PyLinalg
from qat.interop.qiskit import qiskit_to_qlm, BackendToQPU

# pylint: disable = no-member, invalid-name, consider-iterating-dictionary


def add_measures(circ):
    """
    Add measures to a Qiskit circuit.
    """
    for qreg in circ.qregs:
        circ.cregs.append(ClassicalRegister(qreg.size, "c" + qreg.name))

    for qreg, creg in zip(circ.qregs, circ.cregs):
        circ.measure(qreg, creg)

    return circ


def shor_circuit():
    """
    Builds A QuantumCircuit object of shor's algorithm implementation
    found in qiskit aqua

    Args:

    Returns:
        Resulting circuit of the qiskit aqua implementation of Shor's
        algorithm after adding measures on every qubit.
    """
    shor = Shor()
    circ = shor.construct_circuit()
    return add_measures(circ)


def grover_circuit():
    """
    Builds a QuantumCircuit of the qiskit aqua implementation of
    grover's algorithm with set parameters.

    Args:

    Returns:
        Resuling circuit of the qiskit aqua implementation of Grover's
        algorithm after adding measures on every qubit.
    """
    sat_cnf = """
c Example DIMACS 3-sat
p cnf 3 5
-1 -2 -3 0
1 -2 3 0
1 2 -3 0
1 -2 -3 0
-1 2 3 0
"""
#   backend = Aer.get_backend('aer_simulator')
    oracle = SAT(sat_cnf)
    grv = Grover(oracle)
    circ = grv.construct_circuit()

    return add_measures(circ)


def product_pauli_z(q_1, q_2, coeff, n_sys):
    """
    Placeholder
    """
    eye = np.eye((n_sys))
    return WeightedPauliOperator(
        [[
            coeff,
            Pauli(eye[q_1], np.zeros(n_sys)) * Pauli(eye[q_2], np.zeros(n_sys))
        ]]
    )


def ising_hamiltonian(weights, size, n_sys):
    """
    Placeholder
    """
    hamilt = reduce(
        lambda x, y: x + y,
        [
            product_pauli_z(i, j, -weights[i, j], n_sys)
            for (i, j) in itertools.product(range(size), range(size))
        ],
    )
#   hamilt.to_matrix()
    return hamilt


def prepare_init_state(T, qreg, size, n_sys):
    """
    Placeholder
    """
    init_circuit = QuantumCircuit(qreg)
    alpha = 2 * np.arctan(np.exp(-1 / T))
    for i in range(size):
        init_circuit.rx(alpha, qreg[size + i])
        init_circuit.cx(qreg[size + i], qreg[i])

    init_state = Custom(n_sys, circuit=init_circuit)
    return init_state


def get_qaoa():
    """
    Takes care of setting up qiskit aqua's qaoa implementation
    with specific parameters

    Args:

    Returns:
        Built QAOA object from qiskit aqua (c.f qiskit's github for more
        info)
    """
    size = 2
    n_sys = size * 2
    T = 1000
    weights = np.array([[0, 1], [0, 0]])
    p = 2

    hamilt = ising_hamiltonian(weights, size, n_sys)
    qreg = QuantumRegister(n_sys)

    initial_state = prepare_init_state(T, qreg, size, n_sys)
    qaoa = QAOA(hamilt, COBYLA(), p, initial_state)
    return qaoa


def qaoa_circuit():
    """ Builds the QAOA QuantumCircuit using qiskit aqua's implementation

    Args:

    Returns:
        Resulting circuit of the qiskit aqua qaoa algorithm's
        implementation after adding measures on every qubit
    """
    circ = get_qaoa().construct_circuit([math.pi] * 4)[0]
    return add_measures(circ)
    # expected = qaoa.run(quantum_instance)


expected_counts = {}
result_counts = {}


def compare_results(expected, result, aggregate=True):
    """
    TODO
    """
    if aggregate:
        expected_arr = [entry.state.state for entry in expected.raw_data]
        result_arr = [entry.state.state for entry in result.raw_data]
        expected_counts.update(Counter(expected_arr))
        result_counts.update(Counter(result_arr))
        nbshots = sum(expected_counts.values())
        for v in expected_counts.values():
            v = v/nbshots
        nbshots = sum(result_counts.values())
        for v in result_counts.values():
            v = v/nbshots
    else:
        if expected.raw_data[0].probability is None:
            expected_arr = Counter([entry.state.state for entry in
                                    expected.raw_data])
            nbshots = sum(expected_arr.values())
            expected_counts.update(
                {k: v/nbshots for k, v in expected_arr.items()})
        else:
            expected_counts.update(
                {entry.state.state: entry.probability for
                 entry in expected.raw_data})
        result_counts.update({entry.state.state: entry.probability for
                              entry in result.raw_data})

    print("__Expected {}".format(expected_counts))
    print("__Results {}".format(result_counts))
    distance = {}
    print("sizes of expected {} and result {}"
          .format(len(expected_counts), len(result_counts)))
    shared = expected_counts.keys() & result_counts.keys()
    print("Shared keys are {}".format(shared))
    for entry in shared:
        print("for {}: expected:{} |result: {}"
              .format(entry, expected_counts[entry], result_counts[entry]))
    for state in expected_counts.keys():
        if state in result_counts.keys():
            distance[state] = abs(
                expected_counts[state] - result_counts[state])
        else:
            distance[state] = expected_counts[state]
    for state in result_counts.keys():
        if state not in expected_counts.keys():
            distance[state] = result_counts[state]

    return distance


def analyze_distance(distance):
    """
    TODO
    """
    return max(distance.values())


def test_algorithm(circuit, iterations=(1000000)):
    """
    Tests a circuit by submitting it to both aer_simulator and PyLinalg.
    """
    linalg = PyLinalg()
    qlm_circ, _ = qiskit_to_qlm(circuit, sep_measures=True)
    test_job = qlm_circ.to_job(nbshots=0, aggregate_data=False)
    expected = linalg.submit(test_job)

    qiskit_qpu = BackendToQPU(Aer.get_backend('aer_simulator'))

    test_job.nbshots = iterations
    result = qiskit_qpu.submit(test_job)

    dist_calc = compare_results(expected, result, aggregate=False)
    distance = analyze_distance(dist_calc)
    print("Distance is {}".format(distance))
    return distance


SHOR_DISTANCE = test_algorithm(shor_circuit())
GROVER_DISTANCE = test_algorithm(grover_circuit())
# QAOA_DISTANCE = test_algorithm(qaoa_circuit())

print("Shor distance is {}".format(SHOR_DISTANCE))
print("Grover distance is {}".format(GROVER_DISTANCE))
# print("QAOA distance is {}".format(QAOA_DISTANCE))
