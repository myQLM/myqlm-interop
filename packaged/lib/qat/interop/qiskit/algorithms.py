#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@brief 

#@file qat/interop/qiskit/algorithms.py
#@namespace qat.interop.qiskit.algorithms
#@authors Reda Drissi <mohamed-reda.drissi@atos.net>
#@copyright 2019  Bull S.A.S.  -  All rights reserved.
#           This is not Free or Open Source software.
#           Please contact Bull SAS for details about its license.
#           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois

"""
Implementation of ready to use circuits of algorithms taken
            from qiskit aqua


"""
import numpy as np
import itertools
from functools import reduce, partial

from qiskit import Aer, QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Pauli

from qiskit.aqua.operator import Operator
from qiskit.aqua import QuantumInstance

from qiskit.aqua.algorithms import Shor, Grover, QAOA

from qiskit.aqua.components.oracles import LogicalExpressionOracle as SAT
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.components.initial_states.custom import Custom

from qat.interop.qiskit.providers import generate_qlm_result


def add_measures(circ):
    for qreg in circ.qregs:
        circ.cregs.append(ClassicalRegister(qreg.size, "c" + qreg.name))

    for qreg, creg in zip(circ.qregs, circ.cregs):
        circ.measure(qreg, creg)

    return circ


def shor_circuit():
    """ Builds A QuantumCircuit object of shor's algorithm implementation
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
    """ Builds a QuantumCircuit of the qiskit aqua implementation of
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
    # backend = Aer.get_backend('qasm_simulator')
    oracle = SAT(sat_cnf)
    grv = Grover(oracle)
    circ = grv.construct_circuit()

    return add_measures(circ)


#    expected = grv.run(backend)

#    qiskit_circuit = grv.construct_circuit()
#    qlm_circuit = to_qlm_circuit(qiskit_circuit)


def product_pauli_z(q1, q2, coeff, N_sys):
    eye = np.eye((N_sys))
    return Operator(
        [[coeff, Pauli(eye[q1], np.zeros(N_sys)) * Pauli(eye[q2], np.zeros(N_sys))]]
    )


def ising_hamiltonian(weights, N, N_sys):
    H = reduce(
        lambda x, y: x + y,
        [
            product_pauli_z(i, j, -weights[i, j], N_sys)
            for (i, j) in itertools.product(range(N), range(N))
        ],
    )
    H.to_matrix()
    return H


def prepare_init_state(T, qreg, N, N_sys):
    init_circuit = QuantumCircuit(qreg)
    alpha = 2 * np.arctan(np.exp(-1 / T))
    for i in range(N):
        init_circuit.rx(alpha, qreg[N + i])
        init_circuit.cx(qreg[N + i], qreg[i])

    init_state = Custom(N_sys, circuit=init_circuit)
    return init_state


def get_qaoa():
    """
        Takes care of setting up qiskit aqua's qaoa implementation
        with specific parameters
    Args:

    Returns:
        built QAOA object from qiskit aqua (c.f qiskit's github for more
        info)
    """
    N = 2
    N_sys = N * 2
    T = 1000
    weights = np.array([[0, 1], [0, 0]])
    p = 2

    Hc = ising_hamiltonian(weights, N, N_sys)
    qreg = QuantumRegister(N_sys)

    initial_state = prepare_init_state(T, qreg, N, N_sys)
    qaoa = QAOA(Hc, COBYLA(), p, initial_state)
    return qaoa


def qaoa_circuit():
    """ Builds the QAOA QuantumCircuit using qiskit aqua's implementation

    Args:

    Returns:
        Resulting circuit of the qiskit aqua qaoa algorithm's
        implementation after adding measures on every qubit
    """
    import math

    circ = get_qaoa().construct_circuit([math.pi] * 4)[0]
    return add_measures(circ)
    # expected = qaoa.run(quantum_instance)
