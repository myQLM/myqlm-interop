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


Description  Some pyquil algorithms taken from rigetti's grove repo

            - Simon algorithm

            - Deutsch-Jozsa algorithm

            - 3-qubits QFT

            - meyer penny

            - quantum dice

Overview
=========


"""

from math import pi, ceil, log

from pyquil import Program
from pyquil.gates import *
import numpy as np

from grove.simon.simon import Simon, create_1to1_bitmap


def run_simon(qc, bitmap=create_1to1_bitmap("101"), trials=30):
    sm = Simon()
    sm._init_attr(bitmap)
    circ = sm.simon_circuit
    return qc.run_and_measure(circ, qubits=list(circ.get_qubits()), trials=trials)


SWAP_MATRIX = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
r"""The matrix that performs \alpha\ket{ij}\to\alpha\ket{ji}"""

ORACLE_GATE_NAME = "DEUTSCH_JOZSA_ORACLE"


def _gen_bitmap(register_size=5):
    ret = {}
    size_0 = 0
    size_1 = 0
    for i in range(1 << register_size):
        if size_0 == (1 << (register_size - 1)):
            size_1 += 1
            ret[np.binary_repr(i, width=register_size)] = "1"
        elif size_1 == (1 << (register_size - 1)):
            size_0 += 1
            ret[np.binary_repr(i, width=register_size)] = "0"
        else:
            entry = np.random.choice(["0", "1"])
            ret[np.binary_repr(i, width=register_size)] = entry
            if entry == "0":
                size_0 += 1
            else:
                size_1 += 1
    return ret


DEF_JOSZA_MAP = _gen_bitmap()


def run_deutsch_josza(cxn, bitstring_map=DEF_JOSZA_MAP, trials=10):
    """Computes whether bitstring_map represents a constant function, given that it is constant
     or balanced. Constant means all inputs map to the same value, balanced means half of the
     inputs maps to one value, and half to the other.

    :param QVMConnection cxn: The connection object to the Rigetti cloud to run pyQuil programs.
    :param bitstring_map: A dictionary whose keys are bitstrings, and whose values are bits
     represented as strings.
    :type bistring_map: Dict[String, String]
    :return: True if the bitstring_map represented a constant function, false otherwise.
    :rtype: bool
    """
    deutsch_jozsa_circuit, computational_qubits = _init_attr(bitstring_map)
    return cxn.run_and_measure(deutsch_jozsa_circuit, computational_qubits, trials)


def deutsch_josza_is_constant_prob(result):
    bitstring = np.array(result, dtype=int)
    total = 0
    for res in result:
        if all([bit == 1 for bit in res]):
            total += 1
    return total / len(result)


def _init_attr(bit_map):
    """
    Instantiates the necessary Deutsch-Jozsa state.

    :param Dict[String, String] bitstring_map: truth-table of the input bitstring map in
    dictionary format, used to construct the oracle in the Deutsch-Jozsa algorithm.
    :return: None
    :rtype: NoneType
    """
    # self.bit_map = bitstring_map
    n_qubits = len(list(bit_map.keys())[0])
    # We use one extra qubit for making the oracle,
    # and one for storing the answer of the oracle.
    n_ancillas = 2
    qubits = list(range(n_qubits + n_ancillas))
    computational_qubits = qubits[:n_qubits]
    ancillas = qubits[n_qubits:]
    unitary_matrix = _unitary_function(bit_map)
    deutsch_jozsa_circuit = _construct_deutsch_jozsa_circuit(
        qubits, computational_qubits, ancillas, unitary_matrix
    )

    return deutsch_jozsa_circuit, computational_qubits


def _construct_deutsch_jozsa_circuit(
    qubits, computational_qubits, ancillas, unitary_matrix
):
    """
    Builds the Deutsch-Jozsa circuit. Which can determine whether a function f mapping
    :math:`\{0,1\}^n \to \{0,1\}` is constant or balanced, provided that it is one of them.

    :return: A program corresponding to the desired instance of Deutsch Jozsa's Algorithm.
    :rtype: Program
    """
    dj_prog = Program()

    # Put the first ancilla qubit (query qubit) into minus state
    dj_prog.inst(X(ancillas[0]), H(ancillas[0]))

    # Apply Hadamard, Oracle, and Hadamard again
    dj_prog.inst([H(qubit) for qubit in computational_qubits])

    # Build the oracle
    oracle_prog = Program()
    oracle_prog.defgate(ORACLE_GATE_NAME, unitary_matrix)

    scratch_bit = ancillas[1]
    qubits_for_funct = [scratch_bit] + computational_qubits
    oracle_prog.inst(tuple([ORACLE_GATE_NAME] + qubits_for_funct))
    dj_prog += oracle_prog

    # Here the oracle does not leave the computational qubits unchanged, so we use a CNOT to
    # to move the result to the query qubit, and then we uncompute with the dagger.
    dj_prog.inst(CNOT(qubits[0], ancillas[0]))
    dj_prog += oracle_prog.dagger()
    dj_prog.inst([H(qubit) for qubit in computational_qubits])
    return dj_prog


def _unitary_function(mappings):
    """
    Creates a unitary transformation that maps each state to the values specified
    in mappings.

    Some (but not all) of these transformations involve a scratch qubit, so room for one is
    always provided. That is, if given the mapping of n qubits, the calculated transformation
    will be on n + 1 qubits, where the 0th is the scratch bit and the return value
    of the function is left in the 1st.

    :param mappings: Dictionary of the mappings of f(x) on all length n bitstrings, e.g.

        >>> {'00': '0', '01': '1', '10': '1', '11': '0'}

    :type mappings: Dict[String, Int]
    :return: ndarray representing specified unitary transformation.
    :rtype: np.ndarray
    """
    num_qubits = int(np.log2(len(mappings)))
    bitsum = sum([int(bit) for bit in mappings.values()])

    # Only zeros were entered
    if bitsum == 0:
        return np.kron(SWAP_MATRIX, np.identity((1 << (num_qubits - 1))))

    # Half of the entries were 0, half 1
    elif bitsum == (1 << (num_qubits - 1)):
        unitary_funct = np.zeros(shape=((1 << num_qubits), (1 << num_qubits)))
        index_lists = [
            list(range(1 << (num_qubits - 1))),
            list(range(1 << (num_qubits - 1), (1 << num_qubits))),
        ]
        for j in range(1 << num_qubits):
            bitstring = np.binary_repr(j, num_qubits)
            value = int(mappings[bitstring])
            mappings.pop(bitstring)
            i = index_lists[value].pop()
            unitary_funct[i, j] = 1
        return np.kron(np.identity(2), unitary_funct)

    # Only ones were entered
    elif bitsum == (1 << num_qubits):
        x_gate = np.array([[0, 1], [1, 0]])
        return np.kron(SWAP_MATRIX, np.identity(1 << (num_qubits - 1))).dot(
            np.kron(x_gate, np.identity(1 << num_qubits))
        )
    else:
        raise ValueError("f(x) must be constant or balanced")


def QFT3():
    """ Returns the Quantum Fourier Transform of 3 qubits
        pyquil circuit"""
    prog = Program()
    ro = prog.declare("ro", memory_size=3)
    prog += [
        SWAP(0, 2),
        H(0),
        CPHASE(-pi / 2.0, 0, 1),
        H(1),
        CPHASE(-pi / 4.0, 0, 2),
        CPHASE(-pi / 2.0, 1, 2),
        H(2),
    ]

    prog.measure(0, ro[0])
    prog.measure(1, ro[1])
    prog.measure(2, ro[2])
    return prog

def run_qft3(qc, trials=30):
    return qc.run(QFT3(), trials=trials)

def meyer_penny_program():
    """
    Returns the program to simulate the Meyer-Penny Game
    The full description is available in docs/source/examples.rst
    :return: pyQuil Program
    """
    prog = Program()
    ro = prog.declare("ro", memory_size=2)
    picard_register = ro[1]
    answer_register = ro[0]

    then_branch = Program(X(0))
    else_branch = Program(I(0))

    # Prepare Qubits in Heads state or superposition, respectively
    prog.inst(X(0), H(1))
    # Q puts the coin into a superposition
    prog.inst(H(0))
    # Picard makes a decision and acts accordingly
    prog.measure(1, picard_register)
    prog.if_then(picard_register, then_branch, else_branch)
    # Q undoes his superposition operation
    prog.inst(H(0))
    # The outcome is recorded into the answer register
    prog.measure(0, answer_register)

    return prog

def run_meyer_penny(qc, trials=30):
    return qc.run(meyer_penny_program(), trials=trials)

def quantum_die(number_of_sides=6):
    """ Generates a quantum program to roll a die of n faces"""

    prog = Program()
    qbits = int(ceil(log(number_of_sides, 2)))
    ro = prog.declare("ro", "BIT", qbits)
    # Hadamard intialize
    for qbit in range(qubits):
        p += H(qbit)
    # Simple measurements
    for qbit in range(qubits):
        p += MEASURE(qbit, ro[qbit])
    return prog
