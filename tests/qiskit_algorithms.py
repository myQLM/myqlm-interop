#!/usr/bin/env python3
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
from test_algorithms import compare_results, analyze_distance
from qat.interop.qiskit.algorithms import shor_circuit, grover_circuit, qaoa_circuit
from qat.linalg import LinAlg
from qat.mps import MPS
from qat.interop.qiskit.providers import generate_qlm_result, QiskitQPU
from qat.interop.qiskit.converters import to_qlm_circ
from qiskit import Aer

def test_algorithm(circuit, iterations=(1024*1024)):
    linalg = LinAlg()
    mps = MPS(lnnize=True)
    qlm_circ = to_qlm_circ(circuit)
    test_job = qlm_circ.to_job(nbshots=1)
    test_job.nbshots=iterations
    print(test_job.nbshots)
    expected = linalg.submit(test_job)

    qiskit = QiskitQPU(Aer.get_backend('qasm_simulator'))

    result = qiskit.submit(test_job)

    dist_calc = compare_results(expected, result, aggregate=False)
    distance = analyze_distance(dist_calc)
    print("Distance is {}".format(distance))
    return distance

shor_distance = test_algorithm(shor_circuit())
grover_distance = test_algorithm(grover_circuit())
qaoa_distance = test_algorithm(qaoa_circuit())

print("Shor distance is {}".format(shor_distance))
print("Grover distance is {}".format(grover_distance))
print("QAOA distance is {}".format(qaoa_distance))
