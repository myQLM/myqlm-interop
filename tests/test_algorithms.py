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
expected_counts = {}
result_counts = {}
from collections import Counter


def compare_results(expected, result, aggregate=True):
    if aggregate:
        expected_arr = [entry.state.state for entry in expected.raw_data]
        result_arr = [entry.state.state for entry in expected.raw_data]
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
            expected_counts.update({k:v/nbshots for k,v in expected_arr.items()})
        else:
            expected_counts.update({entry.state.state:entry.probability for entry in
                           expected.raw_data})
        result_counts.update({entry.state.state:entry.probability for entry in
                         result.raw_data})

    print("__Expected {}".format(expected_counts))
    print("__Results {}".format(result_counts))
    distance = {}
    print("sizes of expected {} and result {}".format(len(expected_counts), len(result_counts)))
    shared = expected_counts.keys() & result_counts.keys()
    print("Shared keys are {}".format(shared))
    for entry in shared:
        print("for {}: expected:{} |result: {}".format(entry, expected_counts[entry], result_counts[entry]))
    for state in expected_counts.keys():
        if state in result_counts.keys():
            distance[state] = abs(expected_counts[state] - result_counts[state])
        else:
            distance[state] = expected_counts[state]
    for state in result_counts.keys():
        if state not in expected_counts.keys():
            distance[state] = result_counts[state]

    return distance


def analyze_distance(distance):

    import statistics

    return max(distance.values())


if __name__=="__main__":
    from qat.interop.qiskit.algorithms import shor_circuit, grover_circuit
    from qat.linalg import LinAlg
    from qat.mps import MPS
    from qat.interop.qiskit.providers import generate_qlm_result, QiskitQPU
    from qat.interop.qiskit.converters import to_qlm_circ
    from qiskit import Aer

    linalg = LinAlg()
    mps = MPS(lnnize=True)
    shor_circ = shor_circuit()

    circ = to_qlm_circ(shor_circ)
    print(type(circ))
    shor_job = circ.to_job(nbshots=1)
    shor_job.nbshots=1024
    print(shor_job.nbshots)
    shor_res = linalg.submit(shor_job)

    qiskit = QiskitQPU(Aer.get_backend('qasm_simulator'))

    shor_qiskit = qiskit.submit(shor_job)

    print("Result counts {}".format(result_counts))
    print("Expected counts {}".format(expected_counts))
    dist_calc = compare_results(shor_res, shor_qiskit, aggregate=False)
    distance = analyze_distance(dist_calc)
    print("Distance is {}".format(distance))
    print(distance < 0.001)
