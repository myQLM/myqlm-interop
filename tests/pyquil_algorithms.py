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


Description ...

Overview
=========


"""
from collections import Counter

def compare_results(expected, result, aggregate=True):
    if aggregate:
        expected_arr = [entry.state for entry in expected.raw_data]
        result_arr = [entry.state for entry in expected.raw_data]
        expected_counts = Counter(expected_arr)
        result_counts = Counter(result_arr)
    else:
        expected_counts = {entry.state:entry.probability for entry in
                           expected.raw_data}
        result_counts = {entry.state:entry.probability for entry in
                         result.raw_data}

    distance = {}

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

    return statistics.median(list(distance.values()))


if __name__=="__main__":
    from qat.interop.pyquil.algorithms import run_simon
    from qat.linalg import LinAlg
    from qat.mps import MPS
    from qat.interop.pyquil.providers import generate_qlm_result
    from pyquil.api import QVMConnection

    qvm = QVMConnection(endpoint="http://127.0.0.1:15011")
    linalg = LinAlg()
    mps = MPS(lnnize=True)
    qvm_res = run_simon(qvm, trials=10024)

    simon_res = {entry.state:entry.probability for entry in generate_qlm_result(qvm_res).raw_data}

    print(simon_res)
    print(max(simon_res.values()))
