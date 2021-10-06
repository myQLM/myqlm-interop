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

import sys
import pytest
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
    if sys.version_info < (3, 7):
        pytest.skip("python version < 3.7: skipping pyquil_binder tests", allow_module_level=True)

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
