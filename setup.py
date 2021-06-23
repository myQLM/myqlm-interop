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

import os, sys
from setuptools import setup, find_namespace_packages
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main([".", "-v"])
        sys.exit(errno)

setup(
    name="myqlm-interop",
    version="0.0.6",
    author="Atos Quantum Lab",
    license="Atos myQLM EULA",
    packages=find_namespace_packages(include=["qat.*"]),
    scripts=["bin/oqasm2circ"],
    install_requires=["qat-lang>=0.0.6", "numpy", "ply"],
    extras_require={
        "qiskit_binder": ["qiskit==0.19.2", "qiskit-terra==0.14.1",
                          "qiskit-aqua==0.7.1", "qiskit-ignis==0.3.0",
                          "qiskit-aer==0.5.1", "qiskit-ibmq-provider==0.7.1"],
        "projectq_binder": ["projectq==0.5.1"],
        "cirq_binder": ["cirq==0.10.0"],
        "pyquil_binder": ["pyquil==2.20.0", "quantum-grove==1.7.0"],
        "all": ["qiskit==0.19.2", "qiskit-terra==0.14.1", "qiskit-aqua==0.7.1",
                "qiskit-ignis==0.3.0", "qiskit-aer==0.5.1",
                "qiskit-ibmq-provider==0.7.1",
                "projectq==0.5.1", "cirq==0.8.2", "pyquil==2.20.0",
                "quantum-grove==1.7.0"]
    },
    tests_require=["pytest"],
    cmdclass={'test': PyTest},
)
