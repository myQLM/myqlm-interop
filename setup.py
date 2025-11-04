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
from setuptools import setup, find_namespace_packages
from setuptools.command.test import test as TestCommand


# Define dependencies
_QISKIT_DEPS = ["qiskit>=1.0.0;python_version>='3.8'", "qiskit-ibm-runtime>=0.11.1;python_version>='3.8'"]
_PROJECTQ_DEPS = ["projectq>=0.8.0;python_version>='3.8'"]
_CIRQ_DEPS = ["cirq>=1.1.0;python_version>='3.8'"]
_PYQUIL_DEPS = ["pyquil>=3.5.0;python_version>='3.8'", "quantum-grove>=1.7.0;python_version>='3.8'"]


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


def get_description():
    """
    Returns the long description of the current
    package

    Returns:
        str
    """
    with open("README.md", "r", encoding="utf-8") as readme:
        return readme.read()


setup(
    name="myqlm-interop",
    version="1.7.1",
    author="Atos Quantum Lab",
    license="Atos myQLM EULA",
    description="myQLM-interop package",
    long_description=get_description(),
    packages=find_namespace_packages(include=["qat.*"]),
    scripts=["bin/oqasm2circ"],
    install_requires=["qat-lang>=2.2.0", "numpy>=2.0.0", "ply"],
    extras_require={
        "qiskit_binder": _QISKIT_DEPS,
        "projectq_binder": _PROJECTQ_DEPS,
        "cirq_binder": _CIRQ_DEPS,
        "pyquil_binder": _PYQUIL_DEPS,
        "all": [
            *_QISKIT_DEPS,
            *_PROJECTQ_DEPS,
            *_CIRQ_DEPS,
            *_PYQUIL_DEPS
        ]
    },
    tests_require=["pytest"],
    cmdclass={'test': PyTest},
)
