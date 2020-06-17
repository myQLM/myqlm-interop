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
        "projectq_binder": ["projectq==0.4.2"],
        "cirq_binder": ["cirq==0.4.0"],
        "pyquil_binder": ["pyquil==2.7.2", "quantum-grove==1.7.0"],
        "all": ["qiskit==0.19.2", "qiskit-terra==0.14.1", "qiskit-aqua==0.7.1",
                "qiskit-ignis==0.3.0", "qiskit-aer==0.5.1",
                "qiskit-ibmq-provider==0.7.1",
                "projectq==0.4.2", "cirq==0.4.0", "pyquil==2.7.2",
                "quantum-grove==1.7.0"]
    },
    tests_require=["pytest"],
    cmdclass={'test': PyTest},
)
