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
    version="0.0.1",
    author="Atos Quantum Lab",
    license="Atos myQLM EULA",
    packages=find_namespace_packages(include=["qat.*"]),
    scripts=["bin/oqasm2circ"],
    install_requires=["thrift==0.10", "qat-lang>=0.0.8", "numpy", "ply"],
    extras_require={
        "qiskit_binder": ["qiskit==0.7.2", "qiskit-terra==0.8.0",
                          "qiskit-aqua==0.5.0"],
        "projectq_binder": ["projectq==0.4.2"],
        "cirq_binder": ["cirq==0.4.0"],
        "pyquil_binder": ["pyquil==2.7.2", "quantum-grove==1.7.0"],
        "all": ["qiskit==0.7.2", "qiskit-terra==0.8.0", "qiskit-aqua==0.5.0",
                "projectq==0.4.2", "cirq==0.4.0", "pyquil==2.7.2",
                "quantum-grove==1.7.0"]
    },
    tests_require=["pytest"],
    cmdclass = {'test': PyTest},
)
