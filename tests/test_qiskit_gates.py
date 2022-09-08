# -*- coding: utf-8 -*-

"""
@authors    Arnaud Gazda <arnaud.gazda@atos.net>
@copyright  2022 Bull S.A.S. - All rights reserved
            This is not Free or Open Source software.
            Please contact Bull SAS for details about its license.
            Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois

Description: Testing implementation of U1, U2 and U3 gates in the myQLM framework.
             Qiskit documentation provides some equalities tested in this file
"""

from itertools import product
from numpy import linspace, allclose, pi, exp
from qat.core.circuit_builder.matrix_util import default_gate_set
from qat.interop.qiskit.converters import _gen_u2, _gen_u


def test_u1_gate():
    """
    Testing the definition of the U1 gate in the myQLM
    framework.
    This test is based on
    https://qiskit.org/documentation/stubs/qiskit.circuit.library.U1Gate.html
    """
    # gate_set = default_gate_set()


def test_u2_gate():
    """
    Testing the definition of the U2 gate in the myQLM
    framework.
    This test checks the equality U2(φ,λ) = RZ(φ).RY(π/2).RZ(λ)

    .. note::

        According to https://qiskit.org/documentation/stubs/qiskit.circuit.library.U3Gate.html
        the definition of the U3 gate (U2 definition is based on U3) differ between OpenQASM 3.0
        and OpenQASM 2.0

        These is a global phase of exp(i * (φ + λ) / 2) betwen OpenQASM 3.0 and OpenQASM 2.0
        This test consider this global phase
    """
    # Build gate set
    gate_set = default_gate_set()
    rz_def = gate_set["RZ"]
    ry_def = gate_set["RY"]

    # Check matrix
    for phi_, lambda_ in product(linspace(0, pi, 10), linspace(0, pi, 10)):
        # Get rotation gates
        rz_phi = rz_def.matrix_generator(phi_)
        rz_lambda = rz_def.matrix_generator(lambda_)
        ry_pi = ry_def.matrix_generator(pi / 2)
        phase = exp(1j * (phi_ + lambda_) / 2)

        # Build matrices
        expected_gate = phase * rz_phi.dot(ry_pi).dot(rz_lambda)
        actual_gate = _gen_u2(phi_, lambda_)

        # Check equality
        assert allclose(expected_gate, actual_gate), f"Invalid U2({phi_}, {lambda_}) gate"


def test_u3_gate():
    """
    Testing the definition of the U3 gate in the myQLM
    framework
    This test checks the following equalities:
     * U3(ϴ, -π/2, π/2) = RX(ϴ)
     * U3(ϴ, 0, 0) = RY(ϴ)

    Depending on the definition used (the one from OpenQASM 2.0 or OpenQASM 3.0),
    a global phase can appear
    """
    # Build gate set
    gate_set = default_gate_set()
    rx_def = gate_set["RX"]
    ry_def = gate_set["RY"]

    for theta in linspace(0, pi, 10):
        # Get rotation gates
        rx_theta = rx_def.matrix_generator(theta)
        ry_theta = ry_def.matrix_generator(theta)

        # Perform check
        assert allclose(_gen_u(theta, -pi / 2, pi / 2), rx_theta), f"Invalid U3({theta}, -π/2, π/2) gate"
        assert allclose(_gen_u(theta, 0, 0), ry_theta), f"Invalid U3({theta}, 0, 0) gate"
