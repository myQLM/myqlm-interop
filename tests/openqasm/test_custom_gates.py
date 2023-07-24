# -*- coding: utf-8 -*-

"""
@authors    Arnaud GAZDA <arnaud.gazda@atos.net>
@copyright  2023 Bull S.A.S. - All rights reserved
            This is not Free or Open Source software.
            Please contact Bull SAS for details about its license.
            Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois

Description: Compiling a OpenQASM circuit composed of custom gates (defined using myQLM)
"""

import math
import pytest
from qat.lang.AQASM import AbstractGate
from qat.interop.openqasm import OqasmParser


def test_alias():
    """
    Ensure the alias methods behaves as expected
    """
    data = """
    OPENQASM 2.0;

    // Allocating registers
    qreg q[1];
    creg c[1];

    // Apply gates
    p(pi/4) q[0];
    u(pi/4, pi/4, pi/4) q[0];
    """

    parser = OqasmParser(gates={"p": "PH", "u": "U"})
    circuit = parser.compile(data)

    gates = list(circuit.iterate_simple())
    assert len(gates) == 2

    name, angles, qubits = gates[0]
    assert name == "PH"
    assert len(angles) == 1
    assert angles[0] == pytest.approx(math.pi / 4)
    assert qubits == [0]

    name, angles, qubits = gates[1]
    assert name == "U"
    assert qubits == [0]

    assert len(angles) == 3
    for angle in angles:
        assert angle == pytest.approx(math.pi / 4)


def test_custom_gate():
    """
    Ensure a custom gate can be registered
    """
    data = """
    OPENQASM 2.0;

    // Allocating registers
    qreg q[1];
    creg c[1];

    // Apply gates
    custom q[0];
    """

    my_custom = AbstractGate("my_custom", [], arity=1)
    parser = OqasmParser(gates={"custom": my_custom}, include_matrices=False)
    circuit = parser.compile(data)

    gates = list(circuit.iterate_simple())
    assert len(gates) == 1

    name, angles, qubits = gates[0]
    assert name == "my_custom"
    assert angles == []
    assert qubits == [0]
    
