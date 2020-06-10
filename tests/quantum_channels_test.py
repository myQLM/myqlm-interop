# -*- coding: utf-8 -*-

"""
@namespace ...
@authors Léo HUIN <leo.huin@atos.net>
@copyright  2019-2020 Bull S.A.S.  -  All rights reserved.
            This is not Free or Open Source software.
            Please contact Bull SAS for details about its license.
            Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois
@brief Test suite for quantum channels translate functions.
"""

import unittest
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info.operators.channel import Choi, PTM, Kraus, Chi, SuperOp
from qat.interop.qiskit.quantum_channels import array_to_matrix, \
    qchannel_to_qiskit, qiskit_to_qchannel
from qat.comm.quops.ttypes import QuantumChannel, RepresentationType

class TestTraduction(unittest.TestCase):
    """
    Unitary tests for quantum channels translate functions
    """

    def test_from_qiskit(self):
        """
        Test quantum channels created from qiskit are equals to quantum channels
        created by qiskit -> to myqlm -> to qiskit
        """
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.x(2)
        self.assertEqual(Kraus(circuit), qchannel_to_qiskit(qiskit_to_qchannel(Kraus(circuit))))
        self.assertEqual(Chi(circuit), qchannel_to_qiskit(qiskit_to_qchannel(Chi(circuit))))
        self.assertEqual(Choi(circuit), qchannel_to_qiskit(qiskit_to_qchannel(Choi(circuit))))
        self.assertEqual(SuperOp(circuit), qchannel_to_qiskit(qiskit_to_qchannel(SuperOp(circuit))))
        self.assertEqual(PTM(circuit), qchannel_to_qiskit(qiskit_to_qchannel(PTM(circuit))))

    def test_from_myqlm_superop(self):
        """
        Test all combinations of superop quantum channel between myqlm and qiskit
        """
        arr = np.arange(16*16, dtype=complex).reshape((16, 16))
        basis = [array_to_matrix(arr)]
        qchannel = QuantumChannel(representation=RepresentationType.SUPEROP, arity=2, basis=basis)
        qiskit_qchannel = SuperOp(arr)
        self.assertEqual(qchannel, qiskit_to_qchannel(qchannel_to_qiskit(qchannel)))
        self.assertEqual(qiskit_qchannel, qchannel_to_qiskit(qchannel))
        self.assertEqual(qchannel, qiskit_to_qchannel(qiskit_qchannel))

    def test_from_myqlm_chi(self):
        """
        Test all combinations of chi quantum channel between myqlm and qiskit
        """
        arr = np.arange(64*64, dtype=complex).reshape((64, 64))
        basis = [array_to_matrix(arr)]
        qchannel = QuantumChannel(representation=RepresentationType.CHI, arity=3, basis=basis)
        qiskit_qchannel = Chi(arr)
        self.assertEqual(qchannel, qiskit_to_qchannel(qchannel_to_qiskit(qchannel)))
        self.assertEqual(qiskit_qchannel, qchannel_to_qiskit(qchannel))
        self.assertEqual(qchannel, qiskit_to_qchannel(qiskit_qchannel))

    def test_from_myqlm_ptm(self):
        """
        Test all combinations of ptm quantum channel between myqlm and qiskit
        """
        arr = np.arange(4*4, dtype=complex).reshape((4, 4))
        matri = array_to_matrix(arr)
        qchannel = QuantumChannel(representation=RepresentationType.PTM, arity=1, matrix=matri)
        qiskit_qchannel = PTM(arr)
        self.assertEqual(qchannel, qiskit_to_qchannel(qchannel_to_qiskit(qchannel)))
        self.assertEqual(qiskit_qchannel, qchannel_to_qiskit(qchannel))
        self.assertEqual(qchannel, qiskit_to_qchannel(qiskit_qchannel))

    def test_from_myqlm_choi(self):
        """
        Test all combinations of choi quantum channel between myqlm and qiskit
        """
        arr = np.arange(16*16, dtype=complex).reshape((16, 16))
        matri = array_to_matrix(arr)
        qchannel = QuantumChannel(representation=RepresentationType.CHOI, arity=2, matrix=matri)
        qiskit_qchannel = Choi(arr)
        self.assertEqual(qchannel, qiskit_to_qchannel(qchannel_to_qiskit(qchannel)))
        self.assertEqual(qiskit_qchannel, qchannel_to_qiskit(qchannel))
        self.assertEqual(qchannel, qiskit_to_qchannel(qiskit_qchannel))

    def test_from_myqlm_kraus(self):
        """
        Test all combinations of kraus quantum channel between myqlm and qiskit
        """
        arr = np.arange(8*8, dtype=complex).reshape((8, 8))
        kraus_ops = [array_to_matrix(arr)]
        qchannel = QuantumChannel(
            representation=RepresentationType.KRAUS,
            arity=3,
            kraus_ops=kraus_ops
        )
        qiskit_qchannel = Kraus(arr)
        self.assertEqual(qchannel, qiskit_to_qchannel(qchannel_to_qiskit(qchannel)))
        self.assertEqual(qiskit_qchannel, qchannel_to_qiskit(qchannel))
        self.assertEqual(qchannel, qiskit_to_qchannel(qiskit_qchannel))

if __name__ == "__main__":
    unittest.main()
