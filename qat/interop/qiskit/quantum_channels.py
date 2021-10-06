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

from qiskit.quantum_info.operators.channel import Choi, PTM, Kraus, Chi, SuperOp

import numpy as np

from qat.comm.quops.ttypes import QuantumChannel, RepresentationType
from qat.comm.datamodel.ttypes import Matrix, ComplexNumber


def array_to_matrix(array):
    """
    Transform a two dimmentional numpy array to a myqlm Matrix.

    Args:
        array: (ndarray) a two dimmentional numpy array

    Returns:
        (Matrix): a myqlm Matrix
    """

    assert len(array.shape) == 2, "The array must be two dimmentional"
    data = []
    for arr in array:
        for elem in arr:
            data.append(ComplexNumber(np.real(elem), np.imag(elem)))
    matri = Matrix(array.shape[0], array.shape[1], data)
    return matri


def qiskit_to_qchannel(representation):
    """
    Create a myqlm representation of quantum channel from a qiskit representation
    of a quantum channel.

    Args:
        representation: (Kraus|Choi|Chi|SuperOp|PTM) qiskit representation of a quantum channel.

    Returns:
        (QuantumChannel): myqlm representation of a quantum channel.
    """

    qchannel = None
    qiskit_data = representation.data
    # Find what representation it is.
    # Then create the corresponding matrix (kraus_ops|basis|matrix)from the data
    # of the representation.
    # Finally, create the QuantumChannel with the RepresentationType, the arity
    # (got from the qiskit representation) and the matrix.
    if isinstance(representation, Kraus):
        kraus_ops = []
        for arr in qiskit_data:
            kraus_ops.append(array_to_matrix(arr))
        qchannel = QuantumChannel(
            representation=RepresentationType.KRAUS,
            arity=representation.num_qubits,
            kraus_ops=kraus_ops)
    elif isinstance(representation, Chi):
        basis = []
        basis.append(array_to_matrix(qiskit_data))
        qchannel = QuantumChannel(
            representation=RepresentationType.CHI,
            arity=representation.num_qubits,
            basis=basis)
    elif isinstance(representation, SuperOp):
        basis = []
        basis.append(array_to_matrix(qiskit_data))
        qchannel = QuantumChannel(
            representation=RepresentationType.SUPEROP,
            arity=representation.num_qubits,
            basis=basis)
    elif isinstance(representation, PTM):
        matri = array_to_matrix(qiskit_data)
        qchannel = QuantumChannel(
            representation=RepresentationType.PTM,
            arity=representation.num_qubits,
            matrix=matri)
    elif isinstance(representation, Choi):
        matri = array_to_matrix(qiskit_data)
        qchannel = QuantumChannel(
            representation=RepresentationType.CHOI,
            arity=representation.num_qubits,
            matrix=matri)
    return qchannel


def qchannel_to_qiskit(representation):
    """
    Create a qiskit representation of quantum channel from a myqlm representation
    of a quantum channel.

    Args:
        representation: (QuantumChannel) myqlm representation of a quantum channel.

    Returns:
        (Kraus|Choi|Chi|SuperOp|PTM): qiskit representation of a quantum channel.
    """

    rep = representation.representation
    # Find what representation it is.
    # Then create the corresponding matrix and shape it like qiskit is expecting it.
    # Finally, create the qiskit representation from that matrix.
    if rep in (RepresentationType.PTM, RepresentationType.CHOI):
        matri = representation.matrix
        data_re = []
        data_im = []
        for i in range(matri.nRows):
            for j in range(matri.nCols):
                data_re.append(matri.data[i * matri.nRows + j].re + 0.j)
                data_im.append(matri.data[i * matri.nRows + j].im)
        data = np.array(data_re)
        data.imag = np.array(data_im)
        data = data.reshape((matri.nRows, matri.nCols))
        return PTM(data) if (rep == RepresentationType.PTM) else Choi(data)
    if rep in (RepresentationType.CHI, RepresentationType.SUPEROP):
        final_data = []
        for matri in representation.basis:
            data_re = []
            data_im = []
            for i in range(matri.nRows):
                for j in range(matri.nCols):
                    data_re.append(matri.data[i * matri.nRows + j].re + 0.j)
                    data_im.append(matri.data[i * matri.nRows + j].im)
            data = np.array(data_re)
            data.imag = np.array(data_im)
            data = data.reshape((matri.nRows, matri.nCols))
            final_data.append(data)
        if rep == RepresentationType.CHI:
            return Chi(final_data) if len(final_data) > 1 else Chi(final_data[0])
        return SuperOp(final_data) if len(final_data) > 1 else SuperOp(final_data[0])
    if rep == RepresentationType.KRAUS:
        final_data = []
        for matri in representation.kraus_ops:
            data_re = []
            data_im = []
            for i in range(matri.nRows):
                for j in range(matri.nCols):
                    data_re.append(matri.data[i * matri.nRows + j].re + 0.j)
                    data_im.append(matri.data[i * matri.nRows + j].im)
            data = np.array(data_re)
            data.imag = np.array(data_im)
            data = data.reshape((matri.nRows, matri.nCols))
            final_data.append(data)
        return Kraus(final_data)
    return None
