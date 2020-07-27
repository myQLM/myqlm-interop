# -*- coding: utf-8 -*-

"""
@file ...
@namespace ...
@authors Léo HUIN <leo.huin@atos.net>
@copyright  2019-2020 Bull S.A.S.  -  All rights reserved.
            This is not Free or Open Source software.
            Please contact Bull SAS for details about its license.
            Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois
@brief  Pytket plugins
"""

from qat.core import Batch, HardwareSpecs, Result
from qat.core import Topology, TopologyType
from qat.core.plugins import AbstractPlugin
from qat.interop.cirq.converters import qlm_to_cirq, cirq_to_qlm
import importlib.abc
import cirq
import pytket
from pytket import Architecture
from pytket import route
from .cirq_convert import tk_to_cirq, cirq_to_tk

def get_coupling_list_from_specs(specs: HardwareSpecs, nbqbits=None):
    """
    Function that create a coupling list from a HardwareSpecs
    and the number of qbits.

    Args:
        specs: (qat.core.HardwareSpecs) a HardwareSpecs object
        with its topology
        nbqbits: (int) the number of qbits

    Returns:
        (list): list of pair of qbits [(qbit, qbit), (qbit, qbit), ...]
        (str): the description of the topology between
        'ALL_TO_ALL', 'LNN' and 'CUSTOM'
    """
    coupling_list = []
    if specs.topology.type == TopologyType.ALL_TO_ALL:
        assert nbqbits != None, "nbqbits not specified"
        for i in range(nbqbits):
            for j in range(nbqbits):
                if j > i:
                    coupling_list.append([i, j])
        description = "ALL_TO_ALL"
    elif specs.topology.type == TopologyType.LNN:
        assert nbqbits != None, "nbqbits not specified"
        for i in range(1, nbqbits):
            coupling_list.append([i - 1, i])
        description = "LNN"
    else: 
        for i in range(specs.nbqbits):
            for qbit in specs.topology.graph[i]:
                if qbit > i:
                    coupling_list.append([i, qbit])
        description = "CUSTOM"
    return coupling_list, description


class PytketNnizer(AbstractPlugin):
    """
    QLM's wrapper for the Pytket's Nizzers
    """
    def __init__(self):
        super().__init__()

    def compile(self, batch: Batch, specs: HardwareSpecs) -> Batch:
        """
        Method that use Pytket's Nnizers so that batch's circuits match the HardwareSpecs's Topology.
        """
        new_jobs = []
        for job in batch.jobs:
            coupling_list, _ = get_coupling_list_from_specs(specs, job.circuit.nbqbits)
            arch = Architecture(coupling_list)
            circ = cirq_to_tk(qlm_to_cirq(job.circuit))
            new_circ = route(circ, arch)._get_circuit()
            new_circ = tk_to_cirq(new_circ)
            cirq_circ, _ = cirq_to_qlm(new_circ, True)
            new_jobs.append(cirq_circ.to_job())
        return Batch(new_jobs)

    def post_process(self, batch_result):
        return batch_result

    def do_post_processing(self) -> bool:
        return False

    def __call__(self):
        return self
