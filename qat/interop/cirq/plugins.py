# -*- coding: utf-8 -*-

"""
@file ...
@namespace ...
@authors Léo HUIN <leo.huin@atos.net>
@copyright  2019-2020 Bull S.A.S.  -  All rights reserved.
            This is not Free or Open Source software.
            Please contact Bull SAS for details about its license.
            Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois
@brief  Cirq plugins
"""

from qat.core import Batch, HardwareSpecs, Result
from qat.core import Topology, TopologyType
from qat.core.plugins import AbstractPlugin
from qat.interop.cirq.converters import qlm_to_cirq, cirq_to_qlm
import networkx as nx
import importlib.abc
import cirq
import cirq.contrib.routing as ccr

def get_coupling_list_from_specs_cirq(specs: HardwareSpecs, nbqbits=None):
    """
    Function that create a coupling list from a HardwareSpecs
    and the number of qbits.

    Args:
        specs: (qat.core.HardwareSpecs) a HardwareSpecs object
        with its topology
        nbqbits: (int) the number of qbits

    Returns:
        (list): list of pair of cirq.LineQubits [(qbit, qbit), (qbit, qbit), ...]
        (str): the description of the topology between
        'ALL_TO_ALL', 'LNN' and 'CUSTOM'
    """
    coupling_list = []
    if specs.topology.type == TopologyType.ALL_TO_ALL:
        assert nbqbits != None, "nbqbits not specified"
        for i in range(nbqbits):
            for j in range(nbqbits):
                if j > i:
                    coupling_list.append([cirq.LineQubit(i), cirq.LineQubit(j)])
        description = "ALL_TO_ALL"
    elif specs.topology.type == TopologyType.LNN:
        assert nbqbits != None, "nbqbits not specified"
        for i in range(1, nbqbits):
            coupling_list.append([cirq.LineQubit(i - 1), cirq.LineQubit(i)])
        description = "LNN"
    else:
        for i in range(specs.nbqbits):
            for qbit in specs.topology.graph[i]:
                if qbit > i:
                    coupling_list.append([cirq.LineQubit(i), cirq.LineQubit(qbit)])
        description = "CUSTOM"
    return coupling_list, description
            

class CirqNnizer(AbstractPlugin):
    """
    QLM's wrapper for the Cirq's Nizzers
    """
    def __init__(self, method="greedy"):
        """
        Args:
            method: (str) A string that correspond to a cirq routing method.
                See the cirq's documentation
        """
        super().__init__()
        self.method = method

    def compile(self, batch: Batch, specs: HardwareSpecs) -> Batch:
        """
        Method that use Cirq's Nnizers so that batch's circuits match the HardwareSpecs's Topology.
        """
        new_jobs = []
        for job in batch.jobs:
            coupling_list, _ = get_coupling_list_from_specs_cirq(specs, job.circuit.nbqbits)
            g = nx.Graph(coupling_list)
            circ = qlm_to_cirq(job.circuit)
            new_circ = ccr.route_circuit(circ, g, router=None, algo_name=self.method).circuit
            qlm_circ, _ = cirq_to_qlm(new_circ, True)
            new_jobs.append(qlm_circ.to_job())
        return Batch(new_jobs)

    def post_process(self, batch_result):
        return batch_result

    def do_post_processing(self) -> bool:
        return False

    def __call__(self, method=None):
        if method:
            self.method = method
        return self
