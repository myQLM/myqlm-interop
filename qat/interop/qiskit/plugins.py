# -*- coding: utf-8 -*-

"""
@file ...
@namespace ...
@authors Léo HUIN <leo.huin@atos.net>
@copyright  2019-2020 Bull S.A.S.  -  All rights reserved.
            This is not Free or Open Source software.
            Please contact Bull SAS for details about its license.
            Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois
@brief  Qiskit plugins
"""

import qiskit
from qat.core import Batch, HardwareSpecs, Result
from qat.core import Topology, TopologyType
from qat.core.plugins import AbstractPlugin
from .converters import qlm_to_qiskit, qiskit_to_qlm
from qiskit.transpiler.coupling import CouplingMap
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes.routing.basic_swap import BasicSwap
from qiskit.transpiler.passes.routing.stochastic_swap import StochasticSwap
from qiskit.transpiler.passes.routing.lookahead_swap import LookaheadSwap
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit import QuantumCircuit

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


class QiskitNnizer(AbstractPlugin):
    """
    QLM's wrapper for the Qiskit's Nizzers
    """
    def __init__(self, method="basic_swap", trials=20, seed=None, search_depth=4, search_width=4):
        """
        Args:
            method: (str) a string that correspond to a qiskit routing method.
                See the qiskit's documentation
            trials: (int) the number of trials before it fail
            seed: (int) a seed can be provide when using the 'stochastic_swap' method (that is random)
            search_depth: (int) the search depth for he 'lookahead_swap' method
            search_width: (int) the search width for the 'lookahead_swap' method
        """
        super().__init__()
        self.method = method
        self.trials = trials
        self.seed = seed
        self.search_depth = search_depth
        self.search_width = search_width

    def compile(self, batch: Batch, specs: HardwareSpecs) -> Batch:
        """
        Method that use Qiskit's Nnizers so that batch's circuits match the HardwareSpecs's Topology.
        """
        new_jobs = []
        for job in batch.jobs:
            coupling_list, description = get_coupling_list_from_specs(specs, job.circuit.nbqbits)
            coupling_map = CouplingMap(couplinglist=coupling_list, description=description)
            nnizer = None
            if self.method == "basic_swap":
                nnizer = BasicSwap(coupling_map)
            elif self.method == "stochastic_swap":
                nnizer = StochasticSwap(coupling_map, self.trials, self.seed)
            elif self.method == "lookahead_swap":
                nnizer = LookaheadSwap(coupling_map, self.search_depth, self.search_width)
            else:
                raise ValueError("The method '" + self.method + "' does not exist")
            circ = qlm_to_qiskit(job.circuit)
            circ.qregs[0].name = 'q'
            dag = circuit_to_dag(circ)
            new_dag = nnizer.run(dag)
            new_circ = dag_to_circuit(new_dag)
            qlm_circ, _ = qiskit_to_qlm(new_circ, True)
            new_jobs.append(qlm_circ.to_job())
        return Batch(new_jobs)

    def post_process(self, batch_result):
        return batch_result

    def do_post_processing(self) -> bool:
        return False

    def __call__(self, method=None, trials=None, seed=None, search_depth=None, search_width=None):
        if method:
            self.method = method
        if trials:
            self.trials = trials
        if seed:
            self.seed = seed
        if search_depth:
            self.search_depth = search_depth
        if search_width:
            self.search_width = search_width
        return self
