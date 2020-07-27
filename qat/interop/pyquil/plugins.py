# -*- coding: utf-8 -*-

"""
@file ...
@namespace ...
@authors Léo HUIN <leo.huin@atos.net>
@copyright  2019-2020 Bull S.A.S.  -  All rights reserved.
            This is not Free or Open Source software.
            Please contact Bull SAS for details about its license.
            Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois
@brief  Pyquil plugins
"""

from qat.core import Batch, HardwareSpecs, Result
from qat.core import Topology, TopologyType
from qat.core.plugins import AbstractPlugin
from qat.interop.pyquil.converters import qlm_to_pyquil, pyquil_to_qlm
from pyquil.quil import Pragma
from pyquil.pyqvm import PyQVM
from pyquil.api import QuantumComputer
from pyquil.device import NxDevice
from pyquil.api import QVMCompiler
from pyquil.api._compiler import _extract_program_from_pyquil_executable_response
import networkx as nx

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


class PyquilNnizer(AbstractPlugin):
    """
    QLM's wrapper for the Pyquil's Nizzers
    """
    def __init__(self, method="NAIVE", endpoint="tcp://localhost:5555", timeout=30):
        """
        Args:
            method: (str) a string that correspond to a pyquil routing method.
                See the pyquil's documentation
            endpoint: (str) the enpoint of the compiler
            timeout: (int) the time before it timeout
        """
        super().__init__()
        self.method = method
        self.endpoint = endpoint
        self.timeout = timeout

    def compile(self, batch: Batch, specs: HardwareSpecs) -> Batch:
        """
        Method that use Pyquil's Nnizers so that batch's circuits match the HardwareSpecs's Topology.
        """
        new_jobs = []
        for job in batch.jobs:
            program_pragma = Pragma('INITIAL_REWIRING', ['"' + self.method + '"'])
            circ = qlm_to_pyquil(job.circuit, program_pragma)
            qam = PyQVM(job.circuit.nbqbits)
            qbits = list(range(job.circuit.nbqbits))
            coupling_list, _ = get_coupling_list_from_specs(specs, job.circuit.nbqbits)
            topo = nx.from_edgelist(coupling_list)
            device = NxDevice(topo)
            compiler = QVMCompiler(endpoint=self.endpoint, device=device, timeout=self.timeout)
            qc = QuantumComputer(name="qc", qam=qam, device=device, compiler=compiler)
            executable = qc.compile(circ)
            new_circ = _extract_program_from_pyquil_executable_response(executable)
            qlm_circ, _ = pyquil_to_qlm(new_circ, True)
            new_jobs.append(qlm_circ.to_job())
        return Batch(new_jobs)

    def post_process(self, batch_result):
        return batch_result

    def do_post_processing(self) -> bool:
        return False

    def __call__(self, method=None, endpoint=None, timeout=None):
        if method:
            self.method = method
        if endpoint:
            self.endpoint = endpoint
        if timeout:
            self.timeout = timeout
        return self
