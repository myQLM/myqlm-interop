#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@brief 

@namespace ...
@authors Reda Drissi <mohamed-reda.drissi@atos.net>
@copyright 2019  Bull S.A.S.  -  All rights reserved.
           This is not Free or Open Source software.
           Please contact Bull SAS for details about its license.
           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois


Description ...


"""

from pkgutil import extend_path
# Try to find other QAT packages in other folders
__path__ = extend_path(__path__, __name__)


from pkg_resources import parse_version
import qiskit
import warnings

if parse_version(qiskit.__qiskit_version__['qiskit']) > parse_version('0.10.3'):
    warnings.warn("qiskit version {} is not tested, use version 0.10.3"
                  .format(qiskit.__qiskit_version__['qiskit']))

if parse_version(qiskit.__qiskit_version__['qiskit-terra']) > parse_version('0.8.1'):
    warnings.warn("qiskit-terra version {} is not tested, use version 0.8.1"
                  .format(qiskit.__qiskit_version__['qiskit-terra']))

if parse_version(qiskit.__qiskit_version__['qiskit-ignis']) > parse_version('0.1.1'):
    warnings.warn("qiskit-ignis version {} is not tested, use version 0.1.1"
                  .format(qiskit.__qiskit_version__['qiskit-ignis']))

if parse_version(qiskit.__qiskit_version__['qiskit-aer']) > parse_version('0.2.1'):
    warnings.warn("qiskit-aer version {} is not tested, use version 0.2.1"
                  .format(qiskit.__qiskit_version__['qiskit-aer']))

if parse_version(qiskit.__qiskit_version__['qiskit-ibmq-provider']) > parse_version('0.2.2'):
    warnings.warn("qiskit-ibmq-provider version {} is not tested, use version 0.2.2"
                  .format(qiskit.__qiskit_version__['qiskit-ibmq-provider']))

if parse_version(qiskit.__qiskit_version__['qiskit-aqua']) > parse_version('0.5.1'):
    warnings.warn("qiskit-aqua version {} is not tested, use version 0.5.1"
                  .format(qiskit.__qiskit_version__['qiskit-aqua']))


