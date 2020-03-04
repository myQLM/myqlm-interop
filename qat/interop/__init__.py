#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@brief

@authors Reda Drissi <mohamed-reda.drissi@atos.net>
@copyright 2019-2020 Bull S.A.S.  -  All rights reserved.
           This is not Free or Open Source software.
           Please contact Bull SAS for details about its license.
           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois



Overview
=========


"""

from pkgutil import extend_path
# Try to find other QAT packages in other folders
__path__ = extend_path(__path__, __name__)


from qat.core.version import VERSION
from pkg_resources import parse_version

if parse_version(VERSION) < parse_version('0.0.6'):
    import warnings
    warnings.warn("Qiskit and Pyquil providers and algorithms are not "
                  "compatible with version 0.0.5 and below")
