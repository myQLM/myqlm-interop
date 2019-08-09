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
import projectq
import warnings

if parse_version(projectq.__version__) > parse_version('0.4.2'):
    warnings.warn("projectq version {} is not tested, use version 0.4.2"
                  .format(projectq.__version__))


