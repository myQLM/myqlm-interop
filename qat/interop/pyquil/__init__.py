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
import pyquil
import grove
import warnings

if parse_version(pyquil.__version__) > parse_version('2.7.2'):
    warnings.warn("pyquil version {} is not tested, use version 2.7.2"
                  .format(pyquil.__version__))

if parse_version(grove.__version__) > parse_version('1.7.0'):
    warnings.warn("grove version {} is not tested, use version 1.7.0"
                  .format(grove.__version__))
