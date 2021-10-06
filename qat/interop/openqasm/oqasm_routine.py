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


class Routine:

    def __init__(self, src=None):
        if src:
            self.copy_constr(src)
        else:
            self.name = None  # name of the gate
            self.params = []  # parameters
            self.args = []  # qbit arguments
            self.glist = []  # list of gates

    def copy_constr(self, src):
        self.name = src.name
        self.params = src.params
        self.args = src.args
        self.glist = src.glist


class Gate:
    def __init__(self):
        self.name = None  # name of the gate
        self.params = []  # parameters
        self.qblist = []  # qbit arguments


class Element:
    def __init__(self):
        self.name = None
        self.index = None
        self.value = None
