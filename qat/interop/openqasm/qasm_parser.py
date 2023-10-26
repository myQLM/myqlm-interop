# -*- coding: utf-8 -*-
# pylint: skip-file

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

import math  # noqa
import numpy as np
import re
import ply.yacc as yacc
from qat.lang.parser.gates_ast import ASTCircuitBuilder, GateAST
from qat.comm.datamodel.ttypes import OpType
from qat.interop.openqasm.qasm_lexer import OqasmLexer
from qat.interop.openqasm.oqasm_routine import Element, Routine, Gate

from qat.lang.AQASM import AbstractGate
from qat.core.circuit_builder.matrix_util import default_gate_set
from qat.core.gate_set import UnknownGate


class ParsingEOF(Exception):
    """
    Unexpected End Of File
    """

    def __str__(self):
        return "Missing END before EOF"


class ParsingError(Exception):
    """
    Standard exception for parsing errors.
    """

    def __init__(self, m):
        super(ParsingError, self).__init__()
        self.token = m

    def __str__(self):
        base = "Line {} : Parsing error around '{}'"
        return base.format(self.token.lineno, self.token.value)


class ImplementationError(Exception):
    """
    Exception raised when a feature is not supported
    """

    def __init__(self, m):
        super(ImplementationError, self).__init__()


class InvalidParameterNumber(Exception):
    """
    Exception raised when some parametrized gates is used
    with the wrong number of parameters.
    """

    def __init__(self, name, expected, arg_list, lineno):
        super(InvalidParameterNumber, self).__init__()
        self.expected = expected
        self.arg_list = arg_list
        self.name = name
        self.lineno = lineno

    def __str__(self):
        return "line {} : {} requires {} arguments ({} given)".format(
            self.lineno, self.name, self.expected, len(self.arg_list)
        )


def gen_U(theta, phi, lamda):
    """
    Generates the U / U3 gate matrix. The definition of this gate is based on:
    https://qiskit.org/documentation/stubs/qiskit.circuit.library.U3Gate.html (Sept 08, 2022)

    Args:
        theta:
        phi:
        lamda: lambda parameter

    Returns:
        numpy.ndarray U gate matrix
    """
    m11 = np.cos(theta / 2)
    m12 = -np.exp(1j * lamda) * np.sin(theta / 2)
    m21 = np.exp(1j * phi) * np.sin(theta / 2)
    m22 = np.exp(1j * (phi + lamda)) * np.cos(theta / 2)
    return np.array([[m11, m12], [m21, m22]], dtype=np.complex128)


def gen_dummy():
    return np.matrix([[1, 0], [0, 1]], dtype=np.complex128)


def gen_u1(lamda):
    return gen_U(0, 0, lamda)


def gen_u2(phi, lamda):
    return gen_U(math.pi / 2, phi, lamda)


def extract_inc(filename):
    """
    Helper function to inline included files
    """
    ret = None
    with open(filename, "r") as main_file:
        for line in main_file:
            res = re.search("^include .*$", line)
            if res:
                incfile = res.group(0).split(" ", 1)[1][1:-2]
                if incfile == "qelib1.inc":
                    continue
                incstr = extract_inc(incfile)
                with open(filename, "r") as main_split:
                    ret = main_split.read().split(line, 1)
                    ret[0] = ret[0] + incstr
                    ret[0] = ret[0] + ret[1]
        if ret is None:
            with open(filename, "r") as main_ret:
                return main_ret.read()
        else:
            return ret[0]


def eval_exp(rout_params, instr_params, actual_params):
    params_index = []
    eval_params = []
    eval_param_op = []
    # print("Params we got ")
    # print(rout_params)
    # print(instr_params)
    # print(actual_params)
    for param in instr_params:
        rout_index = 0
        if isinstance(param, str):
            if param == "+" or param == "-" or param == "*" or param == "/":
                eval_param_op.append(param)
            else:
                for k in rout_params:
                    if param == k:
                        # matches each local parameter inside the routine to
                        # the parameter value used in the main function
                        # print("appending str")
                        # print(actual_params[rout_index])
                        # print("pre str eval params")
                        # print(eval_params)
                        if len(eval_param_op) == 0:
                            eval_param_op.append(actual_params[rout_index])

                        elif (
                            eval_param_op[len(eval_param_op) - 1] == "+"
                            or eval_param_op[len(eval_param_op) - 1] == "-"
                            or eval_param_op[len(eval_param_op) - 1] == "*"
                            or eval_param_op[len(eval_param_op) - 1] == "/"
                        ):

                            eval_param_op.append(actual_params[rout_index])
                            eval_params.append(eval_param_op)
                            eval_param_op = []
                        else:
                            eval_params.append(eval_param_op)
                            eval_param_op = []
                            eval_param_op.append(actual_params[rout_index])
                        break
                    rout_index += 1
        elif isinstance(param, list):  # non evaluated expression
            p = eval_exp(rout_params, param, actual_params)
            # print("recursie params")
            # print(p)
            # print("pre list eval params")
            # print(eval_params)
            # if len(params_index) == 1:
            # print("found")
            # print(p)
            if len(eval_param_op) == 0:
                if len(p) > 0:
                    eval_param_op.append(p[0])

            elif (
                eval_param_op[len(eval_param_op) - 1] == "+"
                or eval_param_op[len(eval_param_op) - 1] == "-"
                or eval_param_op[len(eval_param_op) - 1] == "*"
                or eval_param_op[len(eval_param_op) - 1] == "/"
            ):
                eval_param_op.append(p[0])
                eval_params.append(eval_param_op)
                eval_param_op = []
            else:
                eval_params.append(eval_param_op)
                eval_param_op = []
                if len(p) > 0:
                    eval_param_op.append(p[0])
            # print("post list eval params")
            # print(eval_params)
        else:
            if len(eval_param_op) == 0:
                eval_param_op.append(param)

            elif (
                eval_param_op[len(eval_param_op) - 1] == "+"
                or eval_param_op[len(eval_param_op) - 1] == "-"
                or eval_param_op[len(eval_param_op) - 1] == "*"
                or eval_param_op[len(eval_param_op) - 1] == "/"
            ):
                eval_param_op.append(param)
                eval_params.append(eval_param_op)
                eval_param_op = []
            else:
                eval_params.append(eval_param_op)
                eval_param_op = []
                eval_param_op.append(param)
    if len(eval_param_op) > 0:
        eval_params.append(eval_param_op)
    # print("eval params")
    # print(eval_params)
    if True:
        # print("evals")
        # print(eval_params)
        for ev in eval_params:
            if len(ev) < 2:
                params_index.append(ev[0])
                continue
            if ev[1] == "-":
                params_index.append(ev[0] - ev[2])
            if ev[1] == "+":
                params_index.append(ev[0] + ev[2])
            if ev[1] == "*":
                params_index.append(ev[0] * ev[2])
            if ev[1] == "/":
                params_index.append(ev[0] / ev[2])
    # print("here is ")
    # print(params_index)
    return params_index


"""
OPENQASM parser.
"""


class OqasmParser(object):
    """
    Parser of OpenQASM 2.0 files. This class provides tools to translate a string object
    (containing OpenQASM 2.0 source code) into a :class:`~qat.core.Circuit`

    .. tab-set::

        .. tab-item:: Basic example

            .. run-block:: python

                from qat.interop.openqasm import OqasmParser

                # Define a dummy circuit
                data = \"\"\"
                OPENQASM 2.0;

                // Allocate qubits and cbits
                qreg q[1];
                creg c[1];

                // Apply gates and measure
                x q[0];
                measure q[0] -> c[0];
                \"\"\"

                # Translate into a myQLM circuit
                parser = OqasmParser()
                circuit = parser.compile(data)
                print("The circuit is composed of gates",
                      list(circuit.iterate_simple()))

        .. tab-item:: Advanced example

            .. run-block:: python

                from qat.lang.AQASM import AbstractGate

                # Define a circuit using OpenQASM
                OPENQASM_CODE = \"\"\"
                OPENQASM 2.0;

                // Allocating registers
                qreg q[1];
                creg c[1];

                // Dummy circuit
                p(pi/4) q[0];
                my_custom_gate q[0];
                \"\"\"

                # Register gates (i.e. "p" gate is an alias for "PH" gate, "my_custom_gate" is defined by an abstract gate)
                custom_gate = AbstractGate("custom_gate", [], arity)
                parser = OqasmParser(gates={"p": "PH", "my_custom_gate": custom_gate}, include_matrices=False)

                # Compile circuit and display it
                for gate, angles, qubits in parser.compile(OQASM_CODE).iterate_simple():
                    if angles:
                        print(f"Apply {gate}{angles} on qubits {qubits}")
                    else:
                        print(f"Apply {gate} on qubits {qubits}")

    Args:
        gates (dict[str, str or :class:`~qat.lang.AQASM.gates.Gate`], optional): definition of custom gates. These gates
            are defined using a dictionary, a key corresponding to the OpenQASM gate identifier and the key being:
            - a str: the name of the equivalent gate in myQLM (e.g. "PH", "X", "U", etc.)
            - a :class:`~qat.lang.AQASM.gates.Gate`: a custom gate
        include_matrices (bool, optional): include matrices in the generated circuit (default: True)
    """

    def __init__(self, gates: dict = None, include_matrices: bool = True):
        self.start = "main"
        self.format_version = False
        self.lineno = 0
        self.compiler = ASTCircuitBuilder(include_matrices=include_matrices)
        U = AbstractGate("U", [float, float, float], arity=1, matrix_generator=gen_U)
        u1 = AbstractGate("U1", [float], arity=1, matrix_generator=gen_u1)
        u2 = AbstractGate("U2", [float, float], arity=1, matrix_generator=gen_u2)
        u3 = AbstractGate("U3", [float, float, float], arity=1, matrix_generator=gen_U)
        gate_set = default_gate_set()
        gate_set.add_signature(U)
        gate_set.add_signature(u1)
        gate_set.add_signature(u2)
        gate_set.add_signature(u3)
        self.compiler.gate_set = gate_set
        """Create the parser."""
        self.lexer = OqasmLexer()
        self.input = ""
        self.lexer.build()
        self.tokens = self.lexer.tokens
        self.precedence = (
            ("left", "+", "-"),
            ("left", "*", "/"),
            ("left", "negative", "positive"),
            ("right", "^"),
        )
        self.parse_deb = False
        self.external_functions = [
            "sin",
            "cos",
            "tan",
            "exp",
            "ln",
            "sqrt",
            "acos",
            "atan",
            "asin",
        ]

        self.standard_gates = {
            "x": "X",
            "y": "Y",
            "z": "Z",
            "s": "S",
            "t": "T",
            "cx": "CNOT",
            "ccx": "CCNOT",
            "sdg": "SDG",
            "tdg": "TDG",
            "pH": "PH",
            "csign": "CSIGN",
            "cz": "CSIGN",
            "rx": "RX",
            "ry": "RY",
            "U": "U",
            "u1": "U1",
            "u2": "U2",
            "u3": "U3",
            "cu1": "CU1",
            "cu2": "CU2",
            "cu3": "CU3",
            "rz": "RZ",
            "swap": "SWAP",
            "crz": "CRZ",
            "cswap": "CSWAP",
            "h": "H",
            "ch": "CH",
            "id": "I",
        }
        self.routines = []
        self.nbcbits = 0
        self.nbqbits = 0
        self.cregs = []
        self.qregs = []

        # Register gate translation
        for gate_name, gate_definition in (gates or {}).items():
            # If gate is a string -> register alias
            if isinstance(gate_definition, str):
                self.standard_gates[gate_name] = gate_definition

            # If gate is a myQLM gate -> register gate
            else:
                self.standard_gates[gate_name] = gate_definition.name
                self.compiler.gate_set.add_signature(gate_definition)

    def add_creg(self, elem):
        self.cregs.append(elem)
        self.nbcbits += elem.index
        self.compiler.nbcbits += elem.index

    def add_qreg(self, elem):
        self.qregs.append(elem)
        self.nbqbits += elem.index
        self.compiler.nbqbits += elem.index

    def get_reg(self, name):
        for qreg in self.qregs:
            if qreg.name == name:
                return qreg
        for creg in self.cregs:
            if creg.name == name:
                return creg

    def creg_exists(self, name):
        for creg in self.cregs:
            if name == creg.name:
                return creg.index
        return 0

    def qreg_exists(self, name):
        for qreg in self.qregs:
            if name == qreg.name:
                return qreg.index
        return 0

    def get_indexed_id(self, name, offset):
        index = 0
        for creg in self.cregs:
            if name == creg.name:
                return index + offset
            else:
                index += creg.index
        index = 0
        for qreg in self.qregs:
            if name == qreg.name:
                return index + offset
            else:
                index += qreg.index
        return None

    ######################
    #  Helper Functions  #
    ######################

    def run(self, fct, arg):
        """
            Executes one of the known external functions
        Args:
            fct : external function name
            arg : argument on which the function operates
        """
        if fct == "sin":
            return math.sin(arg)
        elif fct == "cos":
            return math.cos(arg)
        elif fct == "tan":
            return math.tan(arg)
        elif fct == "exp":
            return math.exp(arg)
        elif fct == "ln":
            return math.log(arg)
        elif fct == "sqrt":
            return math.sqrt(arg)
        elif fct == "acos":
            return math.acos(arg)
        elif fct == "atan":
            return math.atan(arg)
        elif fct == "asin":
            return math.asin(arg)

    def is_reserved(self, name):
        """
            Verifies whether input name is reserved
        Args :
            name to check
        """
        return (
            name in self.external_functions
            or self.is_routine(name)
            or self.qreg_exists(name)
            or self.creg_exists(name)
        )

    def is_routine(self, name, args_size=None, params_size=None):
        """
            Checks whether input routine name already exists
        Args:
            name that needs checking
        """
        # print(str(len(self.routines)))
        for i in self.routines:
            # print("name is "+name+" but we got "+i.name)
            if i.name == name:
                return True
        return False

    def build_gate(self, name, params):
        """
        Build a quantum gate using a name and a set of parameters

        Args:
            name (str): gate name
            params

        Returns:
            GateAST
        """
        # Get number of parameters of the corresponding AST gate
        try:
            ast_gate_nb_params = self.compiler.gate_set[self.standard_gates[name]].nb_args
        except UnknownGate:
            sub_gate = self.standard_gates[name]
            if sub_gate.startswith("C"):
                sub_gate = sub_gate[1:]
            if sub_gate.endswith("DG"):
                sub_gate = sub_gate[:-2]
            ast_gate_nb_params = self.compiler.gate_set[sub_gate].nb_args

        # If gate is called with the right number of parameters
        if len(params) == ast_gate_nb_params:
            if self.standard_gates[name] == "SDG":
                ast = GateAST("S", params)
                return GateAST("DAG", ast)
            if self.standard_gates[name] == "TDG":
                ast = GateAST("T", params)
                return GateAST("DAG", ast)
            if self.standard_gates[name] == "CU1":
                ast = GateAST("U1", params)
                return GateAST("CTRL", ast)
            if self.standard_gates[name] == "CU2":
                ast = GateAST("U2", params)
                return GateAST("CTRL", ast)
            if self.standard_gates[name] == "CU3":
                ast = GateAST("U3", params)
                return GateAST("CTRL", ast)
            if self.standard_gates[name] == "CRZ":
                ast = GateAST("RZ", params)
                return GateAST("CTRL", ast)
            if self.standard_gates[name] == "CH":
                ast = GateAST("H", params)
                return GateAST("CTRL", ast)
            if self.standard_gates[name] == "CSWAP":
                ast = GateAST("SWAP", params)
                return GateAST("CTRL", ast)

            return GateAST(self.standard_gates[name], params)

        # Invalid number of parameters
        raise InvalidParameterNumber(
            self.standard_gates[name],
            ast_gate_nb_params,
            params,
            self.lineno,
        )

    def build_routine(self, routine_name, args, params):
        """
            Apply routine on function call
        Args:
            routine's name to apply
            routine's arguments (bit list)
            routine's parameters
        """
        res_routines = []
        # print("We are getting for this routine")
        # print(args)
        # print(params)
        routine = None
        for i in self.routines:  # let's get the routine first
            if i.name == routine_name:
                routine = i
        # print("Glist of routine ")
        # print(routine.glist)
        for instr in routine.glist:
            args_index = []  # each index's value will hold the value of
            # corresponding bit in our main code passed as args
            # in the routine call
            params_index = []  # same thing for parameters

            for qbit in instr.qblist:
                rout_index = 0
                for k in routine.args:
                    if qbit == k:
                        # matches each local c/qbit in our function to the
                        # qbit value used as a parameter in the main function
                        args_index.append(args[rout_index])
                        break
                    rout_index += 1

            # same thing for parameters
            # print("instr.params")
            # print(instr.params)
            # print("params")
            # print(params)
            if len(instr.params) > 0:
                if len(params) > 0:
                    params_index = eval_exp(routine.params, instr.params, params)
                else:
                    params_index = eval_exp(routine.params, instr.params, instr.params)
            # we should pass on newly mapped args list
            if self.is_routine(instr.name):
                res_routines.extend(
                    self.build_routine(instr.name, args_index, params_index)
                )
            elif instr.name == "measure":
                res_routines.append(
                    self.compiler.build_measure(args_index[0], args_index[1])
                )
            elif instr.name == "reset":
                res_routines.append(
                    self.compiler.build_reset(args_index[0], args_index[0])
                )
            else:
                ast = self.build_gate(instr.name, params_index)
                res_routines.append(self.compiler.build_op_by_ast(ast, args_index))

        return res_routines

    def build_gate_or_routine(self, name, qbit_list, params) -> list:
        """
        Build a gate or a routine based on a name, a list of qubits and a set of
        parameters

        Args:
            name (str): gate or routine name
            qbit_list (list): list of qubits
            params (list): list of parameters

        Returns:
            list: list of gates
        """
        if name in self.standard_gates:
            ast = self.build_gate(name, params)
            return [self.compiler.build_op_by_ast(ast, qbit_list)]

        if self.is_routine(name, len(qbit_list), len(params)):
            return self.build_routine(name, qbit_list, params)

        raise NameError(f"No such gate or routine {name!r} (or wrong number of arguments)")

    # ---- Begin the PLY parser ----

    # -----------------------------------------
    #  mainprogram : FORMAT ';' program
    # -----------------------------------------
    def p_main(self, t):
        """
        main : FORMAT ';' program
        """
        t[0] = self.compiler
        self.lineno += 1

    # ----------------------------------------
    #  program : statement
    #    | program statement
    # ----------------------------------------
    def p_program(self, t):
        """
        program : statement
          | program statement
        """
        # print(self.compiler.ops)
        # ----------------------------------------
        #  statement : decl
        #            | quantum_op ';'
        #            | FORMAT ';'
        #            | IGNORE
        #            | INCLUDE ';'
        # ----------------------------------------
        version = t.stack[1].value.rsplit(" ", 1)[-1]
        version = version.rsplit("\t", 1)[-1]
        version = version.rsplit("\n", 1)[-1]
        if version != "2.0" and not self.format_version:
            self.format_version = True
            print(
                "WARNING: Version {} not fully supported, only version".format(version)
                + " 2.0 is"
            )

    def p_statement_0(self, t):
        """
        statement : quantum_op ';'
        """
        self.compiler.ops.extend(t[1])
        self.lineno += 1

    def p_statement_1(self, t):
        """
        statement : decl
                  | IGNORE
                  | quantum_op error
                  | FORMAT error
        """
        # print("Started statement {} and {}".format(t[1], t[2] if len(t)>2 else "None"))
        if len(t) > 2:
            if t[2] != ";":
                raise SyntaxError(
                    "Missing ';' at end of statement; " + "received " + str(t[2])
                )
        if len(t) == 6:
            # print('got inside the include')
            with open(t[3], "r") as inc:
                self.parse(inc.read())
        self.lineno += 1

    # ----------------------------------------
    #  indexed_ID : ID [ int ]
    # ----------------------------------------
    def p_indexed_id(self, t):
        """
        indexed_ID : ID '[' NNINTEGER ']'
                   | ID '[' NNINTEGER error
                   | ID '[' error
        """
        if len(t) == 4:
            if not isinstance(t[3], int):
                raise ValueError("Expecting an integer index; received", str(t[3]))
        if t[4] != "]":
            raise SyntaxError("Missing ']' in indexed ID", str(t[4]))
        # indexed_ID becomes the name of the structure, and its size
        t[0] = Element()
        t[0].name = t[1]
        t[0].index = t[3]
        t[0].value = self.get_indexed_id(t[1], t[3])

    # ----------------------------------------
    #  decl : qreg_decl
    #       | creg_decl
    #       | gate_decl
    # ----------------------------------------
    def p_decl(self, t):
        """
        decl : qreg_decl ';'
             | creg_decl ';'
             | qreg_decl error
             | creg_decl error
             | gate_decl
        """
        if len(t) > 2:
            if t[2] != ";":
                raise SyntaxError(
                    "Missing ';' in qreg or creg declaration."
                    " Instead received '" + t[2] + "'"
                )
        t[0] = t[1]
        self.lineno += 1

    # ----------------------------------------
    #  qreg_decl : QREG indexed_ID
    # ----------------------------------------
    def p_qreg_decl(self, t):
        """
        qreg_decl : QREG indexed_ID
        """
        # print("we passed here")
        if self.is_reserved(t[2].name):
            raise ValueError(
                "QREG names cannot be reserved words. " + "Received '" + t[2].name + "'"
            )
        if t[2].index < 1:
            raise ValueError("QREG size must be positive")

        # if value isn't None, it means we found an existing register
        # with that name
        if t[2].value is not None:
            raise ValueError("QREG already declared")

        self.add_qreg(t[2])
        # print("Adding :")
        # print(str(t[2].name)+" and "+str(t[2].index)+" and "+str(t[2].value))

    def p_qreg_decl_e(self, t):
        """
        qreg_decl : QREG error
        """
        raise SyntaxError(
            "Expecting indexed ID (ID[int]) in QREG" + " declaration; received", t[2]
        )

    # ----------------------------------------
    #  creg_decl : CREG indexed_ID
    # ----------------------------------------
    def p_creg_decl(self, t):
        """
        creg_decl : CREG indexed_ID
        """
        if self.is_reserved(t[2].name):
            raise ValueError(
                "CREG names cannot be reserved words. " + "Received '" + t[2].name + "'"
            )
        if t[2].index < 1:
            raise ValueError("CREG size must be positive")

        # if value isn't None, then we found an object with the same name
        if t[2].value:
            raise ValueError("CREG already exists")

        self.add_creg(t[2])
        # TODO should be implemented as a key/value pair dict

    def p_creg_decl_e(self, t):
        """
        creg_decl : CREG error
        """
        raise SyntaxError(
            "Expecting indexed ID (ID[int]) in CREG" + " declaration; received", t[2]
        )

    # ----------------------------------------
    #  primary : ID
    #          | indexed_ID
    # ----------------------------------------
    def p_primary(self, t):
        # primaries would be lists to account for entire registers inputed
        """
        primary : ID
                | indexed_ID
        """
        if isinstance(t[1], Element):  # indexed_ID
            if t[1].value is None:  # value is None
                # print("your qreg")
                # print(str(t[1].name) + " and " + str(t[1].index) + " and "+str(t[1].value))
                t[0] = [t[1].name]
                # if not declared, we return the name instead of
                # int helps verify later
            elif self.qreg_exists(t[1].name):
                if t[1].value >= self.nbqbits:
                    raise IndexError(
                        t[1].name + "[" + t[1].index + "] is out of bounds"
                    )
                else:
                    t[0] = [t[1].value]
            elif self.creg_exists(t[1].name):
                if t[1].value >= self.nbcbits:
                    raise IndexError(
                        t[1].name + "[" + t[1].index + "] is out of bounds"
                    )
                else:
                    t[0] = [t[1].value]
            else:
                # print("All according to plan")
                t[0] = [t[1].value]
        else:  # register
            # print("A register maybe?")
            # print(t[1])
            # TODO get the offset of the first element of the reg and its size
            reg = self.get_reg(t[1])
            # returns the Element whose name is t[1]
            if not reg:
                print("WARNING: No such register : " + t[1])
            else:
                offset = self.get_indexed_id(t[1], 0)
                tab = []
                for i in range(0, reg.index):
                    tab.append(offset + i)
                # gets the a[i] value, so that tab contains
                # the full register's values
                t[0] = tab

    # ----------------------------------------
    #  id_list : ID
    #          | id_list ',' ID
    # ----------------------------------------
    def p_id_list_0(self, t):
        """
        id_list : ID
        """
        t[0] = [t[1]]

    def p_id_list_1(self, t):
        """
        id_list : id_list ',' ID
        """
        t[0] = t[1]
        t[0].append(t[3])

    # ----------------------------------------
    # These are routines parameters
    #  gate_id_list : ID
    #               | gate_id_list ',' ID
    # ----------------------------------------
    def p_gate_id_list_0(self, t):
        """
        gate_id_list : ID
        """
        t[0] = [t[1]]

    def p_gate_id_list_1(self, t):
        """
        gate_id_list : gate_id_list ',' ID
        """
        t[0] = t[1]
        t[0].append(t[3])

    # ----------------------------------------
    #  bit_list : bit
    #           | bit_list ',' bit
    # ----------------------------------------
    def p_bit_list_0(self, t):
        """
        bit_list : ID
        """
        t[0] = [t[1]]

    def p_bit_list_1(self, t):
        """
        bit_list : bit_list ',' ID
        """
        t[0] = t[1]
        t[0].append(t[3])

    # ----------------------------------------
    #  primary_list : primary
    #               | primary_list ',' primary
    # ----------------------------------------
    def p_primary_list_0(self, t):
        """
        primary_list : primary
        """
        t[0] = [t[1]]

    def p_primary_list_1(self, t):
        """
        primary_list : primary_list ',' primary
        """
        t[0] = t[1]
        if not isinstance(t[1][0], int) and not isinstance(t[1][0], list):
            raise ValueError("Register " + t[1][0] + " not declared")
        if not isinstance(t[3][0], int):
            raise ValueError("Register " + t[3][0] + " not declared")
        t[1].append(t[3])

    # TODO parse routine from standard gates

    # ----------------------------------------
    #  gate_decl : GATE ID gate_scope                      bit_list gate_body
    #            | GATE ID gate_scope '(' ')'              bit_list gate_body
    #            | GATE ID gate_scope '(' gate_id_list ')' bit_list gate_body
    #
    # ----------------------------------------
    def p_gate_decl(self, t):
        """
        gate_decl : GATE ID gate_scope bit_list gate_body
                  | GATE ID gate_scope '(' ')' bit_list gate_body
                  | GATE ID gate_scope '(' gate_id_list ')' bit_list gate_body
        """
        # print("Doing gate thingy")
        if self.is_reserved(t[2]):
            raise ValueError(
                "Cannot use reserved words, " + "or already declared objects"
            )
        elif t[2] in self.standard_gates:
            t[0] = None
        else:
            r = Routine()
            r.name = t[2]
            r.params = t[5] if len(t) == 9 else []
            r.args = t[len(t) - 2]
            r.glist = t[len(t) - 1]
            # for i in range(0, len(t)):
            # print("element of t " + str(i) + " is ")
            # print(t[i])
            # print("length of t is " + str(len(t)))
            # print("params")
            # print("glist")
            # each instruction's args needs to be checked against
            # the routine's args
            for j in r.glist:  # i will be a Gate
                # print("testing glists")
                for k in j.qblist:
                    # print("testing qblists")
                    if k not in r.args:
                        raise ValueError("Unknown qbit " + k + " for gate " + j.name)
                # print("over "+str(len(r.params))+" "+str(len(j.params)))
                # print("finished qb")
                for k in j.params:
                    # print("testing params")
                    if isinstance(k, str) and k not in r.params:
                        raise ValueError(
                            "Unknown parameter " + k + " for " + "gate " + j.name
                        )
                # print("ended g/q/p")
            # print("ended")
            # print("adding routine "+r.name)
            self.routines.append(r)
            # for i in self.routines:
            # print(i.name)
        self.lineno += 1

    def p_gate_decl_e(self, t):
        """
        gate_decl : GATE ID gate_scope bit_list error
                  | GATE ID gate_scope '(' ')' bit_list error
                  | GATE ID gate_scope '(' gate_id_list ')' bit_list error
        """

    def p_gate_scope(self, t):
        """
        gate_scope :
        """

    # ----------------------------------------
    #  gate_body : '{' gate_op_list '}'
    #            | '{' '}'
    #
    #            | '{' gate_op_list error
    #            | '{' error
    #
    # Error handling: gete_op will throw if there's a problem so we won't
    #                 get here with in the gate_op_list
    # ----------------------------------------
    def p_gate_body_0(self, t):
        """
        gate_body : '{' '}'
        """
        t[0] = []

    # already handled, the build_routine would do nothing
    def p_gate_body_1(self, t):
        """
        gate_body : '{' gate_op_list '}'
        """
        t[0] = t[2]

    # ----------------------------------------
    #  gate_op_list : gate_op
    #               | gate_op_ist gate_op
    #
    # Error handling: gete_op will throw if there's a problem so we won't
    #                 get here with errors
    # ----------------------------------------
    def p_gate_op_list_0(self, t):
        """
        gate_op_list : gate_op
        """
        t[0] = [t[1]]

    def p_gate_op_list_1(self, t):
        """
        gate_op_list : gate_op_list gate_op
        """
        t[0] = t[1]
        t[0].append(t[2])

    # ----------------------------------------
    # These are for use outside of gate_bodies and allow
    # indexed ids everywhere.
    #
    # unitary_op : U '(' exp_list ')'  primary
    #            | CX                  primary ',' primary
    #            | ID                  primary_list
    #            | ID '(' ')'          primary_list
    #            | ID '(' exp_list ')' primary_list
    #
    # Note that it might not be unitary - this is the mechanism that
    # is also used to invoke calls to 'opaque'
    # ----------------------------------------

    def p_unitary_op_1(self, t):
        """
        unitary_op : CX primary ',' primary
        """
        if len(t[2]) != len(t[4]):
            raise ValueError("Registers not of the same size")
        t[0] = []
        for qbit in range(0, len(t[2])):
            quantum_op = Gate()
            quantum_op.name = "cx"
            quantum_op.params = []
            quantum_op.qblist = [t[2][qbit], t[4][qbit]]
            # add operation for each qbit in the register
            t[0].append(quantum_op)
        # TODO: check that if both primary are id, same size
        # TODO: this needs to be checked in other cases too

    def p_unitary_op_2(self, t):
        """
        unitary_op : U '(' exp_list ')' primary
        """
        t[0] = []
        for qbit in range(0, len(t[5])):
            quantum_op = Gate()
            quantum_op.name = t[1]
            quantum_op.params = t[3]
            quantum_op.qblist = [qbit]
            t[0].append(quantum_op)

    def p_unitary_op_3(self, t):
        """
        unitary_op : ID primary_list
                    | ID '(' ')' primary_list
                    | ID '(' exp_list ')' primary_list
        """
        if t[1] not in self.tokens:
            t[2] = t[len(t) - 1]
            # put the last element in t[2] to avoid all the if statements
            # print(t[2])
            max_len = len(max(t[2], key=len))
            # gets the size of the largest register in the input list
            for i in range(0, len(t[2])):
                if len(t[2][i]) > 1 and max_len != len(t[2][i]):
                    raise ValueError(
                        "Registers are not of the "
                        "same size for operation '" + t[1] + "'"
                    )
                # if this passes then all registers are of the same size
                t[0] = []
            for reg in range(0, max_len):
                # cycle through all elements of registers
                unitary_op = Gate()
                unitary_op.name = t[1]
                unitary_op.params = [] if len(t) < 6 else t[3]
                unitary_op.qblist = []
                for i in range(0, len(t[2])):
                    if len(t[2][i]) == 1:  # a single value
                        unitary_op.qblist.append(t[2][i][0])
                    else:  # a register
                        unitary_op.qblist.append(t[2][i][reg])
                t[0].append(unitary_op)

    # ----------------------------------------
    # This is a restricted set of "quantum_op" which also
    # prohibits indexed ids, for use in a gate_body
    #
    # gate_op : U '(' exp_list ')'  ID         ';'
    #         | CX                  ID ',' ID  ';'
    #         | ID                  id_list    ';'
    #         | ID '(' ')'          id_list    ';'
    #         | ID '(' exp_list ')' id_list    ';'
    #         | BARRIER id_list                ';'
    # ----------------------------------------
    def p_gate_op_0(self, t):
        """
        gate_op : U '(' exp_list ')' ID ';'
        """
        t[0] = Gate()
        t[0].name = "U"  # this needs to go back to U
        t[0].params = t[3]
        t[0].qblist = [t[5]]
        self.lineno += 1

    def p_gate_op_0e1(self, t):
        """
        gate_op : U '(' exp_list ')' error
        """
        raise ValueError(
            "InvalID U inside gate definition. Missing bit"
            + "ID or ';' on line :"
            + str(t.lineno)
        )

    def p_gate_op_0e2(self, t):
        """
        gate_op : U '(' exp_list error
        """
        raise SyntaxError("Missing ')' in U invocation in gate definition.")

    def p_gate_op_1(self, t):
        """
        gate_op : CX ID ',' ID ';'
        """
        t[0] = Gate()
        t[0].name = "cx"
        t[0].params = []
        t[0].qblist = [t[2], t[4]]
        self.lineno += 1

    def p_gate_op_1e1(self, t):
        """
        gate_op : CX error
        """
        raise SyntaxError(
            "InvalID CNOT inside gate definition. "
            + "Expected an ID or ',', received '"
            + str(t[2])
            + "'"
        )

    def p_gate_op_1e2(self, t):
        """
        gate_op : CX ID ',' error
        """
        raise SyntaxError(
            "InvalID CNOT inside gate definition. "
            + "Expected an ID or ';', received '"
            + str(t[4])
            + "'"
        )

    def p_gate_op_2(self, t):
        """
        gate_op : ID id_list ';'
                | ID '(' ')' id_list ';'
                | ID '(' exp_list ')' id_list ';'
        """
        t[0] = Gate()
        try:
            self.standard_gates[t[1]]
        except KeyError:
            if t[1] not in self.tokens:
                # not recognized by name_gate
                if not self.is_routine(
                    t[1], len(t[len(t) - 2]), 0 if len(t) < 7 else len(t[3])
                ):
                    # print(" params : "+str(len(t[3]))+ " and args "+str(len(t[len(t)-2])))
                    raise ValueError("No such gate or routine " + t[1])
            # will be a routine, this might change if we handle
            # routine differently than gates
        t[0].name = t[1]
        t[0].params = [] if len(t) < 7 else t[3]
        # in any case there are no params here
        t[0].qblist = t[len(t) - 2]

        # TODO verify:
        # ID is declared as a gate in global scope
        # everything in the id_list is declared as a bit in local scope
        self.lineno += 1

    def p_gate_op_2e0(self, t):
        """
        gate_op : ID  id_list error
                | ID '(' ')' error
                | ID '(' error
        """
        if len(t) == 4:
            if t[2] != "(":
                raise ValueError("InvalID gate invocation inside gate " + "definition.")
            else:
                raise SyntaxError(
                    "InvalID bit list inside gate definition" + " or missing ';'"
                )
        else:
            raise SyntaxError("Unmatched () for gate invocation inside " + "gate")

    def p_gate_op_5(self, t):
        """
        gate_op : BARRIER id_list ';'
        """
        # TODO no idea how to deal with this, really should though
        self.lineno += 1

    def p_gate_op_5e(self, t):
        """
        gate_op : BARRIER error
        """
        raise SyntaxError("InvalID barrier inside gate definition.")

    # ----------------------------------------
    # opaque : OPAQUE ID gate_scope                      bit_list
    #        | OPAQUE ID gate_scope '(' ')'              bit_list
    #        | OPAQUE ID gate_scope '(' gate_id_list ')' bit_list
    #
    # These are like gate declarations only wihtout a body.
    # ----------------------------------------
    def p_opaque_0(self, t):
        """
        opaque : OPAQUE ID gate_scope bit_list
        """
        # TODO: even less of an idea than barrier how to deal with this

    def p_opaque_1(self, t):
        """
        opaque : OPAQUE ID gate_scope '(' ')' bit_list
        """

    def p_opaque_2(self, t):
        """
        opaque : OPAQUE ID gate_scope '(' gate_id_list ')' bit_list
        """

    def p_opaque_1e(self, t):
        """
        opaque : OPAQUE ID gate_scope '(' error
        """
        raise SyntaxError("Poorly formed OPAQUE statement.")

    # ----------------------------------------
    # measure : MEASURE primary ASSIGN primary
    # ----------------------------------------
    def p_measure(self, t):
        """
        measure : MEASURE primary ASSIGN primary
        """
        if len(t[2]) != len(t[4]):
            raise ValueError(
                "registers are of different sizes '"
                + t[2]
                + "' is of size "
                + len(t[2])
                + " and '"
                + t[4]
                + "' is of size "
                + len(t[4])
            )
        op = Gate()
        op.name = "measure"
        op.qblist = [t[2], t[4]]
        t[0] = [op]

    def p_measure_e(self, t):
        """
        measure : MEASURE primary error
        """
        raise SyntaxError("Illegal measure statement." + str(t[3]))

    # ----------------------------------------
    # barrier : BARRIER primary_list
    #
    # Errors are covered by handling erros in primary_list
    # ----------------------------------------
    def p_barrier(self, t):
        """
        barrier : BARRIER primary_list
        """
        # TODO implement barriers, or maybe just apply them

    # ----------------------------------------
    # reset : RESET primary
    # ----------------------------------------
    def p_reset(self, t):
        """
        reset : RESET primary
        """
        t[0] = []
        for qbit in range(0, len(t[2])):
            op = Gate()
            op.name = "reset"
            op.qblist = [t[2][qbit]]
            t[0].append(op)

    # TODO might be an error to take a single argument

    # ----------------------------------------
    # IF '(' ID MATCHES NNINTEGER ')' unitary_op
    # ----------------------------------------
    def p_if(self, t):
        """
        if : IF '(' ID MATCHES NNINTEGER ')' unitary_op
            | IF '(' ID MATCHES NNINTEGER ')' measure
            | IF '(' ID MATCHES NNINTEGER ')' opaque
            | IF '(' ID MATCHES NNINTEGER ')' barrier
            | IF '(' ID MATCHES NNINTEGER ')' reset
            | IF '(' ID error
            | IF '(' ID MATCHES error
            | IF '(' ID MATCHES NNINTEGER error
            | IF error
        """
        if len(t) == 3:
            raise SyntaxError("Ill-formed IF statement. Perhaps a" + " missing '('?")
        if len(t) == 5:
            raise SyntaxError(
                "Ill-formed IF statement.  Expected '==', " + "received '" + str(t[4])
            )
        if len(t) == 6:
            raise SyntaxError(
                "Ill-formed IF statement.  Expected a number, "
                + "received '"
                + str(t[5])
            )
        if len(t) == 7:
            raise SyntaxError("Ill-formed IF statement, unmatched '('")

        if t[7] == "if":
            raise SyntaxError("Nested IF statements not allowed")

        if t[7] == "barrier":
            raise SyntaxError("barrier not permitted in IF statement")

        if t[7] == "measure":
            raise SyntaxError("measure not permitted in IF statement")

        if t[7] == "reset":
            raise SyntaxError("reset not permitted in IF statement")

        # TODO convert the cregister's value from binary to decimal
        # then compare with input number
        c_size = self.creg_exists(t[3])
        if c_size:
            c_index = self.get_indexed_id(t[3], 0)
        else:
            raise ValueError("No such classical bit register")
        if 1 << c_size > t[5]:  # creg size must be able to contain the int
            bit = bin(t[5])
            formula = "AND " * (len(bit) - 3)
            # prefix notation, so we'll put all operators in the beginning
            # print("formula was "+formula+" and index "+str(c_index))

            for i in range(2, len(bit)):
                formula += "NOT " if bit[i] == "0" else ""
                formula += str(c_index + i - 2) + " "  # OP c[i] NOT c[i+1]
                # print("then "+str(i)+" formula is "+formula)

            # print("formula is "+formula)
            # TODO quantum_op is always empty so replenish it
            # TODO might be better to change IF into a unitary_op/measure/etc
            # TODO since IF needs to know whether to apply the gates or not
            for op in t[7]:
                if op.name == "measure":
                    raise ImplementationError("Conditional measures are not supported")
                elif op.name == "reset":
                    raise ImplementationError("Conditional resets are not supported")
                else:
                    new_ops = self.build_gate_or_routine(op.name, op.qblist, op.params)

                    # print("IF routine gives " + str(len(new_ops)))
                    for op in new_ops:
                        op.type = OpType.CLASSICCTRL
                        op.formula = formula
                        self.compiler.ops.append(op)

    # ----------------------------------------
    # These are all the things you can have outside of a gate declaration
    #        quantum_op : unitary_op
    #                   | opaque
    #                   | measure
    #                   | reset
    #                   | barrier
    #                   | if
    #
    # ----------------------------------------
    def p_quantum_op(self, t):
        """
        quantum_op : unitary_op
                   | opaque
                   | measure
                   | barrier
                   | reset
                   | if
        """
        t[0] = []
        if t[1] is None:
            return
        for gat in t[1]:
            if gat.name == "measure":
                op = self.compiler.build_measure(gat.qblist[0], gat.qblist[1])
                t[0].append(op)
            elif gat.name == "reset":
                op = self.compiler.build_reset([gat.qblist[0]], [gat.qblist[0]])
                t[0].append(op)
            else:
                ops = self.build_gate_or_routine(gat.name, gat.qblist, gat.params)
                t[0].extend(ops)

    # ----------------------------------------
    # unary : NNINTEGER
    #       | REAL
    #       | PI
    #       | ID
    #       | '(' expression ')'
    #       | ID '(' expression ')'
    #
    # We will trust 'expression' to throw before we have to handle it here
    # ----------------------------------------
    def p_unary_0(self, t):
        """
        unary : NNINTEGER
        """
        t[0] = t[1]

    def p_unary_1(self, t):
        """
        unary : REAL
        """
        t[0] = t[1]

    def p_unary_2(self, t):
        """
        unary : PI
        """
        t[0] = math.pi

    def p_unary_3(self, t):
        """
        unary : ID
        """
        t[0] = t[1]

    def p_unary_4(self, t):
        """
        unary : '(' expression ')'
        """
        t[0] = t[2]

    def p_unary_6(self, t):
        """
        unary : ID '(' expression ')'
        """
        # note this is a semantic check, not syntactic
        if t[1] not in self.external_functions:
            raise ValueError("Illegal external function call: ", str(t[1]))
        else:
            t[0] = self.run(t[1], t[3])
        # TODO call the function

    # ----------------------------------------
    # Prefix
    # ----------------------------------------

    def p_expression_1(self, t):
        """
        expression : '-' expression %prec negative
                    | '+' expression %prec positive
        """
        if isinstance(t[2], str):
            if t[1] == "-":
                t[0] = [-1, "*", t[2]]
            else:
                t[0] = t[2]
        else:
            if t[1] == "-":
                t[0] = t[2] * (-1)
                # print("value becomes "+str(t[0]))
            else:
                t[0] = t[2]

    def p_expression_0(self, t):
        """
        expression : expression '*' expression
                    | expression '/' expression
                    | expression '+' expression
                    | expression '-' expression
                    | expression '^' expression
        """
        if (
            isinstance(t[1], str)
            or isinstance(t[3], str)
            or isinstance(t[1], list)
            or isinstance(t[3], list)
        ):
            t[0] = [t[1], t[2], t[3]]
        else:
            if t[2] == "*":
                t[0] = t[1] * t[3]
            elif t[2] == "/":
                t[0] = t[1] / t[3]
            elif t[2] == "+":
                t[0] = t[1] + t[3]
            elif t[2] == "-":
                t[0] = t[1] - t[3]
            elif t[2] == "^":
                t[0] = math.pow(t[1], t[3])

    def p_expression_2(self, t):
        """
        expression : unary
        """
        t[0] = t[1]

    # ----------------------------------------
    # exp_list : exp
    #          | exp_list ',' exp
    # ----------------------------------------
    def p_exp_list_0(self, t):
        """
        exp_list : expression
        """
        t[0] = [t[1]]

    def p_exp_list_1(self, t):
        """
        exp_list : exp_list ',' expression
        """
        t[0] = t[1]
        t[0].append(t[3])

    # def p_error(self, t):

    ##########################
    #      Parser build      #
    ##########################
    def build(self, write_tables=False, debug=False, tabmodule="oqasm_tab", **kwargs):
        """Takes care of building a parser

        Args:
            debug: whether to activate debug output or not
            write_tables: generate parser table file or not
            tabmodule: parser tab to use
        Returns:
            Nothing
        """
        self.parser = yacc.yacc(
            module=self,
            write_tables=write_tables,
            tabmodule=tabmodule,
            debug=False,
            errorlog=yacc.NullLogger(),
            **kwargs
        )

    def parse(self, string, debug=False):
        """Parses a given string of openqasm source code

        Args:
            string: input string to parse
            debug: whether to activate debug output or not

        Returns:
            returns 1 if parsing had no issues, otherwise error code
        """
        self.input = string
        self.parser.parse(string, debug=debug)
        return 1

    def compile(
        self, string, write_tables=False, debug=False, tabmodule="oqasm_tab", **kwargs
    ):
        """Compiles a chunk of openqasm code sent as a parameter,
        and returns the corresponding QLM circuit

        Args:
            string: input openqasm code to parse
            debug: whether to activate debug output or not
            write_tables: generate parser table file or not (default False)
            tabmodule: parser tab to use (default oqasm_tab.py)

        Returns:
            Corresponding QLM circuit
        """
        self.build(
            write_tables=write_tables, debug=debug, tabmodule=tabmodule, **kwargs
        )
        self.parse(string, debug=debug)

        return self.compiler.gen_circuit()
