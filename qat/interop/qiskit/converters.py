#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

"""
.. License

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

myQLM provides binders to translate quantum circuits from Qiskit to
myQLM and vice-versa through the :func:`~qat.interop.qiskit.qiskit_to_qlm`
and :func:`~qat.interop.qiskit.qlm_to_qiskit` functions:

.. code-block:: python

    from qat.interop.qiskit import qiskit_to_qlm

    qlm_circuit = qiskit_to_qlm(your_qiskit_circuit)

Or

.. code-block:: python

    from qat.interop.qiskit import qlm_to_qiskit

    qiskit_circuit = qlm_to_qiskit(your_qlm_circuit)
"""

import warnings
import operator
import numpy as np
from sympy.core import Add, Mul, Pow
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.circuit.library.standard_gates import *

from qat.lang.AQASM import Program, QRoutine
from qat.lang.AQASM.gates import AbstractGate, H, X, Y, Z, SWAP, I, S, \
        T, RX, RY, RZ
from qat.core.util import extract_syntax
from qat.core.assertion import assert_qpu
from qat.core.variables import Variable, ArithExpression
from qat.comm.datamodel.ttypes import OpType
from qat.comm.shared.ttypes import ProcessingType


def _get_qindex(circ, name, index):
    """
    Find the qubit index.

    Args:
        circ: The Qiskit QuantumCircuit in question
        name: The name of the quantum register
        index: The qubit's relative index inside the register

    Returns:
        The qubit's absolute index if all registers are concatenated.
    """
    ret = 0
    for reg in circ.qregs:
        if name != reg.name:
            ret += reg.size
        else:
            return ret + index
    return ret + index


def _get_cindex(circ, name, index):
    """
    Find the classical bit index.

    Args:
        circ: The Qiskit QuantumCircuit in question
        name: The name of the classical register
        index: The qubit's relative index inside the register

    Returns:
        The classical bit's absolute index if all registers are concatenated.
    """
    ret = 0
    for reg in circ.cregs:
        if name != reg.name:
            ret += reg.size
        else:
            return ret + index
    return ret + index


def _sympy_arg_to_arith_expr(prog, variables, param, arg):
    """
    Converts a Sympy argument (that may be an expression) into an
    ArithExpression object. Variable objects may be created in the process
    if needed.

    Args:
        prog: QLM Program object on which variables should be added if needed
        variables: list of variables previously added to prog
        param: Parameter or ParameterExpression object
        arg: Sympy argument to convert

    Returns:
        A variable or an ArithExpression object.
    """

    arith_expr = 0
    # if it is an expression
    if isinstance(arg, (Add, Mul, Pow)):
        for i, sub_arg in enumerate(arg.args):
            arith_sub_expr = _sympy_arg_to_arith_expr(
                prog, variables, param, sub_arg
            )
            if i == 0:
                arith_expr = arith_sub_expr
            elif isinstance(arg, Add):
                arith_expr += arith_sub_expr
            elif isinstance(arg, Mul):
                arith_expr *= arith_sub_expr
            elif isinstance(arg, Pow):
                arith_expr **= arith_sub_expr
        return arith_expr

    # if it is not an expression, but a single value, either abstract or not
    else:
        new_param_name = str(arg)
        var = False
        if isinstance(param, Parameter):
            for x_param in param.expr.parameters:
                if x_param.name == new_param_name:
                    # gets the variable or creates it
                    var = _qiskit_to_qlm_param(prog, variables, x_param)
                    break
        elif isinstance(param, ParameterExpression):
            for x_param in param.parameters:
                if x_param.name == new_param_name:
                    # gets the variable or creates it
                    var = _qiskit_to_qlm_param(prog, variables, x_param)
                    break

        # if the value is not abstract
        if not var:
            try:
                var = float(new_param_name)
            except ValueError:
                string = "Unreliable variable expression in Qiskit Parameter"
                raise KeyError(string)
    return var


def _qiskit_to_qlm_param(prog, variables, param):
    """
    Converts a Qiskit Parameter or ParameterExpression into an object that can
    be passed as an argument to a QLM gate.

    Args:
        prog: QLM Program object on which variables should be added if needed
        variables: list of variables previously added to prog
        param: Parameter or ParameterExpression object

    Returns:
        A Variable or a ArithExpression object.
    """
    if isinstance(param, Parameter):
        name = param.name
        for var in variables:
            if var.name == name:
                return var
        var = prog.new_var(float, name)
        variables.append(var)
    elif isinstance(param, ParameterExpression):
        expression = param._symbol_expr
        return _sympy_arg_to_arith_expr(prog, variables, param, expression)
    return var


def _gen_u(theta, phi, lamda):
    """
    Generates the U gate matrix.
    u1/2/3 would be dealt with through setting the appropriate params to 0.

    Args:
        theta:
        phi:
        lamda: lambda parameter

    Returns:
        numpy.ndarray U gate matrix
    """
    m11 = (np.e ** (1j * (phi + lamda) / 2)) * np.cos(theta / 2)
    m12 = (-1) * (np.e ** (1j * (phi - lamda) / 2)) * np.sin(theta / 2)
    m21 = (np.e ** (1j * (phi - lamda) / 2)) * np.sin(theta / 2)
    m22 = (np.e ** (1j * (phi + lamda) / 2)) * np.cos(theta / 2)
    return np.array([[m11, m12], [m21, m22]], dtype=np.complex128)


def _gen_u2(phi, lmbda):
    """
    Returns the corresponding u2 abstract gate.

    Args:
        phi:
        lmbda: lambda parameter

    Returns:
        numpy.ndarray U2 gate matrix
    """
    return _gen_u(0, phi, lmbda)


def _gen_rxx(theta):
    """
    Generates the RXX gate matrix.

    Args:
        theta:

    Returns:
        numpy.ndarry RXX gate matrix
    """
    return np.array([[np.cos(theta / 2), 0, 0, -1j * np.sin(theta / 2)],
                     [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
                     [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
                     [-1j * np.sin(theta / 2), 0, 0, np.cos(theta / 2)]])


def _gen_rzz(theta):
    """
    Generates the RZZ gate matrix.

    Args:
        theta:

    Returns:
        numpy.ndarray RZZ gate matrix
    """
    return np.diag([1, np.exp(1j * theta), np.exp(1j * theta), 1])


def _gen_r(theta, phi):
    """
    Returns the corresponding R abstract gate.

    Args:
        theta:
        phi:

    Returns:
        numpy.ndarray R gate matrix
    """
    return _gen_u(theta, phi - np.pi / 2, phi + np.pi / 2)


def _gen_ms(theta, nb_qubits):
    """
    Returns the corresponding MS gate.

    Args:
        theta:
        nb_qubits: Number of qubits affected by the gate

    Returns:
        QRoutine object representing the MS gate.
    """
    routine = QRoutine()

    for first_qb in range(nb_qubits):
        for second_qb in range(first_qb + 1, nb_qubits):
            routine.apply(RXX(theta), [first_qb, second_qb])

    return routine


U = AbstractGate("U", [float] * 3, arity=1, matrix_generator=_gen_u)
U2 = AbstractGate("U2", [float, float], arity=1, matrix_generator=_gen_u2)
U3 = AbstractGate("U3", [float] * 3, arity=1, matrix_generator=_gen_u)
RXX = AbstractGate("RXX", [float], arity=2, matrix_generator=_gen_rxx)
RZZ = AbstractGate("RZZ", [float], arity=2, matrix_generator=_gen_rzz)
R = AbstractGate("R", [float] * 2, arity=1, matrix_generator=_gen_r)
MS = AbstractGate("MS", [float, int], arity=lambda theta, nb_qbits: nb_qbits,
                  circuit_generator=_gen_ms)


GATE_DIC = {
    "h": H,
    "x": X,
    "y": Y,
    "z": Z,
    "swap": SWAP,
    "iden": I,
    "s": S,
    "sdg": S.dag(),
    "t": T,
    "tdg": T.dag(),
    "rx": RX,
    "ry": RY,
    "rz": RZ,
    "rxx": RXX,
    "rzz": RZZ,
    "u1": RZ,
    "u2": U2,
    "u3": U3,
    "r": R,
    "ms": MS,
    # below: deprecated
    "u0": I,
    "id": I,
    "U": U,
    "xbase": X,
}


def get_gate(gate, params, num_ctrl_qubits=None):
    """
    Generates a pyAQASM gate from the corresponding Qiskit gate.

    Args:
        gate: Name of the Qiskit gate
        num_ctrl_qubits: Number of control qbits
        params: Parameters of the Qiskit gate
    """
    if gate[0] == "c":
        return get_gate(gate[1:], params, num_ctrl_qubits).ctrl()
    if gate == "toffoli":
        return get_gate("ccx", params, num_ctrl_qubits)
    if gate[0:2] == "mc":
        name = gate[2:]
        for _ in range(num_ctrl_qubits):
            name = "c" + name
        return get_gate(name, params, num_ctrl_qubits)
    if len(params) == 0:
        return GATE_DIC[gate]

    return GATE_DIC[gate](*params)


def qiskit_to_qlm(qiskit_circuit, sep_measures=False, **kwargs):
    """
    Converts a Qiskit circuit into a QLM circuit.

    Args:
        qiskit_circuit: The Qiskit circuit to convert
        sep_measures: Separates measures from the
            circuit:

             - if set to :code:`True`, measures won't be included in the resulting
               circuit, qubits to be measured will be put in a list, the resulting
               measureless circuit and this list will be returned in a tuple:
               (resulting_circuit, list_qubits)
             - if set to :code:`False`, measures will be converted normally
               (Default, set to False)

        kwargs: These are the options that you would use on a regular
                to_circ function, to generate a QLM circuit from a PyAQASM
                program these are added for more flexibility,
                for advanced users

    Returns:
        :code:`tuple` or :class:`~qat.core.Circuit`: If :code:`sep_measures` is set
        to:

         - :code:`True`: the result is a tuple composed of a
           :class:`~qat.core.Circuit` and a list of qubits that should be
           measured
         - :code:`False`: the result is a :class:`~qat.core.Circuit`
    """
    prog = Program()
    qbits_num = 0
    to_measure = []
    for reg in qiskit_circuit.qregs:
        qbits_num = qbits_num + reg.size
    qbits = prog.qalloc(qbits_num)

    cbits_num = 0
    for reg in qiskit_circuit.cregs:
        cbits_num = cbits_num + reg.size
    cbits = prog.calloc(cbits_num)
    variables = []
    for gate_op in qiskit_circuit.data:
        if gate_op[0].name == "barrier" or gate_op[0].name == "opaque":
            continue
        qbit_args = []
        cbit_args = []
        prms = []  # gate parameters
        # Get qbit arguments
        for qarg in gate_op[1]:
            qbit_args.append(
                _get_qindex(qiskit_circuit, qarg.register.name, qarg.index))

        # Get cbit arguments
        for carg in gate_op[2]:
            cbit_args.append(
                _get_cindex(qiskit_circuit, carg.register.name, carg.index))

        # Get parameters
        for param in gate_op[0]._params:
            if isinstance(param, (Parameter, ParameterExpression)):
                prms.append(_qiskit_to_qlm_param(prog, variables, param))
            else:
                prms.append(float(param))
        # Apply measure #
        if gate_op[0].name == "measure":
            if sep_measures:
                to_measure.extend(qbit_args)
            else:
                prog.measure([qbits[i] for i in qbit_args],
                             [cbits[i] for i in cbit_args])
        else:
            if gate_op[0].name == "ms":
                # In this case, the process function needs the number of qubits
                prms.append(len(qbit_args))
            # Apply gates #
            num_ctrl_qubits = None
            try:
                num_ctrl_qubits = gate_op[0].num_ctrl_qubits
            except:
                None
            gate = get_gate(gate_op[0].name, prms, num_ctrl_qubits)
            prog.apply(gate, *[qbits[i] for i in qbit_args][:gate.arity])
    if sep_measures:
        return prog.to_circ(**kwargs), list(set(to_measure))

    return prog.to_circ(**kwargs)


def qlm_circ_sep_meas(qiskit_circuit):
    """
    Helper function. Calls qiskit_to_qlm with seperate measures.

    Args:
        qiskit_circuit:
    """
    return qiskit_to_qlm(qiskit_circuit, True)


def _gen_qiskit_gateset(q_circ):
    """
    Generates a dictionnary of Qiskit gate methods applied
    to the Qiskit circuit passed as argument.
    """
    return {
        'H': q_circ.h,
        'X': q_circ.x,
        'Y': q_circ.y,
        'Z': q_circ.z,
        'SWAP': q_circ.swap,
        'I': q_circ.iden,
        'S': q_circ.s,
        'D-S': q_circ.sdg,
        'T': q_circ.t,
        'D-T': q_circ.tdg,
        'RX': q_circ.rx,
        'RY': q_circ.ry,
        'RZ': q_circ.rz,
        'C-H': q_circ.ch,
        'CNOT': q_circ.cx,
        'C-Y': q_circ.cy,
        'CSIGN': q_circ.cz,
        'C-RZ': q_circ.crz,
        'CCNOT': q_circ.ccx,
        'C-SWAP': q_circ.cswap,
        'U': q_circ.u3,
        'U3': q_circ.u3,
        'U2': q_circ.u2,
        'U1': q_circ.u1,
        'U0': q_circ.iden,
        'PH': q_circ.rz,
        'RXX': q_circ.rxx,
        'RZZ': q_circ.rzz,
        'R': q_circ.r,
        'MS': q_circ.ms
    }


SUPPORTED_CTRLS = ["CNOT", "CCNOT", "C-Y", "CSIGN", "C-H", "C-SWAP", "C-RZ"]


def _get_qiskit_gate_from_name(name):
    """
    Return a qiskit gate that corresponds to the name passed as argument.
    """
    gates = {
        'H': HGate,
        'X': XGate,
        'Y': YGate,
        'Z': ZGate,
        'SWAP': SwapGate,
        'I': IGate,
        'S': SGate,
        'D-S': SdgGate,
        'T': TGate,
        'D-T': TdgGate,
        'RX': RXGate,
        'RY': RYGate,
        'RZ': RZGate,
        'C-H': CHGate,
        'CNOT': CnotGate,
        'C-Y': CYGate,
        'CSIGN': CZGate,
        'C-RZ': CRZGate,
        'CCNOT': CCXGate,
        'C-SWAP': CSwapGate,
        'U': U3Gate,
        'U3': U3Gate,
        'U2': U2Gate,
        'U1': U1Gate,
        'U0': IGate,
        'PH': RZGate,
        'RXX': RXXGate,
        'RZZ': RZZGate,
        'R': RGate,
        'MS': MSGate
    }
    try:
        gate = gates[name]
    except:
        gate = None
    return gate


def _variable_to_parameter(param_list, variable=None, variable_name=""):
    """
    Takes either a Variable object or a the name of a Variable object and
    returns the corresponding Parameter object. If variable is specified,
    and it has a value, the parameter's expr is accordingly specified.

    Args:
        param_list: List of Parameter objects already created for this
                circuit translation
        variable: Variable object
        variable_name: String that is the name of a variable

    Returns:
        A Qiskit Parameter object
    """
    if not(variable or variable_name):
        raise Exception("_variable_to_parameter must either take a variable"
                        + " or a variable_name argument")
    param = None
    if variable_name:
        for x_param in param_list:
            if x_param.name == variable_name:
                return x_param
        param = Parameter(variable_name)
        param_list.append(param)

    # look if the parameter has already been defined
    for x_param in param_list:
        if variable is not None and x_param.name == variable.name:
            param = x_param
            # look if the expression of the parameter has already been defined
            if hasattr(param, "expr"):
                return param

    if param is None:
        param = Parameter(variable.name)
        param_list.append(param)

    # looks if the variable has an expression that should be translated
    if variable:
        if isinstance(variable, ParameterExpression):
            arith_expr_list = variable.to_thrift().split()
            param.expr = _arith_expr_list_to_parameter_expression(
                param_list, arith_expr_list, variable)
        else:
            param.expr = variable
    return param


def _arith_expr_list_to_parameter_expression(
        param_list, arith_expr_list, root_expr):
    """
    Takes a list of arguments created from an ArithExpression
    object and returns either a complete ParameterExpression if
    the list is complete, or part of a ParameterExpression if
    part of the list have been previously removed.

    Args:
        param_list: List of Parameter objects previously created for
                this circuit translation
        arith_expr_list: List of strings being the arguments of an
                ArithExpression object
        root_expr: ArithExpression object from which the
                arith_expr_list originated

    Returns:
        May return a ParamterExpression, a Parameter, an int or a float
        depending of the list given as an argument
    """
    arg = arith_expr_list.pop(0)
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv
    }
    if arg in ['+', '-', '*', '/']:
        return ops[arg](
            # in this case, returns the first expression in the list
            _arith_expr_list_to_parameter_expression(
                param_list, arith_expr_list, root_expr),
            # now that the first expression has been removed,
            # returns the second expression of the operation
            _arith_expr_list_to_parameter_expression(
                param_list, arith_expr_list, root_expr))
    if arg == '^':
        first_arg = _arith_expr_list_to_parameter_expression(
            param_list, arith_expr_list, root_expr)
        second_arg = _arith_expr_list_to_parameter_expression(
            param_list, arith_expr_list, root_expr)
        if isinstance(second_arg, int):
            result = 1
            for _ in range(second_arg):
                result *= first_arg
            return result

        raise SyntaxError("Power operation is not supported by Qiskit for "
                          + "abstract variable expressions.")

    if arg == "UMINUS":
        return -_arith_expr_list_to_parameter_expression(
            param_list, arith_expr_list, root_expr)

    # if it is not an operation, it is either a variable or a value
    if arg in root_expr.get_variables():
        return _variable_to_parameter(param_list, variable_name=arg)

    try:
        # try to cast it to float
        value = float(arg)
    except ValueError:
        string = "Unreliable variable expression in ArithExpression"
        raise KeyError(string)
    return value


def qlm_to_qiskit(qlm_circuit, qubits=None):
    """
    Converts a QLM circuit to a Qiskit circuit.

    The supported translatable gates are:
    :code:`H`, :code:`X`, :code:`Y`, :code:`Z`, :code:`SWAP`,
    :code:`I`, :code:`S`, :code:`S.dag()`, :code:`T`, :code:`T.dag()`,
    :code:`RX`, :code:`RY`, :code:`RZ`, :code:`H.ctrl()`, :code:`CNOT`,
    :code:`Y.ctrl()`, :code:`CSIGN`, :code:`RZ.ctrl()`, :code:`CCNOT`,
    :code:`SWAP.ctrl()`, :code:`U`, :code:`RXX`, :code:`RZZ`, :code:`R`,
    :code:`MS`

    Args:
        qlm_circuit: The input QLM circuit to convert
        qubits (list<int>, optional): measured qubits

    Returns:
        A QuantumCircuit Qiskit object resulting from the conversion
    """
    # Init measured qubits
    if qubits is None:
        qubits = list(range(qlm_circuit.nbqbits))

    qreg = QuantumRegister(qlm_circuit.nbqbits)
    creg = None
    param_list = []
    if qlm_circuit.nbcbits > 0:
        creg = ClassicalRegister(max(qlm_circuit.nbcbits, len(qubits)))
        q_circ = QuantumCircuit(qreg, creg)
    else:
        q_circ = QuantumCircuit(qreg)
    dic = _gen_qiskit_gateset(q_circ)
    for gate_op in qlm_circuit:
        if gate_op.type == OpType.GATETYPE:
            name, params = extract_syntax(
                qlm_circuit.gateDic[gate_op.gate], qlm_circuit.gateDic,
                var_dic=qlm_circuit.var_dic)
            nbctrls = name.count('C-')
            # changes variables and expressions to format used by Qiskit
            for index, param in enumerate(params):
                if isinstance(param, Variable):
                    params[index] = _variable_to_parameter(
                        param_list, variable=param)
                elif isinstance(param, ArithExpression):
                    arith_expr_list = param.to_thrift().split()
                    params[index] = _arith_expr_list_to_parameter_expression(
                        param_list, arith_expr_list, param)
            try:
                if name == "MS":
                    q_circ.ms(params[0], [qreg[i] for i in gate_op.qbits])
                else:
                    if (nbctrls > 0 and name not in SUPPORTED_CTRLS):
                        tmp = name
                        count = 0
                        gate = None
                        while True:
                            last = tmp
                            tmp = tmp.replace("C-", "", 1)
                            if last == tmp:
                                raise ValueError(
                                    "Gate {} not supported by Qiskit API".format(name)
                                )
                            else:
                                count += 1
                                gate = _get_qiskit_gate_from_name(tmp)
                                if gate != None:
                                    gate = gate(*params).control(count)
                                    break
                        if gate != None:
                            q_circ.append(gate, [qreg[i] for i in gate_op.qbits])
                    else:
                        dic[name](* params + [qreg[i] for i in gate_op.qbits])
            except KeyError:
                raise ValueError(
                    "Gate {} not supported by Qiskit API".format(name)
                )
        elif gate_op.type == OpType.MEASURE:
            for index in range(len(gate_op.qbits)):
                q_circ.measure(gate_op.qbits[index], gate_op.cbits[index])

    # Adding measures to unify the interface
    for qbit_index, cbit in zip(qubits, creg):
        q_circ.measure(qreg[qbit_index], cbit)
    return q_circ


def job_to_qiskit_circuit(qlm_job):
    """
    Converts the circuit inside a QLM job into a Qiskit circuit.
    This is only a helper function, parameters such as nbshots should
    be extracted from qlm_job alongside this function's call.

    Args:
        qlm_job: The QLM job containing the circuit to convert

    Returns:
        A QuantumCircuit Qiskit object resulting from the conversion
    """
    # Check processing type
    assert_qpu(qlm_job.type == ProcessingType.SAMPLE,
               "Only jobs having a SAMPLE processing type "
               "could be translated into Qiskit circuits")

    # Convert
    return qlm_to_qiskit(qlm_job.circuit, qlm_job.qubits)


def to_qlm_circ(qiskit_circuit, sep_measures=False, **kwargs):
    """ Deprecated, use qiskit_to_qlm. """
    warnings.warn(
        "to_qlm_circ() is deprecated, please use qiskit_to_qlm()",
        FutureWarning,
    )
    return qiskit_to_qlm(qiskit_circuit, sep_measures, **kwargs)


def to_qiskit_circ(qlm_circuit):
    """ Deprecated, use qlm_to_qiskit. """
    warnings.warn(
        "to_qiskit_circ() is deprecated, please use qlm_to_qiskit()",
        FutureWarning,
    )
    return qlm_to_qiskit(qlm_circuit)
