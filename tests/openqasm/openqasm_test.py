#!/usr/bin/env oython
# -*- coding: utf-8 -*-

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

import unittest
import os
from math import pi
from qat.interop.openqasm.qasm_parser import OqasmParser, ImplementationError,\
        extract_inc, InvalidParameterNumber

try:
    from qat.core.util import extract_syntax
except ImportError:
    from qat.core.circ import extract_syntax
from qat.comm.datamodel.ttypes import OpType
from qat.core.gate_set import WrongArgumentsNumber as WrongParams

# name, type, nb_qbits, nb_params
GATE_DATA = [["H", OpType.GATETYPE, 1, 0],
             ["X", OpType.GATETYPE, 1, 0],
             ["Y", OpType.GATETYPE, 1, 0],
             ["Z", OpType.GATETYPE, 1, 0],
             ["I", OpType.GATETYPE, 1, 0],
             ["U", OpType.GATETYPE, 1, 3],
             ["U1", OpType.GATETYPE, 1, 1],
             ["U2", OpType.GATETYPE, 1, 2],
             ["U3", OpType.GATETYPE, 1, 3],
             ["PH", OpType.GATETYPE, 1, 1],
             ["CU1", OpType.GATETYPE, 2, 1],
             ["CU2", OpType.GATETYPE, 2, 2],
             ["CU3", OpType.GATETYPE, 2, 3],
             ["CH", OpType.GATETYPE, 2, 0],
             ["CRZ", OpType.GATETYPE, 2, 1],
             ["S", OpType.GATETYPE, 1, 0],
             ["SDG", OpType.GATETYPE, 1, 0],
             ["T", OpType.GATETYPE, 1, 0],
             ["TDG", OpType.GATETYPE, 1, 0],
             ["RX", OpType.GATETYPE, 1, 1],
             ["RY", OpType.GATETYPE, 1, 1],
             ["RZ", OpType.GATETYPE, 1, 1],
             ["CNOT", OpType.GATETYPE, 2, 0],
             ["CCNOT", OpType.GATETYPE, 3, 0]]

HEADER = "OPENQASM 2.0;\nqreg q[4];\ncreg c[4];\n"


class TestPyAqasmSimple(unittest.TestCase):
    """ Class for PyAQASM tests """

    def test_version_verification(self):
        """ Whether the correct version is parsed correctly """
        data = "\nqreg q[4];\ncreg c[4];\ngate cu3(theta,phi,lambda) c, t\n{\nu1((lambda-phi)/2) t;\ncx c,t;\nu3(-theta/2,0,-(phi+lambda)/2) t;\
        \ncx c,t;\nu3(theta/2,phi,0) t;\n}\ncu3(2.5, 2, 1) q[0], q[1];\nu2(1,2) q[0];\n"

        right_parser = OqasmParser()
        right_parser.build()

        header = "OPENQASM\n\n\n\t\t\t   2.0\n\n\n  ;"
        right_parser.parse(header + data)
        self.assertEqual(right_parser.format_version, False)
        header = "OPENQASM\n\n\n\t\t\t    2.1;"
        del right_parser
        wrong_parser = OqasmParser()
        wrong_parser.build()

        wrong_parser.parse(header + data)
        self.assertEqual(wrong_parser.format_version, True)


    def test_cu_bug(self):
        """ Trying to fix cu1 bug ignoring third parameter"""
        oq_parser = OqasmParser()
        oq_parser.build()
        data = "gate cu3(theta,phi,lambda) c, t\n{\nu1((lambda-phi)/2) t;\ncx c,t;\nu3(-theta/2,0,-(phi+lambda)/2) t;\
                \ncx c,t;\nu3(theta/2,phi,0) t;\n}\ncu3(2.5, 2, 1) q[0], q[1];\nu2(1,2) q[0];\n"
        print(data)
        oq_parser.parse(HEADER + data)
        circ = oq_parser.compiler.gen_circuit()
        print(circ.ops)
    def test__correct_format(self):
        """ Testing how the parser fares with bad formatting"""
        oq_parser = OqasmParser()
        oq_parser.build()
        data = "U(       \n0,\n\n0,0)q\n [\t\n0\n]\n\n;"
        # print(data)
        oq_parser.parse(HEADER + data)
        circ = oq_parser.compiler.gen_circuit()
        self.assertEqual(len(circ.ops), 1)

    def test__recursive_file_inclusion(self):
        """ Testing whether includes are working correctly """
        test1 = "include \"test2\";\n4\n5\n"
        test2 = "1\ninclude \"test3\";\n3\n"
        test3 = "2\n"
        with open("test1", 'w') as f:
            f.write(test1)
            f.close()
        with open("test2", 'w') as f:
            f.write(test2)
            f.close()
        with open("test3", 'w') as f:
            f.write(test3)
            f.close()
        res = extract_inc("test1")
        data = "1\n2\n3\n4\n5\n"
        os.remove("test1")
        os.remove("test2")
        os.remove("test3")
        self.assertEqual(res, data)

    def test__standard_operations(self):
        """ Testing standard gates and operators work correctly """
        oq_parser = OqasmParser()
        oq_parser.build(debug=True)
        reverse_dic = {v: k for k, v in oq_parser.standard_gates.items()}
        print(reverse_dic)
        data = ""
        for op in GATE_DATA:
            if op[3] > 0:
                data += reverse_dic[op[0]] + "(1"
                for p in range(1, op[3]):
                    data += ", 2"
                data += ") q[0]"
            else:
                data += reverse_dic[op[0]] + " q[0]"
            for q in range(1, op[2]):
                data += ", q["+str(q)+"]"
            data += " ;\n"
        print(HEADER + data)
        res = oq_parser.parse(HEADER + data)
        circ = oq_parser.compiler.gen_circuit()
        for op in circ.ops:
            gate_name, gate_params = extract_syntax(circ.gateDic[op.gate],
                                                    circ.gateDic)
            print("gate {} with params {} on qbits {}"
                  .format(gate_name, gate_params, op.qbits))
        self.assertEqual(res, 1)

    def test__non_implemented_if(self):
        """ Testing behavior with non implemented if (measure/reset)"""
        oq_parser = OqasmParser()
        oq_parser.build()
        data1 = "if (c==1) reset q;\n"
        data2 = "if (c==1) measure q -> c;"
        print(data1)
        success = False
        try:
            oq_parser.parse(HEADER+data1)
        except ImplementationError:
            success = True
        self.assertTrue(success, "The parser failed to raise an\
                       Implementation Error for reset")
        success = False
        oq_parser = OqasmParser()
        oq_parser.build()
        try:
            oq_parser.parse(HEADER+data2)
        except ImplementationError:
            success = True
        self.assertTrue(success, "The parser failed to raise an\
                        Implementation Error for measure")

    def test__implemented_if(self):
        """ Testing behavior with implemented if """
        oq_parser = OqasmParser()
        oq_parser.build()
        data = "if (c==13) U(0,pi/2,0) q[1];\nif (c==20) x q[1];\nx q[2];"
        print(data)
        res = oq_parser.parse(HEADER+data)
        circ = oq_parser.compiler.gen_circuit()
        # 20 > 2^4-1 so no op created as it's always false
        self.assertEqual(len(circ.ops), 2)
        for op in circ.ops:
            gate_name, gate_params = extract_syntax(circ.gateDic[op.gate],
                                                    circ.gateDic)
            print("gate {} with params{} on qbits {}"
                  .format(gate_name, gate_params, op.qbits))
            if op.type == OpType.CLASSICCTRL:
                self.assertEqual(op.cbits, None)
                if gate_name == "U":
                    self.assertEqual(op.formula, "AND AND AND 0 1 NOT 2 3 ")
            else:
                self.assertEqual(op.formula, None)
        self.assertEqual(res, 1)

    def test__empty_params_routines(self):
        """ Testing whether gates requiring params work without
            inputing any"""
        oq_parser = OqasmParser()
        oq_parser.build()
        reverse_dic = {v: k for k, v in oq_parser.standard_gates.items()}
        for op in GATE_DATA:
            success = False
            data = ""
            if op[3] > 0:
                if op[0] == "U":
                    data += reverse_dic[op[0]] + "(0) q[0];\n"
                else:
                    data += reverse_dic[op[0]] + " q[0];\n"
                print(data)
                oq_parser = OqasmParser()
                oq_parser.build()
                try:
                    oq_parser.parse(HEADER + data)
                except (InvalidParameterNumber, WrongParams) as e:
                    success = True
                self.assertTrue(success, "The parser failed to raise " +
                                "Invalid Parameter number for "
                                + reverse_dic[op[0]])

    def test__normal_routines(self):
        """ Testing normal routines work correctly """
        oq_parser = OqasmParser()
        oq_parser.build(debug=True)
        reverse_dic = {v: k for k, v in oq_parser.standard_gates.items()}
        data = "gate tst(p) a1, a2, a3, a4 {"
        for op in GATE_DATA[0:4]:
            data += reverse_dic[op[0]] + " a1"
            for q in range(1, op[2]):
                data += ", a"+str(q+1)
            data += ";"
        data += "} tst(pi) q[0], q[1], q[2], q[3];"
        print(data)
        res = oq_parser.parse(HEADER + data, debug=False)
        circ = oq_parser.compiler.gen_circuit()
        for op in circ.ops:
            gate_name, gate_params = extract_syntax(circ.gateDic[op.gate],
                                                    circ.gateDic)
            print("gate {} with params {} on qbits {}"
                  .format(gate_name, gate_params, op.qbits))
        self.assertEqual(res, 1)

    def test__routines_of_routines(self):
        """ Testing routines using other routines """
        oq_parser = OqasmParser()
        oq_parser.build(debug=True)
        reverse_dic = {v: k for k, v in oq_parser.standard_gates.items()}
        data = "gate tst(p) a1,a2,a3,a4 {\n"
        for op in GATE_DATA[0:4]:
            data += reverse_dic[op[0]] + " a1"
            for q in range(1, op[2]):
                data += ", a"+str(q+1)
            data += ";\n"
        data += "}\ngate tst2(p) b1, b2, b3, b4 {\n"
        for op in GATE_DATA[5:7]:
            if op[3] > 0:
                if op[0] == "U":
                    data += reverse_dic[op[0]] + "(p,0,0) b1"
                else:
                    data += reverse_dic[op[0]] + "(p) b1"
            else:
                data += reverse_dic[op[0]] + " b1"
            for q in range(1, op[2]):
                data += ", b"+str(q+1)
            data += ";\n"
        data += "tst(p) b1, b2, b3, b4;\n}\n"\
               + "tst2(pi) q[0], q[1], q[2], q[3];"
        print(data)
        res = oq_parser.parse(HEADER+data)
        circ = oq_parser.compiler.gen_circuit()
        for op in circ.ops:
            gate_name, gate_params = extract_syntax(circ.gateDic[op.gate],
                                                    circ.gateDic)
            print("gate {} with params {} on qbits {}"
                  .format(gate_name, gate_params, op.qbits))
        self.assertEqual(res, 1)

    def test__routines_with_eval_params(self):
        """ Testing routines using arithmetic expressions in params"""
        oq_parser = OqasmParser()
        oq_parser.build()
        reverse_dic = {v: k for k, v in oq_parser.standard_gates.items()}
        data = "gate rp(p) a1, a2 {\n"
        nb_gates = 0
        for op in GATE_DATA:
            if op[3] > 0:
                nb_gates += 1
                if op[0] == "U" or op[0] == "U3":
                    data += "U (-p, 0, 0) a1;\n"
                elif op[0] == "U2":
                    data += "u2 (-p, 0) a1;\n"
                elif op[0] == "CU3":
                    data += "cu3 (-p, 0, 0) a1, a2;\n"
                elif op[0] == "CU2":
                    data += "cu2 (-p, 0) a1, a2;\n"
                elif op[0] == "CU1":
                    data += "cu1 (-p) a1, a2;\n"
                elif op[0] == "CRZ":
                    data += "crz (-p) a1, a2;\n"
                else:
                    data += reverse_dic[op[0]] + "(-p) a1;\n"
        data += "}\nrp(-3*(5+4)) q[0], q[1];\n"
        data += "rp(-pi/2) q[1], q[2];\n"
        data += "rp(-3*5+4) q[2], q[1];\n"
        data += "rp(-3+5*4) q[3], q[1];\n"
        print(data)
        res = oq_parser.parse(HEADER+data)
        circ = oq_parser.compiler.gen_circuit()
        i = 0
        for op in circ.ops:
            gate_name, gate_params = extract_syntax(circ.gateDic[op.gate],
                                                    circ.gateDic)
            print("gate {} with params {} on qbits {}"
                  .format(gate_name, gate_params, op.qbits))
            if i < nb_gates:
                self.assertEqual(gate_params[0], 27)
            elif i >= nb_gates and i < nb_gates*2:
                self.assertEqual(gate_params[0], pi/2)
            elif i >= nb_gates*2 and i < nb_gates*3:
                self.assertEqual(gate_params[0], 11)
            else:
                self.assertEqual(gate_params[0], -17)
            i += 1
        self.assertEqual(res, 1)

    def test_openqasm_examples(self):
        from subprocess import call
        call(["examples/compile_oqasm_examples.sh"])

    def test__rec_routines_eval_params(self):
        """Testing arithmetic expressions in parameters of recursive routines"""
        oq_parser = OqasmParser()
        oq_parser.build()
        reverse_dic = {v: k for k, v in oq_parser.standard_gates.items()}
        data = "gate rp(p) a1, a2{\n"
        nb_gates = 0
        for op in GATE_DATA:
            if op[3] > 0:
                nb_gates += 1
                if op[0] == "U" or op[0] == "u3":
                    data += "U (-p, p, 0) a1;\n"
                elif op[0] == "U2":
                    data += "u2 (-p, p) a1;\n"
                elif op[0] == "U3":
                    data += "u3 (-p, p, p) a1;\n"
                elif op[0] == "CU1":
                    data += "cu1 (-p) a1, a2;\n"
                elif op[0] == "CU2":
                    data += "cu2 (-p, p) a1, a2;\n"
                elif op[0] == "CU3":
                    data += "cu3 (-p, p, p) a1, a2;\n"
                elif op[0] == "CRZ":
                    data += "crz (-p) a1, a2;\n"
                else:
                    data += reverse_dic[op[0]] + "(-p) a1;\n"
        data += "}\ngate rrp(q) a1, a2{\nrp(3*(-q)+2) a1, a2;\n}\n"
        data += "rrp(-pi/2) q[1], q[0];\n"
        data += "rrp(-3*5+4) q[2], q[0];\n"
        data += "rrp(-3+5*4) q[3], q[0];\n"
        data += "rrp(-3*(5+4)) q[3], q[0];\n"
        print(data)
        res = oq_parser.parse(HEADER+data)
        circ = oq_parser.compiler.gen_circuit()
        i = 0
        for op in circ.ops:
            gate_name, gate_params = extract_syntax(circ.gateDic[op.gate],
                                                    circ.gateDic)
            print("gate {} with params {} on qbits {}"
                  .format(gate_name, gate_params, op.qbits))
            if i < nb_gates:
                self.assertEqual(gate_params[0], -(pi*1.5 + 2))
            elif i >= nb_gates and i < nb_gates*2:
                self.assertEqual(gate_params[0], -35)
            elif i >= nb_gates*2 and i < nb_gates*3:
                self.assertEqual(gate_params[0], 49)
            else:
                self.assertEqual(gate_params[0], -83)
            i += 1
        self.assertTrue(res, 1)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
