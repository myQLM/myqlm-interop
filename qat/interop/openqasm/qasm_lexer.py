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

import ply.lex as lex


class OqasmLexer(object):
    """OPENQASM Lexer.

    This is a wrapper around the PLY lexer to support the "include" statement
    by creating a stack of lexers.
    """
    # pylint: disable=invalid-name,missing-docstring,unused-argument
    # pylint: disable=attribute-defined-outside-init

    def __init__(self):
        self.tokens = OqasmLexer.tokens
        self.reserved = OqasmLexer.reserved
    # ---- Beginning of the PLY lexer ----
    literals = r'=()[]{};<>,.+-/*^"'
    reserved = {
        'barrier': 'BARRIER',
        'creg': 'CREG',
        'gate': 'GATE',
        'if': 'IF',
        'measure': 'MEASURE',
        'opaque': 'OPAQUE',
        'qreg': 'QREG',
        'pi': 'PI',
        'reset': 'RESET',
    }
    tokens = [
        'NNINTEGER',
        'REAL',
        'CX',
        'U',
        'FORMAT',
        'ASSIGN',
        'MATCHES',
        'ID',
        'IGNORE',
    ] + list(reserved.values())

    def t_REAL(self, t):
        r'(([0-9]+|([0-9]+)?\.[0-9]+|[0-9]+\.)[eE][+-]?[0-9]+)|(([0-9]+)?\.[0-9]+|[0-9]+\.)'
        t.value = float(t.value)
        return t

    def t_NNINTEGER(self, t):
        r'[1-9]+[0-9]*|0'
        t.value = int(t.value)
        return t

    def t_ASSIGN(self, t):
        '->'
        return t

    def t_MATCHES(self, t):
        '=='
        return t

    def t_IGNORE(self, t):
        r'\"([^\\\"]|\\.)*\"'
        return t
    # The include might be dropped, or ignored, as we probably won't need it

    def t_INCLUDE(self, t):
        'include'
        #
        # Now eat up the next two tokens which must be
        # 1 - the name of the include file, and
        # 2 - a terminating semicolon
        #
        # Then push the current lexer onto the stack, create a new one from
        # the include file, and push it onto the stack.
        #
        # When we hit eof (the t_eof) rule, we pop.
        next_token = self.lexer.token()
        lineno = next_token.lineno
        next_token = self.lexer.token()

        if next_token is None or next_token.value != ';':
            raise SyntaxError('Invalid syntax, missing ";" at line', str(lineno))

        return self.lexer.token()

    def t_FORMAT(self, t):
        r'OPENQASM\s+(\d+)\.(\d+)'
        return t

    def t_COMMENT(self, t):
        r'//.*'
        pass

    def t_CX(self, t):
        'CX'
        return t

    def t_U(self, t):
        'U'
        return t

    def t_ID(self, t):
        r'[a-z][a-zA-Z0-9_]*'
        if t.value in self.reserved:
            t.type = self.reserved[t.value]
            return t
        return t

    def t_newline(self, t):
        r'\n+'
        self.lexer.lineno += len(t.value)

    t_ignore = ' \t'

    def t_error(self, t):
        print("Unable to match any token rule, got -->%s<--" % t.value[0])
        print("Check your OPENQASM source and any include statements.")
        t.lexer.skip(1)

    def build(self, **kwargs):
        """ Builds the lexer """
        self.lexer = lex.lex(module=self, **kwargs)
