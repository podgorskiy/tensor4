# Copyright 2018 Stanislav Pidhorskyi. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys


class TraceParseException(Exception):
    def __init__(self, message):
        self.message = message


class Parser:
    def __init__(self):
        self.text = ""
        self.it = 0
        self.param = None
        self.inputs = None
        self.var = None
        self.op = ''
        self.op_param = None
        self.op_args = None
        self.return_vars = None
        self.scope_type = ''
        self.scope_name = ''
        self.types = ['Float', 'Dynamic', 'Long', 'Tensor']
        self.statements = []

    def parse(self, text):
        print(text)
        self.text = text + '\0'
        try:
            self.expect_graph()
        except Exception as e:
            start = self.text.rfind('\n', 0, self.it)
            end = self.text.find('\n', self.it)
            if start == -1:
                start = 0
            if end == -1:
                end = len(self.text)
            print(('In line:\n%s\n' % self.text[start:end]) + ' ' * (self.it - start) + '^')
            sys.stdout.flush()
            raise e

        return self.inputs, self.statements, self.return_vars

    def expect_graph(self):
        self.skip_spaces()
        if self.accept_id() and self.param == 'graph':
            self.expect_graph_args()
            self.expect_body()
        else:
            raise TraceParseException("Trace should start with 'graph' keyword")

    def expect_graph_args(self):
        self.skip_spaces()
        self.expect_char('(')
        self.skip_spaces()
        self.accept_var_decl_list()
        self.inputs = self.param
        self.skip_spaces()
        self.expect_char(')')
        self.expect_char(':')

    def accept_var_decl_list(self):
        var_decl_list = []
        while self.accept_var_decl():
            var_decl_list.append(self.var)
            if self.accept_char(','):
                continue
            else:
                self.param = var_decl_list
                return True
        return False

    def accept_var_decl(self):
        self.skip_spaces()
        if self.accept_var_name():
            var_name = self.param
            self.skip_spaces()
            if self.accept_char(':'):
                self.skip_spaces()
                if self.accept_type():
                    var_type = self.param
                    self.skip_spaces()
                    self.param = None
                    if self.accept_var_params():
                        self.var = (var_name, var_type, self.param)
                        return True
        return False

    def accept_type(self):
        if self.accept_id():
            if self.param in self.types:
                return True
        return False

    def accept_var_params(self):
        if self.accept_char('('):
            params = []
            while self.accept_uint():
                self.accept_char('!')
                params.append(self.param)
                if self.accept_char(','):
                    self.skip_spaces()
                    continue
                break
            self.skip_spaces()
            if self.accept_char(')'):
                self.param = params
                return True
            return False
        return True

    def expect_body(self):
        self.skip_spaces()
        self.statements = []
        while self.accept_statement():
            self.statements.append(self.param)
        self.expect_return()
        self.skip_spaces()

    def accept_statement(self):
        lhs = []
        if self.accept_var_decl():
            lhs.append(self.var)
            self.skip_spaces()
            while self.accept_char(','):
                if self.accept_var_decl():
                    lhs.append(self.var)
                else:
                    raise TraceParseException("Expected variable after coma.")
            self.skip_spaces()
            self.expect_char('=')
            self.skip_spaces()
            self.expect_op()
            self.skip_spaces()
            if self.accept_char('#'):
                self.skip_till_next_line()

            rhs = (self.op, self.op_param, self.op_args, self.scope_type, self.scope_name)
            self.param = (lhs, rhs)
            return True
        return False

    def expect_op(self):
        self.op = ''
        self.op_param = None
        self.scope_type = ''
        self.scope_name = ''
        if not self.accept_id():
            raise TraceParseException("Expected identifier for operator name.")

        self.op = self.param
        self.param = None
        if self.accept_op_params(False):
            self.op_param = self.param
        else:
            raise TraceParseException("Wrong operator parameters")
        if self.accept_args():
            self.op_args = self.param
        else:
            raise TraceParseException("Wrong operator arguments")
        self.skip_spaces()
        if self.accept_char(','):
            if self.accept_scope():
                pass
            else:
                raise TraceParseException("Wrong scope")

    def accept_op_params(self, require):
        self.skip_spaces()
        if self.accept_char('['):
            pid = 0
            params = {}
            while True:
                name = '_{0}'.format(pid)
                pid += 1
                self.skip_spaces()
                if self.accept_id():
                    name = self.param
                    self.skip_spaces()
                    if self.accept_char('='):
                        pass
                    else:
                        return False
                if self.accept_number():
                    val = self.param
                elif self.accept_char('{') and self.accept_number():
                    val = self.param
                    if self.skip_spaces() and self.accept_char('}'):
                        pass
                    else:
                        return False
                elif self.accept_op_params(True):
                    val = self.param
                elif self.accept_id():
                    val = self.param
                else:
                    break
                params[name] = val
                self.skip_spaces()
                if self.accept_char(','):
                    pass
                else:
                    break
            if self.accept_char(']'):
                self.param = params
                return True
            else:
                return False
        return not require

    def accept_args(self):
        var_list = []
        self.skip_spaces()
        if self.accept_char('('):
            self.skip_spaces()
            if self.accept_var_name():
                var_list.append(self.param)
                while self.skip_spaces() and self.accept_char(','):
                    self.skip_spaces()
                    if self.accept_var_name():
                        var_list.append(self.param)
                    else:
                        return False
            self.skip_spaces()
            if self.accept_char(')'):
                self.param = var_list
                return True
            return False
        return True

    def accept_scope(self):
        if self.skip_spaces() and self.accept_id() and self.param == 'scope:':
            self.skip_spaces()
            types = []
            names = []
            while True:
                self.expect_scope_id()
                types += [self.param]

                if self.accept_char('['):
                    self.expect_scope_id()
                    names += [self.param]
                    self.expect_char(']')

                if not self.accept_char('/'):
                    break
            self.scope_type = ".".join(types)
            self.scope_name = ".".join(names)
            return True
        return False

    def expect_return(self):
        self.skip_spaces()
        if self.accept_id() and self.param == 'return':
            if self.accept_args():
                self.return_vars = self.param
                self.skip_spaces()
            else:
                raise TraceParseException("Expected return statement to return at least one tensor.")
        else:
            raise TraceParseException("Expected return statement.")

    def expect_id(self):
        if not self.accept_id():
            raise TraceParseException("Expected identifier.")

    def expect_scope_id(self):
        if not self.accept_scope_id():
            raise TraceParseException("Expected identifier.")

    def accept_scope_id(self):
        name = ""
        if self.accept_alnum() or self.accept_char(':') or self.accept_char('_') or self.accept_char('-'):
            name += self.param
            while self.accept_alnum() or self.accept_char(':') or self.accept_char('_') or self.accept_char('-'):
                name += self.param
            self.param = name
            return True
        return False

    def accept_id(self):
        name = ""
        if self.accept_alpha() or self.accept_special():
            name += self.param
            while self.accept_alnum() or self.accept_special():
                name += self.param
            self.param = name
            return True
        return False

    def accept_var_name(self):
        if self.accept_char('%'):
            name = '%'
            if self.accept_str('input.'):
                name += self.param
            if self.accept_alnum() or self.accept_special():
                name += self.param
                while self.accept_alnum() or self.accept_special():
                    name += self.param
                self.param = name
                return True
        return False

    def skip_spaces(self):
        while self.accept_space():
            pass
        return True

    def accept_special(self):
        return self.accept_char(':') or self.accept_char('/') or self.accept_char('_') or self.accept_char('-')

    def accept_alnum(self):
        return self.accept_digit() or self.accept_alpha()

    def accept_uint(self):
        self.skip_spaces()
        if self.accept_digit():
            x = int(self.param)
            while self.accept_digit():
                x = x * 10 + int(self.param)
            self.param = x
            return True
        return False

    def accept_sint(self):
        self.skip_spaces()
        sign = False
        current_pos = self.it
        if self.accept_char('+'):
            pass
        if self.accept_char('-'):
            sign = True
        if self.accept_uint():
            x = int(self.param)
            self.param = x * (-1 if sign else 1)
            return True
        self.it = current_pos
        return False

    def accept_number(self):
        current_pos = self.it
        self.skip_spaces()
        int_part = 0
        frac_part = 0
        sign = False
        has_mantisa = False
        has_dot = False
        if self.accept_char('+'):
            pass
        if self.accept_char('-'):
            sign = True
        if self.accept_uint():
            has_mantisa = True
            int_part = self.param
        mantissa = int_part
        if self.accept_char('.'):
            size = 1
            has_dot = True
            while self.accept_digit():
                frac_part = frac_part * 10 + int(self.param)
                size *= 10
            mantissa = float(int_part) + float(frac_part) / size
        if not (has_mantisa or has_dot):
            return False
        if self.accept_char('e') or self.accept_char('E'):
            if self.accept_sint():
                exp = self.param
                exp = pow(10, exp)
                if sign:
                    self.param = -mantissa * exp
                else:
                    self.param = mantissa * exp
                return True
        else:
            if sign:
                self.param = -mantissa
            else:
                self.param = mantissa
            return True
        self.it = current_pos
        return False

    def accept_digit(self):
        return self._accept_f(lambda x: x.isdigit())

    def accept_alpha(self):
        return self._accept_f(lambda x: x.isalpha())

    def accept_space(self):
        return self._accept_f(lambda x: x.isspace())

    def accept_char(self, c):
        return self._accept_f(lambda x: x == c)

    def expect_char(self, c):
        x = self.text[self.it]
        if x != c:
            raise TraceParseException("Expected %c, but got %c" % (c, x))
        self.param = self.text[self.it]
        self.it += 1

    def _accept_f(self, f):
        if f(self.text[self.it]):
            self.param = self.text[self.it]
            self.it += 1
            return True
        return False

    def accept_str(self, s):
        if self.text[self.it:min(len(self.text[self.it:]), self.it + len(s))] == s:
            self.param = s
            self.it += len(s)
            return True
        return False

    def skip_till_next_line(self):
        while not self.accept_char('\n'):
            if self.text[self.it] == '\0':
                break
            self.it += 1
