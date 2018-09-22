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

import struct
import numpy as np
import pickle


class VarTable:
    """
    Holds various data about variable and parameters
    """
    def set_var_type(self, var, dtype, ndim):
        """
        Assignes type information to the variable.

        Arguments:
            var (string of format %<number>): name of the variable in the trace
            dtype (string): type of data, 'Float', 'Int', or 'Double'
            ndim (int): number of dimentions. Range: [1..4]
        """
        assert(var[0] == '%')  # here 'var' is the name in %<number> format used in trace
        type_letter = {'Float': 'f', 'Int': 'i', 'Double': 'd', '?': '?'}
        self.var_types[var] = ("t4::tensor%d%s" % (ndim, type_letter[dtype]), ndim, dtype)

    def inc_ref_count(self, var):
        while var in self.alias:
            var = self.alias[var]
        if var not in self.ref_counts:
            self.ref_counts[var] = 0
        self.ref_counts[var] += 1

    def dec_ref_count(self, var):
        while var in self.alias:
            var = self.alias[var]
        assert(var in self.ref_counts)
        assert(self.ref_counts[var] > 0)
        self.ref_counts[var] -= 1

    def get_clean_list(self):
        to_clean = [k for k, v in self.ref_counts.items()
                if (v == 0 and k not in self.init_list and k not in self.scalar_constants)]
        for v in to_clean:
            del self.ref_counts[v]
        return to_clean

    def get_var_type(self, var):
        """
        Returns type string associated with the variable.

        Arguments:
            var (string of format %<number>): name of the variable in the trace
        """
        while var in self.alias:
            var = self.alias[var]
        assert(var[0] == '%')  # here 'var' is the name in %<number> format used in trace
        if var in self.var_types:
            return self.var_types[var][0]
        else:
            return "t4::tensor?"

    def get_var_dim_dtype(self, var):
        """
        Returns type information associated with the variable.

        Arguments:
            var (string of format %<number>): name of the variable in the trace

        Returns (tuple): ndims, dtype
        """
        while var in self.alias:
            var = self.alias[var]
        assert(var[0] == '%')  # here 'var' is the name in %<number> format used in trace
        if var in self.var_types:
            return self.var_types[var][1:]
        else:
            return 0, "?"

    def to_c_name(self, var, add_ctx=True, dtype=None):
        """
        Returns variable name to use in generated cpp code.

        Arguments:
            var (string of format %<number>): name of the variable in the trace
            add_ctx (bool): if true will add 'ctx.' prefix. Default is True.

        Returns (string): variable alias in cpp code
        """
        assert(var[0] == '%')  # here 'var' is the name in %<number> format used in trace
        if var in self.init_list:
            return ('ctx.' if add_ctx else '') + self.init_list[var][1]
        if var in self.scalar_constants:
            if dtype is None:
                return str(self.scalar_constants[var])
            else:
                if dtype == "Float":
                    return str(float(self.scalar_constants[var])) + "f"
                if dtype == "Int":
                    return str(int(self.scalar_constants[var]))
                if dtype == "Double":
                    return str(float(self.scalar_constants[var]))
        if var in self.alias:
            return self.to_c_name(self.alias[var])
        return "x" + var[1:]

    def bind_var_to_parameter(self, var, parameter):
        """
        Binds variable to named parameter

        Arguments:
            var (string of format %<number>): name of the variable in the trace
            parameter (string): parameter name.
        """
        assert(var[0] == '%')  # here 'var' is the name in %<number> format used in trace
        self.used_weights.add(parameter)
        self.init_list[var] = (parameter, parameter.replace('.', '_'))

    def has_parameter(self, parameter):
        """
        Check if such parameter exists

        Arguments:
            parameter (string): parameter name.
        """
        return parameter in self.model_dict

    def add_scalar_const(self, var_name, value):
        """
        assignes a scalar const value to var name

        Arguments:
            var_name (string): name of the variable.
            value: value (float, int).
        """
        self.scalar_constants[var_name] = value

    def add_alias(self, var_name, var_name2):
        """
        assignes a scalar const value to var name

        Arguments:
            var_name (string): name of the variable.
            var_name2 (string): name of the alias.
        """
        assert(var_name != var_name2)
        self.alias[var_name] = var_name2

    def write_blob(self, filename=None):
        """
        Dumps parameter data

        Arguments:
            filename (string): filename for binary blob
        """

        if filename is None:
            filename = self.class_name + '.bin'

        with open(filename, "wb") as f:
            for w_name in self.used_weights:
                w = self.model_dict[w_name]
                f.write(w_name.encode())
                f.write(bytes([0]))
                if w.dtype == np.float64:
                    f.write(bytes(b'doubl'))
                if w.dtype == np.float32:
                    f.write(bytes(b'float'))
                if w.dtype == np.int32:
                    f.write(bytes(b'int32'))
                if w.dtype == np.int16:
                    f.write(bytes(b'int16'))
                f.write(struct.pack("b", w.ndim))
                for i in range(w.ndim):
                    f.write(struct.pack("i", w.shape[i]))
                f.write(w.tobytes())

    def __init__(self, module):
        """
        Arguments:
            module (subclass of pytorch nn.Module): the module
        """
        # weights that are used. These ones are going to be written to file and loaded
        self.used_weights = set()

        # vars that are assigned to weights
        self.init_list = {}

        # vars that are scalar constants
        self.scalar_constants = {}

        # maps vars to types
        self.var_types = {}

        self.ref_counts = {}

        # aliases for vars
        self.alias = {}

        model_dict = module.state_dict()
        self.class_name = module.__class__.__name__

        self.model_dict = {}

        for name, param in model_dict.items():
            self.model_dict[name] = param.detach().cpu().numpy()
