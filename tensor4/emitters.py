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

import var_table

class EmitterException(Exception):
    def __init__(self, message):
        self.message = message

emitters = {}


def register(emitter, op_name):
    """
    Registers an emitter with given operator name.

    Arguments:
        emitter : class of the emitter
        op_name (string): name of the operator
    """
    emitters[op_name] = emitter


def func_printer(emitter):
    var_name, var_type, var_init = emitter.output
    decl_str = "%s %s = t4::" % (emitter.vtable.get_var_type(var_name), emitter.vtable.to_c_name(var_name))
    _, dtype = emitter.vtable.get_var_dim_dtype(var_name)
    string = decl_str + emitter.name
    if len(emitter.template_parameters) > 0:
        string += "<"
        for a in emitter.template_parameters:
            string += str(a) + ", "
        string = string[:-2]
        string += ">"
    string += "("
    for a in emitter.args:
        string += emitter.vtable.to_c_name(a, dtype=dtype) + ", "
    for a in emitter.parameters:
        if isinstance(a, float):
            p = str(a) + 'f'
        else:
            p = str(a)
        string += p + ", "
    string = string[:-2] + ");"
    if len(emitter.scope_name) > 0:
        string += " //%s" % emitter.scope_name
    return string + "\n"


def binary_op_printer(emitter, op):
    var_name, var_type, var_init = emitter.output
    assert(len(emitter.args) == 2)
    string = "%s %s = %s %s %s;" % (
        emitter.vtable.get_var_type(var_name),
        emitter.vtable.to_c_name(var_name),
        emitter.vtable.to_c_name(emitter.args[0]),
        op,
        emitter.vtable.to_c_name(emitter.args[1]))
    return string + "\n"


class Emitter(object):
    """
    Basic emitter
    """
    def __init__(self, lhs, rhs, vtable):
        self.lhs = lhs
        self.rhs = rhs
        self.output = None
        self.template_parameters = []
        self.parameters = []
        self.name = "None"
        self.vtable = vtable
        self._emit = lambda: func_printer(self)
        self.validated = False
        _, self.param_map, self.args, self.scope_type, self.scope_name = rhs

    def emit(self):
        assert self.validated
        res = self._emit()
        for a in self.args:
            self.vtable.dec_ref_count(a)
        return res

    def validate_arg_return_count(self, arg_count, return_count):
        assert (len(self.lhs) == return_count)
        assert len(self.args) == arg_count, \
            "%s: Expected %d arguments, but received %d" % (self.__class__.__name__, arg_count, len(self.args))
        for a in self.args:
            self.vtable.inc_ref_count(a)
        self.validated = True

    def bind_arg_to_parameter(self, parameter_name, arg_no):
        assert self.vtable.has_parameter(self.scope_name + parameter_name), \
            "Argument %d can not be bind to %s because %s was not found in variable table" \
            % (arg_no, parameter_name, self.scope_name + parameter_name)

        self.vtable.bind_var_to_parameter(self.args[arg_no], self.scope_name + parameter_name)

    def make_output_same_as_first_arg(self):
        ndim, dtype = self.vtable.get_var_dim_dtype(self.args[0])
        self.output = self.lhs[0]
        self.vtable.set_var_type(self.output[0], dtype, ndim)

    def append_parameter(self, name, templated=False, padding=False):
        param = self.param_map[name]
        param_to_append = []
        if isinstance(param, dict):
            for i in range(len(param.items())):
                assert("_%d" % i in param)
                param_to_append.append(param["_%d" % i])
        elif isinstance(param, list):
            for p in param:
                param_to_append.append(p)
        else:
            param_to_append.append(param)
        if padding:
            assert param_to_append[:2] == param_to_append[2:], "Padding must be symmetric"
            param_to_append = param_to_append[:2]
        append_to = self.template_parameters if templated else self.parameters
        append_to += param_to_append

    def get_dim(self):
        return len(self.param_map['kernel_shape'])


class Conv(Emitter):
    def __init__(self, lhs, rhs, vtable):
        Emitter.__init__(self, lhs, rhs, vtable)
        self.make_output_same_as_first_arg()

        self.validate_arg_return_count(3, 1)
        self.bind_arg_to_parameter('.weight', 1)
        self.bind_arg_to_parameter('.bias', 2)

        self.append_parameter('kernel_shape', templated=True)
        self.append_parameter('strides', templated=True)
        self.append_parameter('pads', templated=True, padding=True)
        self.append_parameter('dilations', templated=True)

        self.name = "Conv%dd" % self.get_dim()


class ConvTranspose(Conv):
    def __init__(self, lhs, rhs, vtable):
        Conv.__init__(self, lhs, rhs, vtable)
        self.name = "ConvTranspose%dd" % self.get_dim()


class MaxPool(Emitter):
    def __init__(self, lhs, rhs, vtable):
        Emitter.__init__(self, lhs, rhs, vtable)
        self.make_output_same_as_first_arg()

        self.validate_arg_return_count(1, 1)

        self.append_parameter('kernel_shape', templated=True)
        self.append_parameter('strides', templated=True)
        self.append_parameter('pads', templated=True, padding=True)

        self.name = "MaxPool%dd" % self.get_dim()


class Flatten(Emitter):
    def __init__(self, lhs, rhs, vtable):
        Emitter.__init__(self, lhs, rhs, vtable)
        _, dtype = self.vtable.get_var_dim_dtype(self.args[0])
        self.output = self.lhs[0]
        self.vtable.set_var_type(self.output[0], dtype, 2)
        self.validate_arg_return_count(1, 1)
        self.append_parameter('axis', templated=True)

        self.name = "Flatten"


class Softmax(Emitter):
    def __init__(self, lhs, rhs, vtable):
        Emitter.__init__(self, lhs, rhs, vtable)
        self.make_output_same_as_first_arg()
        self.validate_arg_return_count(1, 1)
        self.append_parameter('axis', templated=True)
        self.name = "Softmax"


class Dropout(Emitter):
    def __init__(self, lhs, rhs, vtable):
        Emitter.__init__(self, lhs, rhs, vtable)
        assert (len(lhs) == 2)
        assert (len(self.args) == 1)
        self.lhs = self.lhs[:1]
        self.make_output_same_as_first_arg()
        self.append_parameter('ratio', templated=False)
        self.validate_arg_return_count(1, 1)

        self.name = "Dropout"


class Gemm(Emitter):
    def __init__(self, lhs, rhs, vtable):
        Emitter.__init__(self, lhs, rhs, vtable)

        if self.param_map["alpha"] == 1 and self.param_map["beta"] == 1 and self.param_map["transB"] == 1 and \
                ("transA" not in self.param_map or self.param_map["transA"] == 0):
            self.make_output_same_as_first_arg()
            self.validate_arg_return_count(3, 1)
            self.bind_arg_to_parameter('.weight', 1)
            self.bind_arg_to_parameter('.bias', 2)
            self.name = "Linear"

        else:
            raise EmitterException("Unknown Gemm operation")


class Pad(Emitter):
    def __init__(self, lhs, rhs, vtable):
        Emitter.__init__(self, lhs, rhs, vtable)
        self.make_output_same_as_first_arg()
        self.validate_arg_return_count(1, 1)
        self.append_parameter('mode', templated=True)
        self.append_parameter('pads')
        self.name = "Pad"


class Relu(Emitter):
    def __init__(self, lhs, rhs, vtable):
        Emitter.__init__(self, lhs, rhs, vtable)
        self.name = "Relu"
        self.validate_arg_return_count(1, 1)
        self.make_output_same_as_first_arg()


class LeakyRelu(Emitter):
    def __init__(self, lhs, rhs, vtable):
        Emitter.__init__(self, lhs, rhs, vtable)
        self.append_parameter('alpha', templated=False)
        self.name = "LeakyRelu"
        self.validate_arg_return_count(1, 1)
        self.make_output_same_as_first_arg()


class Tanh(Emitter):
    def __init__(self, lhs, rhs, vtable):
        Emitter.__init__(self, lhs, rhs, vtable)
        self.name = "Tanh"
        self.validate_arg_return_count(1, 1)
        self.make_output_same_as_first_arg()


class Neg(Emitter):
    def __init__(self, lhs, rhs, vtable):
        Emitter.__init__(self, lhs, rhs, vtable)
        self.name = "Neg"
        self.validate_arg_return_count(1, 1)
        self.make_output_same_as_first_arg()


class Undefined(Emitter):
    def __init__(self, lhs, rhs, vtable):
        Emitter.__init__(self, lhs, rhs, vtable)
        self.validate_arg_return_count(0, 1)
        self._emit = lambda: None


class Constant(Emitter):
    def __init__(self, lhs, rhs, vtable):
        Emitter.__init__(self, lhs, rhs, vtable)
        var_name, var_type, var_init = lhs[0]
        self.validate_arg_return_count(0, 1)
        if self.param_map['value'] != '<Tensor>':
            v = self.param_map['value']
            self.vtable.add_scalar_const(var_name, v)
        self._emit = lambda: None


class AtenExpand(Emitter):
    def __init__(self, lhs, rhs, vtable):
        Emitter.__init__(self, lhs, rhs, vtable)
        var_name, var_type, var_init = lhs[0]
        self.vtable.add_alias(var_name, self.args[0])
        self.validate_arg_return_count(1, 1)
        self._emit = lambda: None


class Mul(Emitter):
    def __init__(self, lhs, rhs, vtable):
        Emitter.__init__(self, lhs, rhs, vtable)
        self.name = "Mul"
        self.validate_arg_return_count(2, 1)
        self.make_output_same_as_first_arg()
        #self._emit = lambda: binary_op_printer(self, '*')


class Add(Emitter):
    def __init__(self, lhs, rhs, vtable):
        Emitter.__init__(self, lhs, rhs, vtable)
        self.name = "Add"
        self.validate_arg_return_count(2, 1)
        self.make_output_same_as_first_arg()
        #self._emit = lambda: binary_op_printer(self, '+')


class BatchNormalization(Emitter):
    def __init__(self, lhs, rhs, vtable):
        Emitter.__init__(self, lhs, rhs, vtable)
        assert (len(lhs) == 5 or len(lhs) == 1)
        assert (len(self.args) == 5)
        self.lhs = self.lhs[:1]
        self.make_output_same_as_first_arg()

        self.append_parameter('epsilon', templated=False)

        self.validate_arg_return_count(5, 1)
        self.bind_arg_to_parameter('.weight', 1)
        self.bind_arg_to_parameter('.bias', 2)
        self.bind_arg_to_parameter('.running_mean', 3)
        self.bind_arg_to_parameter('.running_var', 4)

        #parameters: 'epsilon', 'momentum', 'is_test'
        self.name = "BatchNormalization"


register(Pad, 'onnx::Pad')
register(Neg, 'onnx::Neg')
register(AtenExpand, 'aten::expand')
register(Conv, 'onnx::Conv')
register(ConvTranspose, 'onnx::ConvTranspose')
register(MaxPool, 'onnx::MaxPool')
register(Flatten, 'onnx::Flatten')
register(Softmax, 'onnx::Softmax')
register(Dropout, 'onnx::Dropout')
register(Gemm, 'onnx::Gemm')
register(LeakyRelu, 'onnx::LeakyRelu')
register(Relu, 'onnx::Relu')
register(Tanh, 'onnx::Tanh')
register(Undefined, 'prim::Undefined')
register(Constant, 'onnx::Constant')
register(Mul, 'onnx::Mul')
register(Add, 'onnx::Add')
register(BatchNormalization, 'onnx::BatchNormalization')
