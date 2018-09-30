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

from .trace_parser import Parser
from .emitters import emitters
from .var_table import VarTable
import struct
import numpy as np
import sys
import torch
import torch.onnx
import torch.onnx.utils


class GeneratorException(Exception):
    def __init__(self, message):
        self.message = message


def generate(module, args=tuple(), kwargs=None):
    def write_h(x, *args):
        source_h.write(x % args)
        
    def write_cpp(x, *args):
        #sys.stdout.write(x % args)
        source_cpp.write(x % args)

    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        args = (args,)
    trace, out = torch.jit.get_trace_graph(module, args, kwargs)
    trace = torch.onnx.utils._optimize_graph(trace.graph(), False)

    #print(str(trace))
    p = Parser()
    result = p.parse(str(trace))
    if result is None:
        raise Exception('Parsing error')

    inputs, statements, return_vars = result

    vtable = VarTable(module)
    module_name = module.__class__.__name__

    # assign types for input vars
    for var_name, var_type, var_init in inputs:
        vtable.set_var_type(var_name, var_type, len(var_init))

    # create emitters
    # will populate vtable.init_list with parameters
    emitter_list = []
    for lhs, rhs in statements:
        op = rhs[0]
        if op not in emitters:
            raise GeneratorException('%s does not have an Emitter' % op)
        e = emitters[op](lhs, rhs, vtable)
        emitter_list.append(e)

    with open(module_name + ".cpp", "w") as source_cpp, open(module_name + ".h", "w") as source_h:
        write_h('#include "tensor4.h"' + '\n' * 3)
        write_cpp('#include "%s"' % (module_name + ".h") + '\n' * 3)

        write_h('struct %s\n{\n' % vtable.class_name)

        arguments = []

        for var_name, var_type, var_init in inputs:
            if var_name in vtable.init_list:
                var_cname = vtable.to_c_name(var_name).replace('ctx.', '')

                decl_str = "\t%s %s;\n" % (vtable.get_var_type(var_name), var_cname)

                write_h(decl_str)
            else:
                arguments.append(var_name)

        write_h('};' + '\n' * 3)

        declaration = '%s %sLoad(const char* filename)' % (vtable.class_name, vtable.class_name)
        write_h(declaration + ";\n\n")
        write_cpp(declaration + "\n{\n")
        write_cpp('\t%s ctx;\n', vtable.class_name)
        write_cpp('\tt4::model_dict dict = t4::load(filename);\n')

        for var_name, var_type, var_init in inputs:
            if var_name in vtable.init_list:
                var_cname = vtable.to_c_name(var_name)
                string = "\tdict.load(%s, \"%s\", %s);\n" % (var_cname, vtable.init_list[var_name][0], ', '.join([str(p) for p in var_init]))

                write_cpp(string)

        write_cpp('\treturn ctx;\n}' + '\n' * 3)

        
        if len(return_vars) == 1:
            return_var = '%s ' % vtable.get_var_type(return_vars[0])
        else:
            return_var = 'std::tuple<%s> ' % ', '.join([vtable.get_var_type(x) for x in return_vars])
        write_cpp(return_var)
        write_h(return_var)
        
        declaration = '%sForward(const %s& ctx, %s)' % (
              vtable.class_name, vtable.class_name, ', '.join([vtable.get_var_type(x) + ' ' + vtable.to_c_name(x) for x in arguments]))
        write_h(declaration + ';\n')
        write_cpp(declaration + '\n{\n')

        for e in emitter_list:
            string = e.emit()
            free_list = vtable.get_clean_list()
            if len(free_list) > 0:
                if string is not None:
                    string = string.replace('t4::Relu(', 't4::ReluInplace(')
                    string = string.replace('t4::BatchNormalization(', 't4::BatchNormalizationInplace(')
                    write_cpp('\t%s', string)
                write_cpp("\tt4::release(")
                string = ", ".join([vtable.to_c_name(x) for x in free_list])
                write_cpp("%s);\n", string)
            elif string is not None:
                write_cpp('\t%s', string)

        if len(return_vars) == 1:
            write_cpp('\treturn %s;\n', vtable.to_c_name(return_vars[0]))
        else:
            write_cpp('\treturn std::make_tuple(%s);\n' % ', '.join([vtable.to_c_name(x) for x in return_vars]))

        write_cpp('}\n')

        vtable.write_blob(vtable.class_name + '.bin')

    return out
