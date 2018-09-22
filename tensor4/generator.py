from trace_parser import Parser
from emitters import emitters
from var_table import VarTable
import test_doc
import struct
import numpy as np
import sys
import torch


class GeneratorException(Exception):
    def __init__(self, message):
        self.message = message


def generate(module, args=tuple(), kwargs=None):
    def write(x, *args):
        sys.stdout.write(x % args)
        source.write(x % args)

    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        args = (args,)
    trace, out = torch.jit.get_trace_graph(module, args, kwargs)
    trace = torch.onnx.utils._optimize_graph(trace.graph(), False)

    print(str(trace))
    p = Parser()
    result = p.parse(str(trace))
    if result is None:
        raise Exception('Parsing error')

    inputs, statements, return_vars = result

    vtable = VarTable(module)

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

    with open("source_.cpp", "w") as source:
        write('#include "t4.h"' + '\n' * 3)

        write('struct %s\n{\n' % vtable.class_name)

        arguments = []

        for var_name, var_type, var_init in inputs:
            if var_name in vtable.init_list:
                var_cname = vtable.to_c_name(var_name).replace('ctx.', '')

                decl_str = "\t%s %s;\n" % (vtable.get_var_type(var_name), var_cname)

                write(decl_str)
            else:
                arguments.append(var_name)

        write('};' + '\n' * 3)

        write('%s %sLoad(const char* filename)\n{\n', vtable.class_name, vtable.class_name)
        write('\t%s ctx;\n', vtable.class_name)
        write('\tt4::model_dict dict = t4::load(filename);\n')

        for var_name, var_type, var_init in inputs:
            if var_name in vtable.init_list:
                var_cname = vtable.to_c_name(var_name)
                string = "\tdict.load(%s, \"%s\", %s);\n" % (var_cname, vtable.init_list[var_name][0], ', '.join([str(p) for p in var_init]))

                write(string)

        write('\treturn ctx;\n}' + '\n' * 3)

        if len(return_vars) == 1:
            write('%s ', vtable.get_var_type(return_vars[0]))
        else:
            write('std::tuple<%s> ' % ', '.join([vtable.get_var_type(x) for x in return_vars]))

        write('%sForward(const %s& ctx, %s)\n{\n',
              vtable.class_name, vtable.class_name, ', '.join([vtable.get_var_type(x) + ' ' + vtable.to_c_name(x) for x in arguments]))

        for e in emitter_list:
            string = e.emit()
            if string is not None:
                write('\t%s', string)
            free_list = vtable.get_clean_list()
            if len(free_list) > 0:
                write("\tt4::free(")
                string = ", ".join([vtable.to_c_name(x) for x in free_list])
                write("%s);\n", string)

        if len(return_vars) == 1:
            write('\treturn %s;\n', vtable.to_c_name(return_vars[0]))
        else:
            write('\treturn std::make_tuple(%s);\n' % ', '.join([vtable.to_c_name(x) for x in return_vars]))

        write('}\n')

        vtable.write_blob(vtable.class_name + '.bin')

    return out

#generate(None)
