"""Dump all the shader parameters from the Source SDK code.

Run with a list of paths pointing to the materialsystem/ folders in the SDK.
This produces _shaderdb.py
"""
import glob
import ast
import re
import sys
import collections
from collections import defaultdict
from typing import Dict, List, Set

from srctools.vmt import VarType


Var = collections.namedtuple('Var', [
    'name',
    'type',
    'default',
    'help',
    'flags',
    'filename',
])

VARS: Dict[str, Var] = {}
CONFLICTS: Dict[str, Set[Var]] = collections.defaultdict(set)

# Overrides for some vars that have mismatches.
OVERRIDES: Dict[str, VarType] = {
    'bluramount': VarType.FLOAT,  # Conflict, also int.
    'bloomamount': VarType.VEC4,  # Conflicted, also float.
    # On unlitgeneric is a >dx9 bool, on lightmappedgeneric it's 80, 81, 90, etc.
    'envmapoptional': VarType.INT,
    'color': VarType.COLOR,  # Randomly made vec3 in some shaders.
    'clearcolor': VarType.VEC3,  # Int and vec
    'noisescale': VarType.VEC4,  # Float and Vec4
    'phongexponent': VarType.FLOAT, # Float and int.
    # Defined as float in engine_post, but immediately casted to int.
    'num_lookups': VarType.INT,
    'selfillumfresnelminmaxexp': VarType.VEC4,  # 3 or 4 values depending on shader.
    # 4 floats in Character shader, float in lightmappedgeneric
    'detailscale': VarType.VEC4,
    # Generic parameter, int in weapondecal but that's unused.
    'alpha': VarType.FLOAT,

    # Flags in some, int in others
    'vertexcolor': VarType.INT,
    'notint': VarType.INT,

    # Vec3 in some shaders, but should be logically color.
    'selfillumtint': VarType.COLOR,
    'phongtint': VarType.COLOR,
    'envmaptint': VarType.COLOR,
    'colortint': VarType.COLOR,
    'flow_color': VarType.COLOR,
    'flow_vortex_color': VarType.COLOR,

    # Marked as int in some shaders but really a bool.
    'separatedetailuvs': VarType.BOOL,
    'forcecheap': VarType.BOOL,
    'nodiffusebumplighting': VarType.BOOL,
    'depthblend': VarType.BOOL,
    'phong': VarType.BOOL,
    'showalpha': VarType.BOOL,
}


def process(filename: str, args: str) -> None:
    """Handle the parameters passed to a shader macro."""
    # Make python do the work, by parsing the data.
    ast_tuple = ast.parse(args.lstrip(), filename, 'eval').body
    elements = ast_tuple.elts
    
    if len(elements) == 5:
        ast_name, ast_type, ast_default, ast_help, ast_flags = ast_tuple.elts
        if isinstance(ast_flags, ast.Name):
            flags = ast_flags.id
        else:
            flags = ast.literal_eval(ast_flags)
        
    elif len(elements) == 4:
        ast_name, ast_type, ast_default, ast_help = ast_tuple.elts
        flags = 0
    elif len(elements) == 3:
        ast_name, ast_type, ast_default = ast_tuple.elts
        ast_help = None
        flags = 0
    else: 
        raise ValueError(ast.dump(ast_tuple))

    if isinstance(ast_default, ast.Name):
        default = ast_default.id
    else:
        default = ast.literal_eval(ast_default)
    
    name = '$' + ast_name.id.casefold()
    if ast_name.id.casefold() in OVERRIDES:
        return
    var = Var(
        name=ast_name.id,
        type=VarType(ast_type.id),
        default=default,
        help=ast.literal_eval(ast_help) if ast_help else '',
        flags=flags,
        filename=filename,
    )
    
    # Detect doubled-up vars!
    if name in VARS:
        other = VARS[name]
        # If the same type, it's compatible.
        if var.type == other.type:
            # If the default is different, note that.
            # but we don't care about help etc.
            if var.default != other.default:
                if isinstance(other.default, frozenset):
                    var = var._replace(default=other.default | {var.default})
                else:
                    var = var._replace(default=frozenset({var.default, other.default}))
                    
                if isinstance(other.filename, frozenset):
                    var = var._replace(filename=other.filename | {var.filename})
                else:
                    var = var._replace(filename=frozenset({var.filename, other.filename}))
        else:
            # Different types, major issue.
            CONFLICTS[name].add(var)
    VARS[name] = var


def dump(folder):
    """Process the materialsystem/ folder from a game repro."""
    for file in glob.iglob(folder + '/**/*'):
        if not file.endswith(('.h', '.cpp', '.c')): continue
        with open(file, errors='ignore') as f:
            for line in f:
                if line.strip() == 'BEGIN_SHADER_PARAMS':
                    break
            else:  # No shader params in this file...
                continue
            
            for line in f:
                line = line.strip()
                if line.startswith('SHADER_PARAM'):
                    macro_name, args = line.split('(', 1)
                    if '//' in args:
                        args = args.split('//', 1)[0]
                    args = args.strip().rstrip(');')
                    try:
                        process(file, args)
                    except Exception as e:
                        raise Exception(file, macro_name, args) from e

    # Special case - FLAGS values.
    # These are a bit flag set on any material.
    try:
        f = open(folder + '/shadersystem.cpp')
    except FileNotFoundError:
        return
    with f:
        for line in f:
            line = line.strip()
            if 's_pShaderStateString[] =' in line:
                break
        else:
            raise ValueError('No s_pShaderStateString in shadersystem.cpp?')
        for line in f:
            if '}' in line:
                return
            if '//' in line:
                line = line.split('//')[0]
            var_name = line.strip(' \t\n",').casefold()

            if '$' not in var_name:
                continue  # Other junk (blank lines, {)

            var = Var(
                name=var_name.lstrip('$'),
                type=VarType.FLAG,
                default='0',
                help='FLAG: sets bitfield.',
                flags=0,
                filename='shadersystem.cpp',
            )

            try:
                other_var = VARS[var_name]
            except KeyError:
                VARS[var_name] = var
            else:
                if other_var.type is not VarType.FLAG:
                    CONFLICTS[var_name] |= {var, other_var}


def main():
    """Run this script."""
    for folder in sys.argv[1:]:
        print("Reading from", folder)
        dump(folder)

    for key, confs in CONFLICTS.items():
        for conf in sorted(confs, key=lambda v: v.filename):
            print(conf)
        print()

    for var_name, var_type in OVERRIDES.items():
        VARS['$' + var_name.casefold()] = Var(
            var_name, var_type,
            '', '', '',
            'shaderdb.py',
        )

    print('Default conflicts:')
    for var in VARS.values():
        if isinstance(var.default, frozenset):
            print(var.name+':', list(var.default), '\n', list(var.filename))
            print()

    with open('../_shaderdb.py', 'w') as f:
        f.write('def _shader_db(var_type, DB):\n')
        var_by_type = defaultdict(list)
        for var in VARS.values():
            var_by_type[var.type].append(var.name.casefold())

        f.write('    [{}] = var_type\n'.format(', '.join([
            var_type.name
            for var_type in VarType
        ])))
        for var_type in VarType:
            for var in sorted(var_by_type[var_type]):
                f.write('    DB[{!r}] = {}\n'.format(var, var_type.name))
            f.write('\n')


if __name__ == '__main__':
    main()
