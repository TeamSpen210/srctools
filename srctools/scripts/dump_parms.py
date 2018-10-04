"""Dump all the shader parameters from the Source SDK code.

Run with a list of paths pointing to the materialsystem/ folders in the SDK.
This produces _shaderdb.py
"""
import glob
import ast
import sys
import collections
from collections import defaultdict

from srctools.vmt import VarType

CONFLICTS = collections.defaultdict(list)
types = set()

Var = collections.namedtuple('Var', [
    'name',
    'type',
    'default',
    'help',
    'flags',
    'filename',
])

VARS = {}


def process(filename, args):
    """Handle the parameters passed to a shader macro."""
    flags = 0
    
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
    
    name = '$' + ast_name.id.casefold().title()
    var = Var(
        name=ast_name.id,
        type=VarType(ast_type.id),
        default=ast.literal_eval(ast_default),
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
            CONFLICTS[name].append(var)
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
                    args = args.rstrip(');')
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
            var_name = line.strip(' \t\n",').casefold().title()

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

            if var_name in VARS:
                if VARS[var_name] is not VarType.FLAG:
                    CONFLICTS[var_name].append(var)
            else:
                VARS[var_name] = var


def main():
    """Run this script."""
    for folder in sys.argv[1:]:
        print("Reading from", folder)
        dump(folder)

    for confs in CONFLICTS.values():
        for conf in confs:
            print(conf)
        print()

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

        f.write('''\
    for vt, vars in zip(var_type, (
''')

        for var_type in VarType:
            f.write('''\
        {!r},
'''.format(' '.join(sorted(var_by_type[var_type]))))

        # We split strings at runtime - a list of strings would be too big to
        # constant-fold, but a string will always be stored as a full-size
        # constant.
        f.write('''\
    )):
        for v in vars.split(' '):
            DB[v] = vt
''')

    from srctools import _shaderdb


if __name__ == '__main__':
    main()
