"""Compare the public srctools API footprint to what's described in the docs.

This ensures public things are all listed.
"""
from typing import Iterator
import posixpath
import importlib
import sys
import types
import pkgutil
import enum

from sphinx.util.inventory import InventoryFile

import srctools

# Don't use Cython modules, they're harder to introspect.
sys.modules['srctools._math'] = None
sys.modules['srctools._tokenizer'] = None
sys.modules['srctools._cy_vtf_readwrite'] = None


def search_modules() -> Iterator[types.ModuleType]:
    """List all the srctools modules."""
    for info in pkgutil.iter_modules(srctools.__path__, 'srctools.'):
        if info.name.startswith('srctools._') or info.name in ['srctools.scripts', 'srctools.vec']:
            # Ignore:
            # - Private internals.
            # - Non-API scripts
            # - Deprecated modules.
            continue
        yield importlib.import_module(info.name)


def find_api(mod: types.ModuleType) -> Iterator[str]:
    """Find names in modules."""
    mod_name = mod.__name__
    yield mod_name
    try:
        symbols = mod.__all__
    except AttributeError:
        # Assume underscore = private.
        symbols = [name for name in vars(mod) if not name.startswith('_')]

    for name in symbols:
        try:
            obj = getattr(mod, name)
        except AttributeError:
            print(f'Missing attr: {mod_name}.{name}!', file=sys.stderr)
            continue
        yield f'{mod_name}.{name}'
        if isinstance(obj, type):
            # Recurse.
            is_enum = issubclass(obj, enum.Enum)
            values = vars(obj)
            for child, value in values.items():
                if child.startswith('_'):
                    continue
                # Exclude add_unknown() fields, aliases - those are documented under regular entry.
                if is_enum and (child.isdigit() or (isinstance(value, obj) and value.name != child)):
                    continue
                yield f'{mod_name}.{name}.{child}'
            try:
                annot = obj.__annotations__
                for child in annot:
                    if not child.startswith('_'):
                        yield f'{mod_name}.{name}.{child}'
            except AttributeError:
                pass


def main() -> int:
    """Do comparison."""
    api: set[str] = set()
    missing: set[str] = set()

    with open('build/objects.inv', 'rb') as f:
        inventory = InventoryFile.load(f, 'build/objects.inv', posixpath.join)

    documented: set[str] = set()
    for category, contents in inventory.items():
        documented.update(contents.keys())

    for mod in search_modules():
        print('Module:', mod.__name__)
        for name in find_api(mod):
            api.add(name)
            if name not in documented:
                missing.add(name)
        if not hasattr(mod, '__all__'):
            print(f'No {mod.__name__}.__all__!')

    print(end='', flush=True)
    if missing:
        print('Missing:', file=sys.stderr)
        for name in sorted(missing):
            print(' *', name, file=sys.stderr)
        print('Count:', len(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
