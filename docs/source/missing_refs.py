"""Handle missing references in documentation.."""
import re
from typing import Iterator, Optional, Tuple

from pathlib import Path

from docutils.nodes import Element
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.errors import NoUri


def identify_typevars() -> Iterator[Tuple[str, str]]:
    """Record all typevars in srctools."""
    srctools_folder = Path(__file__, '..', '..', '..', 'src', 'srctools').resolve()
    print(srctools_folder.resolve())
    for mod_name in srctools_folder.glob('*.py'):
        with open(mod_name) as f:
            for line in f:
                # A simple regex should be sufficient to find them all, no need to actually parse.
                match = re.search(r'^\s*(\w+)\s*=\s*(TypeVar|TypeVarTuple|ParamSpec)\(', line)
                if match is not None:
                    yield f'srctools.{mod_name.stem}.{match.group(1)}', 'typing.' + match.group(2)
                    if mod_name.stem == '__init__':
                        yield f'srctools.{match.group(1)}', 'typing.' + match.group(2)


# All our typevars, so we can suppress reference errors for them.
typevars = {
    '_os.PathLike': 'typing.TypeVar',
    'srctools.math._SupportsIndex': 'typing.SupportsIndex',
}

typevars.update(identify_typevars())


def on_missing_reference(
    app: Sphinx,
    env: BuildEnvironment,
    node: pending_xref,
    contnode: Element,
) -> Optional[Element]:
    """Handle missing references."""
    # If this is a typing_extensions object, redirect to typing.
    # Most things there are backports, so the stdlib docs should have an entry.
    if node['reftarget'].startswith('typing_extensions.'):
        new_node = node.copy()
        new_node['reftarget'] = 'typing.' + node['reftarget'][18:]
        return app.emit_firstresult(
            'missing-reference', env,
            new_node, contnode,
            allowed_exceptions=(NoUri,),
        )

    try:
        typevar_type = typevars[node['reftarget']]
    except KeyError:
        pass
    else:
        new_node = node.copy()
        new_node['reftarget'] = typevar_type
        return app.emit_firstresult(
            'missing-reference', env,
            new_node, contnode,
            allowed_exceptions=(NoUri,),
        )

    return None
