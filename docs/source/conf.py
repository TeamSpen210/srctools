# Configuration file for the Sphinx documentation builder.
from collections import defaultdict

from typing import Any, Dict, List, Optional, Tuple, Type
import sys
import enum
import functools

from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.ext.autodoc import ClassDocumenter, AttributeDocumenter, ObjectMembers, bool_option
import attrs

# Prevent Cython modules from importing, Python ones have all the type hints and docstrings.
sys.modules['srctools._tokenizer'] = None
sys.modules['srctools._math'] = None
sys.modules['srctools._cy_vtf_readwrite'] = None
import srctools

# -- Project information -----------------------------------------------------

project = 'srctools'
copyright = '2022, TeamSpen210'
author = 'TeamSpen210'

# The full version, including alpha/beta/rc tags
release = srctools.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Intersphinx ------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}


# Fully-qualified is way too verbose.
autodoc_typehints_format = 'short'
autodoc_member_order = 'bysource'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


def setup(app: Sphinx) -> None:
    """Perform modifications."""
    app.add_autodocumenter(EnumDocumenter)
    app.add_autodocumenter(EnumMemberDocumenter)


@attrs.define
class EnumInfo:
    """Computed member info."""
    cls: Type[enum.Enum]
    aliases: Dict[str, List[str]]
    canonical: List[enum.Enum]
    should_hex: bool = False


@functools.lru_cache(maxsize=None)
def enum_aliases(enum_obj: Type[enum.Enum]) -> EnumInfo:
    """Compute the aliases for this enum."""
    aliases: Dict[str, List[str]] = defaultdict(list)
    canonical: List[enum.Enum] = []
    for name, member in enum_obj.__members__.items():
        if name != member.name:
            aliases[member.name].append(name)
        else:
            canonical.append(member)
    return EnumInfo(enum_obj, dict(aliases), canonical)


class EnumDocumenter(ClassDocumenter):
    """Handle enum documentation specially."""
    objtype = 'srcenum'
    directivetype = ClassDocumenter.objtype
    priority = 10 + ClassDocumenter.priority
    option_spec = {
        **ClassDocumenter.option_spec,
        'hex': bool_option,
    }
    del option_spec['show-inheritance']

    def __init__(self, *args) -> None:
        """Force-enable show-inheritance, we want to show it's an enum."""
        super().__init__(*args)
        self.options = self.options.copy()
        self.options['show-inheritance'] = True

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any) -> bool:
        """We can only document Enums."""
        return isinstance(member, type) and issubclass(member, enum.Enum)

    def filter_members(self, members: ObjectMembers, want_all: bool) -> List[Tuple[str, Any, bool]]:
        """Specially handle enum members."""
        results: List[Tuple[str, object, bool]] = []

        info = enum_aliases(self.object)
        info.should_hex = self.options.hex

        for member in info.canonical:  # Keep in order.
            if member.name.isdigit() and isinstance(member.value, int): ## and str(member.value) == member.name:
                # add_unknown() pseudo-flags, skip.
                continue
            results.append((member.name, member, True))

        # Have super() handle any other members (properties, methods).
        results.extend(super().filter_members([
            obj for obj in members
            if obj.__name__ not in info.cls.__members__
        ], True))
        return results


class NoReprString(str):
    """A string which doesn't quote in the repr."""
    def __repr__(self) -> str:
        return self


class EnumMemberDocumenter(AttributeDocumenter):
    """Special documenter for enum members."""
    objtype = '__srcenummember'
    directivetype = AttributeDocumenter.objtype  # Unchanged.
    priority = 10000

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any) -> bool:
        """We only document members from EnumDocumenter."""
        return isinstance(parent, EnumDocumenter)

    def add_directive_header(self, sig: str) -> None:
        """Alter behaviour of the header."""
        # If we're
        info = enum_aliases(self.parent)
        if info.should_hex:
            self.object = NoReprString(f'0x{int(self.object):x}')
        super().add_directive_header(sig)

    def add_content(self, more_content: Optional[StringList], no_docstring: bool = False) -> None:
        sourcename = self.get_sourcename()
        info = enum_aliases(self.parent)
        try:
            aliases = info.aliases[self.objpath[-1]]
        except KeyError:
            aliases = []
        if aliases:
            self.add_line(
                ('*Aliases:* ' if len(aliases) > 1 else '*Alias:* ')
                +  ', '.join([f'``{name}``' for name in aliases]),
                sourcename,
            )
            self.add_line('', sourcename)
        super().add_content(more_content, no_docstring)
