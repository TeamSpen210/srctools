# Configuration file for the Sphinx documentation builder.
from typing import Optional
import sys
import enum

import sphinx.application

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

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


def setup(app: sphinx.application.Sphinx) -> None:
    """Perform modifications."""
    app.connect('autodoc-skip-member', autodoc_skip)


def autodoc_skip(app, what: str, name: str, obj: object, skip: Optional[bool], options):
    """Skip flags added by const.add_unknown()."""
    # print('Skip?', repr(what), repr(name), repr(obj), repr(skip))
    if what == 'class':
        if isinstance(obj, enum.Flag) and obj.name.isdigit():
            return True
