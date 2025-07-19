# Configuration file for the Sphinx documentation builder.
from pathlib import Path
import sys
import os

from sphinx.application import Sphinx


# Prevent Cython modules from importing, Python ones have all the type hints and docstrings.
sys.modules['srctools._tokenizer'] = None
sys.modules['srctools._math'] = None
sys.modules['srctools._cy_vtf_readwrite'] = None

# -- Project information -----------------------------------------------------

project = 'srctools'
copyright = '2025, TeamSpen210'
author = 'TeamSpen210'

needs_sphinx = '8.2'

# The full version, including alpha/beta/rc tags
# Use RTD version if available, otherwise check project.
try:
    release = os.environ['READTHEDOCS_VERSION_NAME']
except KeyError:
    import importlib.metadata
    release = importlib.metadata.version('srctools')


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinxcontrib_trio',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# `names` should lookup Python objects.
default_role = "py:obj"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

python_trailing_comma_in_multi_line_signatures = True
maximum_signature_line_length = 90

rst_prolog = '''
.. role:: pycode(code)
   :language: python
'''

# -- Intersphinx ------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# -- Extensions ----------

# Fully-qualified is way too verbose.
autodoc_typehints_format = 'short'
autodoc_member_order = 'bysource'

autodoc_type_aliases = {
    "ValidKVs": "ValidKVs",
}


extlinks = {
    'gh-user': ('https://github.com/%s', '@%s'),
    'src-issue': ('https://github.com/TeamSpen210/srctools/issues/%s', 'Issue #%s'),
    'src-pull': ('https://github.com/TeamSpen210/srctools/pulls/%s', 'PR #%s'),
    'ha-issue': ('https://github.com/TeamSpen210/HammerAddons/issues/%s', 'HammerAddons Issue #%s'),
    'bee-app-issue': ('https://github.com/BEEmod/BEE2.4/issues/%s', 'BEE Issue #%s'),
    'bee-pack-issue': ('https://github.com/BEEmod/BEE2-items/issues/%s', 'BEE Issue #%s'),
    # Pre-TF2 SDK commit.
    'sdk-2013': ('https://github.com/ValveSoftware/source-sdk-2013/blob/0d8dceea4310fde5706b3ce1c70609d72a38efdf/sp/src/%s', "%s"),
    'sdk-2013-mp': ('https://github.com/ValveSoftware/source-sdk-2013/blob/0d8dceea4310fde5706b3ce1c70609d72a38efdf/mp/src/%s', "%s"),
    # Current TF2 SDK commit.
    'sdk-2025': ('https://github.com/ValveSoftware/source-sdk-2013/commit/39f6dde8fbc238727c020d13b05ecadd31bda4c0', "%s"),
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

sys.path.append(str(Path(__file__).parent))
from enum_class import EnumDocumenter, EnumMemberDocumenter
from missing_refs import on_missing_reference


# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-process-signature
def autodoc_process_signature(
    app: Sphinx,
    what: str,
    name: str,
    obj: object,
    options: object,
    signature: str,
    return_annotation: str,
) -> tuple[str, str]:
    """Modify found signatures to fix various issues."""
    if signature is not None:
        signature = signature.replace("TypeAliasForwardRef('ValidKVs')", "ValidKVs")
        if 'EmptyMapping' not in name:
            # Make `= EmptyMapping` look like `= ...`, but not in the class' own docs.
            # That way users don't need to know about the singleton when reading other modules.
            signature = signature.replace('srctools.EmptyMapping', '...')
        if name in {'srctools.tokenizer.Tokenizer', 'srctools.tokenizer.IterTokenizer'}:
            # These constructors default to TokenSyntaxError, but the <> in the class repr breaks
            # the sig. Remove that.
            signature = signature.replace(
                "<class 'srctools.tokenizer.TokenSyntaxError'>",
                'TokenSyntaxError',
            )

    return signature, return_annotation


def setup(app: Sphinx) -> None:
    """Perform modifications."""
    app.add_autodocumenter(EnumDocumenter)
    app.add_autodocumenter(EnumMemberDocumenter)
    app.connect("autodoc-process-signature", autodoc_process_signature)
    app.connect('missing-reference', on_missing_reference, -10)
