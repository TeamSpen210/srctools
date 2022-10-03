"""Modifies HTML.

This makes long signatures chop onto multiple lines.
"""
from sphinx.writers.html import HTMLTranslator, Element

MAX_CHARS = 40


class SrctoolsHTMLTranslator(HTMLTranslator):
    """Modifies the generated HTML.

    For long signatures, we chop them onto multiple lines.
    """
    multiline_param: bool = False

    def visit_desc_parameterlist(self, node: Element) -> None:
        """Figure out if the parameter list is long, and if so add a multiline class."""
        super().visit_desc_parameterlist(node)
        # Just use the length of the characters in the sig, it's close enough.
        # That doesn't include the return annotation, but that shouldn't be too much of a problem.
        self.multiline_param = len(node.astext()) > MAX_CHARS
        if self.multiline_param:
            self.body.append('<div class="multiline-sig">')
            self.param_separator = ',<br>'

    def depart_desc_parameterlist(self, node: Element) -> None:
        """Close the multiline class, if required."""
        if self.multiline_param:
            self.body.append('</div>')
        super().depart_desc_parameterlist(node)

    def depart_desc_parameter(self, node: Element) -> None:
        """Add trailing comma if multiline."""
        super().depart_desc_parameter(node)
        if not self.required_params_left and self.multiline_param:
            self.body.append(',<br>')
