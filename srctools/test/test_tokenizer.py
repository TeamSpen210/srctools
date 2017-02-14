from itertools import zip_longest
import pytest

from srctools.test.test_property_parser import parse_test as prop_parse_test
from srctools.property_parser import KeyValError
from srctools.tokenizer import C_Token, C_Tokenizer, Py_Token, Py_Tokenizer, TokenSyntaxError

if C_Token is not None and C_Tokenizer is not None:
    parms = [(C_Token, C_Tokenizer), (Py_Token, Py_Tokenizer)]
    ids = ['Cython', 'Python']
else:
    print('No _tokenizer!')
    parms = [(Py_Token, Py_Tokenizer)]
    ids = ['Python']


@pytest.fixture(params=parms, ids=ids)
def py_c_token(request):
    """Run the test twice, for the Python and C versions."""
    yield request.param

del parms, ids


def check_tokens(tokenizer, tokens):
    """Check the tokenizer produces the given tokens.

    The arguments are either (token, value) tuples or tokens.
    """
    sentinel = object()
    tokenizer_iter = iter(tokenizer)
    tok_test_iter = iter(tokens)
    for i, (token, comp_token) in enumerate(zip_longest(tokenizer_iter, tok_test_iter, fillvalue=sentinel), start=1):
        # Check if either is too short - we need zip_longest() for that.
        if token is sentinel:
            pytest.fail('{}: Tokenizer ended early - needed {}!'.format(
                i,
                [comp_token] + list(tok_test_iter),
            ))
        if comp_token is sentinel:
            pytest.fail('{}: Tokenizer had too many values - extra = {}!'.format(
                i,
                [token] + list(tokenizer_iter),
            ))
        assert len(token) == 2
        assert isinstance(token, tuple)
        if isinstance(comp_token, tuple):
            comp_type, comp_value = comp_token
            assert comp_type is token[0], "got {}, expected {} @ pos {}".format(token[0], comp_type, tokens[i-2: i+1])
            assert comp_value == token[1], "got {!r}, expected {!r} @ pos {}".format(token[1], comp_value, tokens[i-2: i+1])
        else:
            assert token[0] is comp_token, "got {}, expected {} @ pos {}".format(token[0], comp_token, tokens[i-2: i+1])


def test_prop_tokens(py_c_token):
    """Test the tokenizer returns the correct sequence of tokens for this test string."""
    T, Tokenizer = py_c_token
    tokens = [
        T.NEWLINE,
        T.NEWLINE,
        T.NEWLINE,
        T.NEWLINE,
        (T.STRING, "Root1"), T.NEWLINE,
        T.BRACE_OPEN, T.NEWLINE,
        T.NEWLINE,
        T.NEWLINE,
        (T.STRING, "Key"), (T.STRING, "Value"), T.NEWLINE,

        (T.STRING, "Extra"), (T.STRING, "Spaces"), T.NEWLINE,
        T.NEWLINE,
        (T.STRING, "Block"), T.NEWLINE,
        T.BRACE_OPEN, T.NEWLINE,
        (T.STRING, "Empty"), T.NEWLINE,
        T.BRACE_OPEN, T.NEWLINE,
        T.BRACE_CLOSE, T.NEWLINE,
        T.BRACE_CLOSE, T.NEWLINE,
        (T.STRING, "Block"), T.NEWLINE,
        T.BRACE_OPEN, T.NEWLINE,
        (T.STRING, "bare"), T.NEWLINE,
        T.BRACE_OPEN, T.NEWLINE,
        (T.STRING, "block"), (T.STRING, "he\tre"), T.NEWLINE,
        T.BRACE_CLOSE, T.NEWLINE,
        T.BRACE_CLOSE, T.NEWLINE,
        T.BRACE_CLOSE, T.NEWLINE,
        (T.STRING, "Root2"), T.NEWLINE,
        T.BRACE_OPEN, T.NEWLINE,
        (T.STRING, "Name with \" in it"), (T.STRING, "Value with \" inside"), T.NEWLINE,
        (T.STRING, "multiline"), (T.STRING, 'text\n\tcan continue\nfor many "lines" of\n  possibly indented\n\ntext'), T.NEWLINE,
        T.BRACE_CLOSE, T.NEWLINE,
        (T.STRING, "CommentChecks"), T.NEWLINE,
        T.BRACE_OPEN, T.NEWLINE,
        (T.STRING, "after "), (T.STRING, "value"), T.NEWLINE,
        (T.STRING, "FlagBlocks"), (T.STRING, "This"), (T.PROP_FLAG, "test_disabled"), T.NEWLINE,
        (T.STRING, "Flag"), (T.STRING, "allowed"), (T.PROP_FLAG, "!test_disabled"), T.NEWLINE,
        (T.STRING, "FlagAllows"), (T.STRING, "This"), (T.PROP_FLAG, "test_enabled"), T.NEWLINE,
        (T.STRING, "Flag"), (T.STRING, "blocksthis"), (T.PROP_FLAG, "!test_enabled"), T.NEWLINE,
        T.NEWLINE,
        T.BRACE_CLOSE, T.NEWLINE,
    ]

    tok = Tokenizer(prop_parse_test, '', string_bracket=True)
    check_tokens(tok, tokens)
    # Test a list of lines.
    tok = Tokenizer(prop_parse_test.splitlines(keepends=True), '', string_bracket=True)
    check_tokens(tok, tokens)


def test_constructor(py_c_token):
    """Test various argument syntax for the tokenizer."""
    Token, Tokenizer = py_c_token

    Tokenizer('blah')
    Tokenizer('blah', None)
    Tokenizer('blah', '', TokenSyntaxError)
    Tokenizer('blah', '', KeyValError, True)
    Tokenizer('blah', error=KeyValError)
    Tokenizer(['blah', 'blah'], string_bracket=True)