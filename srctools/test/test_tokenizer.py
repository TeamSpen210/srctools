from itertools import zip_longest
import pytest
import codecs

from srctools.test.test_property_parser import parse_test as prop_parse_test
from srctools.property_parser import KeyValError
from srctools.tokenizer import (
    Token,
    Tokenizer,
    C_Tokenizer, Py_Tokenizer,
    escape_text, _py_escape_text,
    TokenSyntaxError,
)

T = Token

# The correct result of parsing prop_parse_test.
# Either the token, or token + value (which must be correct).
prop_parse_tokens = [
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
    (T.STRING, "Block"),
    T.BRACE_OPEN, T.NEWLINE,
    (T.STRING, "Empty"), T.NEWLINE,
    T.BRACE_OPEN, T.NEWLINE,
    T.BRACE_CLOSE, T.BRACE_CLOSE, T.NEWLINE,
    (T.STRING, "Block"), T.NEWLINE,
    T.BRACE_OPEN, T.NEWLINE,
    (T.STRING, "bare"), T.NEWLINE,
    T.BRACE_OPEN, (T.STRING, "block"), (T.STRING, "he\tre"), T.NEWLINE,
    T.BRACE_CLOSE, T.NEWLINE,
    T.BRACE_CLOSE, T.NEWLINE,
    T.BRACE_CLOSE, T.NEWLINE,
    (T.STRING, "Root2"), T.NEWLINE,
    T.BRACE_OPEN, T.NEWLINE,
    (T.STRING, "Name with \" in it"), (T.STRING, "Value with \" inside"), T.NEWLINE,
    (T.STRING, "multiline"), (T.STRING, 'text\n\tcan continue\nfor many "lines" of\n  possibly indented\n\ntext'), T.NEWLINE,
    (T.STRING, "Escapes"), (T.STRING, '\t \n \\d'), T.NEWLINE,
    (T.STRING, "Oneliner"), T.BRACE_OPEN, (T.STRING, 'name'), (T.STRING, 'value'), T.BRACE_CLOSE, T.NEWLINE,
    T.BRACE_CLOSE, T.NEWLINE,
    (T.STRING, "CommentChecks"), T.NEWLINE,
    T.BRACE_OPEN, T.NEWLINE,
    (T.STRING, "after "), (T.STRING, "value"), T.NEWLINE,
    (T.STRING, "FlagBlocks"), (T.STRING, "This"), (T.PROP_FLAG, "test_disabled"), T.NEWLINE,
    (T.STRING, "Flag"), (T.STRING, "allowed"), (T.PROP_FLAG, "!test_disabled"), T.NEWLINE,
    (T.STRING, "FlagAllows"), (T.STRING, "This"), (T.PROP_FLAG, "test_enabled"), T.NEWLINE,
    (T.STRING, "Flag"), (T.STRING, "blocksthis"), (T.PROP_FLAG, "!test_enabled"), T.NEWLINE,
    T.NEWLINE,
    (T.STRING, "Replaced"), (T.STRING, "shouldbe"), T.NEWLINE,
    (T.STRING, "Replaced"), (T.STRING, "toreplace"), (T.PROP_FLAG, "test_enabled"), T.NEWLINE,
    (T.STRING, "Replaced"), (T.STRING, "alsothis"), (T.PROP_FLAG, "test_enabled"), T.NEWLINE,
    T.NEWLINE,
    (T.STRING, "Replaced"), (T.STRING, "shouldbe"), T.NEWLINE,
    (T.STRING, "Replaced"), (T.STRING, "toreplace"), (T.PROP_FLAG, "!test_disabled"), T.NEWLINE,
    (T.STRING, "Replaced"), (T.STRING, "alsothis"), (T.PROP_FLAG, "!test_disabled"), T.NEWLINE,
    T.NEWLINE,
    (T.STRING, "Replaced"), T.NEWLINE,
    T.BRACE_OPEN, T.NEWLINE,
    (T.STRING, "shouldbe"), (T.STRING, "replaced"), T.NEWLINE,
    (T.STRING, "prop2"), (T.STRING, "blah"), T.NEWLINE,
    T.BRACE_CLOSE, T.NEWLINE,
    (T.STRING, "Replaced"), (T.PROP_FLAG, "test_enabled"), T.NEWLINE,
    T.BRACE_OPEN, T.NEWLINE,
    (T.STRING, "lambda"), (T.STRING, "should"), T.NEWLINE,
    (T.STRING, "replace"), (T.STRING, "above"), T.NEWLINE,
    T.BRACE_CLOSE, T.NEWLINE,
    T.NEWLINE,
    (T.STRING, "Replaced"), T.NEWLINE,
    T.BRACE_OPEN, T.NEWLINE,
    (T.STRING, "shouldbe"), (T.STRING, "replaced"), T.NEWLINE,
    (T.STRING, "prop2"), (T.STRING, "blah"), T.NEWLINE,
    T.BRACE_CLOSE, T.NEWLINE,
    (T.STRING, "Replaced"), (T.PROP_FLAG, "!test_disabled"), T.NEWLINE,
    T.BRACE_OPEN, T.NEWLINE,
    (T.STRING, "lambda"), (T.STRING, "should"), T.NEWLINE,
    (T.STRING, "replace"), (T.STRING, "above"), T.NEWLINE,
    T.BRACE_CLOSE, T.NEWLINE,
    T.BRACE_CLOSE, T.NEWLINE,
]

if C_Tokenizer is not None:
    parms = [C_Tokenizer, Py_Tokenizer]
    parms_escape = [escape_text, _py_escape_text]
    ids = ['Cython', 'Python']
else:
    import srctools.tokenizer
    print('No _tokenizer! ' + str(vars(srctools.tokenizer)))
    parms = [Py_Tokenizer]
    parms_escape = [_py_escape_text]
    ids = ['Python']


@pytest.fixture(params=parms, ids=ids)
def py_c_token(request):
    """Run the test twice, for the Python and C versions."""
    yield request.param


@pytest.fixture(params=parms_escape, ids=ids)
def py_c_escape_text(request):
    """Run the test twice with the two escape_text() functions."""
    yield request.param

del parms, ids


def check_tokens(tokenizer, tokens):
    """Check the tokenizer produces the given tokens.

    The arguments are either (token, value) tuples or tokens.
    """
    # Don't show in pytest tracebacks.
    __tracebackhide__ = True

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
    Tokenizer = py_c_token

    tok = Tokenizer(prop_parse_test, '', string_bracket=True)
    check_tokens(tok, prop_parse_tokens)

    # Test a list of lines.
    test_list = prop_parse_test.splitlines(keepends=True)
    # Break this line up semi-randomly, to help test the chunk code.
    test_list[27:28] = [''] + list(test_list[27].partition('"')) + ['', '', '']
    # They should be the same text though!
    assert ''.join(test_list) == prop_parse_test, "Bad test code!"

    tok = Tokenizer(test_list, '', string_bracket=True)
    check_tokens(tok, prop_parse_tokens)

    # Test a special case - empty chunks at the end.
    test_list += ['', '', '']

    tok = Tokenizer(test_list, '', string_bracket=True)
    check_tokens(tok, prop_parse_tokens)


def test_pushback(py_c_token):
    """Test pushing back tokens."""
    Tokenizer = py_c_token
    tok = Tokenizer(prop_parse_test, '', string_bracket=True)
    tokens = []
    for i, (tok_type, tok_value) in enumerate(tok):
        if i % 3 == 0:
            tok.push_back(tok_type, tok_value)
        else:
            tokens.append((tok_type, tok_value))
    check_tokens(tokens, prop_parse_tokens)



def test_bom(py_c_token):
    """Test skipping a UTF8 BOM at the beginning."""
    Tokenizer = py_c_token

    bom = codecs.BOM_UTF8.decode('utf8')

    text = bom + '''\
"blah"
  {
  "tes__t " "2"
    }
'''.replace('__', bom)  # Check the BOM can be inside the contents.

    tokens = [
        (T.STRING, "blah"), T.NEWLINE,
        T.BRACE_OPEN, T.NEWLINE,
        (T.STRING, "tes" + bom + "t "), (T.STRING, "2"), T.NEWLINE,
        T.BRACE_CLOSE, T.NEWLINE,
    ]

    # Check without chunks.
    tok = Tokenizer(text, '')
    check_tokens(tok, tokens)

    # And with chunks.
    tok = Tokenizer(list(text), '')
    check_tokens(tok, tokens)


def test_constructor(py_c_token):
    """Test various argument syntax for the tokenizer."""
    Tokenizer = py_c_token

    Tokenizer('blah')
    Tokenizer('blah', None)
    Tokenizer('blah', '', TokenSyntaxError)
    Tokenizer('blah', '', KeyValError, True)
    Tokenizer('blah', error=KeyValError)
    Tokenizer(['blah', 'blah'], string_bracket=True)


def test_escape_text(py_c_escape_text):
    """Test the Python and C escape_text() functions."""
    assert py_c_escape_text("hello world") == "hello world"
    assert py_c_escape_text("\thello_world") == r"\thello_world"
    assert py_c_escape_text("\\thello_world") == r"\\thello_world"
    assert py_c_escape_text("\\ttest\nvalue\t\\r\t\n") == r"\\ttest\nvalue\t\\r\t\n"
    # BMP characters, and some multiplane chars.
    assert py_c_escape_text("\t╒═\\═╕\n") == r"\t╒═\\═╕\n"
    assert py_c_escape_text("\t♜♞\\🤐♝♛🥌♚♝\\\\♞\n♜") == r"\t♜♞\\🤐♝♛🥌♚♝\\\\♞\n♜"


def test_token_syntax_error():
    """Test the TokenSyntaxError class."""
    # There's no C version - if we're erroring, we don't care about
    # performance much.

    # This pretty much just needs to return the right repr() and str(),
    # and be an Exception.

    assert issubclass(TokenSyntaxError, Exception)

    err = TokenSyntaxError('test message', None, None)
    assert repr(err) == "TokenSyntaxError('test message', None, None)"
    assert err.mess == 'test message'
    assert err.file is None
    assert err.line_num is None
    assert str(err) == '''test message'''

    err = TokenSyntaxError('test message', 'a file', None)
    assert repr(err) == "TokenSyntaxError('test message', 'a file', None)"
    assert err.mess == 'test message'
    assert err.file == 'a file'
    assert err.line_num is None
    assert str(err) == '''test message
Error occurred with file "a file".'''

    err = TokenSyntaxError('test message', 'a file', 45)
    assert repr(err) == "TokenSyntaxError('test message', 'a file', 45)"
    assert err.mess == 'test message'
    assert err.file == 'a file'
    assert err.line_num == 45
    assert str(err) == '''test message
Error occurred on line 45, with file "a file".'''

    err = TokenSyntaxError('test message', None, 250)
    assert repr(err) == "TokenSyntaxError('test message', None, 250)"
    assert err.mess == 'test message'
    assert err.file is None
    assert err.line_num == 250
    assert str(err) == '''test message
Error occurred on line 250.'''
