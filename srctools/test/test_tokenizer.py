from itertools import zip_longest
import pytest
import codecs

from pytest import raises, mark, param

from srctools.test.test_property_parser import parse_test as prop_parse_test
from srctools.property_parser import KeyValError
from srctools.tokenizer import (
    Token,
    Tokenizer,
    Cy_Tokenizer, Py_Tokenizer,
    Cy_IterTokenizer, Py_IterTokenizer,
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
    (T.STRING, "Replaced"), (T.STRING, "shouldbe2"), T.NEWLINE,
    (T.STRING, "Replaced"), (T.STRING, "toreplace2"), (T.PROP_FLAG, "!test_disabled"), T.NEWLINE,
    (T.STRING, "Replaced"), (T.STRING, "alsothis2"), (T.PROP_FLAG, "!test_disabled"), T.NEWLINE,
    T.NEWLINE,
    (T.STRING, "Replaced"), T.NEWLINE,
    T.BRACE_OPEN, T.NEWLINE,
    (T.STRING, "shouldbe3"), (T.STRING, "replaced3"), T.NEWLINE,
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
    (T.STRING, "shouldbe4"), (T.STRING, "replaced4"), T.NEWLINE,
    (T.STRING, "prop2"), (T.STRING, "blah"), T.NEWLINE,
    T.BRACE_CLOSE, T.NEWLINE,
    (T.STRING, "Replaced"), (T.PROP_FLAG, "!test_disabled"), T.NEWLINE,
    T.BRACE_OPEN, T.NEWLINE,
    (T.STRING, "lambda2"), (T.STRING, "should2"), T.NEWLINE,
    (T.STRING, "replace2"), (T.STRING, "above2"), T.NEWLINE,
    T.BRACE_CLOSE, T.NEWLINE,
    T.BRACE_CLOSE, T.NEWLINE,
]

# Additional text not valid as a property.
noprop_parse_test = """
#ﬁmport test
#EXclßÀde value
#caseA\u0345\u03a3test
{ ]]{ }}}[[ {{] + = :: "test" + "ing" == vaLUE
"""

noprop_parse_tokens = [
    T.NEWLINE,
    (T.DIRECTIVE, "fimport"), (T.STRING, "test"), T.NEWLINE,
    (T.DIRECTIVE, "exclssàde"), (T.STRING, "value"), T.NEWLINE,
    (T.DIRECTIVE, "casea\u03b9\u03c3test"), T.NEWLINE,
    T.BRACE_OPEN, T.BRACK_CLOSE, T.BRACK_CLOSE, T.BRACE_OPEN, T.BRACE_CLOSE, T.BRACE_CLOSE, T.BRACE_CLOSE,
    T.BRACK_OPEN, T.BRACK_OPEN, T.BRACE_OPEN, T.BRACE_OPEN, T.BRACK_CLOSE, T.PLUS,
    T.EQUALS, T.COLON, T.COLON, (T.STRING, "test"), T.PLUS, (T.STRING, "ing"),
    T.EQUALS, T.EQUALS, (T.STRING, "vaLUE"), T.NEWLINE
]

ids = ['Cython', 'Python']
if Cy_Tokenizer is not Py_Tokenizer:
    parms = [Cy_Tokenizer, Py_Tokenizer]
    parms_iter = [Cy_IterTokenizer, Py_IterTokenizer]
    parms_escape = [escape_text, _py_escape_text]
else:
    parms = [param(Py_Tokenizer, marks=mark.skip), Py_Tokenizer]
    parms_iter = [param(Py_IterTokenizer, marks=mark.skip('No Cy_IterTokenizer')), Py_IterTokenizer]
    parms_escape = [param(_py_escape_text, marks=mark.skip), _py_escape_text]


@pytest.fixture(params=parms, ids=ids)
def py_c_token(request):
    """Run the test twice, for the Python and C versions."""
    if request.param is AssertionError:
        raise AssertionError('No Cy_Tokenizer!')
    yield request.param


@pytest.fixture(params=parms_escape, ids=ids)
def py_c_escape_text(request):
    """Run the test twice with the two escape_text() functions."""
    if request.param is AssertionError:
        raise AssertionError('No C escape_text()!')
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
            assert comp_type is token[0], "got {}, expected {} @ pos {}={}".format(token[0], comp_type, i, tokens[i-2: i+1])
            assert comp_value == token[1], "got {!r}, expected {!r} @ pos {}={}".format(token[1], comp_value, i, tokens[i-2: i+1])
        else:
            assert token[0] is comp_token, "got {}, expected {} @ pos {}={}".format(token[0], comp_token, i, tokens[i-2: i+1])


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


def test_nonprop_tokens(py_c_token):
    """Test the tokenizer returns the correct sequence of tokens for non-Property strings."""
    Tokenizer = py_c_token

    tok = Tokenizer(noprop_parse_test, '')
    check_tokens(tok, noprop_parse_tokens)

    # Test a list of lines.
    test_list = noprop_parse_test.splitlines(keepends=True)

    tok = Tokenizer(test_list, '')
    check_tokens(tok, noprop_parse_tokens)

    # Test a special case - empty chunks at the end.
    test_list += ['', '', '']

    tok = Tokenizer(test_list, '')
    check_tokens(tok, noprop_parse_tokens)


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

    tok = Tokenizer(noprop_parse_test, '')
    tokens = []
    for i, (tok_type, tok_value) in enumerate(tok):
        if i % 3 == 0:
            tok.push_back(tok_type, tok_value)
        else:
            tokens.append((tok_type, tok_value))
    check_tokens(tokens, noprop_parse_tokens)


def test_call_next(py_c_token):
    """Test that tok() functions, and it can be mixed with iteration."""
    Tokenizer = py_c_token
    tok = Tokenizer('''{ "test" } "test" { + } ''', 'file')

    tok_type, tok_value = tok()
    assert tok_type is Token.BRACE_OPEN and tok_value == '{'

    it1 = iter(tok)

    assert next(it1) == (Token.STRING, "test")
    assert tok() == (Token.BRACE_CLOSE, '}')
    assert next(it1) == (Token.STRING, "test")
    assert next(it1) == (Token.BRACE_OPEN, '{')
    assert tok() == (Token.PLUS, '+')
    # Another iterator doesn't restart.
    assert next(iter(tok)) == (Token.BRACE_CLOSE, '}')
    assert tok() == (Token.EOF, '')

    with pytest.raises(StopIteration):
        next(it1)


def test_star_comments(py_c_token):
    """Test disallowing /* */ comments."""
    Tokenizer = py_c_token

    text = '''\
    "blah"
        {
        "a" "b"
    /*
        "c" "d"
        }
    "second"
        {
    */
        }
    '''

    with pytest.raises(TokenSyntaxError):
        # Default = false
        for tok, tok_value in py_c_token(text):
            pass

    with pytest.raises(TokenSyntaxError):
        for tok, tok_value in py_c_token(text, allow_star_comments=False):
            pass


    check_tokens(py_c_token(text, allow_star_comments=True), [
        (Token.STRING, "blah"), Token.NEWLINE,
        Token.BRACE_OPEN, Token.NEWLINE,
        (Token.STRING, "a"), (Token.STRING, "b"), Token.NEWLINE,
        Token.NEWLINE,
        Token.BRACE_CLOSE, Token.NEWLINE,
    ])

    # Test with one string per chunk:
    for tok, tok_value in py_c_token(list(text), allow_star_comments=True):
        pass

    # Check line number is correct.
    tokenizer = py_c_token(text, allow_star_comments=True)
    for tok, tok_value in tokenizer:
        if tok is Token.BRACE_CLOSE:
            assert 10 == tokenizer.line_num

    # Check unterminated comments are invalid.
    with pytest.raises(TokenSyntaxError):
        for tok, tok_value in py_c_token(text.replace('*/', ''), allow_star_comments=True):
            pass

    # Test some edge cases with multiple asterisks:
    for tok, tok_value in py_c_token('"blah"\n/**/', allow_star_comments=True):
        pass

    for tok, tok_value in py_c_token('"blah"\n/*\n **/', allow_star_comments=True):
        pass


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


def test_obj_config(py_c_token):
    """Test getting and setting configuration attributes."""
    Tokenizer = py_c_token

    assert Tokenizer('').string_bracket is False
    assert Tokenizer('').allow_escapes is True

    assert Tokenizer('', string_bracket=1).string_bracket is True
    assert Tokenizer('', string_bracket=True).string_bracket is True
    assert Tokenizer('', string_bracket=False).string_bracket is False
    assert Tokenizer('', allow_escapes=[]).allow_escapes is False
    assert Tokenizer('', allow_escapes=True).allow_escapes is True
    assert Tokenizer('', allow_escapes=False).allow_escapes is False

    tok = Tokenizer('')
    assert tok.string_bracket is False
    tok.string_bracket = True
    assert tok.string_bracket is True
    tok.string_bracket = False
    assert tok.string_bracket is False

    tok = Tokenizer('')
    assert tok.allow_escapes is True
    tok.allow_escapes = False
    assert tok.allow_escapes is False
    tok.allow_escapes = True
    assert tok.allow_escapes is True


def test_escape_text(py_c_escape_text):
    """Test the Python and C escape_text() functions."""
    assert py_c_escape_text("hello world") == "hello world"
    assert py_c_escape_text("\thello_world") == r"\thello_world"
    assert py_c_escape_text("\\thello_world") == r"\\thello_world"
    assert py_c_escape_text("\\ttest\nvalue\t\\r\t\n") == r"\\ttest\nvalue\t\\r\t\n"
    # BMP characters, and some multiplane chars.
    assert py_c_escape_text("\t╒═\\═╕\n") == r"\t╒═\\═╕\n"
    assert py_c_escape_text("\t♜♞\\🤐♝♛🥌♚♝\\\\♞\n♜") == r"\t♜♞\\🤐♝♛🥌♚♝\\\\♞\n♜"


def test_brackets(py_c_token):
    """Test the [] handling produces the right results."""
    check_tokens(py_c_token('"blah" [ !!text45() ]', string_bracket=True), [
        (Token.STRING, "blah"),
        (Token.PROP_FLAG, " !!text45() ")
    ])

    check_tokens(py_c_token('"blah" [ !!text45() ]', string_bracket=False), [
        (Token.STRING, "blah"),
        Token.BRACK_OPEN,
        (Token.STRING, "!!text45"),
        (Token.PAREN_ARGS, ''),
        Token.BRACK_CLOSE,
    ])

    # Without, we don't care about brackets.
    check_tokens(py_c_token('[ unclosed {}', string_bracket=False), [
        Token.BRACK_OPEN,
        (Token.STRING, 'unclosed'),
        Token.BRACE_OPEN,
        Token.BRACE_CLOSE,
    ])

    # Corner case, check EOF while reading the string.
    check_tokens(py_c_token('[ unclosed', string_bracket=False), [
        Token.BRACK_OPEN,
        (Token.STRING, 'unclosed'),
    ])

    # Without, we don't care about brackets.
    check_tokens(py_c_token('unopened ]', string_bracket=False), [
        (Token.STRING, 'unopened'),
        Token.BRACK_CLOSE,
    ])

def test_invalid_bracket(py_c_token):
    """Test detecting various invalid combinations of [] brackets."""
    with raises(TokenSyntaxError):
        for tok, tok_value in py_c_token('[ unclosed', string_bracket=True):
            pass

    with raises(TokenSyntaxError):
        for tok, tok_value in py_c_token('unopened ]', string_bracket=True):
            pass

    with raises(TokenSyntaxError):
        for tok, tok_value in py_c_token('[ok] bad ]', string_bracket=True):
            pass

    with raises(TokenSyntaxError):
        for tok, tok_value in py_c_token('[ no [ nesting ] ]', string_bracket=True):
            pass

def test_invalid_paren(py_c_token):
    with raises(TokenSyntaxError):
        for tok, tok_value in py_c_token('( unclosed', string_bracket=True):
            pass

    with raises(TokenSyntaxError):
        for tok, tok_value in py_c_token('unopened )', string_bracket=True):
            pass

    with raises(TokenSyntaxError):
        for tok, tok_value in py_c_token('(ok) bad )', string_bracket=True):
            pass

    with raises(TokenSyntaxError):
        for tok, tok_value in py_c_token('( no ( nesting ) )', string_bracket=True):
            pass


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


def test_unicode_error_wrapping(py_c_token):
    """Test that Unicode errors are wrapped into TokenSyntaxError."""
    def raises_unicode():
        yield "line of_"
        yield "text\n"
        raise UnicodeDecodeError('utf8', bytes(100), 1, 2, 'reason')

    tok = py_c_token(raises_unicode())
    assert tok() == (Token.STRING, "line")
    assert tok() == (Token.STRING, "of_text")
    with pytest.raises(TokenSyntaxError) as exc_info:
        list(tok)
    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


def test_early_binary_arg(py_c_token):
    """Test that passing bytes values is caught before looping."""
    with pytest.raises(TypeError):
        py_c_token(b'test')


def test_bare_string(py_c_token):
    """Test the bare name option."""
    Tokenizer = py_c_token

    text = r"""
    "quoted" unquoted \\\\ "\\\\" "quoteagain"
    """
    check_tokens(Tokenizer(text, mark_bare_strings=False), [
        T.NEWLINE,
        (T.STRING, "quoted"),
        (T.STRING, "unquoted"),
        (T.STRING, r"\\\\"),
        (T.STRING, r"\\"),
        (T.STRING, "quoteagain"),
        T.NEWLINE,
    ])
    check_tokens(Tokenizer(text, mark_bare_strings=True), [
        T.NEWLINE,
        (T.STRING, "quoted"),
        (T.BARE_STRING, "unquoted"),
        (T.BARE_STRING, r"\\\\"),
        (T.STRING, r"\\"),
        (T.STRING, "quoteagain"),
        T.NEWLINE,
    ])
