from typing import Any, Callable, Iterable, Iterator, Tuple, Type, Union
from itertools import tee, zip_longest
import codecs
import platform

import pytest

from srctools.keyvalues import KeyValError
# noinspection PyProtectedMember
from srctools.tokenizer import (
    _OPERATOR_VALS as TOK_VALS, BaseTokenizer, Cy_BaseTokenizer, Cy_Tokenizer,
    Py_BaseTokenizer, Py_Tokenizer, Token, Tokenizer, TokenSyntaxError, _py_escape_text,
    escape_text,
)
from test_keyvalues import parse_test as prop_parse_test


T = Token
IS_CPYTHON = platform.python_implementation() == 'CPython'

# See https://github.com/ValveSoftware/source-sdk-2013/blob/0d8dceea4310fde5706b3ce1c70609d72a38efdf/sp/src/tier1/utlbuffer.cpp#L57-L69
ESCAPE_CHARS = "\n \t \v \b \r \f \a \\ ? \' \""
ESCAPE_ENCODED = r"\n \t \v \b \r \f \a \\ \? \' " + r'\"'

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
    (T.STRING, "multi+line"), (T.STRING, 'text\n\tcan continue\nfor many "lines" of\n  possibly indented\n\ntext'), T.NEWLINE,
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
    (T.STRING, "Replaced"), (T.STRING, "to+replace"), (T.PROP_FLAG, "test_enabled"), T.NEWLINE,
    (T.STRING, "Replaced"), (T.STRING, "also+this"), (T.PROP_FLAG, "test_enabled"), T.NEWLINE,
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
    (T.STRING, "otherval"), (T.STRING, "blah"), T.NEWLINE,
    (T.STRING, "shouldnotreplace"), (T.PROP_FLAG, "test_enabled"), T.NEWLINE,
    T.BRACE_OPEN, T.NEWLINE,
    (T.STRING, "key"), (T.STRING, "value1"), T.NEWLINE,
    (T.STRING, "key"), (T.STRING, "value2"), T.NEWLINE,
    T.BRACE_CLOSE, T.NEWLINE,
    (T.STRING, "skipped"), (T.PROP_FLAG, "test_disabled"), T.NEWLINE,
    T.BRACE_OPEN, T.NEWLINE,
    (T.STRING, "ignore"), (T.STRING, "value"), T.NEWLINE,
    T.BRACE_CLOSE, T.NEWLINE,
    T.BRACE_CLOSE, T.NEWLINE,
]

# Additional text not valid as a property.
noprop_parse_test = """
#letter_abcdefghijklmnopqrstuvwxyz_ABCDEFGHIJKLMNOPQRSTUVWXYZ
"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f"
#ﬁmport test
#EXclßÀde value\r
#caseA\u0345\u03a3test
{ ]]{ }}}[[ {{] = "test" "ing" == vaLUE= 4 6
"""

noprop_parse_tokens = [
    T.NEWLINE,
    (T.DIRECTIVE, "letter_abcdefghijklmnopqrstuvwxyz_abcdefghijklmnopqrstuvwxyz"), T.NEWLINE,
    # Test all control characters are valid.
    (T.STRING, "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f"), T.NEWLINE,
    (T.DIRECTIVE, "fimport"), (T.STRING, "test"), T.NEWLINE,
    (T.DIRECTIVE, "exclssàde"), (T.STRING, "value"), T.NEWLINE,
    (T.DIRECTIVE, "casea\u03b9\u03c3test"), T.NEWLINE,
    T.BRACE_OPEN, T.BRACK_CLOSE, T.BRACK_CLOSE, T.BRACE_OPEN, T.BRACE_CLOSE, T.BRACE_CLOSE, T.BRACE_CLOSE,
    T.BRACK_OPEN, T.BRACK_OPEN, T.BRACE_OPEN, T.BRACE_OPEN, T.BRACK_CLOSE,
    T.EQUALS, (T.STRING, "test"), (T.STRING, "ing"),
    T.EQUALS, T.EQUALS, (T.STRING, "vaLUE"), T.EQUALS, (T.STRING, "4"), (T.STRING, "6"), T.NEWLINE
]


if Cy_Tokenizer is not Py_Tokenizer:
    parms = [Cy_Tokenizer, Py_Tokenizer]
    ids = ['Cython', 'Python']
else:
    import srctools.tokenizer
    print('No _tokenizer! ' + str(vars(srctools.tokenizer)))
    parms = [Py_Tokenizer]
    ids = ['Python']


@pytest.fixture(params=parms, ids=ids)
def py_c_token(request: Any) -> Type[Tokenizer]:
    """Run the test twice, for the Python and C versions."""
    return request.param

del parms, ids


def check_tokens(
    tokenizer: Iterable[Tuple[Token, str]],  # Iterable so a list can be passed to check.
    tokens: Iterable[Union[Token, Tuple[Token, str]]],
) -> None:
    """Check the tokenizer produces the given tokens.

    The arguments are either (token, value) tuples or tokens.
    """
    # Don't show in pytest tracebacks.
    __tracebackhide__ = True

    sentinel = object()
    tokenizer_iter, tokenizer_backup = tee(tokenizer, 2)
    tok_test_iter = iter(tokens)
    for i, (token, comp_token) in enumerate(zip_longest(tokenizer_iter, tok_test_iter, fillvalue=sentinel), start=1):
        # Check if either is too short - we need zip_longest() for that.
        if token is sentinel:
            pytest.fail(
                f'{i}: Tokenizer ended early - needed {[comp_token, *tok_test_iter]}, '
                f'got {list(tokenizer_backup)}!'
            )
        if comp_token is sentinel:
            pytest.fail(f'{i}: Tokenizer had too many values - extra = {[token, *tokenizer_iter]}!')
        assert len(token) == 2
        assert isinstance(token, tuple)
        if isinstance(comp_token, tuple):
            comp_type, comp_value = comp_token
            # Compare together, we want identity for the enum. Ignore split-assertion warning,
            # we print out the values ourselves so PyTest assertion rewriting is not necessary.
            assert token[0] is comp_type and token[1] == comp_value, (  # noqa: PT018
                f"got {token[0]}({token[1]!r}), "
                f"expected {comp_type}({comp_value!r}) @ pos {i}={tokens[i - 2: i + 1]}"
            )
        else:
            assert token[0] is comp_token, (
                f"got {token[0]}({token[1]!r}), "
                f"expected {comp_token} @ pos {i}={tokens[i - 2: i + 1]}"
            )


def test_prop_tokens(py_c_token: Type[Tokenizer]) -> None:
    """Test the tokenizer returns the correct sequence of tokens for this test string."""
    Tokenizer = py_c_token

    tok = Tokenizer(prop_parse_test, '', string_bracket=True)
    check_tokens(tok, prop_parse_tokens)

    # Test a list of lines.
    test_list = prop_parse_test.splitlines(keepends=True)
    # Break this line up semi-randomly, to help test the chunk code.
    test_list[27:28] = ['', *test_list[27].partition('"'), '', '', '']
    # They should be the same text though!
    assert ''.join(test_list) == prop_parse_test, "Bad test code!"

    # Test an iterator.
    tok = Tokenizer(iter(prop_parse_test.splitlines(keepends=True)), '', string_bracket=True)
    check_tokens(tok, prop_parse_tokens)

    tok = Tokenizer(test_list, '', string_bracket=True)
    check_tokens(tok, prop_parse_tokens)

    # Test a special case - empty chunks at the end.
    test_list += ['', '', '']

    tok = Tokenizer(test_list, '', string_bracket=True)
    check_tokens(tok, prop_parse_tokens)


def test_nonprop_tokens(py_c_token: Type[Tokenizer]) -> None:
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


def test_pushback(py_c_token: Type[Tokenizer]) -> None:
    """Test pushing back tokens."""
    Tokenizer = py_c_token
    tok = Tokenizer(prop_parse_test, '', string_bracket=True)
    tokens = []
    for i, (tok_type, tok_value) in enumerate(tok):
        if i % 5 in (0, 3, 4):
            tok.push_back(tok_type, tok_value)
        else:
            tokens.append((tok_type, tok_value))
    check_tokens(tokens, prop_parse_tokens)

    tok = Tokenizer(noprop_parse_test, '')
    tokens = []
    for i, (tok_type, tok_value) in enumerate(tok):
        if i % 5 in (0, 3, 4):
            tok.push_back(tok_type, tok_value)
        else:
            tokens.append((tok_type, tok_value))
    check_tokens(tokens, noprop_parse_tokens)


@pytest.mark.parametrize('token, val', [
    (Token.EOF, ''),
    (Token.NEWLINE, '\n'),
    (Token.BRACE_OPEN, '{'),
    (Token.BRACE_CLOSE, '}'),
    (Token.BRACK_OPEN, '['),
    (Token.BRACK_CLOSE, ']'),
    (Token.COLON, ':'),
    (Token.EQUALS, '='),
    (Token.PLUS, '+'),
    (Token.COMMA, ','),
])
def test_pushback_opvalues(py_c_token: Type[Tokenizer], token: Token, val: str) -> None:
    """Test the operator tokens pushback the correct fixed value."""
    tok: Tokenizer = py_c_token(['test data'], string_bracket=False)
    tok.push_back(token, val)
    assert tok() == (token, val)

    tok.push_back(Token.STRING, 'push')
    tok.push_back(Token.DIRECTIVE, 'second')
    assert tok() == (Token.DIRECTIVE, 'second')
    assert tok() == (Token.STRING, 'push')

    # Value is ignored for these token types.
    tok.push_back(token, 'another_val')
    assert tok() == (token, val)


def test_call_next(py_c_token: Type[Tokenizer]) -> None:
    """Test that tok() functions, and it can be mixed with iteration."""
    tok: Tokenizer = py_c_token('''{ "test" } "test" { = } ''', 'file')

    tok_type, tok_value = tok_tup = tok()
    assert tok_type is Token.BRACE_OPEN, tok_tup
    assert tok_value == '{', tok_tup

    it1 = iter(tok)

    assert next(it1) == (Token.STRING, "test")
    assert tok() == (Token.BRACE_CLOSE, '}')
    assert next(it1) == (Token.STRING, "test")
    assert next(it1) == (Token.BRACE_OPEN, '{')
    assert tok() == (Token.EQUALS, '=')
    # Another iterator doesn't restart.
    assert next(iter(tok)) == (Token.BRACE_CLOSE, '}')
    assert tok() == (Token.EOF, '')

    with pytest.raises(StopIteration):
        next(it1)


def test_star_comments(py_c_token: Type[Tokenizer]) -> None:
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
        for _ in Tokenizer(text):
            pass

    with pytest.raises(TokenSyntaxError):
        for _ in Tokenizer(text, allow_star_comments=False):
            pass

    check_tokens(Tokenizer(text, allow_star_comments=True), [
        (Token.STRING, "blah"), Token.NEWLINE,
        Token.BRACE_OPEN, Token.NEWLINE,
        (Token.STRING, "a"), (Token.STRING, "b"), Token.NEWLINE,
        Token.NEWLINE,
        Token.BRACE_CLOSE, Token.NEWLINE,
    ])

    check_tokens(Tokenizer(text, allow_star_comments=True, preserve_comments=True), [
        (Token.STRING, "blah"), Token.NEWLINE,
        Token.BRACE_OPEN, Token.NEWLINE,
        (Token.STRING, "a"), (Token.STRING, "b"), Token.NEWLINE,
        (Token.COMMENT, '''
        "c" "d"
        }
    "second"
        {
    '''), Token.NEWLINE,
        Token.BRACE_CLOSE, Token.NEWLINE,
    ])

    # Test with one string per chunk:
    for _ in Tokenizer(list(text), allow_star_comments=True):
        pass

    # Check line number is correct.
    tokenizer = Tokenizer(text, allow_star_comments=True)
    for tok, tok_value in tokenizer:
        if tok is Token.BRACE_CLOSE:
            assert 10 == tokenizer.line_num

    # Check unterminated comments are invalid.
    with pytest.raises(TokenSyntaxError):
        for _ in Tokenizer(text.replace('*/', ''), allow_star_comments=True):
            pass

    # Test some edge cases with multiple asterisks:
    for _ in Tokenizer('"blah"\n/**/', allow_star_comments=True):
        pass

    for _ in Tokenizer('"blah"\n/*\n **/', allow_star_comments=True):
        pass


def test_bom(py_c_token: Type[Tokenizer]) -> None:
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

    # Test edge cases - characters with the first 1 or 2 bytes of the 3-byte
    # BOM should be kept intact.\xef\xbb\xbf
    matches_1 = b'\xef\xbe\xae'.decode('utf8') + '_test'
    matches_2 = b'\xef\xbb\xae'.decode('utf8') + '_test'
    assert matches_1.encode('utf8')[:1] == codecs.BOM_UTF8[:1]
    assert matches_2.encode('utf8')[:2] == codecs.BOM_UTF8[:2]

    assert next(Tokenizer(bom + 'test')) == (Token.STRING, 'test')
    assert next(Tokenizer(matches_1)) == (Token.STRING, matches_1)
    assert next(Tokenizer(matches_2)) == (Token.STRING, matches_2)

    # Test strings that are less than 3 bytes long.
    check_tokens(Tokenizer(['e']), [(Token.STRING, 'e')])
    check_tokens(Tokenizer(['e', 'x']), [(Token.STRING, 'ex')])
    check_tokens(Tokenizer(['e', ' ', 'f']), [(Token.STRING, 'e'), (Token.STRING, 'f')])


def test_universal_newlines(py_c_token: Type[Tokenizer]) -> None:
    r"""Test that \r characters are replaced by \n, even in direct strings."""
    tokens = [
        (Token.STRING, 'Line'),
        (Token.STRING, 'one\ntwo'),
        Token.NEWLINE,
        (Token.STRING, 'Line\nTwo'),
        Token.NEWLINE,
        Token.COMMA,
        Token.NEWLINE, Token.NEWLINE,
        Token.EQUALS,
        (Token.STRING, 'multi\nline'),
    ]
    text = ['Line "one\ntwo" \r', '"Line\rTwo"', '\r', '\n,', '\r\r', '=', '"multi\r\nline"']
    check_tokens(py_c_token(text), tokens)
    check_tokens(py_c_token(''.join(text)), tokens)


def test_token_has_value() -> None:
    """Check the result of the .has_value property."""
    assert not Token.EOF.has_value
    assert not Token.NEWLINE.has_value
    assert not Token.BRACE_OPEN.has_value
    assert not Token.BRACE_CLOSE.has_value
    assert not Token.BRACK_OPEN.has_value
    assert not Token.BRACK_CLOSE.has_value
    assert not Token.COLON.has_value
    assert not Token.EQUALS.has_value
    assert not Token.PLUS.has_value
    assert not Token.COMMA.has_value

    assert Token.STRING.has_value
    assert Token.PAREN_ARGS.has_value
    assert Token.DIRECTIVE.has_value
    assert Token.PROP_FLAG.has_value
    assert Token.COMMENT.has_value


def test_constructor(py_c_token: Type[Tokenizer]) -> None:
    """Test various argument syntax for the tokenizer."""
    Tokenizer = py_c_token

    Tokenizer('blah')
    Tokenizer('blah', None)
    Tokenizer('blah', '', TokenSyntaxError)
    Tokenizer('blah', error=KeyValError)
    Tokenizer(['blah', 'blah'], string_bracket=True)


def test_tok_filename(py_c_token: Type[Tokenizer]) -> None:
    """Test that objects other than a direct string can be passed as filename."""
    Tokenizer = py_c_token

    class AFilePath:
        """Test path object support."""
        def __fspath__(self) -> str:
            return "path/to/file.vdf"

    class SubStr(str):
        """Subclasses should be accepted."""

    assert Tokenizer('blah', AFilePath()).filename == 'path/to/file.vdf'
    tok = Tokenizer('blah', SubStr('test/path.txt'))
    assert tok.filename == 'test/path.txt'
    assert isinstance(tok.filename, str)

    assert Tokenizer('file', b'binary/filename\xE3\x00.txt').filename == 'binary/filename\\xe3\\x00.txt'


@pytest.mark.parametrize('parm, default', [
    ('string_bracket', False),
    ('allow_escapes', True),
    ('allow_star_comments', False),
    ('preserve_comments', False),
    ('colon_operator', False),
    ('plus_operator', False),
])
def test_obj_config(py_c_token: Type[Tokenizer], parm: str, default: bool) -> None:
    """Test getting and setting configuration attributes."""
    Tokenizer = py_c_token

    assert getattr(Tokenizer(''), parm) is default
    assert getattr(Tokenizer('', **{parm: True}), parm) is True
    assert getattr(Tokenizer('', **{parm: False}), parm) is False
    assert getattr(Tokenizer('', **{parm: 1}), parm) is True
    assert getattr(Tokenizer('', **{parm: []}), parm) is False

    tok = Tokenizer('')
    setattr(tok, parm, False)
    assert getattr(tok, parm) is False

    setattr(tok, parm, True)
    assert getattr(tok, parm) is True

    # Don't check it does force setting them to bools, Python version doesn't
    # need to do that.
    setattr(tok, parm, '')
    assert not getattr(tok, parm)

    setattr(tok, parm, {1, 2, 3})
    assert getattr(tok, parm)


@pytest.mark.parametrize('inp, out', [
    ('', ''),
    ("hello world", "hello world"),
    ("\thello_world", r"\thello_world"),
    ("\\thello_world", r"\\thello_world"),
    ("\\ttest\nvalue\t\\r\t\n", r"\\ttest\nvalue\t\\r\t\n"),
    (ESCAPE_CHARS, ESCAPE_ENCODED),
    # BMP characters, and some multiplane chars.
    ('test: ╒══╕', r'test: ╒══╕'),
    ("♜♞🤐♝♛🥌 chess: ♚♝♞♜", "♜♞🤐♝♛🥌 chess: ♚♝♞♜"),
    ('\t"╒═\\═╕"\n', r'\t\"╒═\\═╕\"\n'),
    ("\t♜♞\\🤐♝♛🥌♚♝\\\\♞\n♜", r"\t♜♞\\🤐♝♛🥌♚♝\\\\♞\n♜"),
])
@pytest.mark.parametrize('func', [_py_escape_text, escape_text], ids=['Py', 'Cy'])
def test_escape_text(inp: str, out: str, func: Callable[[str], str]) -> None:
    """Test the Python and C escape_text() functions."""
    assert func(inp) == out
    # If the same it should reuse the string.
    # But don't check on PyPy etc, may have primitive optimisations.
    if inp == out and IS_CPYTHON:
        assert func(inp) is inp


def test_brackets(py_c_token: Type[Tokenizer]) -> None:
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


@pytest.mark.parametrize('op, tok, option', [
    (':', Token.COLON, 'colon_operator'),
    ('+', Token.PLUS, 'plus_operator'),
], ids=['colon', 'plus'])
def test_conditional_op(py_c_token: Type[Tokenizer], op: str, option: str, tok: Token) -> None:
    """Test : and + can be detected as a string or operator depending on the option."""
    disabled = {option: False}
    enabled = {option: True}

    # Explicit string, unaffected.
    check_tokens(py_c_token(f'"test{op}call"', **disabled), [
        (Token.STRING, f'test{op}call'),
    ])
    check_tokens(py_c_token(f'"test{op}call"', **enabled), [
        (Token.STRING, f'test{op}call'),
    ])

    # Applies to bare strings, also note another char after.
    check_tokens(py_c_token('test%call%{}'.replace('%', op), **disabled), [
        (Token.STRING, f'test{op}call{op}'),
        Token.BRACE_OPEN, Token.BRACE_CLOSE,
    ])
    check_tokens(py_c_token('test%call%{}'.replace('%', op), **enabled), [
        (Token.STRING, 'test'),
        tok,
        (Token.STRING, 'call'),
        tok,
        Token.BRACE_OPEN, Token.BRACE_CLOSE,
    ])

    # Test the string starting with the character.
    check_tokens(py_c_token('{%test%call}'.replace('%', op), **disabled), [
        Token.BRACE_OPEN,
        (Token.STRING, f'{op}test{op}call'),
        Token.BRACE_CLOSE,
    ])
    check_tokens(py_c_token('{%test%call}'.replace('%', op), **enabled), [
        Token.BRACE_OPEN, tok,
        (Token.STRING, 'test'), tok,
        (Token.STRING, 'call'), Token.BRACE_CLOSE,
    ])

    # Test directives
    check_tokens(py_c_token(f'\n#word{op}Two', **disabled), [
        Token.NEWLINE, (Token.DIRECTIVE, f'word{op}two'),
    ])
    check_tokens(py_c_token(f'\n#word{op}Two', **enabled), [
        Token.NEWLINE, (Token.DIRECTIVE, 'word'),
        tok, (Token.STRING, 'Two'),
    ])


def test_invalid_bracket(py_c_token: Type[Tokenizer]) -> None:
    """Test detecting various invalid combinations of [] brackets."""
    with pytest.raises(TokenSyntaxError):
        for tok, tok_value in py_c_token('[ unclosed', string_bracket=True):
            pass

    with pytest.raises(TokenSyntaxError):
        for tok, tok_value in py_c_token('unopened ]', string_bracket=True):
            pass

    with pytest.raises(TokenSyntaxError):
        for tok, tok_value in py_c_token('[ok] bad ]', string_bracket=True):
            pass

    with pytest.raises(TokenSyntaxError):
        for tok, tok_value in py_c_token('[ no [ nesting ] ]', string_bracket=True):
            pass


def test_invalid_paren(py_c_token: Type[Tokenizer]) -> None:
    with pytest.raises(TokenSyntaxError):
        for tok, tok_value in py_c_token('( unclosed', string_bracket=True):
            pass

    with pytest.raises(TokenSyntaxError):
        for tok, tok_value in py_c_token('unopened )', string_bracket=True):
            pass

    with pytest.raises(TokenSyntaxError):
        for tok, tok_value in py_c_token('(ok) bad )', string_bracket=True):
            pass

    with pytest.raises(TokenSyntaxError):
        for tok, tok_value in py_c_token('( no ( nesting ) )', string_bracket=True):
            pass


def test_allow_escapes(py_c_token: Type[Tokenizer]) -> None:
    """Test parsing with and without escapes enabled."""
    check_tokens(py_c_token(r'{ "string\n" "tab\ted" }', allow_escapes=False), [
        Token.BRACE_OPEN,
        (Token.STRING, r"string\n"),
        (Token.STRING, r"tab\ted"),
        Token.BRACE_CLOSE,
    ])
    check_tokens(py_c_token(
        f'{{ "string\\n" "all escapes: {ESCAPE_ENCODED}" }}',
        allow_escapes=True,
    ), [
        Token.BRACE_OPEN,
        (Token.STRING, "string\n"),
        (Token.STRING, f'all escapes: {ESCAPE_CHARS}'),
        Token.BRACE_CLOSE,
    ])

    tok = py_c_token(r'"text\" with quote"', allow_escapes=False)
    assert tok() == (Token.STRING, 'text\\')
    assert tok() == (Token.STRING, 'with')
    assert tok() == (Token.STRING, 'quote')
    with pytest.raises(TokenSyntaxError):
        tok()


def test_preserve_comments(py_c_token: Type[Tokenizer]) -> None:
    """Test the ability to output comments."""
    text = '''
    "a" { "b" } // end-of-"line" comment
    /* multi
    line comment
    */
    
    "c" // Successive
    // Comments
    '''
    check_tokens(py_c_token(text, allow_star_comments=True), [
        Token.NEWLINE,
        (Token.STRING, "a"), Token.BRACE_OPEN, (Token.STRING, "b"), Token.BRACE_CLOSE, Token.NEWLINE,
        Token.NEWLINE,
        Token.NEWLINE,
        (Token.STRING, "c"), Token.NEWLINE,
        Token.NEWLINE,
    ])
    check_tokens(py_c_token(text, allow_star_comments=True, preserve_comments=True), [
        Token.NEWLINE,
        (Token.STRING, "a"), Token.BRACE_OPEN, (Token.STRING, "b"), Token.BRACE_CLOSE,
        (Token.COMMENT, ' end-of-"line" comment'), Token.NEWLINE,
        (Token.COMMENT, ' multi\n    line comment\n    '), Token.NEWLINE,
        Token.NEWLINE,
        (Token.STRING, "c"), (Token.COMMENT, " Successive"), Token.NEWLINE,
        (Token.COMMENT, " Comments"), Token.NEWLINE,
    ])
    # Verify the line numbers are also correct. Note that these are *after* the token.
    tok = py_c_token(text, allow_star_comments=True, preserve_comments=True)
    line_numbers = [tok.line_num for _ in tok.skipping_newlines()]
    assert line_numbers == [2, 2, 2, 2, 2, 5, 7, 7, 8]


def test_token_syntax_error() -> None:
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


def test_tok_error(py_c_token: Type[Tokenizer]) -> None:
    """Test the tok.error() helper."""
    tok: Tokenizer = py_c_token(['test'], 'filename.py')
    tok.line_num = 45
    assert tok.error('basic') == TokenSyntaxError('basic', 'filename.py', 45)
    assert tok.error('Error with } and { brackets') == TokenSyntaxError(
        'Error with } and { brackets', 'filename.py', 45,
    )

    tok.line_num = 3782
    assert tok.error('Arg {0}, {2} and {1} formatted', 'a', 'b', 'c') == TokenSyntaxError(
        'Arg a, c and b formatted', 'filename.py', 3782,
    )
    tok.filename = None
    assert tok.error('Param: {:.6f}, {!r}, {}', 1/3, "test", test_tok_error) == TokenSyntaxError(
        f"Param: {1/3:.6f}, 'test', {test_tok_error}", None, 3782,
    )


error_messages = {
    Token.STRING: 'Unexpected string = "%"!',
    Token.PROP_FLAG: 'Unexpected property flags = [%]!',
    Token.PAREN_ARGS: 'Unexpected parentheses block = (%)!',
    Token.DIRECTIVE: 'Unexpected directive "#%"!',
    Token.COMMENT: 'Unexpected comment "//%"!',
    Token.EOF: 'File ended unexpectedly!',
    Token.NEWLINE: 'Unexpected newline!',
    Token.BRACE_OPEN: 'Unexpected "{" character!',
    Token.BRACE_CLOSE: 'Unexpected "}" character!',
    Token.BRACK_OPEN: 'Unexpected "[" character!',
    Token.BRACK_CLOSE: 'Unexpected "]" character!',
    Token.COLON: 'Unexpected ":" character!',
    Token.COMMA: 'Unexpected "," character!',
    Token.EQUALS: 'Unexpected "=" character!',
    Token.PLUS: 'Unexpected "+" character!',
}


@pytest.mark.parametrize('token', Token)
def test_tok_error_messages(py_c_token: Type[Tokenizer], token: Token) -> None:
    """Test the tok.error() handler with token types."""
    fmt = error_messages[token]  # If KeyError, needs to be updated.
    tok: Tokenizer = py_c_token(['test'], 'fname')
    tok.line_num = 23
    assert tok.error(token) == TokenSyntaxError(fmt.replace('%', ''), 'fname', 23)
    assert tok.error(token, 'the value') == TokenSyntaxError(fmt.replace('%', 'the value'), 'fname', 23)
    with pytest.raises(TypeError):
        tok.error(token, 'val1', 'val2')


def test_unicode_error_wrapping(py_c_token: Type[Tokenizer]) -> None:
    """Test that Unicode errors are wrapped into TokenSyntaxError."""
    def raises_unicode() -> Iterator[str]:
        yield "line of_"
        yield "text\n"
        raise UnicodeDecodeError('utf8', bytes(100), 1, 2, 'reason')

    tok = py_c_token(raises_unicode())
    assert tok() == (Token.STRING, "line")
    assert tok() == (Token.STRING, "of_text")
    with pytest.raises(TokenSyntaxError) as exc_info:
        list(tok)
    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


def test_early_binary_arg(py_c_token: Type[Tokenizer]) -> None:
    """Test that passing bytes values is caught before looping."""
    with pytest.raises(TypeError):
        py_c_token(b'test')


def test_block_iter(py_c_token: Type[Tokenizer]) -> None:
    """Test the Tokenizer.block() helper."""
    # First two correct usages:
    tok = py_c_token('''\
    "test"
    {
    "blah" "value"
    
    "another" "value"
    }
    ''')
    assert tok() == (Token.STRING, "test")
    bl = tok.block("tester")
    assert next(bl) == "blah"
    assert next(bl) == "value"
    assert next(bl) == "another"
    assert next(bl) == "value"
    with pytest.raises(StopIteration):
        next(bl)

    tok = py_c_token(' "blah" "value" } ')
    bl = tok.block("tester", False)
    assert next(bl) == "blah"
    assert next(bl) == "value"
    with pytest.raises(StopIteration):
        next(bl)

    # Completes correctly with no values.
    assert list(py_c_token('{}').block('')) == []

    # We can remove tokens halfway through on the original tokenizer.
    tok = py_c_token(' { \n\n"legal" { = } block } ')
    bl = tok.block("test")
    assert next(bl) == 'legal'
    assert tok() == (Token.BRACE_OPEN, '{')
    assert tok() == (Token.EQUALS, '=')
    assert tok() == (Token.BRACE_CLOSE, '}')
    assert next(bl) == 'block'
    with pytest.raises(StopIteration):
        next(bl)

    # Now errors.

    # Not an open brace, also it must defer to the first next() call.
    b = py_c_token(' hi ').block('blah', consume_brace=True)
    with pytest.raises(TokenSyntaxError):
        next(b)

    # Also with implicit consume_brace
    b = py_c_token(' hi ').block('blah')
    with pytest.raises(TokenSyntaxError):
        next(b)

    # Open brace where there shouldn't.
    b = py_c_token('{').block('blah', consume_brace=False)
    with pytest.raises(TokenSyntaxError):
        next(b)

    # Two open braces, only consume one.
    b = py_c_token('{ {').block('blah')
    with pytest.raises(TokenSyntaxError):
        next(b)

    # And one in the middle.
    b = py_c_token('{ "test" { "never-here" } ').block('blah')
    assert next(b) == "test"
    with pytest.raises(TokenSyntaxError):
        next(b)

    # Running off the end uses the block in the result.
    b = py_c_token('{ blah "blah" ').block('SpecialBlockName')
    assert next(b) == "blah"
    assert next(b) == "blah"
    with pytest.raises(TokenSyntaxError, match='SpecialBlockName'):
        next(b)


# noinspection PyCallingNonCallable
@pytest.mark.parametrize(
    'Tok',
    [Cy_BaseTokenizer, Py_BaseTokenizer],
    ids=['Cython', 'Python'],
)
def test_subclass_base(Tok: Type[BaseTokenizer]) -> None:
    """Test subclassing of the base tokenizer."""
    class Sub(Tok):
        def __init__(self, tok):
            super().__init__('filename', TokenSyntaxError)
            self.__tokens = iter(tok)

        def _get_token(self) -> Tuple[Token, str]:
            try:
                tok = next(self.__tokens)
            except StopIteration:
                return Token.EOF, ''
            if isinstance(tok, tuple):
                return tok
            else:
                return tok, TOK_VALS[tok]

    check_tokens(Sub(noprop_parse_tokens), noprop_parse_tokens)

    # Do some peeks, pushbacks, etc to check they work.
    tok = Sub(prop_parse_tokens)
    it = iter(tok)
    assert next(it) == (Token.NEWLINE, '\n')
    assert next(tok.skipping_newlines()) == (Token.STRING, 'Root1')
    tok.push_back(Token.DIRECTIVE, '#test')
    assert tok() == (Token.DIRECTIVE, '#test')
    assert tok() == (Token.NEWLINE, '\n')
    assert tok.peek() == (Token.BRACE_OPEN, '{')
    tok.push_back(Token.STRING, "extra")
    assert tok.peek() == (Token.STRING, "extra")
    assert tok() == (Token.STRING, "extra")
    assert tok.peek() == (Token.BRACE_OPEN, '{')
    assert next(iter(tok)) == (Token.BRACE_OPEN, '{')
    skip = tok.skipping_newlines()
    assert next(skip) == (Token.STRING, 'Key')
    assert next(it) == (Token.STRING, 'Value')
    assert next(skip) == (Token.STRING, 'Extra')
    assert next(skip) == (Token.STRING, 'Spaces')
    assert next(tok.skipping_newlines()) == (Token.STRING, 'Block')
