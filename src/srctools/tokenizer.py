"""Parses text into groups of tokens.

This is used internally for parsing KV1, text DMX, FGDs, VMTs, etc. If available this will be
replaced with a faster Cython-optimised version.

The :py:class:`BaseTokenizer` class implements various helper functions for navigating through the
token stream. The :py:class:`Tokenizer` class then takes text file objects, a full string or an
iterable of strings and actually parses it into tokens, while :py:class:`IterTokenizer` allows
transforming the stream before the destination receives it.

Once the tokenizer is created, either iterate over it or call the tokenizer to fetch the next
token/value pair. One token of lookahead is supported, accessed by the
:py:func:`BaseTokenizer.peek()` and  :py:func:`BaseTokenizer.push_back()` methods. They also track
the current line number as data is read, letting you ``raise BaseTokenizer.error(...)`` to easily
produce an exception listing the relevant line number and filename.

Character escapes match matches `utilbuffer.cpp <https://github.com/ValveSoftware/source-sdk-2013/blob/0d8dceea4310fde5706b3ce1c70609d72a38efdf/sp/src/tier1/utlbuffer.cpp#L57-L69>`_ in the SDK.
Specifically, the following characters are escaped:
`\\\\n`, `\\\\t`, `\\\\v`, `\\\\b`, `\\\\r`, `\\\\f`, `\\\\a`, `\\`, `?`, `'` and `"`.
"""
import re
from typing import (
    TYPE_CHECKING, Final, Iterable, Iterator, List, NoReturn, Optional, Tuple, Type,
    Union,
)
from typing_extensions import Self, TypeAlias, overload
from enum import Enum
from os import fspath as _conv_path
import abc

from srctools import StringPath


__all__ = [
    'TokenSyntaxError', 'BARE_DISALLOWED',
    'Token', 'BaseTokenizer', 'Tokenizer', 'IterTokenizer',
    'escape_text', 'format_exc_fileinfo',
]


def format_exc_fileinfo(msg: str, file: Optional[StringPath], line_num: Optional[int]) -> str:
    """If a line number or file is provided, include those in the error message."""
    if file is None and line_num is None:
        return msg
    parts = [msg]
    if line_num is not None:
        parts.append(f'\nError occurred on line {line_num}')
        if file is not None:
            parts.append(f', with file "{file}".')
        else:
            parts.append('.')
    elif file is not None:
        parts.append(f'\nError occurred with file "{file}".')
    else:
        # We checked for both being none above!
        raise AssertionError((msg, file, line_num))
    return ''.join(parts)


class TokenSyntaxError(Exception):
    """An error that occurred when parsing a file.

    Normally this is created via :py:func:`BaseTokenizer.error()` which formats text into the error
    and includes the filename/line number from the tokenizer.

    The string representation will include the provided file and line number if present.
    """
    mess: str
    """The error message that occurred."""
    file: Optional[StringPath]
    """The filename of the file being parsed, or ``None`` if not known."""
    line_num: Optional[int]
    """The line where the error occurred, or ``None`` if not applicable (EOF, for instance)."""

    def __init__(
        self,
        message: str,
        file: Optional[StringPath] = None,
        line: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.mess = message
        self.file = file
        self.line_num = line

    def __repr__(self) -> str:
        return f'TokenSyntaxError({self.mess!r}, {self.file!r}, {self.line_num!r})'

    # This is mutable.
    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TokenSyntaxError):
            return (
                self.mess == other.mess and
                self.file == other.file and
                self.line_num == other.line_num
            )
        return NotImplemented

    def __str__(self) -> str:
        """Generate the complete error message.

        This includes the line number and file, if available.
        """
        return format_exc_fileinfo(self.mess, self.file, self.line_num)


class Token(Enum):
    """A token type produced by the tokenizer."""
    EOF = 0  #: Produced indefinitely after the end of the file is reached.
    STRING = 1  #: Quoted or unquoted text.
    NEWLINE = 2  #: Produced at the end of every line.
    PAREN_ARGS = 3  #: Parenthesised ``(data)``.
    DIRECTIVE = 4  #: ``#name`` (automatically casefolded).
    COMMENT = 5  #: A ``//`` or ``/* */`` comment.

    BRACE_OPEN = 6  #: A ``{`` character.
    BRACE_CLOSE = 7  #: A ``}`` character.

    PROP_FLAG = 11  #: A ``[!flag]``
    BRACK_OPEN = 12  #: A ``[`` character. Only used if ``PROP_FLAG`` is not.
    BRACK_CLOSE = 13  #: A ``]`` character.

    COLON = 14  #: A ``:`` character, if :py:attr:`~Tokenizer.colon_operator` is enabled.
    EQUALS = 15  #: A ``=`` character.
    PLUS = 16  #: A ``+`` character, if :py:attr:`Tokenizer.plus_operator` is enabled.
    COMMA = 17  #: A ``,`` character.

    @property
    def has_value(self) -> bool:
        """If true, this type has an associated value."""
        return self.value in (1, 3, 4, 5, 11)


_OPERATOR_VALS = {
    Token.EOF: '',
    Token.NEWLINE: '\n',

    Token.BRACE_OPEN: '{',
    Token.BRACE_CLOSE: '}',

    Token.BRACK_OPEN: '[',
    Token.BRACK_CLOSE: ']',

    Token.COLON: ':',
    Token.EQUALS: '=',
    Token.PLUS: '+',
    Token.COMMA: ',',
}


_OPERATORS = {
    '{': Token.BRACE_OPEN,
    '}': Token.BRACE_CLOSE,

    '=': Token.EQUALS,
    ',': Token.COMMA,
}


# See https://github.com/ValveSoftware/source-sdk-2013/blob/0d8dceea4310fde5706b3ce1c70609d72a38efdf/sp/src/tier1/utlbuffer.cpp#L57-L69
ESCAPES = {
    'n': '\n',
    't': '\t',
    'v': '\v',
    'b': '\b',
    'r': '\r',
    'f': '\f',
    'a': '\a',

    # If present, disables special parsing.
    '"': '"',
    "'": "'",
    '/': '/',
    '\\': '\\',
    '?': '?',
}
ESCAPES_INV = {char: f'\\{sym}' for sym, char in ESCAPES.items()}
ESCAPE_RE = re.compile('|'.join(map(re.escape, ESCAPES_INV)))

#: Characters not allowed for bare strings. These must be quoted.
BARE_DISALLOWED: Final = frozenset('"\'{};,=[]()\r\n\t ')


class BaseTokenizer(abc.ABC):
    """Provides an interface for processing text into tokens.

     It then provides tools for using those to parse data. This is an :external:py:class:`abc.ABC`,
     a subclass must be used to provide a source for the tokens.
    """
    error_type: Type[TokenSyntaxError]
    """The exception class to produce if an error occurs. This must be a subtype of 
    :py:class:`TokenSyntaxError`, since it is passed the line number and filename in addition to
    the error message. The :py:meth:`error()` method can be used to intelligently construct an
    instance to raise.
    """
    filename: Optional[str]
    """The filename that is being parsed. This is passed along to the error class, 
    to produce relevant errors.
    """
    line_num: int
    """The line number of the last token. Can be changed, but is automatically updated whenever
    :py:attr:`Token.NEWLINE` tokens are seen.
    """

    #: If set, this token will be returned next.
    _pushback: List[Tuple[Token, str]]

    def __init__(
        self,
        filename: Optional[StringPath],
        error: Type[TokenSyntaxError],
    ) -> None:
        if filename is not None:
            self.filename = _conv_path(filename)
            if isinstance(self.filename, bytes):
                # We only use this for display, so if bytes convert.
                # Call repr() then strip the b'', so we get the
                # automatic escaping of unprintable characters.
                self.filename = repr(self.filename)[2:-1]
        else:
            self.filename = None

        if error is None:
            self.error_type = TokenSyntaxError
        else:
            if not issubclass(error, TokenSyntaxError):
                raise TypeError(f'Invalid error instance "{type(error).__name__}"!')
            self.error_type = error

        self._pushback = []
        self.line_num = 1

    @overload
    def error(self, __message: Token) -> TokenSyntaxError: ...

    @overload
    def error(self, __message: Token, __value: str) -> TokenSyntaxError: ...

    @overload
    def error(self, __message: str, *args: object) -> TokenSyntaxError: ...

    def error(self, message: Union[str, Token], *args: object) -> TokenSyntaxError:
        """Raise a syntax error exception.

        This returns the :py:class:`TokenSyntaxError` instance, with
        line number and filename attributes filled in.
        The message can be a :py:class:`Token` with the associated string value to produce a
        wrong token error, or a string which will be `{}-formatted`_ with the positional args
        if they are present.

        .. _{}-formatted: https://docs.python.org/3/library/string.html#formatstrings
        """
        if isinstance(message, Token):
            if len(args) > 1:
                raise TypeError(f'Token {message.name} passed with multiple values: {args}')
            tok_val = '' if len(args) == 0 else args[0]

            if message is Token.PROP_FLAG:
                message = f'Unexpected property flags = [{tok_val}]!'
            elif message is Token.PAREN_ARGS:
                message = f'Unexpected parentheses block = ({tok_val})!'
            elif message is Token.STRING:
                message = f'Unexpected string = "{tok_val}"!'
            elif message is Token.DIRECTIVE:
                message = f'Unexpected directive "#{tok_val}"!'
            elif message is Token.COMMENT:
                message = f'Unexpected comment "//{tok_val}"!'
            elif message is Token.EOF:
                message = 'File ended unexpectedly!'
            elif message is Token.NEWLINE:
                message = 'Unexpected newline!'
            else:
                message = f'Unexpected "{_OPERATOR_VALS[message]}" character!'
        elif args:
            message = message.format(*args)
        return self.error_type(
            message,
            self.filename,
            self.line_num,
        )

    def __reduce__(self) -> NoReturn:
        """Disallow pickling Tokenizers.

        The files themselves usually are not pickleable, or are very
        large strings.
        There is also the issue with recreating the C/Python versions.
        """
        raise TypeError('Cannot pickle Tokenizers!')

    @abc.abstractmethod
    def _get_token(self) -> Tuple[Token, str]:
        """Compute the next token, must be implemented by subclasses."""
        raise NotImplementedError

    def __call__(self) -> Tuple[Token, str]:
        """Compute and fetch the next token."""
        if self._pushback:
            return self._pushback.pop()
        return self._get_token()

    def __iter__(self) -> Self:
        """Tokenizers are their own iterator."""
        return self

    def __next__(self) -> Tuple[Token, str]:
        """Iterate to produce a token, stopping at EOF."""
        tok_and_val = self()
        if tok_and_val[0] is Token.EOF:
            raise StopIteration
        return tok_and_val

    def push_back(self, tok: Token, value: Optional[str] = None) -> None:
        """Return a token, so it will be reproduced when called again.

        The value is required for :py:const:`Token.STRING`, :py:const:`~Token.PAREN_ARGS` and \
        :py:const:`~Token.PROP_FLAG`, but ignored for other token types.
        """
        if not isinstance(tok, Token):
            raise ValueError(repr(tok) + ' is not a Token!')

        try:
            value = _OPERATOR_VALS[tok]
        except KeyError:
            if value is None:
                raise ValueError(f'Value required for {tok.name!r}!') from None

        self._pushback.append((tok, value))

    def peek(self) -> Tuple[Token, str]:
        """Peek at the next token, without removing it from the stream."""
        tok_and_val = self()
        self._pushback.append(tok_and_val)
        return tok_and_val

    def skipping_newlines(self) -> Iterator[Tuple[Token, str]]:
        """Iterate over the tokens, skipping newlines."""
        while True:
            tok_and_val = tok, tok_value = self()
            if tok is Token.EOF:
                return
            elif tok is not Token.NEWLINE:
                yield tok_and_val

    def block(self, name: str, consume_brace: bool = True) -> Iterator[str]:
        """Helper iterator for parsing keyvalue style blocks.

        This will first consume a ``{``. Then it will skip newlines, and output
        each string section found. When ``}`` is found it terminates, anything else
        produces an appropriate error.
        This is safely re-entrant, and tokens can be taken or put back as required.
        """
        if consume_brace:
            self.expect(Token.BRACE_OPEN)
        while True:
            tok, tok_value = self()
            if tok is Token.EOF:
                raise self.error(f'Unclosed {name} block!')
            elif tok is Token.BRACE_CLOSE:
                return
            elif tok is Token.STRING:
                yield tok_value
            elif tok is not Token.NEWLINE:
                raise self.error(tok)

    def expect(self, token: Token, skip_newline: bool = True) -> str:
        """Consume the next token, which should be the given type.

        If it is not, this raises an error.
        If skip_newline is true, newlines will be skipped over. This
        does not apply if the desired token is newline.
        """

        if token is Token.NEWLINE:
            skip_newline = False

        next_token, value = self()

        while skip_newline and next_token is Token.NEWLINE:
            next_token, value = self()

        if next_token is not token:
            raise self.error(
                'Expected {}, but got {}!',
                token,
                next_token,
            )
        return value


class Tokenizer(BaseTokenizer):
    """Processes text data into groups of tokens.

    This mainly groups strings and removes comments.

    Due to many inconsistencies in Valve's parsing of files,
    several options are available to control whether different
    syntaxes are accepted.
    """
    _chunk_iter: Iterator[str]
    _cur_chunk: str
    _char_index: int

    string_bracket: bool
    """If set, `[bracket]` blocks are parsed as a single string-like block. \
    If disabled these are parsed as :py:const:`~Token.BRACK_OPEN`, :py:const:`~Token.STRING` \
    then :py:const:`~Token.BRACK_CLOSE`.
    """
    allow_escapes: bool
    """This determines whether ``\\n``-style escapes are expanded."""
    allow_star_comments: bool
    """If enabled, this allows ``/* */`` comments. Otherwise, an immediate error is produced."""
    preserve_comments: bool
    """:py:const:`Token.COMMENT` are produced if this is set."""
    colon_operator: bool
    """This controls whether ``:`` produces :py:const:`~Token.COLON` tokens, or is treated as
      part of a bare string."""
    plus_operator: bool
    """This controls whether ``+`` produces :py:const:`~Token.PLUS` tokens, or is treated as
      part of a bare string."""
    _last_was_cr: bool

    def __init__(
        self,
        data: Union[str, Iterable[str]],
        filename: Optional[StringPath] = None,
        error: Type[TokenSyntaxError] = TokenSyntaxError,
        *,
        string_bracket: bool = False,
        allow_escapes: bool = True,
        allow_star_comments: bool = False,
        preserve_comments: bool = False,
        colon_operator: bool = False,
        plus_operator: bool = False,
    ) -> None:
        # If a file-like object, automatically use the configured name.
        if filename is None and hasattr(data, 'name'):
            filename = data.name  # pyright: ignore - Can't handle hasattr()

        super().__init__(filename, error)

        # Catch passing direct bytes far in advance.
        if isinstance(data, bytes):
            raise TypeError(
                'Cannot parse binary data! Decode to the desired encoding, '
                'or wrap in io.TextIOWrapper() to decode gradually.'
            )

        # If it's a literal string, there's no point iterating over that.
        # So just keep it as a single chunk, and set the iterator to immediately
        # quit.
        if isinstance(data, str):
            self._cur_chunk = data
            self._chunk_iter = iter(())
        else:
            self._cur_chunk = ''
            self._chunk_iter = iter(data)
        self._char_index = -1

        self.string_bracket = bool(string_bracket)
        self.allow_escapes = bool(allow_escapes)
        self.allow_star_comments = bool(allow_star_comments)
        self.colon_operator = bool(colon_operator)
        self.plus_operator = bool(plus_operator)
        self.preserve_comments = bool(preserve_comments)
        self._last_was_cr = False

    def _next_char(self) -> Optional[str]:
        """Return the next character, or None if no more characters are there."""
        self._char_index += 1
        try:
            return self._cur_chunk[self._char_index]
        except IndexError:
            # Retrieve a chunk from the iterable.
            # Skip empty chunks (shouldn't be there.)
            try:
                for chunk in self._chunk_iter:
                    if isinstance(chunk, bytes):
                        raise ValueError('Cannot parse binary data!')
                    if not isinstance(chunk, str):
                        raise ValueError("Data was not a string!")
                    if chunk:
                        self._cur_chunk = chunk
                        self._char_index = 0
                        return chunk[0]
            except UnicodeDecodeError as exc:
                raise self.error("Could not decode file!") from exc
            # Out of characters after empty chunks
            return None

    def _get_token(self) -> Tuple[Token, str]:
        """Return the next token, value pair."""
        value_chars: List[str]
        while True:
            next_char = self._next_char()
            if next_char is None:  # EOF, use a dummy string.
                return Token.EOF, ''
            # First try simple operators.
            try:
                return _OPERATORS[next_char], next_char
            except KeyError:
                pass
            # Handle newlines, converting \r and \r\n to \n.
            if next_char == '\r':
                self._last_was_cr = True
                self.line_num += 1
                return Token.NEWLINE, '\n'
            elif next_char == '\n':
                # Consume the \n in \r\n.
                if self._last_was_cr:
                    self._last_was_cr = False
                    continue
                self.line_num += 1
                return Token.NEWLINE, '\n'
            else:
                self._last_was_cr = False

            if next_char in ' \t':
                # Ignore whitespace..
                continue

            elif next_char == '/':
                if (comm := self._handle_comment()) is not None:
                    return comm
            elif next_char == '"':
                return self._handle_string()
            elif next_char == '[':
                # FGDs use [] for grouping, Properties use it for flags.
                if not self.string_bracket:
                    return Token.BRACK_OPEN, '['

                value_chars = []
                while True:
                    next_char = self._next_char()
                    if next_char == ']':
                        return Token.PROP_FLAG, ''.join(value_chars)
                    # Must be one line!
                    elif next_char == '\n':
                        raise self.error(
                            'Reached end of line '
                            'without closing "]"!'
                        )
                    elif next_char == '[':
                        # Don't allow nesting, that's bad.
                        raise self.error('Cannot nest [] brackets!')
                    elif next_char is None:
                        raise self.error(
                            'Unterminated property flag!\n\n'
                            'Like "name" "value" [flag_without_end'
                        )
                    value_chars.append(next_char)

            elif next_char == '(':
                # Parentheses around text...
                value_chars = []
                while True:
                    next_char = self._next_char()
                    if next_char == ')':
                        return Token.PAREN_ARGS, ''.join(value_chars)
                    elif next_char == '\n':
                        self.line_num += 1
                    elif next_char == '(':
                        raise self.error('Cannot nest () brackets!')
                    elif next_char is None:
                        raise self.error('Unterminated parentheses!')
                    value_chars.append(next_char)

            # Ignore Unicode Byte Order Mark on first lines
            elif next_char == '\uFEFF' and self.line_num == 1:
                continue
                # If not on line 1 we fall out of the if,
                # and get an unexpected char error.

            elif next_char == ':' and self.colon_operator:
                return Token.COLON, ':'

            elif next_char == '+' and self.plus_operator:
                return Token.PLUS, '+'

            elif next_char == ']':
                if self.string_bracket:
                    # If string_bracket is set (using PROP_FLAG), this is a
                    # syntax error - we don't have an open one to close!
                    raise self.error('No open [] to close with "]"!')
                return Token.BRACK_CLOSE, ']'

            elif next_char == ')':
                raise self.error('No open () to close with ")"!')

            elif next_char == '#':  # A #name "directive", which we casefold.
                value_chars = []
                while True:
                    next_char = self._next_char()
                    if (
                        next_char in BARE_DISALLOWED
                        or (next_char == ':' and self.colon_operator)
                        or (next_char == '+' and self.plus_operator)
                    ):
                        # We need to repeat this, so we return the ending char next.
                        # If it's not allowed, that'll error on next call.
                        self._char_index -= 1
                        return Token.DIRECTIVE, ''.join(value_chars)
                    elif next_char is None:
                        # A directive could be the last value in the file.
                        return Token.DIRECTIVE, ''.join(value_chars)
                    else:
                        value_chars.append(next_char.casefold())

            # Bare names
            elif next_char not in BARE_DISALLOWED:
                value_chars = [next_char]
                while True:
                    next_char = self._next_char()
                    if (
                        next_char in BARE_DISALLOWED
                        or (next_char == ':' and self.colon_operator)
                        or (next_char == '+' and self.plus_operator)
                    ):
                        # We need to repeat this, so we return the ending char next.
                        # If it's not allowed, that'll error on next call.
                        self._char_index -= 1
                        return Token.STRING, ''.join(value_chars)
                    elif next_char is None:
                        # Bare names at the end are actually fine.
                        # It could be a value for the last prop.
                        return Token.STRING, ''.join(value_chars)
                    else:
                        value_chars.append(next_char)

            else:
                raise self.error('Unexpected character "{}"!', next_char)

    def _handle_comment(self) -> Optional[Tuple[Token, str]]:
        """Handle a comment. The last character read was the initial slash."""
        # The next must be either a slash (//) or star (/*)
        comment_next = self._next_char()
        comment_buf: Optional[List[str]] = [] if self.preserve_comments else None
        if comment_next == '*':
            # /* comment.
            if self.allow_star_comments:
                comment_start = self.line_num
                while True:
                    next_char = self._next_char()
                    if next_char is None:
                        raise self.error(
                            'Unclosed /* comment '
                            '(starting on line {})!',
                            comment_start,
                        )
                    elif next_char == '\n':
                        self.line_num += 1
                        if comment_buf is not None:
                            comment_buf.append(next_char)
                    elif next_char == '*':
                        # Check next, next character!
                        next_next_char = self._next_char()
                        if next_next_char is None:
                            raise self.error(
                                'Unclosed /* comment '
                                '(starting on line {})!',
                                comment_start,
                            )
                        elif next_next_char == '/':
                            break
                        else:
                            # We need to reparse this, to ensure
                            # "**/" parses correctly!
                            self._char_index -= 1
                    elif comment_buf is not None:
                        comment_buf.append(next_char)
                if comment_buf is not None:
                    return Token.COMMENT, ''.join(comment_buf)
            else:
                raise self.error(
                    '/**/-style comments are not allowed!'
                )
        elif comment_next != '/':
            raise self.error(
                'Single slash found, '
                'instead of two for a comment (// or /* */)!'
                if self.allow_star_comments else
                'Single slash found, '
                'instead of two for a comment (//)!'
            )
        else:
            # Skip to end of line
            comment_buf = [] if self.preserve_comments else None
            while True:
                next_char = self._next_char()
                if next_char == '\n' or next_char is None:
                    break
                if comment_buf is not None:
                    comment_buf.append(next_char)

            # We want to produce the token for the end character.
            self._char_index -= 1
            if comment_buf is not None:
                return Token.COMMENT, ''.join(comment_buf)
        return None  # Swallow the comment.

    def _handle_string(self) -> Tuple[Token, str]:
        """Handle a quoted string definition. The last character was a quote."""
        value_chars: List[str] = []
        last_was_cr = False
        while True:
            next_char = self._next_char()
            if next_char == '"':
                return Token.STRING, ''.join(value_chars)
            elif next_char == '\r':
                self.line_num += 1
                last_was_cr = True
                value_chars.append('\n')
                continue
            elif next_char == '\n':
                if last_was_cr:
                    last_was_cr = False
                    continue
                self.line_num += 1
            else:
                last_was_cr = False

            if next_char == '\\' and self.allow_escapes:
                # Escape text
                escape = self._next_char()
                if escape is None:
                    raise self.error('No character to escape!')
                elif escape == '\n':
                    continue  # Allow \ at the end of a line to skip.
                try:
                    next_char = ESCAPES[escape]
                except KeyError:
                    # Instead of checking for EOF first, do it here since None won't be in
                    # the dict. That way the happy path doesn't have to check.
                    if escape is None:
                        raise self.error('Unterminated string!') from None
                    else:
                        next_char = '\\' + escape
                        # raise self.error('Unknown escape "\\{}" in {}!', escape, self._cur_chunk)
            if next_char is None:
                raise self.error('Unterminated string!')
            else:
                value_chars.append(next_char)


class IterTokenizer(BaseTokenizer):
    """Wraps a token iterator to provide the tokenizer interface.

    This is useful to pre-process a token stream before parsing it with other
    code.
    """
    source: Iterator[Tuple[Token, str]]

    def __init__(
        self,
        source: Iterable[Tuple[Token, str]],
        filename: StringPath = '',
        error: Type[TokenSyntaxError] = TokenSyntaxError,
    ) -> None:
        super().__init__(filename, error)
        self.source = iter(source)

    def __repr__(self) -> str:
        if self.error_type is TokenSyntaxError:
            return f'IterTokenizer({self.source!r}, {self.filename!r})'
        else:
            return f'IterTokenizer({self.source!r}, {self.filename!r}, {self.error_type!r})'

    def _get_token(self) -> Tuple[Token, str]:
        try:
            return next(self.source)
        except StopIteration:
            return Token.EOF, ''


def escape_text(text: str) -> str:
    r"""Escape special characters and backslashes, so tokenising reproduces them."""
    return ESCAPE_RE.sub(lambda match: ESCAPES_INV[match.group()], text)


# This is available as both C and Python versions, plus the unprefixed
# best version.
# For static typing, make it think they're the same.
Py_BaseTokenizer: TypeAlias = BaseTokenizer
Cy_BaseTokenizer: TypeAlias = BaseTokenizer
Py_Tokenizer: TypeAlias = Tokenizer
Cy_Tokenizer: TypeAlias = Tokenizer
Py_IterTokenizer: TypeAlias = IterTokenizer
Cy_IterTokenizer: TypeAlias = IterTokenizer

# Maintain this for testing.
_py_escape_text = escape_text
cy_escape_text = escape_text

# Do it this way, so static analysis ignores this.
if not TYPE_CHECKING:
    _glob = globals()
    try:
        from . import _tokenizer
    except ImportError:
        pass
    else:
        _name = ''
        for _name in ['Tokenizer', 'BaseTokenizer', 'IterTokenizer']:
            _glob[_name] = _glob['Cy_' + _name] = getattr(_tokenizer, _name)
        _glob['escape_text'] = _glob['cy_escape_text'] = _tokenizer.escape_text
        del _glob, _name, _tokenizer
