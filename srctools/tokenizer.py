"""Parses text into groups of tokens.

This is used internally for parsing files.
"""
from enum import Enum
from os import fspath as _conv_path, PathLike
from typing import (
    Union, Optional, Type, Any,
    Iterable, Iterator,
    Tuple, List,
)


class TokenSyntaxError(Exception):
    """An error that occurred when parsing a file.

    mess = The error message that occurred.
    file = The filename passed to Property.parse(), if it exists
    line_num = The line where the error occurred.
    """
    def __init__(
        self,
        message: str,
        file: Optional[str],
        line: Optional[int],
    ) -> None:
        super().__init__()
        self.mess = message
        self.file = file
        self.line_num = line

    def __repr__(self) -> str:
        return 'TokenSyntaxError({!r}, {!r}, {!r})'.format(
            self.mess,
            self.file,
            self.line_num,
            )

    def __str__(self) -> str:
        """Generate the complete error message.

        This includes the line number and file, if available.
        """
        mess = self.mess
        if self.line_num:
            mess += '\nError occurred on line ' + str(self.line_num)
            if self.file:
                mess += ', with file'
            else:
                mess += '.'
        if self.file:
            if not self.line_num:
                mess += '\nError occurred with file'
            mess += ' "' + self.file + '".'
        return mess


class Token(Enum):
    """A token type produced by the tokenizer."""
    EOF = 0  # Ran out of text.
    STRING = 1  # Quoted or unquoted text
    NEWLINE = 2  # \n
    PAREN_ARGS = 3  # (data)

    BRACE_OPEN = 5
    BRACE_CLOSE = 6

    PROP_FLAG = 10  # [!flag]
    BRACK_OPEN = 11  # only if above is not used
    BRACK_CLOSE = 12

    COLON = 13
    EQUALS = 14
    PLUS = 15
    COMMA = 16

    @property
    def has_value(self) -> bool:
        """If true, this type has an associated value."""
        return self.value in (1, 3, 10)

_PUSHBACK_VALS = {
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

    ':': Token.COLON,
    '=': Token.EQUALS,
    '+': Token.PLUS,
    ',': Token.COMMA,
}


ESCAPES = {
    'n': '\n',
    't': '\t',
    # Escape function of the following
    '"': '"',
    '/': '/',
    '\\': '\\',

    # \ at end of line to ignore the newline.
    '\n': '',
}

# Characters not allowed for bare names on a line.
BARE_DISALLOWED = set('"\'{};:,[]()\n\t ')


class Tokenizer:
    """Processes text data into groups of tokens.

    This mainly groups strings and removes comments.

    Due to many inconsistencies in Valve's parsing of files,
    several options are available to control whether different
    syntaxes are accepted:
        * string_bracket parses [bracket] blocks as a single string-like block.
          If disabled these are parsed as BRACK_OPEN, STRING, BRACK_CLOSE.
        * allow_escapes controls whether \\n-style escapes are expanded.
        * allow_star_comments if enabled allows /* */ comments.
    """
    def __init__(
        self,
        data: Union[str, Iterable[str]],
        filename: Union[str, PathLike]=None,
        error: Type[TokenSyntaxError]=TokenSyntaxError,
        string_bracket: bool=False,
        allow_escapes: bool=True,
        allow_star_comments: bool=False,
    ) -> None:
        # Catch passing direct bytes far in advance.
        if isinstance(data, bytes):
            raise TypeError(
                'Cannot parse binary data! Decode to the desired encoding, '
                'or wrap in io.TextIOWrapper() to decode gradually.'
            )

        if isinstance(data, str):
            self.cur_chunk = data
            self.chunk_iter = iter(())  # type: Iterator[str]
        else:
            self.cur_chunk = ''
            self.chunk_iter = iter(data)
        self.char_index = -1

        if filename is not None:
            self.filename = _conv_path(filename)
            # If a file-like object, automatically use the configured name.
        elif hasattr(data, 'name'):
            self.filename = data.name  # type: ignore  # hasattr()
        else:
            self.filename = None

        if error is None:
            self.error_type = TokenSyntaxError  # type: Type[TokenSyntaxError]
        else:
            if not issubclass(error, TokenSyntaxError):
                raise TypeError('Invalid error instance "{}"!'.format(type(error).__name__))
            self.error_type = error

        self.string_bracket = bool(string_bracket)
        self.allow_escapes = bool(allow_escapes)
        self.allow_star_comments = bool(allow_star_comments)
        # If set, this token will be returned next.
        self._pushback = None  # type: Optional[Tuple[Token, str]]
        self.line_num = 1

    def error(self, message: Union[str, Token], *args) -> TokenSyntaxError:
        """Raise a syntax error exception.

        This returns the TokenSyntaxError instance, with
        line number and filename attributes filled in.
        The message can be a Token to indicate a wrong token,
        or a string which will be {}-formatted with the positional args
        if they are present.
        """
        if isinstance(message, Token):
            message = 'Unexpected token {}!'.format(message.name)
        elif args:
            message = message.format(*args)
        return self.error_type(
            message,
            self.filename,
            self.line_num,
        )

    def __reduce__(self) -> Any:
        """Disallow pickling Tokenizers.

        The files themselves usually are not pickleable, or are very
        large strings.
        There is also the issue with recreating the C/Python versions.
        """
        raise NotImplementedError('Cannot pickle Tokenizers!')

    def _next_char(self) -> Optional[str]:
        """Return the next character, or None if no more characters are there."""
        self.char_index += 1
        try:
            return self.cur_chunk[self.char_index]
        except IndexError:
            # Retrieve a chunk from the iterable.
            try:
                chunk = self.cur_chunk = next(self.chunk_iter)
            except StopIteration:
                # Out of characters
                return None
            except UnicodeDecodeError as exc:
                raise self.error("Could not decode file!") from exc

            # Specifically catch passing binary data.
            if isinstance(chunk, bytes):
                raise ValueError('Cannot parse binary data!')
            if not isinstance(chunk, str):
                raise ValueError("Data was not a string!")

            self.char_index = 0

            try:
                return chunk[0]
            except IndexError:
                # Skip empty chunks (shouldn't be there.)
                try:
                    for chunk in self.chunk_iter:
                        if isinstance(chunk, bytes):
                            raise ValueError('Cannot parse binary data!')
                        if not isinstance(chunk, str):
                            raise ValueError("Data was not a string!")
                        if chunk:
                            self.cur_chunk = chunk
                            return chunk[0]
                except UnicodeDecodeError as exc:
                    raise self.error("Could not decode file!") from exc
                # Out of characters after empty chunks
                return None

    def __call__(self) -> Tuple[Token, str]:
        """Return the next token, value pair."""
        if self._pushback is not None:
            next_val = self._pushback
            self._pushback = None
            return next_val

        while True:
            next_char = self._next_char()
            if next_char is None:  # EOF, use a dummy string.
                return Token.EOF, ''
            # First try simple operators.
            try:
                return _OPERATORS[next_char], next_char
            except KeyError:
                pass
            if next_char == '\n':
                self.line_num += 1
                return Token.NEWLINE, '\n'

            elif next_char in ' \t':
                # Ignore whitespace..
                continue

            # Comments
            elif next_char == '/':
                # The next must be another slash! (//)
                comment_next = self._next_char()
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
                            elif next_char == '*':
                                # Check next next character!
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
                                    self.char_index -= 1
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
                    while True:
                        next_char = self._next_char()
                        if next_char == '\n' or next_char is None:
                            break
                    # We want to produce the token for the end character.
                    self.char_index -= 1

            # Strings
            elif next_char == '"':
                value_chars = []  # type: List[str]
                while True:
                    next_char = self._next_char()
                    if next_char == '"':
                        return Token.STRING, ''.join(value_chars)
                    elif next_char == '\n':
                        self.line_num += 1
                    elif next_char == '\\' and self.allow_escapes:
                        # Escape text
                        escape = self._next_char()
                        if escape is None:
                            raise self.error('No character to escape!')
                        try:
                            next_char = ESCAPES[escape]
                        except KeyError:
                            if escape is None:
                                raise self.error('Unterminated string!')
                            else:
                                next_char = '\\' + escape
                                # raise self.error('Unknown escape "\\{}" in {}!', escape, self.cur_chunk)
                    if next_char is None:
                        raise self.error('Unterminated string!')
                    else:
                        value_chars.append(next_char)

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

            elif next_char == ']':
                if self.string_bracket:
                    # If string_bracket is set (using PROP_FLAG), this is a
                    # syntax error - we don't have an open one to close!
                    raise self.error('No open [] to close with "]"!')
                return Token.BRACK_CLOSE, ']'

            elif next_char == ')':
                raise self.error('No open () to close with ")"!')

            # Bare names
            elif next_char not in BARE_DISALLOWED:
                value_chars = [next_char]
                while True:
                    next_char = self._next_char()
                    if next_char in BARE_DISALLOWED:
                        # We need to repeat this so we return the ending
                        # char next. If it's not allowed, that'll error on
                        # next call.
                        self.char_index -= 1
                        return Token.STRING, ''.join(value_chars)
                    elif next_char is None:
                        # Bare names at the end are actually fine.
                        # It could be a value for the last prop.
                        return Token.STRING, ''.join(value_chars)
                    else:
                        value_chars.append(next_char)

            else:
                raise self.error('Unexpected character "{}"!', next_char)

    def __iter__(self) -> Iterator[Tuple[Token, str]]:
        # Call ourselves until EOF is returned
        return iter(self, (Token.EOF, ''))

    def push_back(self, tok: Token, value: str=None) -> None:
        """Return a token, so it will be reproduced when called again.

        Only one token can be pushed back at once.
        The value is required for STRING, PAREN_ARGS and PROP_FLAGS, but ignored
        for other token types.
        """
        if self._pushback is not None:
            raise ValueError('Token already pushed back!')
        if not isinstance(tok, Token):
            raise ValueError(repr(tok) + ' is not a Token!')

        try:
            value = _PUSHBACK_VALS[tok]
        except KeyError:
            if value is None:
                raise ValueError('Value required for {!r}!'.format(tok.name)) from None

        self._pushback = (tok, value)

    def peek(self) -> Tuple[Token, str]:
        """Peek at the next token, without removing it from the stream."""
        tok_and_val = self()
        # We know this is a valid pushback value, and any existing value was
        # just removed. So unconditionally assign.
        self._pushback = tok_and_val
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

        This will first consume a {. Then it will skip newlines, and output
        each string section found. When } is found it terminates, anything else
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

    def expect(self, token: Token, skip_newline: bool=True) -> str:
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


def escape_text(text: str) -> str:
    r"""Escape special characters and backslashes, so tokenising reproduces them.

    Specifically, \, ", tab, and newline.
    """
    return (
        text.
        replace('\\', '\\\\').
        replace('"', '\\"').
        replace('\t', '\\t').
        replace('\n', '\\n')
    )


# This is available as both C and Python versions, plus the unprefixed
# best version.
Py_Tokenizer = Tokenizer

# This is for static typing help, so it thinks they're the same.
C_Tokenizer = Tokenizer

# Maintain this for testing.
_py_escape_text = escape_text

# Make the actual assignment hidden to type checkers.
try:
    # noinspection all
    from srctools._tokenizer import Tokenizer, Tokenizer as C_Tokenizer, escape_text  # type: ignore
except ImportError:
    pass
