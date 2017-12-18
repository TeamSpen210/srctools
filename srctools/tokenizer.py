"""Parses text into groups of tokens.

This is used internally for parsing files.
"""
from enum import Enum

from typing import (
    Union, Optional,
    Callable, Iterable, Iterator,
    Tuple,
    Type,
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
            line: Optional[int]
            ) -> None:
        super().__init__()
        self.mess = message
        self.file = file
        self.line_num = line

    def __repr__(self):
        return 'TokenSyntaxError({!r}, {!r}, {!r})'.format(
            self.mess,
            self.file,
            self.line_num,
            )

    def __str__(self):
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

    @property
    def has_value(self):
        """If true, this type has an associated value."""
        return self.value in (1, 3, 10)

_PUSHBACK_VALS = {
    Token.EOF: None,
    Token.NEWLINE: '\n',

    Token.BRACE_OPEN: '{',
    Token.BRACE_CLOSE: '}',

    Token.BRACK_OPEN: '[',
    Token.BRACK_CLOSE: ']',

    Token.COLON: ':',
    Token.EQUALS: '=',
    Token.PLUS: '+',
}


_OPERATORS = {
    '{': Token.BRACE_OPEN,
    '}': Token.BRACE_CLOSE,

    ']': Token.BRACK_CLOSE,  # Won't be used if PROP_FLAG

    ':': Token.COLON,
    '=': Token.EQUALS,
    '+': Token.PLUS,

    # None is returned when no more characters...
    None: Token.EOF,
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
BARE_DISALLOWED = '"\'{};:[]()\n\t '


class Tokenizer:
    """Processes text data into groups of tokens.

    This mainly groups strings and removes comments.
    """
    def __init__(
        self,
        data: Union[str, Iterable[str]],
        filename: str=None,
        error: Type[TokenSyntaxError]=TokenSyntaxError,
        string_bracket=False,
        allow_escapes=True,
    ):
        if isinstance(data, bytes):
            raise ValueError(
                'Cannot parse binary data! Decode to the desired encoding, '
                'or wrap in io.TextIOWrapper() to decode as needed.'
            )

        if isinstance(data, str):
            self.cur_chunk = data
            self.chunk_iter = iter(())
        else:
            self.cur_chunk = ''
            self.chunk_iter = iter(data)
        self.char_index = -1
        self.filename = filename

        if error is None:
            self.error_type = TokenSyntaxError
        else:
            if not issubclass(error, TokenSyntaxError):
                raise TypeError('Invalid error instance "{}"!'.format(type(error).__name__))
            self.error_type = error

        self.string_bracket = string_bracket
        self.allow_escapes = allow_escapes
        # If set, this token will be returned next.
        self._pushback = None  # type: Optional[Tuple[Token, str]]
        self.line_num = 1

        # If a file-like object, this is automatic.
        if not filename and hasattr(data, 'name'):
            self.filename = data.name

    def error(self, message: Union[str, Token], *args):
        """Raise a syntax error exception.

        This returns the TokenSyntaxError instance, with
        line number and filename attributes filled in.
        The message can be a Token to indicate a wrong token,
        or a string which will be formatted with the positional args.
        """
        if isinstance(message, Token):
            message = 'Unexpected token {}!'.format(message.name)
        else:
            message = message.format(*args)
        return self.error_type(
            message,
            self.filename,
            self.line_num,
        )

    def __reduce__(self):
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
            if isinstance(chunk, bytes):
                raise ValueError('Cannot parse binary data!')
            if not isinstance(chunk, str):
                raise ValueError("Data was not a string!")
            self.char_index = 0

            try:
                return chunk[0]
            except IndexError:
                # Skip empty chunks (shouldn't be there.)
                for chunk in self.chunk_iter:
                    if isinstance(chunk, bytes):
                        raise ValueError('Cannot parse binary data!')
                    if not isinstance(chunk, str):
                        raise ValueError("Data was not a string!")
                    if chunk:
                        self.cur_chunk = chunk
                        return chunk[0]
                # Out of characters after empty chunks
                return None

    def __call__(self) -> Tuple[Token, str]:
        """Return the next token, value pair."""
        if self._pushback is not None:
            next_char = self._pushback
            self._pushback = None
            return next_char

        while True:
            next_char = self._next_char()
            # First try simple operators & EOF.
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
                if comment_next != '/':
                    raise self.error(
                        'Single slash found, '
                        'instead of two for a comment (//)!'
                    )
                # Skip to end of line
                while True:
                    next_char = self._next_char()
                    if next_char == '\n' or next_char is None:
                        break
                # We want to produce the token for the end character.
                self.char_index -= 1

            # Strings
            elif next_char == '"':
                value_chars = []
                while True:
                    next_char = self._next_char()
                    if next_char == '"':
                        return Token.STRING, ''.join(value_chars)
                    elif next_char == '\n':
                        self.line_num += 1
                    elif next_char == '\\' and self.allow_escapes:
                        # Escape text
                        escape = self._next_char()
                        try:
                            next_char = ESCAPES[escape]
                        except KeyError:
                            if escape is None:
                                raise self.error('Unterminated string!')
                            else:
                                next_char = '\\' + escape
                                # raise self.error('Unknown escape "\\{}" in {}!', escape, self.cur_chunk)
                    elif next_char is None:
                        raise self.error('Unterminated string!')
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
                        raise self.error(Token.NEWLINE)
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
                    elif next_char is None:
                        raise self.error('Unterminated parentheses!')
                    value_chars.append(next_char)

            # Ignore Unicode Byte Order Mark on first lines
            elif next_char == '\uFEFF' and self.line_num == 1:
                continue
                # If not on line 1 we fall out of the if,
                # and get an unexpected char error.

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

    def __iter__(self) -> Iterator[Tuple[Token, Optional[str]]]:
        # Call ourselves until EOF is returned
        return iter(self, (Token.EOF, None))

    def push_back(self, tok: Token, value: str=None):
        """Return a token, so it will be reproduced when called again.

        Only one token can be pushed back at once.
        The value should be the original value, or None
        """
        if self._pushback is not None:
            raise ValueError('Token already pushed back!')
        if not isinstance(tok, Token):
            raise ValueError(repr(tok) + ' is not a Token!')

        try:
            real_value = _PUSHBACK_VALS[tok]
        except KeyError:
            if value is None:
                value = ''
            elif not isinstance(value, str):
                raise ValueError('Invalid value provided ({!r}) for {}!'.format(
                    value, tok.name
                )) from None
        else:
            if value is None:
                value = real_value
            elif real_value != value:
                raise ValueError('Invalid value provided ({!r}) for {}!'.format(
                    value, tok.name
                )) from None

        self._pushback = (tok, value)

    def peek(self) -> Tuple[Token, str]:
        """Peek at the next token, without removing it from the stream."""
        tok_and_val = self()
        # We know this is a valid pushback value, and any existing value was
        # just removed. So unconditionally assign.
        self._pushback = tok_and_val
        return tok_and_val


    def skipping_newlines(self):
        """Iterate over the tokens, skipping newlines."""
        while True:
            tok_and_val = tok, tok_value = self()
            if tok is Token.EOF:
                return
            elif tok is not Token.NEWLINE:
                yield tok_and_val

    def expect(self, token: Token, skip_newline=True):
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


# This is available as both C and Python versions, plus the unprefixed
# best version.
Py_Tokenizer = Tokenizer

# This is for static typing help, so it thinks they're the same.
C_Tokenizer = Tokenizer

# Make the actual assignment hidden to type checkers.
try:
    # noinspection all
    from srctools._tokenizer import Tokenizer, Tokenizer as C_Tokenizer  # type: ignore
except ImportError:
    C_Tokenizer = None  # type: ignore
