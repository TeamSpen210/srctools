"""Parses text into groups of tokens.

This is used internally for parsing files.
"""
from enum import Enum

from typing import (
    Union, Optional,
    Callable, Iterable, Iterator,
    Tuple
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

    BRACE_OPEN = '{'
    BRACE_CLOSE = '}'

    PROP_FLAG = 10  # [!flag]
    BRACK_OPEN = 11  # only if above is not used
    BRACK_CLOSE = ']'  # Won't be used if PROP_FLAG

    COLON = ':'
    EQUALS = '='
    PLUS = '+'


OPERATORS = {
    token.value: token
    for token in Token
    if isinstance(token.value, str)
}

# Returned when no more characters...
OPERATORS[None] = Token.EOF

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
BARE_DISALLOWED = '"\'{}<>();:[]\n\t '


class Tokenizer:
    """Processes text data into groups of tokens.

    This mainly groups strings and removes comments.
    """
    def __init__(
        self,
        data: Union[str, Iterable[str]],
        filename: str=None,
        error: Callable[
            [str, Optional[int], Optional[str]],
            TokenSyntaxError,
        ]=TokenSyntaxError,
        string_bracket=False,
    ):
        if isinstance(data, str):
            self.cur_chunk = data
            self.chunk_iter = iter(())
        else:
            self.cur_chunk = ''
            self.chunk_iter = iter(data)
        self.char_index = -1
        self.filename = filename
        self.error_type = error
        self.string_bracket = string_bracket
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

    def _next_char(self) -> Optional[str]:
        """Return the next character, or None if no more characters are there."""
        self.char_index += 1
        try:
            return self.cur_chunk[self.char_index]
        except IndexError:
            # Retrieve a chunk from the iterable.
            try:
                self.cur_chunk = next(self.chunk_iter)
            except StopIteration:
                # Out of characters
                return None
            self.char_index = 0
            try:
                return self.cur_chunk[0]
            except IndexError:
                # Skip empty chunks (shouldn't be there.)
                for chunk in self.chunk_iter:
                    if chunk:
                        self.cur_chunk = chunk
                        return chunk[0]
                # Out of characters after empty chunks
                return None

    def __call__(self) -> Tuple[Token, str]:
        """Return the next token, value pair."""
        while True:
            next_char = self._next_char()
            # First try simple operators & EOF.
            try:
                return OPERATORS[next_char], next_char
            except KeyError:
                pass
            if next_char == '\n':
                self.line_num += 1
                return Py_Token.NEWLINE, '\n'

            elif next_char in ' \t':
                # Ignore whitespace..
                continue

            # Comments
            elif next_char == '/':
                # The next must be another slash! (//)
                comment_next = self._next_char()
                if comment_next != '/':
                    raise self.error('Single slash found!')
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
                        return Py_Token.STRING, ''.join(value_chars)
                    elif next_char == '\n':
                        self.line_num += 1
                    elif next_char == '\\':
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
                    return Py_Token.BRACK_OPEN, '['

                value_chars = []
                while True:
                    next_char = self._next_char()
                    if next_char == ']':
                        return Py_Token.PROP_FLAG, ''.join(value_chars)
                    # Must be one line!
                    elif next_char == '\n':
                        raise self.error(Py_Token.NEWLINE)
                    elif next_char is None:
                        raise self.error('Unterminated property flag!')
                    value_chars.append(next_char)

            elif next_char == '(':
                # Parentheses around text...
                value_chars = []
                while True:
                    next_char = self._next_char()
                    if next_char == ')':
                        return Py_Token.PAREN_ARGS, ''.join(value_chars)
                    elif next_char == '\n':
                        self.line_num += 1
                    elif next_char is None:
                        raise self.error('Unterminated parentheses!')
                    value_chars.append(next_char)

            # Ignore Unicode Byte Order Mark on first lines
            elif next_char == '\uFEFF' and self.line_num == 1:
                continue

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
                        return Py_Token.STRING, ''.join(value_chars)
                    elif next_char is None:
                        # Bare names at the end are actually fine.
                        # It could be a value for the last prop.
                        return Py_Token.STRING, ''.join(value_chars)
                    else:
                        value_chars.append(next_char)

            else:
                raise self.error('Unexpected character "{}"!', next_char)

    def __iter__(self) -> Iterator[Tuple[Token, Optional[str]]]:
        # Call ourselves until EOF is returned
        return iter(self, (Py_Token.EOF, None))

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

# These are available as both C and Python versions, plus the unprefixed
# best version.
Py_Token = Token
Py_Tokenizer = Tokenizer

# These are for static typing help, so it knows they're the same.
C_Token = Token
C_Tokenizer = Tokenizer
try:
    # noinspection all
    from srctools._tokenizer import Token, Tokenizer
except ImportError:
    C_Token = C_Tokenizer = None  # type: ignore
else:
    C_Token = Token  # type: ignore
    C_Tokenizer = Tokenizer  # type: ignore
