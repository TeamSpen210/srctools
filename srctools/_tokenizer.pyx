#cython: language_level=3
"""Cython version of the Tokenizer class."""
from enum import Enum

class Token(Enum):
    """A token type produced by the tokenizer."""
    EOF = 0  # Ran out of text.
    STRING = 1  # Quoted or unquoted text
    NEWLINE = 2  # \n
    BRACE_OPEN = '{'
    BRACE_CLOSE = '}'

    PAREN_OPEN = '('
    PAREN_CLOSE = ')'

    PROP_FLAG = 10  # [!flag]
    BRACK_OPEN = 11  # only if above is not used
    BRACK_CLOSE = ']'  # Won't be used if PROP_FLAG

    COLON = ':'

# Characters not allowed for bare names on a line.
DEF BARE_DISALLOWED = '"\'{}<>();:[]'


cdef class Tokenizer:
    """Processes text data into groups of tokens.

    This mainly groups strings and removes comments.
    """
    cdef str cur_chunk
    cdef object chunk_iter
    cdef int char_index
    # Class to call when errors occur..
    cdef object error_type

    cdef readonly str filename
    cdef readonly int line_num
    cdef readonly bint string_bracket

    cdef object _tok_EOF
    cdef object _tok_STRING
    cdef object _tok_PROP_FLAG
    cdef object _tok_NEWLINE
    cdef object _tok_BRACE_OPEN
    cdef object _tok_BRACE_CLOSE
    cdef object _tok_BRACK_OPEN
    cdef object _tok_BRACK_CLOSE

    def __init__(self, data, filename, error=None, bint string_bracket=False):
        if isinstance(data, str):
            self.cur_chunk = data
            self.chunk_iter = iter(())
        else:
            self.cur_chunk = ''
            self.chunk_iter = iter(data)
        self.char_index = -1
        self.filename = filename
        if error is None:
            from srctools.tokenizer import TokenSyntaxError
            self.error_type = TokenSyntaxError
        else:
            self.error_type = error
        self.string_bracket = string_bracket
        self.line_num = 1

        # If a file-like object, this is automatic.
        if not filename and hasattr(data, 'name'):
            self.filename = data.name

        self._tok_STRING = Token.STRING
        self._tok_PROP_FLAG = Token.PROP_FLAG
        self._tok_NEWLINE = Token.NEWLINE
        self._tok_EOF = (Token.EOF, None)
        self._tok_BRACE_OPEN = (Token.BRACE_OPEN, '{')
        self._tok_BRACE_CLOSE = (Token.BRACE_CLOSE, '}')
        self._tok_BRACK_OPEN = (Token.BRACK_OPEN, '[')
        self._tok_BRACK_CLOSE = (Token.BRACK_CLOSE, ']')

    def error(self, message, *args):
        """Raise a syntax error exception.

        This returns the TokenSyntaxError instance, with
        line number and filename attributes filled in.
        The message can be a Token to indicate a wrong token,
        or a string which will be formatted with the positional args.
        """
        if isinstance(message, Token):
            message = 'Unexpected token {}!'.format(message.name)
        else:
            message = message.forget(*args)
        return self._error(message)

    cdef _error(self, str message):
        """C-private self.error()."""
        return self.error_type(
            message,
            self.filename,
            self.line_num,
        )

    cdef str _next_char(self):
        """Return the next character, or None if no more characters are there."""
        cdef str chunk

        self.char_index += 1
        if self.char_index < len(self.cur_chunk):
            return self.cur_chunk[self.char_index]

        # Retrieve a chunk from the iterable.
        chunk = next(self.chunk_iter, None)
        if chunk is None:
            return None
        self.cur_chunk = chunk
        self.char_index = 0

        if len(chunk) > 0:
            return chunk[0]

        # Skip empty chunks (shouldn't be there.)
        for chunk in self.chunk_iter:
            if len(chunk) > 0:
                self.cur_chunk = chunk
                return chunk[0]
        # Out of characters after empty chunks
        return None

    def __call__(self):
        """Return the next token, value pair."""
        return self._next_token()

    cdef _next_token(self):
        cdef:
            str next_char

        while True:
            next_char = self._next_char()
            if next_char is None:
                return self._tok_EOF
            if next_char == '{':
                return self._tok_BRACE_OPEN
            if next_char == '}':
                return self._tok_BRACE_CLOSE
            if next_char == ':':
                return self._tok_COLON
            if next_char == ']':
                return self._tok_BRACK_CLOSE

            # First try simple operators & EOF.

            if next_char == '\n':
                self.line_num += 1
                return self._tok_NEWLINE, '\n'

            elif next_char in ' \t':
                # Ignore whitespace..
                continue

            # Comments
            elif next_char == '/':
                # The next must be another slash! (//)
                if self._next_char() != '/':
                    raise self._error('Single slash found!')
                # Skip to end of line
                while True:
                    next_char = self._next_char()
                    if next_char is None or next_char == '\n':
                        break
                # We want to produce the token for the end character.
                self.char_index -= 1

            # Strings
            elif next_char == '"':
                value_chars = []
                while True:
                    next_char = self._next_char()
                    if next_char == '"':
                        return self._tok_STRING, ''.join(value_chars)
                    elif next_char == '\n':
                        self.line_num += 1
                    elif next_char == '\\':
                        # Escape text
                        escape = self._next_char()
                        if escape == 'n':
                            next_char = '\n'
                        elif escape == 't':
                            next_char = '\t'
                        elif escape in '"\\/':
                            # This actually escapes the functions of these..
                            next_char = escape
                        elif escape == '\n':
                            # \ at end of line to ignore the newline.
                            next_char = ''
                        else:
                            if escape is None:
                                raise self._error('Unterminated string!')
                            else:
                                next_char = '\\' + escape
                                # raise self.error('Unknown escape "\\{}" in {}!', escape, self.cur_chunk)
                    elif next_char is None:
                        raise self._error('Unterminated string!')
                    value_chars.append(next_char)

            elif next_char == '[':
                # FGDs use [] for grouping, Properties use it for flags.
                if not self.string_bracket:
                    return self._tok_BRACE_OPEN

                value_chars = []
                while True:
                    next_char = self._next_char()
                    if next_char == ']':
                        return self._tok_PROP_FLAG, ''.join(value_chars)
                    # Must be one line!
                    elif next_char == '\n':
                        raise self.error(self._tok_NEWLINE)
                    elif next_char is None:
                        raise self._error('Unterminated property flag!')
                    value_chars.append(next_char)

            # Bare names
            elif next_char not in BARE_DISALLOWED:
                value_chars = [next_char]
                while True:
                    next_char = self._next_char()
                    if next_char in BARE_DISALLOWED:
                        raise self._error(f'Unexpected character "{next_char}"!')
                    elif next_char in ' \t\n':
                        # We need to repeat this so we return the newline.
                        self.char_index -= 1
                        return self._tok_STRING, ''.join(value_chars)
                    elif next_char is None:
                        # Bare names at the end are actually fine.
                        # It could be a value for the last prop.
                        return self._tok_STRING, ''.join(value_chars)
                    else:
                        value_chars.append(next_char)

            else:
                raise self._error(f'Unexpected character "{next_char}"!')

    def __iter__(self):
        # Call ourselves until EOF is returned
        return iter(self, self._tok_EOF)

    def expect(self, object token):
        """Consume the next token, which should be the given type.

        If it is not, this raises an error.
        """
        cdef tuple next_token = self._next_token()
        if next_token[0] is not token:
            raise self._error(f'Expected {token}, but got {next_token[0]}!')
