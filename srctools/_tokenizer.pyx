#cython: language_level=3, embedsignature=True, auto_pickle=False
"""Cython version of the Tokenizer class."""
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
import array

cdef extern from *:
    unicode PyUnicode_FromStringAndSize(const char *u, Py_ssize_t size)

# Import the Token enum from the Python file, and cache references
# to all the parts.
# Also grab the exception object.

cdef object Token, TokenSyntaxError
from srctools.tokenizer import Token,  TokenSyntaxError

# Cdef-ed globals become static module vars, which aren't in the module
# dict. We can ensure these are of the type we specify - and are quick to
# lookup.
cdef:
    object STRING = Token.STRING
    object PAREN_ARGS = Token.PAREN_ARGS
    object PROP_FLAG = Token.PROP_FLAG   # [!flag]

    object EOF = Token.EOF
    object NEWLINE = Token.NEWLINE

    # Iterator that immediately raises StopIteration.
    object EMPTY_ITER = iter('')

    # Reuse a single tuple for these, since the value is constant.
    tuple EOF_TUP = (Token.EOF, None)
    tuple NEWLINE_TUP = (Token.NEWLINE, '\n')

    tuple COLON_TUP = (Token.COLON, ':')
    tuple EQUALS_TUP = (Token.EQUALS, '=')
    tuple PLUS_TUP = (Token.PLUS, '+')

    tuple BRACE_OPEN_TUP = (Token.BRACE_OPEN, '{')
    tuple BRACE_CLOSE_TUP = (Token.BRACE_CLOSE, '}')

    tuple BRACK_OPEN_TUP = (Token.BRACK_OPEN, '[')
    tuple BRACK_CLOSE_TUP = (Token.BRACK_CLOSE, ']')


# Characters not allowed for bare names on a line.
# Convert to tuple to only check the chars.
DEF BARE_DISALLOWED = tuple('"\'{};:[]()\n\t ')


@cython.final  # No point in inheriting from this.
cdef class Tokenizer:
    """Processes text data into groups of tokens.

    This mainly groups strings and removes comments.
    """
    cdef str cur_chunk
    cdef object chunk_iter
    # Class to call when errors occur..
    cdef object error_type

    cdef public str filename

    cdef int char_index # Position inside cur_chunk

    cdef public int line_num
    cdef public bint string_bracket
    cdef public bint allow_escapes

    cdef object pushback_tok
    cdef object pushback_val


    def __init__(
        self,
        data not None,
        filename=None,
        error=None,
        bint string_bracket=False,
        bint allow_escapes=True,
    ):
        # Early warning for this particular error.
        if isinstance(data, bytes) or isinstance(data, bytearray):
            raise ValueError(
                'Cannot parse binary data! Decode to the desired encoding, '
                'or wrap in io.TextIOWrapper() to decode as needed.'
            )

        # For direct strings, we can immediately assign that as our chunk,
        # and then set the iterable to an empty iterator.
        if isinstance(data, str):
            self.cur_chunk = data
            self.chunk_iter = EMPTY_ITER
        else:
            # The first next_char() call will pull out a chunk.
            self.cur_chunk = ''
            # This checks that it is indeed iterable.
            self.chunk_iter = iter(data)

        # We initially add one, so it'll be 0 next.
        self.char_index = -1

        if filename:
            self.filename = str(filename)
        else:
            # If a file-like object, automatically set to the filename.
            try:
                self.filename = str(data.name)
            except AttributeError:
                # If not, a Falsey value is excluded by the exception message.
                self.filename = ''

        if error is None:
            self.error_type = TokenSyntaxError
        else:
            if not issubclass(error, TokenSyntaxError):
                raise TypeError(f'Invalid error instance "{type(error).__name__}"!')
            self.error_type = error
        self.string_bracket = string_bracket
        self.allow_escapes = allow_escapes

        self.pushback_tok = self.pushback_val = None

        self.line_num = 1

    def __reduce__(self):
        """Disallow pickling Tokenizers.

        The files themselves usually are not pickleable, or are very
        large strings.
        There is also the issue with recreating the C/Python versions.
        """
        raise NotImplementedError('Cannot pickle Tokenizers!')

    def error(self, message, *args):
        """Raise a syntax error exception.

        This returns the TokenSyntaxError instance, with
        line number and filename attributes filled in.
        The message can be a Token to indicate a wrong token,
        or a string which will be formatted with the positional args.
        """
        if isinstance(message, Token):
            message = f'Unexpected token {message.name}!'
        else:
            message = message.format(*args)
        return self._error(message)

    cdef _error(self, str message):
        """C-private self.error()."""
        return self.error_type(
            message,
            self.filename,
            self.line_num,
        )

    # We check all the getitem[] accesses, so don't have Cython recheck.
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Py_UCS4 _next_char(self) except -2:
        """Return the next character, or -1 if no more characters are there."""
        cdef str chunk
        cdef object chunk_obj

        self.char_index += 1
        if self.char_index < len(self.cur_chunk):
            return self.cur_chunk[self.char_index]

        # Retrieve a chunk from the iterable.
        chunk_obj = next(self.chunk_iter, None)
        if chunk_obj is None:
            return -1

        if isinstance(chunk_obj, bytes):
            raise ValueError('Cannot parse binary data!')
        if not isinstance(chunk_obj, str):
            raise ValueError("Data was not a string!")

        self.cur_chunk = chunk = <str>chunk_obj
        self.char_index = 0

        if len(chunk) > 0:
            return (<str>chunk)[0]

        # Skip empty chunks (shouldn't be there.)
        for chunk_obj in self.chunk_iter:
            if isinstance(chunk_obj, bytes):
                raise ValueError('Cannot parse binary data!')
            if not isinstance(chunk_obj, str):
                raise ValueError("Data was not a string!")

            chunk = <str>chunk_obj

            if len(chunk) > 0:
                self.cur_chunk = chunk
                return (<str>chunk)[0]
        # Out of characters after empty chunks
        return -1

    def __call__(self):
        """Return the next token, value pair."""
        return self.next_token()

    cdef next_token(self):
        """Return the next token, value pair - this is the C version."""
        cdef:
            list value_chars
            Py_UCS4 next_char
            Py_UCS4 escape_char

        if self.pushback_tok is not None:
            output = self.pushback_tok, self.pushback_val
            self.pushback_tok = self.pushback_val = None
            return output

        while True:
            next_char = self._next_char()
            if next_char == -1:
                return EOF_TUP

            elif next_char == '{':
                return BRACE_OPEN_TUP
            elif next_char == '}':
                return BRACE_CLOSE_TUP
            elif next_char == ':':
                return COLON_TUP
            elif next_char == '+':
                return PLUS_TUP
            elif next_char == '=':
                return EQUALS_TUP
            elif next_char == ']':
                return BRACK_CLOSE_TUP

            # First try simple operators & EOF.

            elif next_char == '\n':
                self.line_num += 1
                return NEWLINE_TUP

            elif next_char in ' \t':
                # Ignore whitespace..
                continue

            # Comments
            elif next_char == '/':
                # The next must be another slash! (//)
                if self._next_char() != '/':
                    raise self._error(
                        'Single slash found, '
                        'instead of two for a comment (//)!'
                    )
                # Skip to end of line
                while True:
                    next_char = self._next_char()
                    if next_char == -1 or next_char == '\n':
                        break

                # We want to produce the token for the end character -
                # EOF or NEWLINE.
                self.char_index -= 1

            # Strings
            elif next_char == '"':
                value_chars = []
                while True:
                    next_char = self._next_char()
                    if next_char == -1:
                        raise self._error('Unterminated string!')
                    elif next_char == '"':
                        return STRING, ''.join(value_chars)
                    elif next_char == '\n':
                        self.line_num += 1
                    elif next_char == '\\' and self.allow_escapes:
                        # Escape text
                        escape_char = self._next_char()
                        if escape_char == -1:
                            raise self._error('Unterminated string!')

                        if escape_char == 'n':
                            next_char = '\n'
                        elif escape_char == 't':
                            next_char = '\t'
                        elif escape_char == '\n':
                            # \ at end of line ignores the newline.
                            continue
                        elif escape_char in ('"', '\\', '/'):
                            # For these, we escape to give the literal value.
                            next_char = escape_char
                        else:
                            # For unknown escape_chars, escape the \ automatically.
                            value_chars.append('\\' + escape_char)
                            continue
                            # raise self.error('Unknown escape_char "\\{}" in {}!', escape_char, self.cur_chunk)
                    value_chars.append(next_char)

            elif next_char == '[':
                # FGDs use [] for grouping, Properties use it for flags.
                if not self.string_bracket:
                    return BRACK_OPEN_TUP

                value_chars = []
                while True:
                    next_char = self._next_char()
                    if next_char == -1:
                        raise self._error(
                            'Unterminated property flag!\n\n'
                            'Like "name" "value" [flag_without_end'
                        )
                    elif next_char == ']':
                        return PROP_FLAG, ''.join(value_chars)
                    # Must be one line!
                    elif next_char == '\n':
                        raise self.error(NEWLINE)
                    value_chars.append(next_char)

            elif next_char == '(':
                # Parentheses around text...
                value_chars = []
                while True:
                    next_char = self._next_char()
                    if next_char == -1:
                        raise self._error('Unterminated parentheses!')
                    elif next_char == ')':
                        return PAREN_ARGS, ''.join(value_chars)
                    elif next_char == '\n':
                        self.line_num += 1
                    value_chars.append(next_char)

            # Ignore Unicode Byte Order Mark on first lines
            elif next_char == '\uFEFF':
                if self.line_num == 1:
                    continue
                # else, we fall out of the if, and get an unexpected char
                # error.

            else: # Not-in can't be in a switch, so we need to nest this.
                # Bare names
                if next_char not in BARE_DISALLOWED:
                    value_chars = [next_char]
                    while True:
                        next_char = self._next_char()
                        if next_char == -1:
                            # Bare names at the end are actually fine.
                            # It could be a value for the last prop.
                            return STRING, ''.join(value_chars)

                        elif next_char in BARE_DISALLOWED:
                            # We need to repeat this so we return the ending
                            # char next. If it's not allowed, that'll error on
                            # next call.
                            # We need to repeat this so we return the newline.
                            self.char_index -= 1
                            return STRING, ''.join(value_chars)
                        else:
                            value_chars.append(next_char)
                else:
                    raise self._error(f'Unexpected character "{next_char}"!')

    def __iter__(self):
        # Call ourselves until EOF is returned
        return iter(self, EOF_TUP)

    def push_back(self, object tok not None, str value=None):
        """Return a token, so it will be reproduced when called again.

        Only one token can be pushed back at once.
        The value should be the original value, or None
        """
        if self.pushback_tok is not None:
            raise ValueError('Token already pushed back!')
        if not isinstance(tok, Token):
            raise ValueError(repr(tok) + ' is not a Token!')

        cdef int tok_val = tok.value
        cdef str real_value

        if tok_val == 0: # EOF
            real_value = None
        elif tok_val in (1, 3, 10):  # STRING, PAREN_ARGS, PROP_FLAG
            real_value = ''
        elif tok_val == 2:  # NEWLINE
            real_value = '\n'
        elif tok_val == 5:  # BRACE_OPEN
            real_value = '{'
        elif tok_val == 6:  # BRACE_CLOSE
            real_value = '}'
        elif tok_val == 11:  # BRACK_OPEN
            real_value = '['
        elif tok_val == 12:  # BRACK_CLOSE
            real_value = ']'
        elif tok_val == 13:  # COLON
            real_value = ':'
        elif tok_val == 14:  # EQUALS
            real_value = '='
        elif tok_val == 15:  # PLUS
            real_value = '+'
        else:
            raise ValueError('Unknown token value!')

        # If no value provided, use the default (operators)
        if value is None:
            value = real_value
        # A type which needs a value provided...
        elif real_value == '':
            value = '' if value is None else value
        elif not isinstance(value, str):
            raise ValueError(
                f'Invalid value provided ({value!r}) for {tok.name}!'
            ) from None

        self.pushback_tok = tok
        self.pushback_val = value


    def peek(self):
        """Peek at the next token, without removing it from the stream."""
        # We know this is a valid pushback value, and any existing value was
        # just removed. So unconditionally assign.
        self.pushback_tok, self.pushback_val = tok_and_val = <tuple>self.next_token()

        return tok_and_val

    def skipping_newlines(self):
        """Iterate over the tokens, skipping newlines."""
        return NewlinesIter.__new__(NewlinesIter, self)

    def expect(self, object token, bint skip_newline=True):
        """Consume the next token, which should be the given type.

        If it is not, this raises an error.
        If skip_newline is true, newlines will be skipped over. This
        does not apply if the desired token is newline.
        """
        if token is NEWLINE:
            skip_newline = False

        next_token, value = <tuple>self.next_token()

        while skip_newline and next_token is NEWLINE:
            next_token, value = <tuple>self.next_token()

        if next_token is not token:
            raise self._error(f'Expected {token}, but got {next_token}!')
        return value


cdef class NewlinesIter:
    """Iterate over the tokens, skipping newlines."""
    cdef Tokenizer tok

    def __cinit__(self, Tokenizer tok not None):
        self.tok = tok

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            tok_and_val = self.tok.next_token()

            if tok_and_val is EOF_TUP:
                raise StopIteration
            elif tok_and_val is not NEWLINE_TUP:
                return tok_and_val

    def __reduce__(self):
        """This cannot be pickled - the Python version does not have this class."""
        raise NotImplementedError('Cannot pickle NewlinesIter!')

# Remove this class from the module, so it's not directly exposed.
del globals()['NewlinesIter']


@cython.nonecheck(False)
def escape_text(str text not None):
    r"""Escape special characters and backslashes, so tokenising reproduces them.

    Specifically, \, ", tab, and newline.
    """
    # UTF8 = ASCII for those chars, so we can replace in that form.
    cdef bytes enc_text = text.encode('utf8')
    cdef Py_ssize_t final_len = len(enc_text)
    cdef char letter
    for letter in enc_text:
        if letter in (b'\\', b'"', b'\t', b'\n'):
            final_len += 1

    cdef char * out_buff
    cdef int i = 0
    try:
        out_buff = <char *>PyMem_Malloc(final_len+1)
        for letter in enc_text:
            if letter == b'\\':
                out_buff[i] = b'\\'
                i += 1
                out_buff[i] = b'\\'
            elif letter == b'"':
                out_buff[i] = b'\\'
                i += 1
                out_buff[i] = b'"'
            elif letter == b'\t':
                out_buff[i] = b'\\'
                i += 1
                out_buff[i] = b't'
            elif letter == b'\n':
                out_buff[i] = b'\\'
                i += 1
                out_buff[i] = b'n'
            else:
                out_buff[i] = letter
            i += 1
        out_buff[final_len] = b'\0'
        return PyUnicode_FromStringAndSize(out_buff, final_len)
    finally:
        PyMem_Free(out_buff)
