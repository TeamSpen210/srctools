# cython: language_level=3, embedsignature=True, auto_pickle=False
# cython: binding=True
"""Cython version of the Tokenizer class."""
from cpython.mem cimport PyMem_Free, PyMem_Malloc, PyMem_Realloc
from libc.stdint cimport uint16_t, uint_fast8_t
cimport cython


cdef extern from *:
    ctypedef unsigned char uchar "unsigned char"  # Using it a lot, this causes it to not be a typedef at all.
    const char* PyUnicode_AsUTF8AndSize(str string, Py_ssize_t *size) except NULL
    str PyUnicode_FromStringAndSize(const char *u, Py_ssize_t size)
    str PyUnicode_FromKindAndData(int kind, const void *buffer, Py_ssize_t size)

cdef object os_fspath
from os import fspath as os_fspath


# Import the Token enum from the Python file, and cache references
# to all the parts.
# Also grab the exception object.

cdef object Token, TokenSyntaxError
from srctools.tokenizer import Token, TokenSyntaxError


__all__ = ['BaseTokenizer', 'Tokenizer', 'IterTokenizer', 'escape_text']

# Cdef-ed globals become static module vars, which aren't in the module
# dict. We can ensure these are of the type we specify - and are quick to
# lookup.
cdef:
    object STRING = Token.STRING
    object PAREN_ARGS = Token.PAREN_ARGS
    object PROP_FLAG = Token.PROP_FLAG  # [!flag]
    object DIRECTIVE = Token.DIRECTIVE  # #name (automatically casefolded)
    object COMMENT = Token.COMMENT

    object EOF = Token.EOF
    object NEWLINE = Token.NEWLINE

    object COLON = Token.COLON
    object EQUALS = Token.EQUALS
    object PLUS = Token.PLUS
    object COMMA = Token.COMMA

    object BRACE_OPEN = Token.BRACE_OPEN
    object BRACE_CLOSE = Token.BRACE_CLOSE

    object BRACK_OPEN = Token.BRACK_OPEN
    object BRACK_CLOSE = Token.BRACK_CLOSE

    # Reuse a single tuple for these, since the value is constant.
    tuple EOF_TUP = (Token.EOF, '')
    tuple NEWLINE_TUP = (Token.NEWLINE, '\n')

    tuple COLON_TUP = (COLON, ':')
    tuple EQUALS_TUP = (EQUALS, '=')
    tuple PLUS_TUP = (PLUS, '+')
    tuple COMMA_TUP = (COMMA, ',')

    tuple BRACE_OPEN_TUP = (BRACE_OPEN, '{')
    tuple BRACE_CLOSE_TUP = (BRACE_CLOSE, '}')

    tuple BRACK_OPEN_TUP = (BRACK_OPEN, '[')
    tuple BRACK_CLOSE_TUP = (BRACK_CLOSE, ']')

    uchar *EMPTY_BUF = b''  # Initial value, just so it's valid.

# Characters not allowed for bare names on a line.
# TODO: Make this an actual constant value, but that won't do the switch optimisation.
DEF BARE_DISALLOWED = b'"\'{};,=[]()\r\n\t '

# Pack flags into a bitfield.
cdef extern from *:
    """
struct TokFlags {
    unsigned char string_brackets: 1;
    unsigned char allow_escapes: 1;
    unsigned char colon_operator: 1;
    unsigned char plus_operator: 1;
    unsigned char allow_star_comments: 1;
    unsigned char preserve_comments: 1;
    unsigned char file_input: 1;
    unsigned char last_was_cr: 1;
};
    """
    cdef struct TokFlags:
        bint string_brackets
        bint allow_escapes
        bint colon_operator
        bint plus_operator
        bint allow_star_comments
        bint preserve_comments
        # If set, the file_iter is a bound read() method.
        bint file_input
        # Records if the last character was a \r, so the next swallows \n.
        bint last_was_cr

# The number of characters to read from a file each time.
cdef enum:
    FILE_BUFFER = 1024

# noinspection PyMissingTypeHints
cdef class BaseTokenizer:
    """Provides an interface for processing text into tokens.

     It then provides tools for using those to parse data.
     This is an abstract class, a subclass must be used to provide a source
     for the tokens.
    """
    # Class to call when errors occur...
    cdef object error_type

    cdef str filename

    cdef list pushback

    cdef public int line_num
    cdef TokFlags flags

    def __init__(self, filename, error):
        # Use os method to convert to string.
        # We know this isn't a method, so skip Cython's optimisation.
        if filename is not None:
            with cython.optimize.unpack_method_calls(False):
                fname = os_fspath(filename)
            if isinstance(fname, bytes):
                # We only use this for display, so if bytes convert.
                # Call repr() then strip the b'', so we get the
                # automatic escaping of unprintable characters.
                fname = (<str>repr(fname))[2:-1]
            self.filename = str(fname)
        else:
            self.filename = None

        if error is None:
            self.error_type = TokenSyntaxError
        else:
            if not issubclass(error, TokenSyntaxError):
                raise TypeError(f'Invalid error instance "{type(error).__name__}"' '!')
            self.error_type = error

        self.pushback = []
        self.line_num = 1
        self.flags = {
            'string_brackets': 0,
            'allow_escapes': 0,
            'colon_operator': 0,
            'plus_operator': 0,
            'allow_star_comments': 0,
            'preserve_comments': 0,
            'file_input': 0,
            'last_was_cr': 0,
        }

    def __reduce__(self):
        """Disallow pickling Tokenizers.

        The files themselves usually are not pickleable, or are very
        large strings.
        There is also the issue with recreating the C/Python versions.
        """
        raise TypeError('Cannot pickle Tokenizers!')

    @property
    def filename(self):
        """Retrieve the filename used in error messages."""
        return self.filename

    @filename.setter
    def filename(self, fname):
        """Change the filename used in error messages."""
        if fname is None:
            self.filename = None
        else:
            with cython.optimize.unpack_method_calls(False):
                fname = os_fspath(fname)
            if isinstance(fname, bytes):
                # We only use this for display, so if bytes convert.
                # Call repr() then strip the b'', so we get the
                # automatic escaping of unprintable characters.
                fname = (<str> repr(fname))[2:-1]
            self.filename = str(fname)

    @property
    def error_type(self):
        """Return the TokenSyntaxError subclass raised when errors occur."""
        return self.error_type

    @error_type.setter
    def error_type(self, value):
        """Alter the TokenSyntaxError subclass raised when errors occur."""
        if not issubclass(value, TokenSyntaxError):
            raise TypeError(f'The error type must be a TokenSyntaxError subclass, not {type(value).__name__}!.')
        self.error_type = value

    def error(self, message, *args):
        """Raise a syntax error exception.

        This returns the TokenSyntaxError instance, with
        line number and filename attributes filled in.
        Either pass a token and optionally the value to give a generic message,
        or a string which will be {}-formatted with the positional args
        if they are present.
        """
        cdef str tok_val, str_msg
        if type(message) is Token:  # We know no subclasses exist..
            if len(args) > 1:
                raise TypeError(f'Token {message.name} passed with multiple values: {args}')

            tok_val = '' if len(args) == 0 else args[0]
            if tok_val is None:
                raise TypeError('Token value should not be None!')

            if message is PROP_FLAG:
                str_msg = f'Unexpected property flags = [{tok_val}]!'
            elif message is PAREN_ARGS:
                str_msg = f'Unexpected parentheses block = ({tok_val})!'
            elif message is STRING:
                str_msg = f'Unexpected string = "{tok_val}"!'
            elif message is DIRECTIVE:
                str_msg = f'Unexpected directive "#{tok_val}"!'
            elif message is COMMENT:
                str_msg = f'Unexpected comment "//{tok_val}"!'
            elif message is EOF:
                str_msg = 'File ended unexpectedly!'
            elif message is NEWLINE:
                str_msg = 'Unexpected newline!'
            elif message is BRACE_OPEN:
                str_msg = 'Unexpected "{" character!'
            elif message is BRACE_CLOSE:
                str_msg = 'Unexpected "}" character!'
            elif message is BRACK_OPEN:
                str_msg = 'Unexpected "[" character!'
            elif message is BRACK_CLOSE:
                str_msg = 'Unexpected "]" character!'
            elif message is COLON:
                str_msg = 'Unexpected ":" character!'
            elif message is EQUALS:
                str_msg = 'Unexpected "=" character!'
            elif message is PLUS:
                str_msg = 'Unexpected "+" character!'
            elif message is COMMA:
                str_msg = 'Unexpected "," character!'
            else:
                raise RuntimeError(message)
        elif args:
            str_msg = message.format(*args)
        else:
            str_msg = message
        return self._error(str_msg)

    # Don't unpack, error_type should be a class.
    @cython.optimize.unpack_method_calls(False)
    cdef inline _error(self, message: str):
        """C-private self.error()."""
        return self.error_type(
            message,
            self.filename,
            self.line_num,
        )

    def __call__(self):
        """Return the next token, value pair."""
        return self.next_token()

    cdef next_token(self):
        """Call the Python-overridable method.
        
        This also implements pushback.
        """
        if self.pushback:
            return self.pushback.pop()

        return self._get_token()

    def _get_token(self):
        """Compute the next token, must be implemented by subclasses."""
        raise NotImplementedError

    def __iter__(self):
        """Tokenizers are their own iterator."""
        return self

    def __next__(self):
        """Iterate to produce a token, stopping at EOF."""
        tok_and_val = self.next_token()
        if (<tuple?> tok_and_val)[0] is EOF:
            raise StopIteration
        return tok_and_val

    def push_back(self, object tok not None, str value=None):
        """Return a token, so it will be reproduced when called again.

        The value is required for STRING, PAREN_ARGS and PROP_FLAGS, but ignored
        for other token types.
        """
        if not isinstance(tok, Token):
            raise ValueError(f'{tok!r} is not a Token!')

        # Read this directly to skip the 'value' descriptor.
        cdef int tok_val = tok._value_

        if tok_val == 0: # EOF
            value = ''
        elif tok_val in (1, 3, 4, 5, 11):  # STRING, PAREN_ARGS, DIRECTIVE, COMMENT, PROP_FLAG
            # Value parameter is required.
            if value is None:
                raise ValueError(f'Value required for {tok!r}' '!')
        elif tok_val == 2:  # NEWLINE
            value = '\n'
        elif tok_val == 6:  # BRACE_OPEN
            value = '{'
        elif tok_val == 7:  # BRACE_CLOSE
            value = '}'
        elif tok_val == 12:  # BRACK_OPEN
            value = '['
        elif tok_val == 13:  # BRACK_CLOSE
            value = ']'
        elif tok_val == 14:  # COLON
            value = ':'
        elif tok_val == 15:  # EQUALS
            value = '='
        elif tok_val == 16:  # PLUS
            value = '+'
        elif tok_val == 17: # COMMA
            value = ','
        else:
            raise ValueError(f'Unknown token {tok!r}')

        self.pushback.append((tok, value))

    def peek(self):
        """Peek at the next token, without removing it from the stream."""
        tok_and_val = <tuple>self.next_token()
        self.pushback.append(tok_and_val)

        return tok_and_val

    def skipping_newlines(self):
        """Iterate over the tokens, skipping newlines."""
        return _NewlinesIter.__new__(_NewlinesIter, self)

    def block(self, str name, consume_brace=True):
        """Helper iterator for parsing keyvalue style blocks.

        This will first consume a {. Then it will skip newlines, and output
        each string section found. When } is found it terminates, anything else
        produces an appropriate error.
        This is safely re-entrant, and tokens can be taken or put back as required.
        """
        return BlockIter.__new__(BlockIter, self, name, consume_brace)

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
            raise self._error(f'Expected {token}, but got {next_token}' '!')
        return value


cdef class Tokenizer(BaseTokenizer):
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
    cdef object cur_chunk
    cdef object chunk_iter
    cdef Py_ssize_t char_index # Position inside cur_chunk

    # Private buffer, to hold string parts we're constructing.
    # Tokenizers are expected to be temporary, so we just never shrink.
    cdef Py_ssize_t buf_size  # 2 << x
    cdef Py_ssize_t buf_pos
    cdef uchar *val_buffer
    cdef const uchar *chunk_buf
    cdef Py_ssize_t chunk_size

    def __cinit__(self):
        self.buf_size = 128
        self.val_buffer = <uchar *>PyMem_Malloc(self.buf_size * sizeof(uchar))
        self.buf_pos = 0
        if self.val_buffer is NULL:
            raise MemoryError

    def __dealloc__(self):
        PyMem_Free(self.val_buffer)

    def __init__(
        self,
        data not None,
        object filename=None,
        error=None,
        *,
        bint string_bracket: bool = False,
        bint allow_escapes: bool = True,
        bint allow_star_comments: bool = False,
        bint preserve_comments: bool = False,
        bint colon_operator: bool = False,
        bint plus_operator: bool = False,
    ):
        # Early warning for this particular error.
        if isinstance(data, bytes) or isinstance(data, bytearray):
            raise TypeError(
                'Cannot parse binary data! Decode to the desired encoding, '
                'or wrap in io.TextIOWrapper() to decode gradually.'
            )

        cdef TokFlags flags = {
            'string_brackets': string_bracket,
            'allow_escapes': allow_escapes,
            'allow_star_comments': allow_star_comments,
            'preserve_comments': preserve_comments,
            'colon_operator': colon_operator,
            'plus_operator': plus_operator,
            'file_input': 0,
            'last_was_cr': 0,
        }

        # For direct strings, we can immediately assign that as our chunk,
        # and then set the iterable to indicate EOF after that.
        if type(data) is str:
            self.cur_chunk = data
            self.chunk_buf = <const uchar *>PyUnicode_AsUTF8AndSize(data, &self.chunk_size)
            self.chunk_iter = None
        else:
            # The first next_char() call will pull out a chunk.
            self.cur_chunk = ''
            self.chunk_size = 0
            self.chunk_buf = EMPTY_BUF

            # If a file, use the read method to pull bulk data.
            try:
                self.chunk_iter = data.read
            except AttributeError:
                # This checks that it is indeed iterable.
                self.chunk_iter = iter(data)
            else:
                flags.file_input = True

        # We initially add one, so it'll be 0 next.
        self.char_index = -1
        self.buf_reset()

        if not filename:
            # If we're given a file-like object, automatically set the filename.
            try:
                filename = data.name
            except AttributeError:
                # If not, a Falsey filename means nothing is added to any
                # KV exception message.
                filename = None

        BaseTokenizer.__init__(self, filename, error)
        self.flags = flags

        # We want to strip a UTF BOM from the start of the file, if it matches.
        # So pull off the first three characters, and if they don't match,
        # rebuild the cur_chunk to allow them.
        # The BOM is b'\xef\xbb\xbf'.
        if self._next_char()[0] != 0xef:
            self.char_index -= 1
        elif self._next_char()[0] != 0xbb:
            self.char_index -= 2
        elif self._next_char()[0] != 0xbf:
            self.char_index -= 3

    @property
    def string_bracket(self) -> bool:
        """Controls whether [bracket] blocks are parsed as a single string-like block.

        If disabled these are parsed as BRACK_OPEN, STRING, BRACK_CLOSE.
        """
        return self.flags.string_brackets

    @string_bracket.setter
    def string_bracket(self, bint value) -> None:
        self.flags.string_brackets = value

    @property
    def allow_escapes(self) -> bool:
        """Controls whether backslash escapes will be parsed."""
        return self.flags.allow_escapes

    @allow_escapes.setter
    def allow_escapes(self, bint value) -> None:
        self.flags.allow_escapes = value

    @property
    def allow_star_comments(self) -> bool:
        """Controls whether /**/ style comments will be enabled."""
        return self.flags.allow_star_comments

    @allow_star_comments.setter
    def allow_star_comments(self, bint value) -> None:
        self.flags.allow_star_comments = value

    @property
    def preserve_comments(self) -> bool:
        """Controls whether comments will be output as tokens."""
        return self.flags.preserve_comments

    @preserve_comments.setter
    def preserve_comments(self, bint value) -> None:
        self.flags.preserve_comments = value

    @property
    def colon_operator(self) -> bool:
        """Controls whether : characters are treated as a COLON token, or part of strings."""
        return self.flags.colon_operator

    @colon_operator.setter
    def colon_operator(self, bint value) -> None:
        self.flags.colon_operator = value

    @property
    def plus_operator(self) -> bool:
        """Controls whether + characters are treated as a PLUS token, or part of strings."""
        return self.flags.plus_operator

    @plus_operator.setter
    def plus_operator(self, bint value) -> None:
        self.flags.plus_operator = value

    cdef inline bint buf_reset(self) except False:
        """Reset the temporary buffer."""
        # Don't bother resizing or clearing, the next append will overwrite.
        self.buf_pos = 0
        return True

    cdef inline int buf_add_char(self, char new_char) except -1:
        """Add a character to the temporary buffer, reallocating if needed."""
        # Temp, so if memory alloc failure occurs we're still in a valid state.
        cdef uchar *newbuf
        cdef Py_ssize_t new_size
        if self.buf_pos >= self.buf_size:
            new_size = self.buf_size * 2
            new_buf = <uchar *>PyMem_Realloc(
                self.val_buffer,
                new_size * sizeof(uchar),
            )
            if new_buf:
                self.buf_size = new_size
                self.val_buffer = new_buf
            else:
                raise MemoryError

        self.val_buffer[self.buf_pos] = new_char
        self.buf_pos += 1

    cdef str buf_get_text(self):
        """Decode the buffer, and return the text."""
        out = PyUnicode_FromStringAndSize(<char *>self.val_buffer, self.buf_pos)
        # Don't bother resizing or clearing, the next append will overwrite.
        self.buf_pos = 0
        return out

    # We check all the getitem[] accesses, so don't have Cython recheck.
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef (uchar, bint) _next_char(self):
        """Return the next character, and a flag to indicate if more characters are present."""
        cdef str chunk
        cdef object chunk_obj

        self.char_index += 1
        if self.char_index < self.chunk_size:
            return self.chunk_buf[self.char_index], False

        if self.chunk_iter is None:
            return b'\x00', True

        if self.flags.file_input:
            try:
                self.cur_chunk = self.chunk_iter(FILE_BUFFER)
            except UnicodeDecodeError as exc:
                raise self._error(f"Could not decode file:\n{exc!s}") from exc
            self.char_index = 0

            if type(self.cur_chunk) is str:
                self.chunk_buf = <const uchar *>PyUnicode_AsUTF8AndSize(self.cur_chunk, &self.chunk_size)
            else:
                raise ValueError('Expected string, got ' + type(self.cur_chunk).__name__)

            if self.chunk_size > 0:
                return self.chunk_buf[0], False
            else:
                self.chunk_iter = None
                return b'\x00', True

        # Retrieve a chunk from the iterable.
        # Skip empty chunks (shouldn't be there.)
        # Use manual next to avoid re-calling iter() here,
        # or using list/tuple optimisations.
        while True:
            try:
                chunk_obj = next(self.chunk_iter, None)
            except UnicodeDecodeError as exc:
                raise self._error(f"Could not decode file:\n{exc!s}") from exc
            if chunk_obj is None:
                # Out of characters after empty chunks
                self.chunk_iter = None
                return  b'\x00', True

            if isinstance(chunk_obj, bytes):
                raise ValueError('Cannot parse binary data!')
            if type(chunk_obj) is not str:
                raise ValueError("Data was not a string!")

            if len(<str>chunk_obj) > 0:
                self.cur_chunk = chunk_obj
                self.char_index = 0
                self.chunk_buf = <const uchar *>PyUnicode_AsUTF8AndSize(self.cur_chunk, &self.chunk_size)
                return self.chunk_buf[0], False

    def _get_token(self):
        """Compute the next token."""
        return self.next_token()

    cdef next_token(self):
        """Return the next token, value pair - this is the C version."""
        cdef:
            uchar next_char
            uchar escape_char
            uchar peek_char
            bint is_eof
            int start_line
            bint ascii_only
            uchar decode[5]
            bint last_was_cr
            bint save_comments

        # Implement pushback directly for efficiency.
        if self.pushback:
            return self.pushback.pop()

        while True:
            next_char, is_eof = self._next_char()
            # First try simple operators & EOF.
            if is_eof:
                return EOF_TUP

            if next_char == b'{':
                return BRACE_OPEN_TUP
            elif next_char == b'}':
                return BRACE_CLOSE_TUP
            elif next_char == b'=':
                return EQUALS_TUP
            elif next_char == b',':
                return COMMA_TUP

            # Handle newlines, converting \r and \r\n to \n.
            if next_char == b'\r':
                self.line_num += 1
                self.flags.last_was_cr = True
                return NEWLINE_TUP
            elif next_char == b'\n':
                # Consume the \n in \r\n.
                if self.flags.last_was_cr:
                    self.flags.last_was_cr = False
                    continue
                self.line_num += 1
                return NEWLINE_TUP
            else:
                self.flags.last_was_cr = False

            if next_char in b' \t':
                # Ignore whitespace...
                continue

            # Comments
            elif next_char == b'/':
                # The next must be another slash! (//)
                next_char, is_eof = self._next_char()
                if next_char == b'*': # /* comment.
                    if self.flags.allow_star_comments:
                        start_line = self.line_num
                        save_comments = self.flags.preserve_comments
                        while True:
                            next_char, is_eof = self._next_char()
                            if is_eof:
                                raise self._error(
                                    f'Unclosed /* comment '
                                    f'(starting on line {start_line})!',
                                )
                            if next_char == b'\n':
                                self.line_num += 1
                                if save_comments:
                                    self.buf_add_char(next_char)
                            elif next_char == b'*':
                                # Check next, next character!
                                peek_char, is_eof = self._next_char()
                                if is_eof:
                                    raise self._error(
                                        f'Unclosed /* comment '
                                        f'(starting on line {start_line})!',
                                    )
                                elif peek_char == b'/':
                                    break
                                else:
                                    # We need to reparse this, to ensure
                                    # "**/" parses correctly!
                                    self.char_index -= 1
                            else:
                                if save_comments:
                                    self.buf_add_char(next_char)
                        if save_comments:
                            return COMMENT, self.buf_get_text()
                    else:
                        raise self._error(
                            '/**/-style comments are not allowed!'
                        )
                elif next_char == b'/':
                    # Skip to end of line.
                    save_comments = self.flags.preserve_comments
                    while True:
                        next_char, is_eof = self._next_char()
                        if is_eof or next_char == b'\n':
                            break
                        if save_comments:
                            self.buf_add_char(next_char)

                    # We want to produce the token for the end character -
                    # EOF or NEWLINE.
                    self.char_index -= 1
                    if save_comments:
                        return COMMENT, self.buf_get_text()
                else:
                    raise self._error(
                        'Single slash found, '
                        'instead of two for a comment (// or /* */)!'
                        if self.flags.allow_star_comments else
                        'Single slash found, '
                        'instead of two for a comment (//)!'
                    )

            # Strings
            elif next_char == b'"':
                self.buf_reset()
                last_was_cr = False
                while True:
                    next_char, is_eof = self._next_char()
                    if is_eof:
                        raise self._error('Unterminated string!')
                    if next_char == b'"':
                        return STRING, self.buf_get_text()
                    elif next_char == b'\r':
                        self.line_num += 1
                        last_was_cr = True
                        self.buf_add_char(b'\n')
                        continue
                    elif next_char == b'\n':
                        if last_was_cr:
                            last_was_cr = False
                            continue
                        self.line_num += 1
                    else:
                        last_was_cr = False

                    if next_char == b'\\' and self.flags.allow_escapes:
                        # Escape text
                        escape_char, is_eof = self._next_char()
                        if is_eof:
                            raise self._error('Unterminated string!')

                        # See this code:
                        # https://github.com/ValveSoftware/source-sdk-2013/blob/0d8dceea4310fde5706b3ce1c70609d72a38efdf/sp/src/tier1/utlbuffer.cpp#L57-L69
                        if escape_char == b'a':
                            next_char = b'\a'
                        elif escape_char == b'b':
                            next_char = b'\b'
                        elif escape_char == b't':
                            next_char = b'\t'
                        elif escape_char == b'n':
                            next_char = b'\n'
                        elif escape_char == b'v':
                            next_char = b'\v'
                        elif escape_char == b'f':
                            next_char = b'\f'
                        elif escape_char == b'r':
                            next_char = b'\r'
                        elif escape_char == b'\n':
                            # \ at end of line ignores the newline.
                            continue
                        elif escape_char in (b'"', b'\\', b'/', b"'", b'?'):
                            # For these, we escape to give the literal value.
                            next_char = escape_char
                        else:
                            # For unknown escape_chars, escape the \ automatically.
                            self.buf_add_char(b'\\')
                            self.buf_add_char(escape_char)
                            continue
                            # raise self.error('Unknown escape_char "\\{}" in {}!', escape_char, self._cur_chunk)
                    self.buf_add_char(next_char)

            elif next_char == b'[':
                # FGDs use [] for grouping, Properties use it for flags.
                if not self.flags.string_brackets:
                    return BRACK_OPEN_TUP

                self.buf_reset()
                while True:
                    next_char, is_eof = self._next_char()
                    # Must be one line!
                    if is_eof or next_char == b'\n':
                        raise self._error(
                            'Reached end of line '
                            'without closing "]"!'
                        )
                    if next_char == b'[':
                        # Don't allow nesting, that's bad.
                        raise self._error('Cannot nest [] brackets!')
                    elif next_char == b']':
                        return PROP_FLAG, self.buf_get_text()
                    self.buf_add_char(next_char)

            elif next_char == b']':
                if self.flags.string_brackets:
                    # If string_bracket is set (using PROP_FLAG), this is a
                    # syntax error - we don't have an open one to close!
                    raise self._error('No open [] to close with "]"!')
                return BRACK_CLOSE_TUP

            elif next_char == b'(':
                # Parentheses around text...
                self.buf_reset()
                while True:
                    next_char, is_eof = self._next_char()
                    if is_eof:
                        raise self._error('Unterminated parentheses!')
                    if next_char == b'(':
                        raise self._error('Cannot nest () brackets!')
                    elif next_char == b')':
                        return PAREN_ARGS, self.buf_get_text()
                    elif next_char == b'\n':
                        self.line_num += 1
                    self.buf_add_char(next_char)

            elif next_char == b')':
                raise self._error('No open () to close with ")"!')

            # Directives
            elif next_char == b'#':
                self.buf_reset()
                ascii_only = True
                while True:
                    next_char, is_eof = self._next_char()
                    if is_eof:
                        # A directive could be the last value in the file.
                        if ascii_only:
                            return DIRECTIVE, self.buf_get_text()
                        else:
                            return DIRECTIVE, self.buf_get_text().casefold()

                    elif (
                        next_char in BARE_DISALLOWED
                        or (next_char == b':' and self.flags.colon_operator)
                        or (next_char == b'+' and self.flags.plus_operator)
                    ):
                        # We need to repeat this, so we return the ending
                        # char next. If it's not allowed, that'll error on
                        # next call.
                        self.char_index -= 1
                        # And return the directive.
                        if ascii_only:
                            return DIRECTIVE, self.buf_get_text()
                        else:
                            # Have to go through Unicode lowering.
                            return DIRECTIVE, self.buf_get_text().casefold()
                    elif next_char >= 128:
                        # This is non-ASCII, run through the full
                        # Unicode-compliant conversion.
                        ascii_only = False
                        self.buf_add_char(next_char)
                    else:
                        # If ASCII, use bitwise math to convert over.
                        if b'A' <= next_char <= b'Z':
                            self.buf_add_char(next_char + 0x20)
                        else:
                            self.buf_add_char(next_char)

            else:  # These complex checks can't be in a switch, so we need to nest this.
                if next_char == b':' and self.flags.colon_operator:
                    return COLON_TUP
                if next_char == b'+' and self.flags.plus_operator:
                    return PLUS_TUP
                # Bare names
                if next_char not in BARE_DISALLOWED:
                    self.buf_reset()
                    self.buf_add_char(next_char)
                    while True:
                        next_char, is_eof = self._next_char()
                        if is_eof:
                            # Bare names at the end are actually fine.
                            # It could be a value for the last prop.
                            return STRING, self.buf_get_text()

                        elif (
                            next_char in BARE_DISALLOWED
                            or (next_char == b':' and self.flags.colon_operator)
                            or (next_char == b'+' and self.flags.plus_operator)
                        ):  # We need to repeat this so we return the ending
                            # char next. If it's not allowed, that'll error on
                            # next call.
                            self.char_index -= 1
                            return STRING, self.buf_get_text()
                        else:
                            self.buf_add_char(next_char)
                else:
                    # Add in a few more bytes, so we can decode the UTF8 fully.
                    decode = [
                        next_char,
                        self._next_char()[0],
                        self._next_char()[0],
                        self._next_char()[0],
                        0x00,
                    ]
                    raise self._error(f'Unexpected characters "{decode[:4].decode("utf8", "backslashreplace")}"' '!')


cdef class IterTokenizer(BaseTokenizer):
    """Wraps a token iterator to provide the tokenizer interface.

    This is useful to pre-process a token stream before parsing it with other
    code.
    """
    cdef public object source
    def __init__(self, source, filename='', error=None) -> None:
        BaseTokenizer.__init__(self, filename, error)
        self.source = iter(source)

    def __repr__(self):
        if self.error_type is TokenSyntaxError:
            return f'{self.__class__.__name__}({self.source!r}, {self.filename!r})'
        else:
            return f'{self.__class__.__name__}({self.source!r}, {self.filename!r}, {self.error_type!r})'

    def _get_token(self):
        """Compute the next token."""
        return self.next_token()

    cdef next_token(self):
        """Implement pushback directly for efficiency."""
        if self.pushback:
            return self.pushback.pop()

        try:
            return next(self.source)
        except StopIteration:
            return EOF_TUP


# This is entirely internal, users shouldn't access this.
@cython.final
@cython.embedsignature(False)
@cython.internal
cdef class _NewlinesIter:
    """Iterate over the tokens, skipping newlines."""
    cdef BaseTokenizer tok

    def __cinit__(self, BaseTokenizer tok not None):
        self.tok = tok

    def __repr__(self):
        return f'<srctools.tokenizer.BaseTokenizer.skipping_newlines() at {id(self):X}>'

    def __init__(self, tok):
        raise TypeError("Cannot create '_NewlinesIter' instances")

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            tok_and_val = self.tok.next_token()
            token = (<tuple?>tok_and_val)[0]

            # Only our code is doing next_token here, so the tuples are
            # going to be this same instance.
            if token is EOF:
                raise StopIteration
            elif token is not NEWLINE:
                return tok_and_val

    def __reduce__(self):
        """This cannot be pickled - the Python version does not have this class."""
        raise NotImplementedError('Cannot pickle _NewlinesIter!')


@cython.final
@cython.embedsignature(False)
@cython.internal
cdef class BlockIter:
    """Helper iterator for parsing keyvalue style blocks."""
    cdef BaseTokenizer tok
    cdef str name
    cdef bint expect_brace

    def __cinit__(self, BaseTokenizer tok, str name, bint expect_brace, *):
        self.tok = tok
        self.name = name
        self.expect_brace = expect_brace

    def __repr__(self):
        return f'<srctools.tokenizer.BaseTokenizer.block() at {id(self):X}>'

    def __init__(self, tok):
        raise TypeError("Cannot create 'BlockIter' instances directly, use Tokenizer.block().")

    def __iter__(self):
        return self

    def __next__(self):
        if self.expect_brace:
            self.expect_brace = False
            next_token = <tuple> self.tok.next_token()[0]
            while next_token is NEWLINE:
                next_token = <tuple> self.tok.next_token()[0]
            if next_token is not BRACE_OPEN:
                raise self.tok._error(f'Expected BRACE_OPEN, but got {next_token}' '!')

        while True:
            token, value = <tuple>self.tok.next_token()

            if token is EOF:
                raise self.tok._error(f'Unclosed {self.name} block!')
            elif token is STRING:
                return value
            elif token is BRACE_CLOSE:
                raise StopIteration
            elif token is not NEWLINE:
                raise self.tok.error(token, value)

    def __reduce__(self):
        """This cannot be pickled - the Python version does not have this class."""
        raise NotImplementedError('Cannot pickle BlockIter!')


cdef inline Py_ssize_t _write_escape(
    uchar *out_buff,
    Py_ssize_t off,
    uchar symbol,
) noexcept:
    # Escape a single letter.
    out_buff[off] = b'\\'
    off += 1
    out_buff[off] = symbol
    return off


@cython.nonecheck(False)
def escape_text(str text not None: str) -> str:
    r"""Escape special characters and backslashes, so tokenising reproduces them.

    This matches utilbuffer.cpp in the SDK.
    The following characters are escaped: \n, \t, \v, \b, \r, \f, \a, \, /, ?, ', ".
    """
    # UTF8 = ASCII for the chars we care about, so we can just loop over the
    # UTF8 data.
    cdef Py_ssize_t size = 0
    cdef Py_ssize_t final_size = 0
    cdef Py_ssize_t i, j
    cdef uchar letter
    cdef const uchar *in_buf = <const uchar *>PyUnicode_AsUTF8AndSize(text, &size)
    final_size = size

    # First loop to compute the full string length, and check if we need to
    # escape at all.
    for i in range(size):
        if in_buf[i] in b'\n\t\v\b\r\f\a\\?\'"':
            final_size += 1

    if size == final_size:  # Unchanged, return original
        return text

    cdef uchar *out_buff
    j = 0
    try:
        out_buff = <uchar *>PyMem_Malloc(final_size+1 * sizeof(uchar))
        if out_buff is NULL:
            raise MemoryError
        for i in range(size):
            letter = in_buf[i]
            # b'ntvbrfa?\'"'
            if letter == b'\n':
                j = _write_escape(out_buff, j, b'n')
            elif letter == b'\t':
                j = _write_escape(out_buff, j, b't')
            elif letter == b'\v':
                j = _write_escape(out_buff, j, b'v')
            elif letter == b'\b':
                j = _write_escape(out_buff, j, b'b')
            elif letter == b'\r':
                j = _write_escape(out_buff, j, b'r')
            elif letter == b'\f':
                j = _write_escape(out_buff, j, b'f')
            elif letter == b'\a':
                j = _write_escape(out_buff, j, b'a')
            elif letter == b'?':
                j = _write_escape(out_buff, j, b'?')
            elif letter == b'\\':
                j = _write_escape(out_buff, j, b'\\')
            elif letter == b'"':
                j = _write_escape(out_buff, j, b'"')
            elif letter == b"'":
                j = _write_escape(out_buff, j, b"'")
            else:
                out_buff[j] = letter
            j += 1
        out_buff[final_size] = b'\0'
        return PyUnicode_FromStringAndSize(<char *>out_buff, final_size)
    finally:
        PyMem_Free(out_buff)


# This is a replacement for a method in VPK, which is very slow normally
# since it has to accumulate character by character.
cdef class _VPK_IterNullstr:
    """Read a null-terminated ASCII string from the file.

    This continuously yields strings, with empty strings
    indicting the end of a section.
    """
    cdef object file
    cdef uchar *chars
    cdef Py_ssize_t size
    cdef Py_ssize_t used

    def __cinit__(self):
        self.used = 0
        self.size = 64
        self.chars = <uchar *>PyMem_Malloc(self.size)
        if self.chars is NULL:
            raise MemoryError

    def __dealloc__(self):
        PyMem_Free(self.chars)

    def __init__(self, file):
        self.file = file

    def __iter__(self):
        return self

    def __next__(self):
        cdef bytes data
        cdef uchar *temp
        while True:
            data = self.file.read(1)
            if len(data) == 0:
                res = self.chars[:self.used]
                self.used = 0
                raise Exception(f'Reached EOF without null-terminator in {res!r}' '!')
            elif len(data) > 1:
                raise ValueError('Asked to read 1 byte, got multiple?')
            elif (<const char *>data)[0] == 0x00:
                # Blank strings are saved as ' '
                if self.used == 1 and self.chars[0] == b' ':
                    self.used = 0
                    return ''
                if self.used == 0:  # Blank string, this ends the array.
                    self.used = 0
                    raise StopIteration
                else:
                    res = self.chars[:self.used].decode('ascii', 'surrogateescape')
                    self.used = 0
                    return res
            else:
                if self.used == self.size:
                    self.size *= 2
                    temp = <uchar *>PyMem_Realloc(self.chars, self.size)
                    if temp == NULL:
                        raise MemoryError
                    self.chars = temp
                self.chars[self.used] = (<const char *>data)[0]
                self.used += 1


# This is a replacement for _engine_db.make_lookup, it's called an extreme number of times
# and can usefully be optimised.
cdef class _EngineStringTable:
    cdef object read_func
    cdef list inv_list

    def __init__(self, file, list inv_list):
        self.read_func = file.read
        self.inv_list = inv_list

    @cython.optimize.unpack_method_calls(False)
    @cython.wraparound(False)  # Unsigned integers.
    def __call__(self):
        cdef bytes byt = self.read_func(2)
        cdef uchar *buf = byt
        # Unpack an unsigned 16-bit integer.
        cdef uint16_t offset = (buf[0] << 8) | buf[1]
        return self.inv_list[offset]


# Override the tokenizer's name to match the public one.
# This fixes all the methods too, though not in exceptions.

from cpython.object cimport PyTypeObject


cdef extern from *:  # Cython flag indicating if PyTypeObject is safe to access.
    cdef bint USE_TYPE_INTERNALS "CYTHON_USE_TYPE_SLOTS"
if USE_TYPE_INTERNALS:
    (<PyTypeObject *>BaseTokenizer).tp_name = b"srctools.tokenizer.BaseTokenizer"
    (<PyTypeObject *>Tokenizer).tp_name = b"srctools.tokenizer.Tokenizer"
    (<PyTypeObject *>_NewlinesIter).tp_name = b"srctools.tokenizer._skip_newlines_iterator"
try:
    escape_text.__module__ = 'srctools.tokenizer'
except Exception:
    pass  # Perfectly fine.
