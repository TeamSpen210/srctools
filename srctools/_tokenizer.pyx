# cython: language_level=3, embedsignature=True, auto_pickle=False
# cython: binding=True
"""Cython version of the Tokenizer class."""
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdint cimport uint_fast8_t

cdef extern from *:
    ctypedef unsigned char uchar "unsigned char"  # Using it a lot, this causes it to not be a typedef at all.
    const uchar* PyUnicode_AsUTF8AndSize(str string, Py_ssize_t *size) except NULL
    str PyUnicode_FromStringAndSize(const uchar *u, Py_ssize_t size)
    str PyUnicode_FromKindAndData(int kind, const void *buffer, Py_ssize_t size)

cdef object os_fspath
from os import fspath as os_fspath

# Import the Token enum from the Python file, and cache references
# to all the parts.
# Also grab the exception object.

cdef object Token, TokenSyntaxError
from srctools.tokenizer import Token,  TokenSyntaxError

__all__ = ['BaseTokenizer', 'Tokenizer', 'IterTokenizer', 'escape_text']

# Cdef-ed globals become static module vars, which aren't in the module
# dict. We can ensure these are of the type we specify - and are quick to
# lookup.
cdef:
    object STRING = Token.STRING
    object PAREN_ARGS = Token.PAREN_ARGS
    object PROP_FLAG = Token.PROP_FLAG  # [!flag]
    object DIRECTIVE = Token.DIRECTIVE  # #name (automatically casefolded)

    object EOF = Token.EOF
    object NEWLINE = Token.NEWLINE

    object BRACE_OPEN = Token.BRACE_OPEN
    object BRACE_CLOSE = Token.BRACE_CLOSE

    # Reuse a single tuple for these, since the value is constant.
    tuple EOF_TUP = (Token.EOF, '')
    tuple NEWLINE_TUP = (Token.NEWLINE, '\n')

    tuple COLON_TUP = (Token.COLON, ':')
    tuple EQUALS_TUP = (Token.EQUALS, '=')
    tuple PLUS_TUP = (Token.PLUS, '+')
    tuple COMMA_TUP = (Token.COMMA, ',')

    tuple BRACE_OPEN_TUP = (BRACE_OPEN, '{')
    tuple BRACE_CLOSE_TUP = (BRACE_CLOSE, '}')

    tuple BRACK_OPEN_TUP = (Token.BRACK_OPEN, '[')
    tuple BRACK_CLOSE_TUP = (Token.BRACK_CLOSE, ']')

    uchar *EMPTY_BUF = b''  # Initial value, just so it's valid.

# Characters not allowed for bare names on a line.
# Convert to tuple to only check the chars.
DEF BARE_DISALLOWED = b'"\'{};,[]()\n\t '

# Controls what syntax is allowed
DEF FL_STRING_BRACKETS     = 1<<0
DEF FL_ALLOW_ESCAPES       = 1<<1
DEF FL_ALLOW_STAR_COMMENTS = 1<<2
DEF FL_COLON_OPERATOR      = 1<<3
# If set, the file_iter is a bound read() method.
DEF FL_FILE_INPUT          = 1<<4

DEF FILE_BUFFER = 1024
DEF CHR_EOF = 0x03  # Indicate the end of the file.

# noinspection PyMissingTypeHints
cdef class BaseTokenizer:
    """Provides an interface for processing text into tokens.

     It then provides tools for using those to parse data.
     This is an abstract class, a subclass must be used to provide a source
     for the tokens.
    """
    # Class to call when errors occur..
    cdef object error_type

    cdef str filename

    cdef object pushback_tok
    cdef object pushback_val

    cdef public int line_num
    cdef uint_fast8_t flags

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

        self.pushback_tok = self.pushback_val = None
        self.line_num = 1
        self.flags = 0

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
            if len(args) == 1 and (message is STRING or message is PAREN_ARGS or message is PROP_FLAG or message is DIRECTIVE):
                tok_val = <str?>args[0]
                str_msg = f'Unexpected token {message.name}({tok_val})!'
            else:
                str_msg = f'Unexpected token {message.name}' '!'
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
        if self.pushback_tok is not None:
            output = self.pushback_tok, self.pushback_val
            self.pushback_tok = self.pushback_val = None
            return output

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

        Only one token can be pushed back at once.
        The value is required for STRING, PAREN_ARGS and PROP_FLAGS, but ignored
        for other token types.
        """
        if self.pushback_tok is not None:
            raise ValueError('Token already pushed back!')
        if not isinstance(tok, Token):
            raise ValueError(f'{tok!r} is not a Token!')

        # Read this directly to skip the 'value' descriptor.
        cdef int tok_val = tok._value_

        if tok_val == 0: # EOF
            value = ''
        elif tok_val in (1, 3, 4, 10):  # STRING, PAREN_ARGS, DIRECTIVE, PROP_FLAG
            # Value parameter is required.
            if value is None:
                raise ValueError(f'Value required for {tok!r}' '!')
        elif tok_val == 2:  # NEWLINE
            value = '\n'
        elif tok_val == 5:  # BRACE_OPEN
            value = '{'
        elif tok_val == 6:  # BRACE_CLOSE
            value = '}'
        elif tok_val == 11:  # BRACK_OPEN
            value = '['
        elif tok_val == 12:  # BRACK_CLOSE
            value = ']'
        elif tok_val == 13:  # COLON
            value = ':'
        elif tok_val == 14:  # EQUALS
            value = '='
        elif tok_val == 15:  # PLUS
            value = '+'
        elif tok_val == 16: # COMMA
            value = ','
        else:
            raise ValueError(f'Unknown token {tok!r}')

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
        bint string_bracket=False,
        bint allow_escapes=True,
        bint allow_star_comments=False,
        bint colon_operator=False,
    ):
        # Early warning for this particular error.
        if isinstance(data, bytes) or isinstance(data, bytearray):
            raise TypeError(
                'Cannot parse binary data! Decode to the desired encoding, '
                'or wrap in io.TextIOWrapper() to decode gradually.'
            )

        cdef int flags = 0
        if string_bracket:
            flags |= FL_STRING_BRACKETS
        if allow_escapes:
            flags |= FL_ALLOW_ESCAPES
        if allow_star_comments:
            flags |= FL_ALLOW_STAR_COMMENTS
        if colon_operator:
            flags |= FL_COLON_OPERATOR

        # For direct strings, we can immediately assign that as our chunk,
        # and then set the iterable to indicate EOF after that.
        if type(data) is str:
            self.cur_chunk = data
            self.chunk_buf = PyUnicode_AsUTF8AndSize(data, &self.chunk_size)
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
                flags |= FL_FILE_INPUT

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
        self.flags |= flags

        # We want to strip a UTF BOM from the start of the file, if it matches.
        # So pull off the first three characters, and if they don't match,
        # rebuild the cur_chunk to allow them.
        # The BOM is b'\xef\xbb\xbf'.
        if self._next_char() != 0xef:
            self.char_index -= 1
        elif self._next_char() != 0xbb:
            self.char_index -= 2
        elif self._next_char() != 0xbf:
            self.char_index -= 3

    @property
    def string_bracket(self) -> bool:
        """Check if [bracket] blocks are parsed as a single string-like block.

        If disabled these are parsed as BRACK_OPEN, STRING, BRACK_CLOSE.
        """
        return self.flags & FL_STRING_BRACKETS != 0

    @string_bracket.setter
    def string_bracket(self, bint value) -> None:
        """Set if [bracket] blocks are parsed as a single string-like block.

        If disabled these are parsed as BRACK_OPEN, STRING, BRACK_CLOSE.
        """
        if value:
            self.flags |= FL_STRING_BRACKETS
        else:
            self.flags &= ~FL_STRING_BRACKETS

    @property
    def allow_escapes(self) -> bool:
        """Check if backslash escapes will be parsed."""
        return self.flags & FL_ALLOW_ESCAPES != 0

    @allow_escapes.setter
    def allow_escapes(self, bint value) -> None:
        """Set if backslash escapes will be parsed."""
        if value:
            self.flags |= FL_ALLOW_ESCAPES
        else:
            self.flags &= ~FL_ALLOW_ESCAPES

    @property
    def allow_star_comments(self) -> bool:
        """Check if /**/ style comments will be enabled."""
        return self.flags & FL_ALLOW_STAR_COMMENTS != 0

    @allow_star_comments.setter
    def allow_star_comments(self, bint value) -> None:
        """Set if /**/ style comments are enabled."""
        if value:
            self.flags |= FL_ALLOW_STAR_COMMENTS
        else:
            self.flags &= ~FL_ALLOW_STAR_COMMENTS

    @property
    def colon_operator(self) -> bool:
        """Check if : characters are treated as a COLON token, or part of strings."""
        return self.flags & FL_COLON_OPERATOR != 0

    @colon_operator.setter
    def colon_operator(self, bint value) -> None:
        """Set if : characters are treated as a COLON token, or part of strings."""
        if value:
            self.flags |= FL_COLON_OPERATOR
        else:
            self.flags &= ~FL_COLON_OPERATOR

    cdef inline void buf_reset(self):
        """Reset the temporary buffer."""
        # Don't bother resizing or clearing, the next append will overwrite.
        self.buf_pos = 0

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
        out = PyUnicode_FromStringAndSize(self.val_buffer, self.buf_pos)
        # Don't bother resizing or clearing, the next append will overwrite.
        self.buf_pos = 0
        return out

    # We check all the getitem[] accesses, so don't have Cython recheck.
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef uchar _next_char(self) except? CHR_EOF:
        """Return the next character, or 0 if no more characters are there."""
        cdef str chunk
        cdef object chunk_obj

        self.char_index += 1
        if self.char_index < self.chunk_size:
            return self.chunk_buf[self.char_index]

        if self.chunk_iter is None:
            return CHR_EOF

        if self.flags & FL_FILE_INPUT:
            self.cur_chunk = self.chunk_iter(FILE_BUFFER)
            self.char_index = 0

            if type(self.cur_chunk) is str:
                self.chunk_buf = PyUnicode_AsUTF8AndSize(self.cur_chunk, &self.chunk_size)
            else:
                raise ValueError('Expected string, got ' + type(self.cur_chunk).__name__)

            if self.chunk_size > 0:
                return self.chunk_buf[0]
            else:
                self.chunk_iter = None
                return CHR_EOF

        # Retrieve a chunk from the iterable.
        # Skip empty chunks (shouldn't be there.)
        # Use manual next to avoid re-calling iter() here,
        # or using list/tuple optimisations.
        while True:
            try:
                chunk_obj = next(self.chunk_iter, None)
            except UnicodeDecodeError as exc:
                raise self._error("Could not decode file!") from exc
            if chunk_obj is None:
                # Out of characters after empty chunks
                self.chunk_iter = None
                return CHR_EOF

            if isinstance(chunk_obj, bytes):
                raise ValueError('Cannot parse binary data!')
            if type(chunk_obj) is not str:
                raise ValueError("Data was not a string!")

            if len(<str>chunk_obj) > 0:
                self.cur_chunk = chunk_obj
                self.char_index = 0
                self.chunk_buf = PyUnicode_AsUTF8AndSize(self.cur_chunk, &self.chunk_size)
                return self.chunk_buf[0]

    cdef next_token(self):
        """Return the next token, value pair - this is the C version."""
        cdef:
            uchar next_char
            uchar escape_char
            uchar peek_char
            int start_line
            bint ascii_only
            uchar decode[5]

        if self.pushback_tok is not None:
            output = self.pushback_tok, self.pushback_val
            self.pushback_tok = self.pushback_val = None
            return output

        while True:
            next_char = self._next_char()
            if next_char == CHR_EOF:
                return EOF_TUP

            elif next_char == b'{':
                return BRACE_OPEN_TUP
            elif next_char == b'}':
                return BRACE_CLOSE_TUP
            elif next_char == b'+':
                return PLUS_TUP
            elif next_char == b'=':
                return EQUALS_TUP
            elif next_char == b',':
                return COMMA_TUP
            # First try simple operators & EOF.

            elif next_char == b'\n':
                self.line_num += 1
                return NEWLINE_TUP

            elif next_char in b' \t':
                # Ignore whitespace..
                continue

            # Comments
            elif next_char == b'/':
                # The next must be another slash! (//)
                next_char = self._next_char()
                if next_char == b'*': # /* comment.
                    if self.flags & FL_ALLOW_STAR_COMMENTS:
                        start_line = self.line_num
                        while True:
                            next_char = self._next_char()
                            if next_char == CHR_EOF:
                                raise self._error(
                                    f'Unclosed /* comment '
                                    f'(starting on line {start_line})!',
                                )
                            elif next_char == b'\n':
                                self.line_num += 1
                            elif next_char == b'*':
                                # Check next next character!
                                peek_char = self._next_char()
                                if peek_char == CHR_EOF:
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
                        raise self._error(
                            '/**/-style comments are not allowed!'
                        )
                elif next_char == b'/':
                    # Skip to end of line
                    while True:
                        next_char = self._next_char()
                        if next_char == CHR_EOF or next_char == b'\n':
                            break

                    # We want to produce the token for the end character -
                    # EOF or NEWLINE.
                    self.char_index -= 1
                else:
                    raise self._error(
                        'Single slash found, '
                        'instead of two for a comment (// or /* */)!'
                        if self.flags & FL_ALLOW_STAR_COMMENTS else
                        'Single slash found, '
                        'instead of two for a comment (//)!'
                    )

            # Strings
            elif next_char == b'"':
                self.buf_reset()
                while True:
                    next_char = self._next_char()
                    if next_char == CHR_EOF:
                        raise self._error('Unterminated string!')
                    elif next_char == b'"':
                        return STRING, self.buf_get_text()
                    elif next_char == b'\n':
                        self.line_num += 1
                    elif next_char == b'\\' and self.flags & FL_ALLOW_ESCAPES:
                        # Escape text
                        escape_char = self._next_char()
                        if escape_char == CHR_EOF:
                            raise self._error('Unterminated string!')

                        if escape_char == b'n':
                            next_char = b'\n'
                        elif escape_char == b't':
                            next_char = b'\t'
                        elif escape_char == b'\n':
                            # \ at end of line ignores the newline.
                            continue
                        elif escape_char in (b'"', b'\\', b'/'):
                            # For these, we escape to give the literal value.
                            next_char = escape_char
                        else:
                            # For unknown escape_chars, escape the \ automatically.
                            self.buf_add_char(b'\\')
                            self.buf_add_char(escape_char)
                            continue
                            # raise self.error('Unknown escape_char "\\{}" in {}!', escape_char, self.cur_chunk)
                    self.buf_add_char(next_char)

            elif next_char == b'[':
                # FGDs use [] for grouping, Properties use it for flags.
                if not self.flags & FL_STRING_BRACKETS:
                    return BRACK_OPEN_TUP

                self.buf_reset()
                while True:
                    next_char = self._next_char()
                    if next_char == b'[':
                        # Don't allow nesting, that's bad.
                        raise self._error('Cannot nest [] brackets!')
                    elif next_char == b']':
                        return PROP_FLAG, self.buf_get_text()
                    # Must be one line!
                    elif next_char == CHR_EOF or next_char == b'\n':
                        raise self._error(
                            'Reached end of line '
                            'without closing "]"!'
                        )
                    self.buf_add_char(next_char)

            elif next_char == b']':
                if self.flags & FL_STRING_BRACKETS:
                    # If string_bracket is set (using PROP_FLAG), this is a
                    # syntax error - we don't have an open one to close!
                    raise self._error('No open [] to close with "]"!')
                return BRACK_CLOSE_TUP

            elif next_char == b'(':
                # Parentheses around text...
                self.buf_reset()
                while True:
                    next_char = self._next_char()
                    if next_char == CHR_EOF:
                        raise self._error('Unterminated parentheses!')
                    elif next_char == b'(':
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
                    next_char = self._next_char()
                    if next_char == CHR_EOF:
                        # A directive could be the last value in the file.
                        if ascii_only:
                            return DIRECTIVE, self.buf_get_text()
                        else:
                            return DIRECTIVE, self.buf_get_text().casefold()

                    elif (
                        next_char in BARE_DISALLOWED or
                        (next_char == b':' and self.flags & FL_COLON_OPERATOR)
                    ):
                        # We need to repeat this so we return the ending
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
                        # If ASCII, use bit math to convert over.
                        if b'A' <= next_char <= b'Z':
                            self.buf_add_char(next_char + 0x20)
                        else:
                            self.buf_add_char(next_char)

            else:  # These complex checks can't be in a switch, so we need to nest this.
                if next_char == b':' and FL_COLON_OPERATOR & self.flags:
                    return COLON_TUP
                # Bare names
                if next_char not in BARE_DISALLOWED:
                    self.buf_reset()
                    self.buf_add_char(next_char)
                    while True:
                        next_char = self._next_char()
                        if next_char == CHR_EOF:
                            # Bare names at the end are actually fine.
                            # It could be a value for the last prop.
                            return STRING, self.buf_get_text()

                        elif (
                            next_char in BARE_DISALLOWED or
                            (next_char == b':' and FL_COLON_OPERATOR & self.flags)
                        ):  # We need to repeat this so we return the ending
                            # char next. If it's not allowed, that'll error on
                            # next call.
                            # We need to repeat this so we return the newline.
                            self.char_index -= 1
                            return STRING, self.buf_get_text()
                        else:
                            self.buf_add_char(next_char)
                else:
                    # Add in a few more bytes so we can decode the UTF8 fully.
                    decode = [
                        next_char,
                        self._next_char(),
                        self._next_char(),
                        self._next_char(),
                        0x00,
                    ]
                    raise self._error(f'Unexpected character "{decode[:4].decode("utf8", "backslashreplace")}"' '!')


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
            return f'IterTokenizer({self.source!r}, {self.filename!r})'
        else:
            return f'IterTokenizer({self.source!r}, {self.filename!r}, {self.error_type!r})'

    cdef next_token(self):
        if self.pushback_tok is not None:
            output = self.pushback_tok, self.pushback_val
            self.pushback_tok = self.pushback_val = None
            return output

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
        raise TypeError("Cannot create 'BlockIter' instances")

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


@cython.nonecheck(False)
def escape_text(str text not None: str) -> str:
    r"""Escape special characters and backslashes, so tokenising reproduces them.

    Specifically, \, ", tab, and newline.
    """
    # UTF8 = ASCII for the chars we care about, so we can just loop over the
    # UTF8 data.
    cdef Py_ssize_t size = 0
    cdef Py_ssize_t final_size = 0
    cdef int i, j
    cdef uchar letter
    cdef const uchar *in_buf = PyUnicode_AsUTF8AndSize(text, &size)
    final_size = size

    # First loop to compute the full string length, and check if we need to
    # escape at all.
    for i in range(size):
        if in_buf[i] in b'\\"\t\n':
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
            if letter == b'\\':
                out_buff[j] = b'\\'
                j += 1
                out_buff[j] = b'\\'
            elif letter == b'"':
                out_buff[j] = b'\\'
                j += 1
                out_buff[j] = b'"'
            elif letter == b'\t':
                out_buff[j] = b'\\'
                j += 1
                out_buff[j] = b't'
            elif letter == b'\n':
                out_buff[j] = b'\\'
                j += 1
                out_buff[j] = b'n'
            else:
                out_buff[j] = letter
            j += 1
        out_buff[final_size] = b'\0'
        return PyUnicode_FromStringAndSize(out_buff, final_size)
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
