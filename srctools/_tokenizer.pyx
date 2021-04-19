# cython: language_level=3, embedsignature=True, auto_pickle=False
# cython: binding=True
"""Cython version of the Tokenizer class."""
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef extern from *:
    unicode PyUnicode_FromStringAndSize(const char *u, Py_ssize_t size)
    unicode PyUnicode_FromKindAndData(int kind, const void *buffer, Py_ssize_t size)

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

    tuple BRACE_OPEN_TUP = (BRACE_OPEN, '{')
    tuple BRACE_CLOSE_TUP = (BRACE_CLOSE, '}')

    tuple BRACK_OPEN_TUP = (Token.BRACK_OPEN, '[')
    tuple BRACK_CLOSE_TUP = (Token.BRACK_CLOSE, ']')


# Characters not allowed for bare names on a line.
# Convert to tuple to only check the chars.
DEF BARE_DISALLOWED = tuple('"\'{};:[]()\n\t ')

# Controls what syntax is allowed
DEF FL_STRING_BRACKETS     = 0b0001
DEF FL_ALLOW_ESCAPES       = 0b0010
DEF FL_ALLOW_STAR_COMMENTS = 0b0100
# If set, the file_iter is a bound read() method.
DEF FL_FILE_INPUT          = 0b1000

DEF FILE_BUFFER = 1024

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
    cdef inline _error(self, str message):
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
        # Call ourselves until EOF is returned
        return iter(self, EOF_TUP)

    def push_back(self, object tok not None, str value=None):
        """Return a token, so it will be reproduced when called again.

        Only one token can be pushed back at once.
        The value is required for STRING, PAREN_ARGS and PROP_FLAGS, but ignored
        for other token types.
        """
        if self.pushback_tok is not None:
            raise ValueError('Token already pushed back!')
        if not isinstance(tok, Token):
            raise ValueError(repr(tok) + ' is not a Token!')

        # Read this directly to skip the 'value' descriptor.
        cdef int tok_val = tok._value_
        cdef str real_value

        if tok_val == 0: # EOF
            real_value = ''
        elif tok_val in (1, 3, 4, 10):  # STRING, PAREN_ARGS, DIRECTIVE, PROP_FLAG
            # The value can be anything, so just accept this.
            self.pushback_tok = tok
            self.pushback_val = value
            return
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
            raise ValueError(f'Unknown token {tok!r}')

        if value is None:
            raise ValueError(f'Value required for {tok!r}' '!') from None

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
    cdef str cur_chunk
    cdef object chunk_iter
    cdef int char_index # Position inside cur_chunk

    cdef int flags

    # Private buffer, to hold string parts we're constructing.
    # Tokenizers are expected to be temporary, so we just never shrink.
    cdef Py_ssize_t buf_size  # 2 << x
    cdef Py_ssize_t buf_pos
    cdef Py_UCS4* val_buffer

    def __cinit__(self):
        self.buf_size = 128
        self.val_buffer = <Py_UCS4 *>PyMem_Malloc(self.buf_size * sizeof(Py_UCS4))
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
    ):
        # Early warning for this particular error.
        if isinstance(data, bytes) or isinstance(data, bytearray):
            raise TypeError(
                'Cannot parse binary data! Decode to the desired encoding, '
                'or wrap in io.TextIOWrapper() to decode gradually.'
            )
            
        self.flags = (
            FL_STRING_BRACKETS * string_bracket |
            FL_ALLOW_ESCAPES * allow_escapes |
            FL_ALLOW_STAR_COMMENTS * allow_star_comments |
            0
        )

        # For direct strings, we can immediately assign that as our chunk,
        # and then set the iterable to indicate EOF after that.
        if isinstance(data, str):
            self.cur_chunk = data
            self.chunk_iter = None
        else:
            # The first next_char() call will pull out a chunk.
            self.cur_chunk = ''
            
            # If a file, use the read method to pull bulk data.
            try:
                self.chunk_iter = data.read
            except AttributeError:
                # This checks that it is indeed iterable.
                self.chunk_iter = iter(data)
            else:
                self.flags |= FL_FILE_INPUT

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

    cdef inline void buf_reset(self):
        """Reset the temporary buffer."""
        # Don't bother resizing or clearing, the next append will overwrite.
        self.buf_pos = 0

    cdef inline int buf_add_char(self, Py_UCS4 uchar) except -1:
        """Add a character to the temporary buffer, reallocating if needed."""
        # Temp, so if memory alloc failure occurs we're still in a valid state.
        cdef Py_UCS4 *newbuf
        cdef Py_ssize_t new_size
        if self.buf_pos >= self.buf_size:
            new_size = self.buf_size * 2
            new_buf = <Py_UCS4 *>PyMem_Realloc(
                self.val_buffer,
                new_size * sizeof(Py_UCS4),
            )
            if new_buf:
                self.buf_size = new_size
                self.val_buffer = new_buf
            else:
                raise MemoryError

        self.val_buffer[self.buf_pos] = uchar
        self.buf_pos += 1

    cdef object buf_get_text(self):
        """Decode the buffer, and return the text."""
        # Convert the buffer directly to a string. 4 = UCS4 mode.
        out = PyUnicode_FromKindAndData(4, self.val_buffer, self.buf_pos)
        # Don't bother resizing or clearing, the next append will overwrite.
        self.buf_pos = 0
        return out

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
            
        if self.chunk_iter is None:
            return -1  # EOF
            
        if self.flags & FL_FILE_INPUT:
            self.cur_chunk = <str?>self.chunk_iter(FILE_BUFFER)
            self.char_index = 0

            if len(self.cur_chunk) > 0:
                return self.cur_chunk[0]
            else:
                self.chunk_iter = None
                return -1

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
                return -1

            if isinstance(chunk_obj, bytes):
                raise ValueError('Cannot parse binary data!')
            if not isinstance(chunk_obj, str):
                raise ValueError("Data was not a string!")

            if len(<str ?>chunk_obj) > 0:
                self.cur_chunk = <str>chunk_obj
                self.char_index = 0
                return (<str>chunk_obj)[0]

    cdef next_token(self):
        """Return the next token, value pair - this is the C version."""
        cdef:
            Py_UCS4 next_char
            Py_UCS4 escape_char
            Py_UCS4 peek_char
            int start_line

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
                next_char = self._next_char()
                if next_char == '*': # /* comment.
                    if self.flags & FL_ALLOW_STAR_COMMENTS:
                        start_line = self.line_num
                        while True:
                            next_char = self._next_char()
                            if next_char == -1:
                                raise self._error(
                                    f'Unclosed /* comment '
                                    f'(starting on line {start_line})!',
                                )
                            elif next_char == '\n':
                                self.line_num += 1
                            elif next_char == '*':
                                # Check next next character!
                                peek_char = self._next_char()
                                if peek_char == -1:
                                    raise self._error(
                                        f'Unclosed /* comment '
                                        f'(starting on line {start_line})!',
                                    )
                                elif peek_char == '/':
                                    break
                                else:
                                    # We need to reparse this, to ensure
                                    # "**/" parses correctly!
                                    self.char_index -= 1
                    else:
                        raise self._error(
                            '/**/-style comments are not allowed!'
                        )
                elif next_char == '/':
                    # Skip to end of line
                    while True:
                        next_char = self._next_char()
                        if next_char == -1 or next_char == '\n':
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
            elif next_char == '"':
                self.buf_reset()
                while True:
                    next_char = self._next_char()
                    if next_char == -1:
                        raise self._error('Unterminated string!')
                    elif next_char == '"':
                        return STRING, self.buf_get_text()
                    elif next_char == '\n':
                        self.line_num += 1
                    elif next_char == '\\' and self.flags & FL_ALLOW_ESCAPES:
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
                            self.buf_add_char('\\')
                            self.buf_add_char(escape_char)
                            continue
                            # raise self.error('Unknown escape_char "\\{}" in {}!', escape_char, self.cur_chunk)
                    self.buf_add_char(next_char)

            elif next_char == '[':
                # FGDs use [] for grouping, Properties use it for flags.
                if not self.flags & FL_STRING_BRACKETS:
                    return BRACK_OPEN_TUP

                self.buf_reset()
                while True:
                    next_char = self._next_char()
                    if next_char == '[':
                        # Don't allow nesting, that's bad.
                        raise self._error('Cannot nest [] brackets!')
                    elif next_char == ']':
                        return PROP_FLAG, self.buf_get_text()
                    # Must be one line!
                    elif next_char == '\n' or next_char == -1:
                        raise self._error(
                            'Reached end of line '
                            'without closing "]"!'
                        )
                    self.buf_add_char(next_char)

            elif next_char == ']':
                if self.flags & FL_STRING_BRACKETS:
                    # If string_bracket is set (using PROP_FLAG), this is a
                    # syntax error - we don't have an open one to close!
                    raise self._error('No open [] to close with "]"!')
                return BRACK_CLOSE_TUP

            elif next_char == '(':
                # Parentheses around text...
                self.buf_reset()
                while True:
                    next_char = self._next_char()
                    if next_char == -1:
                        raise self._error('Unterminated parentheses!')
                    elif next_char == '(':
                        raise self._error('Cannot nest () brackets!')
                    elif next_char == ')':
                        return PAREN_ARGS, self.buf_get_text()
                    elif next_char == '\n':
                        self.line_num += 1
                    self.buf_add_char(next_char)

            elif next_char == ')':
                raise self._error('No open () to close with ")"!')

            # Directives
            elif next_char == '#':
                self.buf_reset()
                while True:
                    next_char = self._next_char()
                    if next_char == -1:
                        # A directive could be the last value in the file.
                        return DIRECTIVE, self.buf_get_text()

                    elif next_char in BARE_DISALLOWED:
                        # We need to repeat this so we return the ending
                        # char next. If it's not allowed, that'll error on
                        # next call.
                        self.char_index -= 1
                        return DIRECTIVE, self.buf_get_text()
                    else:
                        # Lower() is far cheaper, but only valid for ASCII.
                        if next_char < 128:
                            self.buf_add_char(next_char.lower())
                        else:
                            # Might result in multiple output characters.
                            for next_char in <str>next_char.casefold():
                                self.buf_add_char(next_char)

            # Ignore Unicode Byte Order Mark on first lines
            elif next_char == '\uFEFF':
                if self.line_num == 1:
                    continue
                # else, we fall out of the if, and get an unexpected char
                # error.

            else: # Not-in can't be in a switch, so we need to nest this.
                # Bare names
                if next_char not in BARE_DISALLOWED:
                    self.buf_reset()
                    self.buf_add_char(next_char)
                    while True:
                        next_char = self._next_char()
                        if next_char == -1:
                            # Bare names at the end are actually fine.
                            # It could be a value for the last prop.
                            return STRING, self.buf_get_text()

                        elif next_char in BARE_DISALLOWED:
                            # We need to repeat this so we return the ending
                            # char next. If it's not allowed, that'll error on
                            # next call.
                            # We need to repeat this so we return the newline.
                            self.char_index -= 1
                            return STRING, self.buf_get_text()
                        else:
                            self.buf_add_char(next_char)
                else:
                    raise self._error(f'Unexpected character "{next_char}"' '!')


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
    # First loop to compute the full string length, and check if we need to 
    # escape at all.
    cdef Py_ssize_t final_len = 0
    cdef Py_UCS4 str_letter
    for str_letter in text:
        if str_letter in ('\\', '"', '\t', '\n'):
            final_len += 1

    if final_len == 0:  # Unchanged, return original
        return text

    # UTF8 = ASCII for the chars we care about, so we can replace in that form.
    cdef bytes enc_text = text.encode('utf8')
    final_len += len(enc_text)

    cdef char * out_buff
    cdef char byt_letter
    cdef int i = 0
    try:
        out_buff = <char *>PyMem_Malloc(final_len+1)
        if out_buff is NULL:
            raise MemoryError
        for byt_letter in enc_text:
            if byt_letter == b'\\':
                out_buff[i] = b'\\'
                i += 1
                out_buff[i] = b'\\'
            elif byt_letter == b'"':
                out_buff[i] = b'\\'
                i += 1
                out_buff[i] = b'"'
            elif byt_letter == b'\t':
                out_buff[i] = b'\\'
                i += 1
                out_buff[i] = b't'
            elif byt_letter == b'\n':
                out_buff[i] = b'\\'
                i += 1
                out_buff[i] = b'n'
            else:
                out_buff[i] = byt_letter
            i += 1
        out_buff[final_len] = b'\0'
        return PyUnicode_FromStringAndSize(out_buff, final_len)
    finally:
        PyMem_Free(out_buff)

cdef extern from *:  # Allow ourselves to access one of the feature flag macros.
    cdef bint USE_TYPE_INTERNALS "CYTHON_USE_TYPE_SLOTS"

# Override the tokenizer's name to match the public one.
# This fixes all the methods too, though not in exceptions.
from cpython.object cimport PyTypeObject
if USE_TYPE_INTERNALS:
    (<PyTypeObject *>BaseTokenizer).tp_name = b"srctools.tokenizer.BaseTokenizer"
    (<PyTypeObject *>Tokenizer).tp_name = b"srctools.tokenizer.Tokenizer"
    (<PyTypeObject *>_NewlinesIter).tp_name = b"srctools.tokenizer.BaseTokenizer.skipping_newlines"
    escape_text.__module__ = 'srctools.tokenizer'
