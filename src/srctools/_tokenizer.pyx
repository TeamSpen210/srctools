# cython: language_level=3, embedsignature=True, auto_pickle=False
# cython: binding=True
"""Cython version of the Tokenizer class."""
from cpython.mem cimport PyMem_Free, PyMem_Malloc, PyMem_Realloc
from libc.stdint cimport uint_fast8_t, uint16_t, uint_fast32_t
cimport cython

cdef object os_fspath
from os import fspath as os_fspath

from .pythoncapi_compat cimport (
    PyUnicodeWriter, PyUnicodeWriter_Discard, PyUnicodeWriter_Create, PyUnicodeWriter_Finish,
    PyUnicodeWriter_WriteChar, PyUnicodeWriter_WriteASCII,
)

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

    object PAREN_OPEN = Token.PAREN_OPEN
    object PAREN_CLOSE = Token.PAREN_CLOSE

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

    tuple PAREN_OPEN_TUP = (PAREN_OPEN, '(')
    tuple PAREN_CLOSE_TUP = (PAREN_CLOSE, ')')

# Characters not allowed for bare names on a line.
# TODO: Make this an actual constant value, but that won't do the switch optimisation.
DEF BARE_DISALLOWED = b'"\'{};,=[]()\r\n\t '

# Pack flags into a bitfield.
cdef extern from *:
    """
struct TokFlags {
    unsigned char string_brackets: 1;
    unsigned char string_parens: 1;
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
        bint string_parens
        bint allow_escapes
        bint colon_operator
        bint plus_operator
        bint allow_star_comments
        bint preserve_comments
        # If set, the file_iter is a bound read() method.
        bint file_input
        # Records if the last character was a \r, so the next swallows \n.
        bint last_was_cr

# The number of characters to read from a file each time. Just an object, we're passing to Python
# code.
cdef object FILE_BUFFER = 1024

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
            'string_parens': 1,
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
            # Unroll all these so the strings are cached, and this hopefully is just a lookup.
            elif message is BRACE_OPEN:
                str_msg = 'Unexpected "{" character!'
            elif message is BRACE_CLOSE:
                str_msg = 'Unexpected "}" character!'
            elif message is BRACK_OPEN:
                str_msg = 'Unexpected "[" character!'
            elif message is BRACK_CLOSE:
                str_msg = 'Unexpected "]" character!'
            elif message is PAREN_OPEN:
                str_msg = 'Unexpected "(" character!'
            elif message is PAREN_CLOSE:
                str_msg = 'Unexpected ")" character!'
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
        elif tok_val == 8:  # PAREN_OPEN
            value = '('
        elif tok_val == 9:  # PAREN_CLOSE
            value = ')'
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

    def peek(self, bint consume_newlines: bool = False):
        """Peek at the next token, without removing it from the stream.

        :param consume_newlines: Skip over newlines until a non-newline is found.
               All tokens are preserved.
        """
        cdef tuple tok_and_val = <tuple?>self.next_token()
        cdef list tokens
        if consume_newlines and tok_and_val[0] is NEWLINE:
            tokens = [tok_and_val]
            while True:
                tok_and_val = <tuple?>self.next_token()
                tokens.append(tok_and_val)
                if tok_and_val[0] is not NEWLINE:
                    tokens.reverse()
                    self.pushback.extend(tokens)
                    return tok_and_val
        else:
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
    # chunk_iter is either a file.read() method or iterator object.
    # cur_chunk is the current string part, keeping that alive.
    cdef str cur_chunk
    cdef object chunk_iter
    cdef object _periodic_callback
    cdef Py_ssize_t char_index # Position inside cur_chunk

    # The in-progress string we're constructing, or NULL when we finished.
    cdef PyUnicodeWriter* writer

    def __cinit__(self):
        self.writer = NULL

    def __dealloc__(self):
        if self.writer != NULL:
            # The compat version needs the null check.
            PyUnicodeWriter_Discard(self.writer)
            self.writer = NULL

    def __init__(
        self,
        data not None,
        object filename=None,
        error=None,
        *,
        object periodic_callback = None,
        bint string_bracket: bool = False,
        bint string_parens: bool = True,
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
            'string_parens': string_parens,
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
                flags.file_input = True

        if periodic_callback is None or callable(periodic_callback):
            self._periodic_callback = periodic_callback
        else:
            raise TypeError(f"periodic_callback must be a callable or None, got {periodic_callback!r}")

        # We initially add one, so it'll be 0 next.
        self.char_index = -1

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
        # So read the first, then backtrack if it's not a BOM.
        first_char, starts_eof = self._next_char()
        if starts_eof or first_char != 0xFEFF:
            self.char_index -= 1

    @property
    def periodic_callback(self):
        """If set, is called periodically when parsing lines. Useful to abort parsing operations."""
        return self._periodic_callback

    @periodic_callback.setter
    def periodic_callback(self, callback):
        if callback is None or callable(callback):
            self._periodic_callback = callback
        raise TypeError(f"periodic_callback must be a callable or None, got {callback!r}")

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
    def string_parens(self) -> bool:
        """Controls whether (parenthesed) blocks are parsed as a single string-like block.

        If disabled these are parsed as PAREN_OPEN, STRING, PAREN_CLOSE.
        """
        return self.flags.string_parens

    @string_parens.setter
    def string_parens(self, bint value) -> None:
        self.flags.string_parens = value

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

    cdef inline bint writer_reset(self) except False:
        """Reset the writer."""
        if self.writer != NULL:
            # The compat version needs the null check.
            PyUnicodeWriter_Discard(self.writer)
            self.writer = NULL  # If Create() raises, ensure we don't have a dead writer here.
        self.writer = PyUnicodeWriter_Create(0)
        return True

    cdef str writer_get(self):
        """Fetch the result from the buffer, resetting."""
        assert self.writer != NULL
        out = PyUnicodeWriter_Finish(self.writer)
        self.writer = NULL
        return out

    # We check all the getitem[] accesses, so don't have Cython recheck.
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef (Py_UCS4, bint) _next_char(self):
        """Return the next character, and a flag to indicate if more characters are present."""
        cdef str chunk
        cdef object chunk_obj

        self.char_index += 1
        if self.char_index < len(self.cur_chunk):
            return self.cur_chunk[self.char_index], False

        if self.chunk_iter is None:
            return '\0', True

        if self.flags.file_input:
            try:
                chunk_obj = self.chunk_iter(FILE_BUFFER)
            except UnicodeDecodeError as exc:
                raise self._error(f"Could not decode file:\n{exc!s}") from exc

            if isinstance(chunk_obj, bytes):
                raise TypeError('Cannot parse binary data!')
            if type(chunk_obj) is not str:
                raise TypeError(f'Expected string, got {type(chunk_obj).__name__}')

            self.cur_chunk = chunk_obj
            self.char_index = 0

            if len(self.cur_chunk) > 0:
                return self.cur_chunk[0], False
            else:  # read() returning blank = EOF.
                self.chunk_iter = None
                return '\0', True

        # Retrieve a chunk from the iterable.
        # Skip empty chunks (shouldn't be there.)
        # Use manual next to avoid re-calling iter() here,
        # or using list/tuple optimisations.
        while True:
            try:
                chunk_obj = next(self.chunk_iter)
            except UnicodeDecodeError as exc:
                raise self._error(f"Could not decode file:\n{exc!s}") from exc
            except StopIteration:
                # Out of characters after empty chunks
                self.chunk_iter = None
                return  '\0', True

            if isinstance(chunk_obj, bytes):
                raise TypeError('Cannot parse binary data!')
            if type(chunk_obj) is not str:
                raise TypeError(f'Expected string, got {type(chunk_obj).__name__}')

            if len(<str>chunk_obj) > 0:
                self.cur_chunk = chunk_obj
                self.char_index = 0
                return self.cur_chunk[0], False

    def _get_token(self):
        """Compute the next token."""
        return self.next_token()

    cdef _inc_line_number(self):
        """Increment the line number count."""
        self.line_num += 1
        if self._periodic_callback is not None and self.line_num % 10 == 0:
            self._periodic_callback()

    cdef next_token(self):
        """Return the next token, value pair - this is the C version."""
        cdef:
            Py_UCS4 next_char
            Py_UCS4 escape_char
            Py_UCS4 peek_char
            bint is_eof
            int start_line
            bint ascii_only
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

            if next_char == '{':
                return BRACE_OPEN_TUP
            elif next_char == '}':
                return BRACE_CLOSE_TUP
            elif next_char == '=':
                return EQUALS_TUP
            elif next_char == ',':
                return COMMA_TUP

            # Handle newlines, converting \r and \r\n to \n.
            if next_char == '\r':
                self._inc_line_number()
                self.flags.last_was_cr = True
                return NEWLINE_TUP
            elif next_char == '\n':
                # Consume the \n in \r\n.
                if self.flags.last_was_cr:
                    self.flags.last_was_cr = False
                    continue
                self._inc_line_number()
                return NEWLINE_TUP
            else:
                self.flags.last_was_cr = False

            if next_char in ' \t':
                # Ignore whitespace...
                continue

            # Comments
            elif next_char == '/':
                # The next must be another slash! (//)
                next_char, is_eof = self._next_char()
                if next_char == '*': # /* comment.
                    if self.flags.allow_star_comments:
                        start_line = self.line_num
                        save_comments = self.flags.preserve_comments
                        if save_comments:
                            self.writer_reset()
                        while True:
                            next_char, is_eof = self._next_char()
                            if is_eof:
                                raise self._error(
                                    f'Unclosed /* comment '
                                    f'(starting on line {start_line})!',
                                )
                            if next_char == '\n':
                                self._inc_line_number()
                                if save_comments:
                                    PyUnicodeWriter_WriteChar(self.writer, next_char)
                            elif next_char == '*':
                                # Check next, next character!
                                peek_char, is_eof = self._next_char()
                                if is_eof:
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
                                if save_comments:
                                    PyUnicodeWriter_WriteChar(self.writer, next_char)
                        if save_comments:
                            return COMMENT, self.writer_get()
                    else:
                        raise self._error(
                            '/**/-style comments are not allowed!'
                        )
                elif next_char == '/':
                    # Skip to end of line.
                    save_comments = self.flags.preserve_comments
                    if save_comments:
                        self.writer_reset()
                    while True:
                        next_char, is_eof = self._next_char()
                        if is_eof or next_char == '\n':
                            break
                        if save_comments:
                            PyUnicodeWriter_WriteChar(self.writer, next_char)

                    # We want to produce the token for the end character -
                    # EOF or NEWLINE.
                    self.char_index -= 1
                    if save_comments:
                        return COMMENT, self.writer_get()
                else:
                    raise self._error(
                        'Single slash found, '
                        'instead of two for a comment (// or /* */)!'
                        if self.flags.allow_star_comments else
                        'Single slash found, '
                        'instead of two for a comment (//)!'
                    )

            # Strings
            elif next_char == '"':
                self.writer_reset()
                last_was_cr = False
                while True:
                    next_char, is_eof = self._next_char()
                    if is_eof:
                        raise self._error('Unterminated string!')
                    if next_char == '"':
                        return STRING, self.writer_get()
                    elif next_char == '\r':
                        self._inc_line_number()
                        last_was_cr = True
                        PyUnicodeWriter_WriteChar(self.writer, '\n')
                        continue
                    elif next_char == '\n':
                        if last_was_cr:
                            last_was_cr = False
                            continue
                        self._inc_line_number()
                    else:
                        last_was_cr = False

                    if next_char == '\\' and self.flags.allow_escapes:
                        # Escape text
                        escape_char, is_eof = self._next_char()
                        if is_eof:
                            raise self._error('Unterminated string!')

                        # See this code:
                        # https://github.com/ValveSoftware/source-sdk-2013/blob/0d8dceea4310fde5706b3ce1c70609d72a38efdf/sp/src/tier1/utlbuffer.cpp#L57-L69
                        if escape_char == 'a':
                            next_char = '\a'
                        elif escape_char == 'b':
                            next_char = '\b'
                        elif escape_char == 't':
                            next_char = '\t'
                        elif escape_char == 'n':
                            next_char = '\n'
                        elif escape_char == 'v':
                            next_char = '\v'
                        elif escape_char == 'f':
                            next_char = '\f'
                        elif escape_char == 'r':
                            next_char = '\r'
                        elif escape_char == '\n':
                            # \ at end of line ignores the newline.
                            continue
                        elif escape_char in ('"', '\\', '/', "'", '?'):
                            # For these, we escape to give the literal value.
                            next_char = escape_char
                        else:
                            # For unknown escape_chars, escape the \ automatically.
                            PyUnicodeWriter_WriteChar(self.writer, '\\')
                            PyUnicodeWriter_WriteChar(self.writer, escape_char)
                            continue
                            # raise self.error('Unknown escape_char "\\{}" in {}!', escape_char, self._cur_chunk)
                    PyUnicodeWriter_WriteChar(self.writer, next_char)

            elif next_char == '[':
                # FGDs use [] for grouping, Properties use it for flags.
                if not self.flags.string_brackets:
                    return BRACK_OPEN_TUP

                self.writer_reset()
                while True:
                    next_char, is_eof = self._next_char()
                    # Must be one line!
                    if is_eof or next_char == '\n':
                        raise self._error(
                            'Reached end of line '
                            'without closing "]"!'
                        )
                    if next_char == '[':
                        # Don't allow nesting, that's bad.
                        raise self._error('Cannot nest [] brackets!')
                    elif next_char == ']':
                        return PROP_FLAG, self.writer_get()
                    PyUnicodeWriter_WriteChar(self.writer, next_char)

            elif next_char == ']':
                if self.flags.string_brackets:
                    # If string_bracket is set (using PROP_FLAG), this is a
                    # syntax error - we don't have an open one to close!
                    raise self._error('No open [] to close with "]"!')
                return BRACK_CLOSE_TUP

            elif next_char == '(':
                # Parentheses around text...
                # Some code might want to parse this individually.
                if not self.flags.string_parens:
                    return PAREN_OPEN_TUP

                self.writer_reset()
                while True:
                    next_char, is_eof = self._next_char()
                    if is_eof:
                        raise self._error('Unterminated parentheses!')
                    if next_char == '(':
                        raise self._error('Cannot nest () brackets!')
                    elif next_char == ')':
                        return PAREN_ARGS, self.writer_get()
                    elif next_char == '\n':
                        self._inc_line_number()
                    PyUnicodeWriter_WriteChar(self.writer, next_char)

            elif next_char == ')':
                if self.flags.string_parens:
                    # If string_parens is set (using PAREN_ARGS), this is a
                    # syntax error - we don't have an open one to close!
                    raise self._error('No open () to close with ")"!')
                return PAREN_CLOSE_TUP

            # Directives
            elif next_char == '#':
                self.writer_reset()
                # If it's entirely ascii, we can do the conversion inline, skip calling
                # unicode logic.
                ascii_only = True
                while True:
                    next_char, is_eof = self._next_char()
                    if is_eof:
                        # A directive could be the last value in the file.
                        if ascii_only:
                            return DIRECTIVE, self.writer_get()
                        else:
                            return DIRECTIVE, self.writer_get().casefold()

                    elif (
                        next_char in BARE_DISALLOWED
                        or (next_char == ':' and self.flags.colon_operator)
                        or (next_char == '+' and self.flags.plus_operator)
                    ):
                        # We need to repeat this, so we return the ending
                        # char next. If it's not allowed, that'll error on
                        # next call.
                        self.char_index -= 1
                        # And return the directive.
                        if ascii_only:
                            return DIRECTIVE, self.writer_get()
                        else:
                            # Have to go through Unicode lowering.
                            return DIRECTIVE, self.writer_get().casefold()
                    elif next_char >= 128:
                        # This is non-ASCII, run through the full
                        # Unicode-compliant conversion.
                        ascii_only = False
                        PyUnicodeWriter_WriteChar(self.writer, next_char)
                    else:
                        # If ASCII, use bitwise math to convert over.
                        if 'A' <= next_char <= 'Z':
                            PyUnicodeWriter_WriteChar(self.writer, (<uint_fast32_t>next_char + 0x20))
                        else:
                            PyUnicodeWriter_WriteChar(self.writer, next_char)

            else:  # These complex checks can't be in a switch, so we need to nest this.
                if next_char == ':' and self.flags.colon_operator:
                    return COLON_TUP
                if next_char == '+' and self.flags.plus_operator:
                    return PLUS_TUP
                # Bare names
                if next_char not in BARE_DISALLOWED:
                    self.writer_reset()
                    PyUnicodeWriter_WriteChar(self.writer, next_char)
                    while True:
                        next_char, is_eof = self._next_char()
                        if is_eof:
                            # Bare names at the end are actually fine.
                            # It could be a value for the last prop.
                            return STRING, self.writer_get()

                        elif (
                            next_char in BARE_DISALLOWED
                            or (next_char == ':' and self.flags.colon_operator)
                            or (next_char == '+' and self.flags.plus_operator)
                        ):  # We need to repeat this so we return the ending
                            # char next. If it's not allowed, that'll error on
                            # next call.
                            self.char_index -= 1
                            return STRING, self.writer_get()
                        else:
                            PyUnicodeWriter_WriteChar(self.writer, next_char)
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
    Py_UCS4 *out_buff,
    Py_ssize_t off,
    Py_UCS4 symbol,
) noexcept:
    # Escape a single letter.
    out_buff[off] = '\\'
    off += 1
    out_buff[off] = symbol
    return off


@cython.nonecheck(False)
def escape_text(str text not None: str, bint multiline: bool = False) -> str:
    r"""Escape special characters and backslashes, so tokenising reproduces them.

    This matches utilbuffer.cpp in the SDK.
    The following characters are escaped: \t, \v, \b, \r, \f, \a, \, ', ".
    / and ? are accepted as escapes, but not produced since they're unambiguous.
    In addition, \n is escaped only if `multiline` is false.
    """
    cdef Py_ssize_t size = len(text)
    cdef Py_ssize_t final_size = size
    cdef Py_UCS4 letter

    # First loop to compute the full string length, and check if we need to
    # escape at all.
    for letter in text:
        if letter in '\t\v\b\r\f\a\\\'"':
            final_size += 1
        elif letter == '\n' and not multiline:
            final_size += 1

    if size == final_size:  # Unchanged, return original
        return text

    cdef PyUnicodeWriter* writer = PyUnicodeWriter_Create(final_size)
    try:
        for letter in text:
            # b'tvbrfa?\'"'
            if letter == '\t':
                PyUnicodeWriter_WriteASCII(writer, b"\\t", 2)
            elif letter == '\v':
                PyUnicodeWriter_WriteASCII(writer, b"\\v", 2)
            elif letter == '\b':
                PyUnicodeWriter_WriteASCII(writer, b"\\b", 2)
            elif letter == '\r':
                PyUnicodeWriter_WriteASCII(writer, b"\\r", 2)
            elif letter == '\f':
                PyUnicodeWriter_WriteASCII(writer, b"\\f", 2)
            elif letter == '\a':
                PyUnicodeWriter_WriteASCII(writer, b"\\a", 2)
            elif letter == '\\':
                PyUnicodeWriter_WriteASCII(writer, b"\\\\", 2)
            elif letter == '"':
                PyUnicodeWriter_WriteASCII(writer, b"\\\"", 2)
            elif letter == "'":
                PyUnicodeWriter_WriteASCII(writer, b"\\'", 2)
            else:
                if letter == "\n" and not multiline:
                    PyUnicodeWriter_WriteASCII(writer, b"\\n", 2)
                else:
                    PyUnicodeWriter_WriteChar(writer, letter)

        return PyUnicodeWriter_Finish(writer)
    except:
        PyUnicodeWriter_Discard(writer)
        raise


# This is a replacement for a method in VPK, which is very slow normally
# since it has to accumulate character by character.
cdef class _VPK_IterNullstr:
    """Read a null-terminated ASCII string from the file.

    This continuously yields strings, with empty strings
    indicting the end of a section.
    """
    cdef object file
    cdef unsigned char *chars
    cdef Py_ssize_t size
    cdef Py_ssize_t used

    def __cinit__(self):
        self.used = 0
        self.size = 64
        self.chars = <unsigned char *>PyMem_Malloc(self.size)
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
        cdef unsigned char *temp
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
                    temp = <unsigned char *>PyMem_Realloc(self.chars, self.size)
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
        cdef unsigned char *buf = byt
        # Unpack an unsigned 16-bit integer.
        cdef uint16_t offset = (buf[1] << 8) | buf[0]
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
