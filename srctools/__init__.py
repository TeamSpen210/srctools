import os as _os
import string as _string
import tempfile as _tempfile
from collections import abc as _abc

from typing import Union as _Union

__all__ = [
    'Vec', 'Vec_tuple', 'parse_vec_str',

    'NoKeyError', 'KeyValError', 'Property',

    'VMF', 'Entity', 'Solid', 'Side', 'Output', 'UVAxis',

    'clean_line', 'is_plain_text', 'blacklist', 'whitelist',
    'bool_as_int', 'conv_bool', 'conv_int', 'conv_float',

    'EmptyMapping', 'AtomicWriter',
]

_FILE_CHARS = set(_string.ascii_letters + _string.digits + '-_ .|')


def clean_line(line: str):
    """Removes extra spaces and comments from the input."""
    if isinstance(line, bytes):
        line = line.decode()  # convert bytes to strings if needed
    if '//' in line:
        line = line.split('//', 1)[0]
    return line.strip()


def is_plain_text(name, valid_chars=_FILE_CHARS):
    """Check to see if any characters are not in the whitelist.

    """
    for char in name:
        if char not in valid_chars:
            return False
    return True


def whitelist(string, valid_chars=_FILE_CHARS, rep_char='_'):
    """Replace any characters not in the whitelist with the replacement char."""
    chars = list(string)
    for ind, char in enumerate(chars):
        if char not in valid_chars:
            chars[ind] = rep_char
    return ''.join(chars)


def blacklist(string, invalid_chars='', rep_char='_'):
    """Replace any characters in the blacklist with the replacement char."""
    chars = list(string)
    for ind, char in enumerate(chars):
        if char in invalid_chars:
            chars[ind] = rep_char
    return ''.join(chars)


def escape_quote_split(line):
    """Split quote values on a line, handling \\" correctly."""
    out_strings = []
    was_backslash = False  # Last character was a backslash
    cur_part = []  # The current chunk of text

    for char in line:
        if char == '\\':
            was_backslash = True
            cur_part.append('\\')
            continue

        if char == '"':
            if was_backslash:
                cur_part.pop()  # Consume the backslash, then drop to append it.
            else:
                out_strings.append(''.join(cur_part))
                cur_part.clear()
                continue
        # Backslash only applies for character..
        was_backslash = False
        cur_part.append(char)

    # Part after the last quotation
    out_strings.append(''.join(cur_part))
    return out_strings


def bool_as_int(val):
    """Convert a True/False value into '1' or '0'.

    Valve uses these strings for True/False in editoritems and other
    config files.
    """
    if val:
        return '1'
    else:
        return '0'


BOOL_LOOKUP = {
    False: False,
    0: False,
    '0': False,
    'no': False,
    'false': False,
    'n': False,
    'f': False,

    1: True,
    True: True,
    '1': True,
    'yes': True,
    'true': True,
    'y': True,
    't': True,
}


def conv_bool(val: _Union[str, bool, None], default=False):
    """Converts a string to a boolean, using a default if it fails.

    Accepts any of '0', '1', 'false', 'true', 'yes', 'no'.
    If Val is None, this always returns the default.
    0, 1, True and False will be passed through unchanged.
    """
    if val is None:
        return default
    try:
        # Lookup bools, ints, and normal strings
        return BOOL_LOOKUP[val]
    except KeyError:
        # Try again with casefolded strings
        return BOOL_LOOKUP.get(val.casefold(), default)


def conv_float(val, default=0.0):
    """Converts a string to an float, using a default if it fails.

    """
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def conv_int(val: str, default=0):
    """Converts a string to an integer, using a default if it fails.

    """
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


class EmptyMapping(_abc.MutableMapping):
    """A Mapping class which is always empty.

    Any modifications will be ignored.
    This is used for default arguments, since it then ensures any changes
    won't be kept, as well as allowing default.items() calls and similar.
    """
    __slots__ = []

    def __init__(self):
        pass

    def __call__(self):
        # Just in case someone tries to instantiate this
        return self

    def __getitem__(self, key):
        """All key acesses fail."""
        raise KeyError(key)

    def __setitem__(self, key, value):
        """All key setting suceeds."""
        pass

    def __delitem__(self, key):
        """All key deletions fail."""
        raise KeyError(key)

    def __contains__(self, key):
        """EmptyMapping does not have any keys."""
        return False

    def get(self, key, default=None):
        """get() or setdefault() always returns the default item."""
        return default

    def __bool__(self):
        """EmptyMapping is falsey."""
        return False

    def __len__(self):
        """EmptyMapping is 0 long."""
        return 0

    def __iter__(self):
        """Iteration yields no values."""
        return iter(())


    # Mutable functions
    setdefault = get

    @staticmethod
    def update(*args, **kargs):
        """Runs {}.update() on arguments."""
        # Check arguments are correct, and raise appropriately.
        # Also consume args[0] if an iterator - this raises if args > 1.
        {}.update(*args, **kargs)

    __marker = object()

    def pop(self, key, default=__marker):
        """Returns the default value, or raises KeyError if not present."""
        if default is self.__marker:
            raise KeyError(key)
        return default


    def popitem(self):
        """Popitem() raises, since no items are in EmptyMapping."""
        raise KeyError('EmptyMapping is empty')

EmptyMapping = EmptyMapping()  # We only want the one instance


class AtomicWriter:
    """Atomically overwrite a file.

    Use as a context manager - the returned temporary file
    should be written to. When cleanly exiting, the file will be transfered.
    If an exception occurs in the body, the temporary data will be discarded.

    This is not reentrant, but can be repeated - starting the context manager
    clears the file.
    """
    def __init__(self, filename, is_bytes=False):
        """Create an AtomicWriter.
        is_bytes sets text or bytes writing mode. The file is always writable.
        """
        self.filename = filename
        self.dir = _os.path.dirname(filename)
        self.is_bytes = is_bytes
        self.temp = None

    def make_tempfile(self):
        """Create the temporary file object."""
        if self.temp is not None:
            # Already open - close and delete the current file.
            self.temp.close()
            _os.remove(self.temp.name)

        # Create folders if needed..
        _os.makedirs(self.dir, exist_ok=True)

        self.temp = open(
            self.filename,
            'wb' if self.is_bytes else 'wt',
        )

    def __enter__(self):
        """Delegate to the underlying temporary file handler."""
        self.make_tempfile()
        return self.temp.__enter__()

    def __exit__(self, exc_type, exc_value, tback):
        # Pass to tempfile, which also closes().
        temp_path = self.temp.name
        self.temp.__exit__(exc_type, exc_value, tback)
        self.temp = None
        if exc_type is not None:
            # An exception occurred, clean up.
            try:
                _os.remove(temp_path)
            except FileNotFoundError:
                pass
        else:
            # No exception, commit changes
            _os.replace(temp_path, self.filename)

        return False  # Don't cancel the exception.


# Import these, so people can reference 'srctools.Vec' instead of
# 'srctools.vec.Vec'.
# Should be done after other code, so everything's initialised.
from srctools.vec import Vec, Vec_tuple, parse_vec_str
from srctools.property_parser import NoKeyError, KeyValError, Property
from srctools.vmf import VMF, Entity, Solid, Side, Output, UVAxis
from srctools.vpk import VPK
from srctools.filesys import FileSystem, FileSystemChain, get_filesystem
