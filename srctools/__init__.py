import string as _string
from collections import abc as _abc

from typing import Union as _Union

__all__ = [
    'Vec', 'Vec_tuple', 'parse_vec_str',

    'NoKeyError', 'KeyValError', 'Property',

    'VMF', 'Entity', 'Solid', 'Side', 'Output', 'UVAxis',

    'clean_line', 'is_plain_text', 'blacklist', 'whitelist',
    'bool_as_int', 'conv_bool', 'conv_int', 'conv_float',

    'EmptyMapping',
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
    'FALSE': False,
    'n': False,
    'f': False,

    1: True,
    True: True,
    '1': True,
    'yes': True,
    'true': True,
    'TRUE': True,
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

    def __call__(self):
        # Just in case someone tries to instantiate this
        return self

    def __getitem__(self, item):
        raise KeyError

    def __contains__(self, item):
        return False

    def get(self, item, default=None):
        return default

    def __bool__(self):
        return False

    def __next__(self):
        raise StopIteration

    def __len__(self):
        return 0

    def __iter__(self):
        return self

    items = values = keys = __iter__

    # Mutable functions
    setdefault = get

    def update(*args, **kwds):
        pass

    __setitem__ = __delitem__ = clear = update

    __marker = object()

    def pop(self, key, default=__marker):
        if default is self.__marker:
            raise KeyError
        return default

    def popitem(self):
        raise KeyError

EmptyMapping = EmptyMapping()  # We only want the one instance


# Import these, so people can reference 'srctools.Vec' instead of
# 'srctools.vec.Vec'.
from srctools.vec import Vec, Vec_tuple, parse_vec_str
from srctools.property_parser import NoKeyError, KeyValError, Property
from srctools.vmf import VMF, Entity, Solid, Side, Output, UVAxis
