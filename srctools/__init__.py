import itertools as _itertools
import os as _os
from typing import (
    Union, Type, TypeVar, Iterator, Sequence, List, Container,
    IO, Optional,
    Mapping, KeysView, ValuesView, ItemsView,
    MutableMapping, AbstractSet, Set,
    Any,
    overload,
    Iterable,
    Tuple,
)
from types import TracebackType
from collections import deque


__all__ = [
    'Vec', 'Vec_tuple', 'parse_vec_str',
    'Angle', 'Matrix',

    'NoKeyError', 'KeyValError', 'Property',
    'VMF', 'Entity', 'Solid', 'Side', 'Output', 'UVAxis',

    'clean_line', 'is_plain_text', 'blacklist', 'whitelist',
    'bool_as_int', 'conv_bool', 'conv_int', 'conv_float',
    'BOOL_LOOKUP',

    'FileSystem', 'FileSystemChain', 'get_filesystem',

    'EmptyMapping', 'AtomicWriter',

    'FGD', 'VPK',
    'VTF',
    'SurfaceProp', 'SurfChar',
    'GameID',
]

# import string
# _FILE_CHARS = frozenset(string.ascii_letters + string.digits + '-_ .|')
_FILE_CHARS = frozenset(
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '0123456789'
    '-_ .|'
)
ValT = TypeVar('ValT')


def clean_line(line: str) -> str:
    """Removes extra spaces and comments from the input."""
    if isinstance(line, bytes):
        line = line.decode()  # convert bytes to strings if needed
    if '//' in line:
        line = line.split('//', 1)[0]
    return line.strip()


def is_plain_text(
    name: str,
    valid_chars: Container[str]=_FILE_CHARS,
) -> bool:
    """Check to see if any characters are not in the whitelist.

    """
    for char in name:
        if char not in valid_chars:
            return False
    return True


def whitelist(
    string: str,
    valid_chars: Container[str]=_FILE_CHARS,
    rep_char: str='_',
) -> str:
    """Replace any characters not in the whitelist with the replacement char."""
    chars = list(string)
    for ind, char in enumerate(chars):
        if char not in valid_chars:
            chars[ind] = rep_char
    return ''.join(chars)


def blacklist(
    string: str,
    invalid_chars: Container[str]=(),
    rep_char: str='_',
) -> str:
    """Replace any characters in the blacklist with the replacement char."""
    chars = list(string)
    for ind, char in enumerate(chars):
        if char in invalid_chars:
            chars[ind] = rep_char
    return ''.join(chars)


def escape_quote_split(line: str) -> List[str]:
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

SequenceT = TypeVar('SequenceT', bound=Sequence)


def partition(coll: SequenceT, size: int) -> Iterator[SequenceT]:
    """Break up the collection into groups of at most size"""
    if len(coll) <= size:
        yield coll
        return

    for start in range(0, len(coll), size):
        yield coll[start:start + size]


def bool_as_int(val: object) -> str:
    """Convert a True/False value into '1' or '0'.

    Valve uses these strings for True/False in editoritems and other
    config files.
    """
    if val:
        return '1'
    else:
        return '0'


BOOL_LOOKUP: Mapping[str, bool] = {
    '0': False,
    'no': False,
    'false': False,
    'n': False,
    'f': False,
    '1': True,
    'yes': True,
    'true': True,
    'y': True,
    't': True,
}

@overload
def conv_bool(val: Union[str, bool, None]) -> bool: ...
@overload
def conv_bool(val: Union[str, bool, None], default: ValT) -> Union[ValT, bool]: ...
def conv_bool(val: Union[str, bool, None], default: ValT = False) -> Union[ValT, bool]:
    """Converts a string to a boolean, using a default if it fails.

    Accepts any of '0', '1', 'false', 'true', 'yes', 'no'.
    If Val is None, this always returns the default.
    0, 1, True and False will be passed through unchanged.
    """
    if val is None:
        return default
    if isinstance(val, int):
        return val != 0
    try:
        # Lookup bools, ints, and normal strings
        return BOOL_LOOKUP[val]
    except KeyError:
        # Try again with casefolded strings
        return BOOL_LOOKUP.get(val.casefold(), default)


@overload
def conv_float(val: str) -> float: ...
@overload
def conv_float(val: str, default: ValT) -> Union[ValT, float]: ...
def conv_float(val: str, default: ValT = 0.0) -> Union[ValT, float]:
    """Converts a string to an float, using a default if it fails.

    """
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


@overload
def conv_int(val: str) -> int: ...
@overload
def conv_int(val: str, default: ValT) -> Union[ValT, int]: ...
def conv_int(val: str, default: ValT = 0) -> Union[ValT, int]:
    """Converts a string to an integer, using a default if it fails.

    """
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


class _EmptyMapping(MutableMapping[Any, Any]):
    """A Mapping class which is always empty.

    Any modifications will be ignored.
    This is used for default arguments, since it then ensures any changes
    won't be kept, as well as allowing default.items() calls and similar.
    """
    __slots__ = ()

    def __call__(self) -> '_EmptyMapping':
        # Just in case someone tries to instantiate this
        return self

    def __repr__(self) -> str:
        return "srctools.EmptyMapping"

    def __reduce__(self) -> str:
        return 'EmptyMapping'

    def __getitem__(self, key: Any) -> Any:
        """All key acesses fail."""
        raise KeyError(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        """All key setting suceeds."""
        pass

    def __delitem__(self, key: Any) -> None:
        """All key deletions fail."""
        raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        """EmptyMapping does not have any keys."""
        return False

    @overload
    def get(self, key: Any) -> None: ...
    @overload
    def get(self, key: Any, default: ValT) -> ValT: ...
    def get(self, key: Any, default: ValT=None) -> Optional[ValT]:
        """get() always returns the default item."""
        return default

    def __bool__(self) -> bool:
        """EmptyMapping is falsey."""
        return False

    def __len__(self) -> int:
        """EmptyMapping is 0 long."""
        return 0

    def __iter__(self) -> Iterator[Any]:
        """Iteration yields no values."""
        return iter(())

    def keys(self) -> KeysView[Any]:
        """Return an empty keys() view singleton."""
        return EmptyKeysView

    def items(self) -> ItemsView[Any, Any]:
        """Return an empty items() view singleton."""
        return EmptyItemsView

    def values(self) -> ValuesView[Any]:
        """Return an empty values() view singleton."""
        return EmptyValuesView

    # Mutable functions
    def setdefault(self, key: Any, default: ValT=None) -> ValT:
        """setdefault() always returns the default item, but does not store it."""
        return default

    @overload
    def update(self, __m: Mapping[Any, Any], **kwargs: Any) -> None: ...
    @overload
    def update(self, __m: Iterable[Tuple[Any, Any]], **kwargs: Any) -> None: ...

    def update(self, *args, **kargs: Any) -> None:
        """Runs {}.update() on arguments."""
        # Check arguments are correct, and raise appropriately.
        # Also consume args[0] if an iterator - this raises if args > 1.
        {}.update(*args, **kargs)

    __marker = object()

    def pop(self, key: Any, default: ValT = __marker) -> ValT:
        """Returns the default value, or raises KeyError if not present."""
        if default is self.__marker:
            raise KeyError(key)
        return default

    def popitem(self) -> Any:
        """Popitem() raises, since no items are in EmptyMapping."""
        raise KeyError('EmptyMapping is empty')

class _EmptySetView:
    """Common code between EmptyKeysView and EmptyItemsView."""
    __slots__ = ()

    def __len__(self) -> int:
        """This contains no keys/items."""
        return 0

    def __contains__(self, key: Any) -> bool:
        """All keys/items are not present."""
        return False

    def __iter__(self) -> Iterator[Any]:
        """Iteration produces no values."""
        return iter(())

    # Set ops. A lot can share implementations.

    def _only_full(self, other) -> Any:
        """Only nonempty sets pass."""
        if not isinstance(other, AbstractSet):
            return NotImplemented
        return len(other) > 0

    def _only_empty(self, other) -> Any:
        """Only empty sets pass."""
        if not isinstance(other, AbstractSet):
            return NotImplemented
        return len(other) == 0

    def _always_true(self, other) -> Any:
        """Any set passes this comparison."""
        return True if isinstance(other, AbstractSet) else NotImplemented

    def _always_false(self, other) -> Any:
        """Any set fails this comparison."""
        return False if isinstance(other, AbstractSet) else NotImplemented

    def _binop_copy(self, other: Iterable[ValT]) -> Set[ValT]:
        """A binary operation which returns all the other values."""
        if not isinstance(other, Iterable):
            return NotImplemented
        return set(other)

    def _binop_drop(self, other: Iterable[ValT]) -> Set[ValT]:
        """A binary operation producing no values."""
        if not isinstance(other, Iterable):
            return NotImplemented
        deque(other, 0)  # Consume all values.
        return set()

    __eq__ = _only_empty
    __ne__ = _only_full
    __lt__ = _only_full
    __le__ = _always_true
    __gt__ = _always_false
    __ge__ = _only_empty

    __and__ = _binop_drop
    __rand__ = _binop_drop
    __or__ = _binop_copy
    __ror__ = _binop_copy
    __sub__ = _binop_drop
    __rsub__ = _binop_copy
    __xor__ = _binop_copy
    __rxor__ = _binop_copy

    del _only_empty, _only_full, _always_false, _binop_drop, _binop_copy

    def isdisjoint(self, other) -> bool:
        """This set is always disjoint."""
        iter(other)  # Check it's iterable.
        return True

    def __hash__(self) -> int:
        """These are immutable, so they can be hashable."""
        return _set_hash


class _EmptyKeysView(_EmptySetView, KeysView[Any]):
    """A Mapping view implementation that always acts empty, and supports set operations."""
    __slots__ = ()

    def __repr__(self) -> str:
        return 'srctools.EmptyKeysView'

    def __reduce__(self) -> str:
        return 'EmptyKeysView'


class _EmptyItemsView(_EmptySetView, ItemsView[Any, Any]):
    """A Mapping view implementation that always acts empty, and supports set operations."""
    __slots__ = ()

    def __repr__(self) -> str:
        return 'srctools.EmptyItemsView'

    def __reduce__(self) -> str:
        return 'EmptyItemsView'


class _EmptyValuesView(ValuesView[Any]):
    """A Mapping.values() implementation that always acts empty. This is not a set."""
    __slots__ = ()
    def __repr__(self) -> str:
        return 'srctools.EmptyValuesView'

    def __reduce__(self) -> str:
        return 'EmptyValuesView'

    def __contains__(self, key: Any) -> bool:
        """All values are not present."""
        return False

    def __len__(self) -> int:
        """This contains no values."""
        return 0

    def __iter__(self) -> Iterator[Any]:
        """Iteration produces no values."""
        return iter(())


_set_hash = hash(frozenset())
EmptyMapping = _EmptyMapping()
EmptyKeysView = _EmptyKeysView(EmptyMapping)
EmptyItemsView = _EmptyItemsView(EmptyMapping)
EmptyValuesView = _EmptyValuesView(EmptyMapping)


class AtomicWriter:
    """Atomically overwrite a file.

    Use as a context manager - the returned temporary file
    should be written to. When cleanly exiting, the file will be transferred.
    If an exception occurs in the body, the temporary data will be discarded.

    This is not reentrant, but can be repeated - starting the context manager
    clears the file.
    """
    def __init__(self, filename: _os.PathLike, is_bytes: bool=False, encoding: str='utf8') -> None:
        """Create an AtomicWriter.
        is_bytes sets text or bytes writing mode. The file is always writable.
        """
        self.filename = filename
        self.dir = _os.path.dirname(filename)
        self.encoding = encoding
        self._temp_name: Optional[str] = None
        self.is_bytes = is_bytes
        self.temp: Optional[IO] = None

    def make_tempfile(self) -> None:
        """Create the temporary file object."""
        if self.temp is not None:
            # Already open - close and delete the current file.
            self.temp.close()
            _os.remove(self.temp.name)

        # Create folders if needed..
        if self.dir:
            _os.makedirs(self.dir, exist_ok=True)

        for i in _itertools.count(start=1):
            self._temp_name = _os.path.join(self.dir, 'tmp_{}'.format(i))
            try:
                if self.is_bytes:
                    self.temp = open(self._temp_name, 'xb')
                else:
                    self.temp = open(self._temp_name, 'xt', encoding=self.encoding)
                break
            except FileExistsError:
                pass

    def __enter__(self):
        """Delegate to the underlying temporary file handler."""
        self.make_tempfile()
        return self.temp.__enter__()

    def __exit__(
        self,
        exc_type: Type[BaseException],
        exc_value: BaseException,
        tback: TracebackType,
    ) -> None:
        # Pass to tempfile, which also closes().
        if self.temp is not None:
            self.temp.__exit__(exc_type, exc_value, tback)
            self.temp = None
        if self._temp_name is None:
            # Exit without enter?
            return None
        if exc_type is not None:
            # An exception occurred, clean up.
            try:
                _os.remove(self._temp_name)
            except FileNotFoundError:
                pass
        else:
            # No exception, commit changes
            _os.replace(self._temp_name, self.filename)

        return None  # Don't cancel the exception.


# Import these, so people can reference 'srctools.Vec' instead of
# 'srctools.vec.Vec'.
# Should be done after other code, so everything's initialised.
# Not all classes are imported, just most-used ones.
from srctools.math import Vec, Vec_tuple, parse_vec_str, lerp, Angle, Matrix
from srctools.property_parser import NoKeyError, KeyValError, Property
from srctools.filesys import FileSystem, FileSystemChain, get_filesystem
from srctools.vmf import VMF, Entity, Solid, Side, Output, UVAxis
from srctools.vpk import VPK
from srctools.fgd import FGD
from srctools.const import GameID
from srctools.surfaceprop import SurfaceProp, SurfChar
from srctools.vtf import VTF
