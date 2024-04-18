from typing import (
    TYPE_CHECKING, AbstractSet, Any, Container, Final, Generic, ItemsView, Iterable,
    Iterator, KeysView, List, Mapping, MutableMapping, NoReturn, Optional, Sequence, Set,
    Tuple, Type, TypeVar, Union, ValuesView, overload,
)
from typing_extensions import Literal, TypeAlias
from collections import deque
from pathlib import Path
from types import TracebackType
import io
import itertools as _itertools
import os as _os
import sys as _sys
import warnings

from useful_types import SupportsKeysAndGetItem


__version__: str
if not TYPE_CHECKING:
    try:
        from ._version import __version__
    except ImportError:
        __version__ = '<unknown>'
    else:
        # Discard the now-useless module. Use globals so static analysis ignores this.
        del _sys.modules[globals().pop('_version').__name__]

__all__ = [
    '__version__',
    'Vec', 'FrozenVec', 'parse_vec_str', 'lerp',
    'Angle', 'FrozenAngle', 'Matrix', 'FrozenMatrix',

    'NoKeyError', 'KeyValError', 'Keyvalues',
    'VMF', 'Entity', 'Solid', 'Side', 'Output', 'UVAxis',

    'clean_line', 'is_plain_text', 'blacklist', 'whitelist',
    'bool_as_int', 'conv_bool', 'conv_int', 'conv_float',
    'BOOL_LOOKUP',

    'FileSystem', 'FileSystemChain', 'get_filesystem',

    'EmptyMapping', 'StringPath',

    'FGD', 'VPK',
    'VTF',
    'SurfaceProp', 'SurfChar',
    'GameID',

    # Submodules:
    'binformat', 'bsp', 'cmdseq', 'const', 'dmx', 'fgd', 'filesys', 'game',  # pyright: ignore
    'instancing', 'logger', 'math', 'mdl', 'packlist', 'particles', 'keyvalues', 'run',  # pyright: ignore
    'smd', 'sndscript', 'surfaceprop', 'tokenizer', 'vmf', 'vmt', 'vpk', 'vtf',  # pyright: ignore
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
# Pathlike can only be subscripted in 3.9+
StringPath: TypeAlias = Union[str, '_os.PathLike[str]']


def clean_line(line: str) -> str:
    """Removes extra spaces and comments from the input."""
    if isinstance(line, bytes):
        line = line.decode()  # convert bytes to strings if needed
    if '//' in line:
        line = line.split('//', 1)[0]
    return line.strip()


def is_plain_text(
    name: str,
    valid_chars: Container[str] = _FILE_CHARS,
) -> bool:
    """Check to see if any characters are not in the whitelist.

    """
    for char in name:
        if char not in valid_chars:
            return False
    return True


def whitelist(
    string: str,
    valid_chars: Container[str] = _FILE_CHARS,
    rep_char: str = '_',
) -> str:
    """Replace any characters not in the whitelist with the replacement char."""
    chars = list(string)
    for ind, char in enumerate(chars):
        if char not in valid_chars:
            chars[ind] = rep_char
    return ''.join(chars)


def blacklist(
    string: str,
    invalid_chars: Container[str] = (),
    rep_char: str = '_',
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


def partition(coll: Sequence[ValT], size: int) -> Iterator[Sequence[ValT]]:
    """Break up the collection into groups of at most size.

    Practically speaking slicing should give the same type, but that isn't
    guaranteed.
    """
    if len(coll) <= size:
        yield coll
        return

    for start in range(0, len(coll), size):
        yield coll[start:start + size]


def bool_as_int(val: object) -> Literal['0', '1']:
    """Convert a True/False value into '1' or '0'.

    Valve uses these strings for True/False in editoritems and other
    config files.
    """
    if val:
        return '1'
    else:
        return '0'


BOOL_LOOKUP: Final[Mapping[str, bool]] = {
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
def conv_bool(val: Union[str, bool, None], default: Union[ValT, bool] = False) -> Union[ValT, bool]:
    """Converts a string to a boolean, using a default if it fails.

    Accepts any of ``0``, ``1``, ``false``, ``true``, ``yes``, ``no``.
    If val is ``None``, this always returns the default.
    ``0``, ``1``, ``True`` and ``False`` will be passed through unchanged.
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
        try:
            val = val.casefold()
        except AttributeError:  # Non-string?
            return default
        try:
            return BOOL_LOOKUP[val]
        except KeyError:
            return default

@overload
def conv_float(val: Union[int, float, str]) -> float: ...
@overload
def conv_float(val: Union[int, float, str], default: ValT) -> Union[ValT, float]: ...
def conv_float(val: Union[int, float, str], default: Union[ValT, float] = 0.0) -> Union[ValT, float]:
    """Converts a string to a float, using a default if it fails.

    The value may also be an integer or float, which is also converted.
    """
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


@overload
def conv_int(val: Union[int, float, str]) -> int: ...
@overload
def conv_int(val: Union[int, float, str], default: ValT) -> Union[ValT, int]: ...
def conv_int(val: Union[int, float, str], default: Union[ValT, int] = 0) -> Union[ValT, int]:
    """Converts a string to an integer, using a default if it fails.

    The value may also be an integer or float, which is also converted.
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
    def get(self, key: Any, default: Optional[ValT] = None) -> Optional[ValT]:
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

    @overload  # type: ignore[override]
    def setdefault(self, key: Any) -> None: ...
    # This is stricter than Mapping[Any, Any]. That returns Any, we always return the default.
    @overload
    def setdefault(self, key: Any, default: ValT) -> ValT: ...

    def setdefault(self, key: Any, default: Optional[ValT] = None) -> Optional[ValT]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """setdefault() always returns the default item, but does not store it."""
        return default

    @overload
    def update(self, m: SupportsKeysAndGetItem[Any, Any], /, **kwargs: Any) -> None: ...
    @overload
    def update(self, m: Iterable[Tuple[Any, Any]], /, **kwargs: Any) -> None: ...
    @overload
    def update(self, **kwargs: Any) -> None: ...

    def update(self, *args: Any, **kargs: Any) -> None:  # type: ignore  # Mypy bug?
        """Runs {}.update() on arguments."""
        # Check arguments are correct, and raise appropriately.
        # Also consume args[0] if an iterator - this raises if args > 1.
        {}.update(*args, **kargs)

    # Also stricter than Mapping[Any, Any], since we NoReturn if not passed a default.
    __marker: Final[Any] = object()
    @overload
    def pop(self, key: Any, /) -> NoReturn: ...
    @overload
    def pop(self, key: Any, /, default: ValT) -> ValT: ...

    def pop(self, key: Any, default: ValT = __marker) -> ValT:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Returns the default value, or raises KeyError if not present."""
        if default is self.__marker:
            raise KeyError(key)
        return default

    def popitem(self) -> NoReturn:
        """Popitem() raises, since no items are in EmptyMapping."""
        raise KeyError('EmptyMapping is empty')


class _EmptySetView(AbstractSet[Any]):
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

    def __ne__(self, other: object) -> bool:
        """All nonempty sets are non-equal."""
        if not isinstance(other, AbstractSet):
            return NotImplemented
        return len(other) > 0

    def __eq__(self, other: object) -> bool:
        """Only empty sets are equal."""
        if not isinstance(other, AbstractSet):
            return NotImplemented
        return len(other) == 0

    def __le__(self, other: object) -> bool:
        """We are <= to all sets."""
        if isinstance(other, AbstractSet):
            return True
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        """We are never > any set."""
        if isinstance(other, AbstractSet):
            return False
        return NotImplemented

    def __or__(self, other: Iterable[ValT]) -> Set[ValT]:
        """A binary operation which returns all the other values."""
        if not isinstance(other, Iterable):
            return NotImplemented
        return set(other)

    def __and__(self, other: Iterable[ValT]) -> Set[ValT]:
        """A binary operation producing no values."""
        if not isinstance(other, Iterable):
            return NotImplemented
        deque(other, 0)  # Consume all values.
        return set()

    def isdisjoint(self, other: Iterable[Any]) -> bool:
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


_set_hash: Final = hash(frozenset())
EmptyMapping: _EmptyMapping = _EmptyMapping()
EmptyKeysView: KeysView[Any] = _EmptyKeysView(EmptyMapping)
EmptyItemsView: ItemsView[Any, Any] = _EmptyItemsView(EmptyMapping)
EmptyValuesView: ValuesView[Any] = _EmptyValuesView(EmptyMapping)
IOKindT = TypeVar('IOKindT', io.BufferedWriter, io.TextIOWrapper)


class AtomicWriter(Generic[IOKindT]):
    """Atomically overwrite a file.

    Use as a context manager - the returned temporary file
    should be written to. When cleanly exiting, the file will be transferred.
    If an exception occurs in the body, the temporary data will be discarded.

    This is not reentrant, but can be repeated - starting the context manager
    clears the file.
    """
    filename: Path
    encoding: str
    _temp_name: Optional[Path]
    is_bytes: bool
    temp: Optional[IOKindT]

    @overload
    def __init__(
        self: 'AtomicWriter[io.BufferedWriter]', filename: StringPath,
        is_bytes: Literal[True],
    ) -> None: ...

    @overload
    def __init__(
        self: 'AtomicWriter[io.TextIOWrapper]', filename: StringPath,
        is_bytes: Literal[False] = False, encoding: str = 'utf8',
    ) -> None: ...

    def __init__(
        self,
        filename: StringPath,
        is_bytes: bool = False,
        encoding: str = 'utf8',
    ) -> None:
        """Create an AtomicWriter.
        is_bytes sets text or bytes writing mode. The file is always writable.
        """
        self.filename = Path(filename)
        self.encoding = encoding
        self._temp_name = None
        self.is_bytes = is_bytes
        self.temp = None

    def make_tempfile(self) -> None:
        """Create the temporary file object."""
        if self.temp is not None:
            # Already open - close and delete the current file.
            self.temp.close()
            Path(self.temp.name).unlink()

        # Create folders if needed.
        self.filename.parent.mkdir(parents=True, exist_ok=True)

        for i in _itertools.count(start=1):
            self._temp_name = self.filename.with_name(f'tmp_{i}')
            try:
                if self.is_bytes:  # type checkers can't narrow self from this!
                    self.temp = self._temp_name.open('xb')  # type: ignore
                else:
                    self.temp = self._temp_name.open('xt', encoding=self.encoding)  # type: ignore
                break
            except FileExistsError:
                pass

    def __enter__(self) -> IOKindT:
        """Delegate to the underlying temporary file handler."""
        self.make_tempfile()
        assert self.temp is not None
        return self.temp.__enter__()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        tback: Optional[TracebackType],
    ) -> None:
        # Delegate down to close the file like normal.
        if self.temp is not None:
            self.temp.__exit__(exc_type, exc_value, tback)
            self.temp = None
        if self._temp_name is None:
            # Exit without enter?
            return None
        if exc_type is not None:
            # An exception occurred, clean up.
            try:
                self._temp_name.unlink()
            except FileNotFoundError:
                pass
        else:
            # No exception, commit changes
            self._temp_name.replace(self.filename)

        return None  # Don't cancel the exception.


# Import these, so people can reference 'srctools.Vec' instead of 'srctools.math.Vec'.
# Should be done after other code, so everything's initialised.
# Not all classes are imported, just most-used ones.
# This shouldn't be used in our modules, to ensure the order here doesn't matter.
# isort: off
from .math import (
    FrozenVec, Vec,
    parse_vec_str, lerp,
    FrozenMatrix, Matrix,
    FrozenAngle, Angle,
)
# Deprecated.
from .math import Vec_tuple  # noqa
from srctools.keyvalues import NoKeyError, KeyValError, Keyvalues
from srctools.filesys import FileSystem, FileSystemChain, get_filesystem
from srctools.vmf import VMF, Entity, Solid, Side, Output, UVAxis
from srctools.vpk import VPK
from srctools.fgd import FGD
from srctools.const import GameID
from srctools.surfaceprop import SurfaceProp, SurfChar
from srctools.vtf import VTF

_py_conv_int = _cy_conv_int = conv_int
_py_conv_float = _cy_conv_float = conv_float
_py_conv_bool = _cy_conv_bool = conv_bool


if TYPE_CHECKING:
    Property = Keyvalues  #: :deprecated: Use srctools.Keyvalues.
else:
    def __getattr__(name: str) -> object:
        if name == 'Property':
            warnings.warn(
                'srctools.Property is renamed to srctools.Keyvalues',
                DeprecationWarning,
                stacklevel=2,
            )
            return Keyvalues
        raise AttributeError(name)

    try:
        from ._math import conv_int, conv_float, conv_bool
    except ImportError:
        pass
    else:
        _cy_conv_int = conv_int
        _cy_conv_float = conv_float
        _cy_conv_bool = conv_bool
