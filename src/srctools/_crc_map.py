"""A specialised mapping, which is parsed from binary files using CRCs as keys.

This matches the binary format for choreo VCDs and closed captions/subtitles.

These files just store a CRC of the key, used to perform each lookup. This means keys are not
initially available once parsed.
"""
import abc
from collections.abc import MutableMapping
from typing import ClassVar, Dict, Optional, Tuple, TypeVar, Union

from srctools.binformat import checksum


ValueT = TypeVar('ValueT')


class CollisionError(Exception):
    """Raised if a checksum collision error is detected."""
    def __init__(self, key1: str, key2: str, checksum: int) -> None:
        self.key1 = key1
        self.key2 = key2
        self.checksum = checksum

    def __str__(self) -> str:
        return f'Checksum collision: {self.key1!r} != {self.key2!r}, checksum={self.checksum:X}'


class ChecksumMap(MutableMapping[str, ValueT], abc.ABC):
    """A specialised mapping which handles the data loss in binary choreo or subtitle files.

    The mapping stores fully parsed values (with the key available) seperately from unknown
    checksums. When accessed, the checksum is located if possible. Parsed values are preferred
    to looking for checksums, if both are present.

    Iteration excludes values with only checksums known, use iter_checksums() to get all values.
    """
    # Checksum -> key (if known) and value.
    _values: Dict[int, Tuple[Optional[str], ValueT]]

    encoding: ClassVar[str] = 'utf8'

    def __init__(self) -> None:
        self._values = {}

    def _normalise_key(self, key: str) -> str:
        """Perform normalisation of the key, such as converting filename slashes."""
        return key

    def _try_parse(self, key: str, check: int) -> ValueT:
        """Hook to allow lazily parsing values.

        This should parse this value, store it then return, or raise KeyError if not found.
        """
        raise KeyError(key) from None

    def _clear_unparsed(self, check: int) -> bool:
        """If a lazily parsed value with this checksum is present, clear it."""
        return False

    def clear(self) -> None:
        """Clear all values."""
        self._values.clear()

    def __len__(self) -> int:
        """Return the number of entries."""
        return len(self._values)

    def __getitem__(self, key: str, /) -> ValueT:
        key = self._normalise_key(key)
        check = checksum(key.encode(self.encoding))
        try:
            exist_key, value = self._values[check]
        except KeyError:
            return self._try_parse(key, check)
        if exist_key is None:
            # Key wasn't known, record that.
            self._values[check] = key, value
        elif key != exist_key:
            raise CollisionError(key, exist_key, check)
        return value

    def __setitem__(self, key: Union[str, int], value: ValueT, /) -> None:
        """Set an item, via either a known key or checksum."""
        if isinstance(key, str):
            key = self._normalise_key(key)
            check = checksum(key.encode(self.encoding))
            try:
                exist_key = self._values[check][0]
            except KeyError:
                self._clear_unparsed(check)
            else:
                if exist_key is not None and exist_key != key:
                    raise CollisionError(key, exist_key, check)
            self._values[check] = key, value
            self._clear_unparsed(check)
        else:
            check = key
            # If already known, preserve the key.
            try:
                exist_key = self._values[check][0]
            except KeyError:
                exist_key = None
                self._clear_unparsed(check)
            self._values[check] = exist_key, value

    def __delitem__(self, key: Union[str, int], /) -> None:
        """Remove the value with this key or checksum."""
        if isinstance(key, str):
            key = self._normalise_key(key)
            check = checksum(key.encode(self.encoding))
            try:
                exist_key = self._values.pop(check)[0]
            except KeyError:
                if not self._clear_unparsed(check):
                    raise KeyError(key) from None
            else:
                if exist_key is not None and exist_key != key:
                    raise CollisionError(key, exist_key, check)
        else:
            # We only have a checksum, so can't collide.
            try:
                self._values.pop(key)
            except KeyError:
                if not self._clear_unparsed(key):
                    raise KeyError(key) from None

    def __iter__(self) -> str:
        """Iterate over all known keys. Does not include checksum-only values."""
        for key, value in self._values.values():
            if key is not None:
                yield key
