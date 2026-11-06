"""Common types and concrete classes shared by different modules."""
import types

from typing import Protocol, TypeVar, Literal


AnyStr_co = TypeVar("AnyStr_co", str, bytes, covariant=True)
AnyStr_contra = TypeVar("AnyStr_contra", str, bytes, contravariant=True)


class FileR(Protocol[AnyStr_co]):
    """A readable file."""
    def read(self, count: int = ..., /) -> AnyStr_co: ...
    def __enter__(self, /) -> 'FileR[AnyStr_co]': ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
        /,
    ) -> bool | None: ...


class FileWText(Protocol):
    """A writable text file."""
    def write(self, data: str, /) -> object: ...


class FileWBinary(Protocol):
    """A writable binary file."""
    def write(self, data: bytes | bytearray, /) -> object: ...


class FileSeek(Protocol):
    """A seekable file."""
    def tell(self) -> int: ...
    def seek(self, pos: int, whence: Literal[0, 1, 2] = 0, /) -> object: ...


class FileRSeek(FileR[AnyStr_co], FileSeek, Protocol):
    """A readable and seekable file."""


class FileWTextSeek(FileWText, FileSeek, Protocol):
    """A writable and seekable text file."""


class FileWBinarySeek(FileWBinary, FileSeek, Protocol):
    """A writable and seekable binary file."""
