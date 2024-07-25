"""Implements a consistent interface for reading files.

This allows accessing different backing file sources interchangably.
Files are case-insensitive, and both slashes are converted to '/'.

With a filesystem object, you can index it to locate files inside (raising
:external:py:class:`FileNotFoundError` if missing). Once a :py:class:`File` object is obtained,
call :py:func:`~File.open_bin()` or :py:func:`~File.open_str()` to read the contents. Writing is not
supported.

* :py:class:`FileSystem` is the base :external:py:class:`~abc.ABC`, which can be subclassed to
  define additional subsystems.
* :py:class:`RawFileSystem` provides access to a directory, optionally prohibiting access to parent
  folders.
* :py:class:`ZipFileSystem` and :py:class:`VPKFileSystem` provide access to their respective
  archives.
* :py:class:`VirtualFileSystem` redirects to a mapping object, for fake file contents already
  in memory.
* :py:class:`FileSystemChain` contains multiple other systems, concatenating them together like
  how Source treats its search paths.

See the :py:class:`srctools.game.Game` class for parsing ``gameinfo.txt`` files into filesystems.
To mount a :py:class:`~srctools.bsp.BSP` file, use ``ZipFileSystem(bsp.pakfile)``.
"""
from typing import (
    Any, BinaryIO, Dict, Final, Generic, Iterator, List, Mapping, Optional, Set, TextIO,
    Tuple, Union, cast,
)
from typing_extensions import Self, TypeVar, deprecated
from zipfile import ZipFile, ZipInfo
import io
import os

from srctools import StringPath
from srctools.keyvalues import Keyvalues
from srctools.vpk import VPK, FileInfo as VPKFile


__all__ = [
    'File', 'FileSystem',
    'get_filesystem',
    'CACHE_KEY_INVALID', 'RootEscapeError',

    'RawFileSystem', 'VPKFileSystem', 'ZipFileSystem',
    'VirtualFileSystem', 'FileSystemChain',
]

FileSysT_co = TypeVar('FileSysT_co', bound='FileSystem', default='FileSystem', covariant=True)
CACHE_KEY_INVALID: Final = -1
"""This is returned from :py:meth:`FileSystem.cache_key()` to indicate no key could be computed."""

# This is the type of File._data. It should only be used by subclasses.
_FileDataT = TypeVar('_FileDataT', default=Any)


def get_filesystem(path: str) -> 'FileSystem[Any]':
    """Return a filesystem given a path.

    If the path is a directory this returns a RawFileSystem.
    Otherwise, it returns a VPK or zip, depending on extension.
    """
    if os.path.isdir(path):
        return RawFileSystem(path)
    ext = path[-4:]
    if ext == '.zip':
        return ZipFileSystem(path)
    if ext == '.vpk':
        return VPKFileSystem(path)
    raise ValueError(f'Unrecognised filesystem for "{path}"')


class RootEscapeError(ValueError):
    """Raised when a path tries to refer to a file outside the root of a filesystem."""
    root: str
    path: str

    def __init__(self, root: str, path: str) -> None:
        """Raised when a path tries to refer to a file outside the root of a filesystem."""
        super().__init__(root, path)
        self.root = root
        self.path = path

    def __str__(self) -> str:
        """Format a specific error."""
        return f'Path "{self.path}" escaped "{self.root}"!'


class File(Generic[FileSysT_co]):
    """Represents a file in a system. Should only be created by filesystems."""
    sys: FileSysT_co
    path: str
    # This is _FileDataT of the filesys, but leave as Any here so the File
    # doesn't need to expose that TypeVar. FileSystem._get_data does the type
    # check.
    _data: Any

    def __init__(
        self,
        system: FileSysT_co,
        path: str,
        data: Any,
    ) -> None:
        """Create a File.

        :param system: should be the filesystem which matches.
        :param path: is the relative path for the file.
        :param data: is filesystem-specific data, allowing directly opening the file.
        """
        self.sys = system
        self.path = path
        self._data = data

    def __fspath__(self) -> str:
        """This can be interpreted as a path."""
        return self.path

    def open_bin(self) -> BinaryIO:
        """Return a file-like object in bytes mode.

        This should be closed when done.
        """
        return self.sys.open_bin(self)

    def open_str(self, encoding: str = 'utf8') -> TextIO:
        """Return a file-like object in unicode mode.

        This should be closed when done.
        """
        return self.sys.open_str(self, encoding)

    def cache_key(self) -> int:
        """Return a checksum or last-modified date suitable for caching.

        This allows preventing reparsing the file. If not possible, return -1.
        """
        # File/Filesystem may use each other's internals.
        # noinspection PyProtectedMember
        return self.sys._get_cache_key(self)


class FileSystem(Generic[_FileDataT]):
    """Base class for different systems defining the interface."""
    path: str
    _ref_count: int

    def __init__(self, path: StringPath) -> None:
        self.path = os.fspath(path)
        self._ref_count = 0

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.path!r})'

    @deprecated('References concept removed, filesystems are always open.')
    def open_ref(self) -> None:
        """:deprecated: No longer needs to be called."""

    @deprecated('References concept removed, filesystems are always open.')
    def close_ref(self) -> None:
        """:deprecated: No longer needs to be called."""

    @deprecated('References concept removed, filesystems are always open.')
    def _check_open(self) -> None:
        """Ensure self._ref is valid."""

    def read_kv1(self, path: Union[str, File[Self]], encoding: str = 'utf8') -> Keyvalues:
        """Read a Keyvalues1 file from the filesystem.

        This handles opening and closing files.
        """
        with self.open_str(path, encoding) as file:
            return Keyvalues.parse(
                file,
                f'{self.path}:{path.path if isinstance(path, File) else path}',
            )

    @deprecated('Use FileSystem.read_kv1() instead.')
    def read_prop(self, path: str, encoding: str = 'utf8') -> Keyvalues:
        """Read a Keyvalues1 file from the filesystem.

        :deprecated: Use :py:meth:`~FileSystem.read_kv1()`.
        """
        return self.read_kv1(path, encoding)

    def __eq__(self, other: object) -> bool:
        """Filesystems are equal if they have the same type and same path."""
        if not isinstance(other, type(self)):
            return NotImplemented  # If both ours -> False
        return os.path.normpath(self.path) == os.path.normpath(other.path)

    def __hash__(self) -> int:
        return hash(type(self).__name__ + os.path.normpath(self.path))

    @deprecated('References concept removed, filesystems are always open.')
    def __enter__(self) -> Self:
        """:deprecated: No longer needs to be used as a context manager."""
        return self

    @deprecated('References concept removed, filesystems are always open.')
    def __exit__(self, *args: object) -> None:
        """:deprecated: No longer needs to be used as a context manager."""
        return None

    def __iter__(self) -> Iterator[File[Self]]:
        """Iteration yields each file."""
        return self.walk_folder('')

    def __getitem__(self, name: str) -> File[Self]:
        return self._get_file(name)

    def __contains__(self, name: str) -> bool:
        return self._file_exists(name)

    @classmethod
    def _get_data(cls, file: File[Self]) -> _FileDataT:
        """Accessor for file._data, to show the relationship to the type
        checker.

        It should only be available to that filesystem, and produces the type.
        """
        # File/Filesystem may use each other's internals.
        # noinspection PyProtectedMember
        return cast(_FileDataT, file._data)

    # The following should be overridden:
    def _file_exists(self, name: str) -> bool:
        try:
            self._get_file(name)
            return True
        except FileNotFoundError:
            return False

    def _get_file(self, name: str) -> File[Self]:
        """Return a specific file."""
        raise NotImplementedError

    def walk_folder(self, folder: str = '') -> Iterator[File[Self]]:
        """Iterate over all files in the specified subfolder, yielding each."""
        raise NotImplementedError

    def open_str(self, name: Union[str, File[Self]], encoding: str = 'utf8') -> TextIO:
        """Open a file in unicode mode or raise FileNotFoundError.

        This should be closed when done.
        """
        raise NotImplementedError

    def open_bin(self, name: Union[str, File[Self]]) -> BinaryIO:
        """Open a file in bytes mode or raise FileNotFoundError.

        This should be closed when done.
        """
        raise NotImplementedError

    def _get_cache_key(self, file: File[Self]) -> int:
        """Return a checksum or last-modified date suitable for caching.

        This allows preventing reparsing the file. If not possible, return CACHE_KEY_INVALID (-1).
        """
        return CACHE_KEY_INVALID


class FileSystemChain(FileSystem[File[FileSystem[Any]]]):
    """Chains several filesystem into one prioritised whole.

    Each system can additionally be filtered to only allow access to files inside a subfolder. These
    will appear as if they are at the root level.
    """

    def __init__(self, *systems: Union[FileSystem[Any], Tuple[FileSystem[Any], str]]) -> None:
        super().__init__('')
        self.systems: List[Tuple[FileSystem[Any], str]] = []
        for sys in systems:
            if isinstance(sys, tuple):
                self.add_sys(*sys)
            else:
                self.add_sys(sys)

    def __repr__(self) -> str:
        return '{}(\n{})'.format(
            self.__class__.__name__,
            ",\n ".join(map(repr, self.systems)),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FileSystemChain):
            return NotImplemented
        return self.systems == other.systems

    def __hash__(self) -> int:
        return hash(tuple(self.systems))

    @classmethod
    def get_system(cls, file: File[Self]) -> FileSystem[Any]:
        """Retrieve the system for a File, if it was produced from a FileSystemChain."""
        if not isinstance(file.sys, FileSystemChain):
            raise ValueError('File is not from a FileSystemChain..')
        return cls._get_data(file).sys

    def add_sys(
        self,
        sys: FileSystem[Any],
        prefix: str = '',
        *,
        priority: bool = False,
    ) -> None:
        """Add a filesystem to the list.

        :param priority: If True, the system will be added to the start, instead of the end.
        :param sys: The filesystem to add.
        :param prefix: If specified, only this subfolder's contents will be accessible, instead of the whole system.
        """
        if priority:
            self.systems.insert(0, (sys, prefix))
        else:
            self.systems.append((sys, prefix))

    def _get_file(self, name: str) -> File[Self]:
        """Search for a file on each filesystem in turn."""
        for sys, prefix in self.systems:
            full_name = os.path.join(prefix, name).replace('\\', '/')
            try:
                file_info = sys._get_file(full_name)
            except FileNotFoundError:
                continue
            # Pass the original file instance, so we can open
            # from the original system.
            return File(self, full_name, file_info)
        raise FileNotFoundError(name)

    def open_str(self, name: Union[str, File[Self]], encoding: str = 'utf8') -> TextIO:
        """Open a file in unicode mode or raise FileNotFoundError.

        This should be closed when done.
        """
        if isinstance(name, File):
            return self._get_data(name).open_str(encoding)
        return self._get_file(name).open_str(encoding)

    def open_bin(self, name: Union[str, File[Self]]) -> BinaryIO:
        """Open a file in bytes mode or raise FileNotFoundError.

        This should be closed when done.
        """
        if isinstance(name, File):
            return self._get_data(name).open_bin()
        return self._get_file(name).open_bin()

    def walk_folder(self, folder: str = '') -> Iterator[File[Self]]:
        """Walk folders, not repeating files.

        This requires temporarily storing the visited paths, to prevent revisiting them. If repeated
        access is not problematic, prefer :py:func:`~FileSystemChain.walk_folder_repeat()`.
        """
        done: Set[str] = set()
        for file in self.walk_folder_repeat(folder):
            folded = file.path.casefold()
            if folded in done:
                continue
            done.add(folded)
            yield file

    def walk_folder_repeat(self, folder: str = '') -> Iterator[File[Self]]:
        """Walk folders, but allow repeating files.

        If a file is contained in multiple systems, it will be yielded for each. The first is the
        highest-priority. Using this instead of  :py:func:`~FileSystemChain.walk_folder()` is
        cheaper, since a set of visited files must be maintained.
        """
        for sys, prefix in self.systems:
            full_folder = os.path.join(prefix, folder).replace('\\', '/')
            for file in sys.walk_folder(full_folder):
                yield File(
                    self,
                    os.path.relpath(file.path, prefix).replace('\\', '/'),
                    file,
                )

    def _get_cache_key(self, file: File[Self]) -> int:
        """Return the last modified time of this file.

        If individual timestamps are not stored, the modification time of the
        filesystem is returned instead."""
        # Delegate to the original File stored in ours.
        if not isinstance(file.sys, FileSystemChain):
            raise ValueError('File is not from a FileSystemChain..')
        return self._get_data(file).cache_key()


class VirtualFileSystem(FileSystem[str]):
    """Access a dict as if it were a filesystem.

    The dict should map file paths to either bytes or strings.
    The encoding arg specifies how text data is presented if open_bin()
    is called.
    """
    _mapping: Mapping[str, Tuple[str, Union[str, bytes, bytearray, memoryview]]]

    def __init__(self, mapping: Mapping[str, Union[str, bytes, bytearray, memoryview]], encoding: str = 'utf8') -> None:
        super().__init__('<virtual>')
        self._mapping = {
            self._clean_path(filename): (filename, data)
            for filename, data in
            dict(mapping).items()
        }
        self.bytes_encoding = encoding

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VirtualFileSystem):
            return NotImplemented
        return (
            self.bytes_encoding == other.bytes_encoding and
            self._mapping == other._mapping
        )

    def __hash__(self) -> int:
        return hash(self.bytes_encoding) ^ hash(tuple(self._mapping.values()))

    @classmethod
    def _clean_path(cls, path: Union[str, File[Self]]) -> str:
        """Convert paths to one representation."""
        if isinstance(path, File):
            path = path.path
        return os.path.normpath(path).replace('\\', '/').casefold()

    def open_bin(self, name: Union[str, File[Self]]) -> BinaryIO:
        """Return a bytes buffer for a 'file'."""
        try:
            filename, data = self._mapping[self._clean_path(name)]
        except KeyError:
            raise FileNotFoundError(name) from None
        if isinstance(data, str):
            data = data.encode(self.bytes_encoding)
        return io.BytesIO(data)

    def open_str(
        self,
        name: Union[str, File[Self]],
        encoding: str = 'utf8',
    ) -> TextIO:
        """Return a string buffer for a 'file'.

        This performs universal newlines conversion.
        The encoding argument is ignored for files which are
        originally text.
        """
        try:
            filename, data = self._mapping[self._clean_path(name)]
        except KeyError:
            raise FileNotFoundError(name) from None
        if isinstance(data, str):
            # None = universal newlines mode directly.
            # No encoding is needed obviously.
            return io.StringIO(data, newline=None)
        else:
            # Decode on the fly, with universal newlines.
            return io.TextIOWrapper(
                io.BytesIO(data),
                encoding=encoding,
            )

    def walk_folder(self, folder: str = '') -> Iterator[File[Self]]:
        """Return all files that are 'subfolders' of the provided folder."""
        folder = self._clean_path(folder)

        for filename, data in self._mapping.values():
            if filename.startswith(folder):
                yield File(self, filename, filename)

    def _file_exists(self, name: str) -> bool:
        return self._clean_path(name) in self._mapping

    def _get_file(self, name: str) -> File[Self]:
        """Access the specified file."""
        try:
            filename, data = self._mapping[self._clean_path(name)]
        except KeyError:
            raise FileNotFoundError(name) from None
        return File(self, filename, filename)


class RawFileSystem(FileSystem[str]):
    """Accesses files in a real folder.

    This can prohibit access to folders above the root.
    """
    def __init__(self, path: StringPath, constrain_path: bool = True) -> None:
        super().__init__(os.path.abspath(path))
        self.constrain_path = constrain_path

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}({self.path!r}, '
            f'constrain_path={self.constrain_path})'
        )

    def _resolve_path(self, path: str) -> str:
        """Get the absolute path."""
        abs_path = os.path.abspath(os.path.join(self.path, path))
        if self.constrain_path and not abs_path.startswith(self.path):
            raise RootEscapeError(self.path, path)
        return abs_path

    def walk_folder(self, folder: str = '') -> Iterator[File[Self]]:
        """Yield files in a folder."""
        path = self._resolve_path(folder)
        for dirpath, dirnames, filenames in os.walk(path):
            for file in filenames:
                rel_path = os.path.relpath(
                    os.path.join(dirpath, file),
                    self.path,
                ).replace('\\', '/')
                yield File(self, rel_path, rel_path)

    def open_str(self, name: Union[str, File[Self]], encoding: str = 'utf8') -> TextIO:
        """Open a file in unicode mode or raise FileNotFoundError.

        This should be closed when done.
        """
        if isinstance(name, File):
            name = self._get_data(name)
        return open(self._resolve_path(name), encoding=encoding)

    def open_bin(self, name: Union[str, File[Self]]) -> BinaryIO:
        """Open a file in bytes mode or raise FileNotFoundError.

        This should be closed when done.
        """
        if isinstance(name, File):
            name = self._get_data(name)
        return open(self._resolve_path(name), mode='rb')

    def _file_exists(self, name: str) -> bool:
        # We don't need this, but it should match other filesystems.
        return os.path.isfile(self._resolve_path(name))

    def _get_file(self, name: str) -> File[Self]:
        if os.path.isfile(self._resolve_path(name)):
            name = name.replace('\\', '/')
            return File(self, name, name)
        raise FileNotFoundError(name)

    def _get_cache_key(self, file: File[Self]) -> int:
        """Our cache key is the last modification time."""
        try:
            return os.stat(self._resolve_path(file.path)).st_mtime_ns
        except FileNotFoundError:
            return -1


class ZipFileSystem(FileSystem[ZipInfo]):
    """Accesses files in a zip file."""
    _no_close: bool
    zip: ZipFile

    def __init__(self, path: StringPath, zipfile: Optional[ZipFile] = None) -> None:
        super().__init__(path)

        if zipfile is not None:
            # The user owns the responsibility to close the zipfile, not us.
            self._no_close = True
            self.zip = zipfile
        else:
            self._no_close = False
            self.zip = ZipFile(path)

        self._name_to_info: Dict[str, ZipInfo] = {
            info.filename.casefold(): info
            for info in self.zip.infolist()
            # Some zip files include entries for the directories too.
            # They have a trailing slash.
            if not info.filename.endswith('/')
        }

    def walk_folder(self, folder: str = '') -> Iterator[File[Self]]:
        """Yield files in a folder."""
        # \\ is not allowed in zips.
        folder = folder.replace('\\', '/').casefold()
        for filename, fileinfo in self._name_to_info.items():
            if filename.startswith(folder):
                yield File(self, fileinfo.filename, fileinfo)

    def open_bin(self, name: Union[str, File[Self]]) -> BinaryIO:
        """Open a file in bytes mode or raise FileNotFoundError.

        The filesystem needs to be open while accessing this.
        """
        # We need the ZipInfo object, either direct from a File or via a lookup
        # on the spot.
        if isinstance(name, File):
            info = self._get_data(name)
        else:
            name = name.replace('\\', '/')
            try:
                info = self._name_to_info[name.casefold()]
            except KeyError:
                raise FileNotFoundError(f'{self.path}:{name}') from None

        # Type of open() is IO[bytes], basically the same.
        return cast(BinaryIO, self.zip.open(info))

    def open_str(
        self,
        name: Union[str, File[Self]],
        encoding: str = 'utf8',
    ) -> io.TextIOWrapper:
        """Open a file in unicode mode or raise FileNotFoundError.

        The filesystem needs to be open while accessing this.
        """
        # Zips only open in binary, so just open that, then wrap to decode.
        return io.TextIOWrapper(self.open_bin(name), encoding)

    def _get_file(self, name: str) -> File[Self]:
        name = name.replace('\\', '/')
        try:
            info = self._name_to_info[name.casefold()]
        except KeyError:
            raise FileNotFoundError(f'{self.path}:{name}') from None
        return File(self, name, info)

    def _file_exists(self, name: str) -> bool:
        return name.replace('\\', '/').casefold() in self._name_to_info

    def _get_cache_key(self, file: File[Self]) -> int:
        """Return the CRC of the VPK file."""
        return self._get_data(file).CRC


class VPKFileSystem(FileSystem[VPKFile]):
    """Accesses files in a VPK file."""
    vpk: VPK
    _name_to_file: Dict[str, VPKFile]

    def __init__(self, path: StringPath) -> None:
        super().__init__(path)
        self.vpk = VPK(self.path)
        # Used to enforce case-insensitivity.
        self._name_to_file = {
            file.filename.replace('\\', '/').casefold(): file
            for file in self.vpk
        }

    def _file_exists(self, name: str) -> bool:
        return name.casefold().replace('\\', '/') in self._name_to_file

    def _get_file(self, name: str) -> File[Self]:
        key = name.casefold().replace('\\', '/')
        try:
            file = self._name_to_file[key]
        except KeyError:
            raise FileNotFoundError(name) from None
        return File(self, key, file)

    def walk_folder(self, folder: str = '') -> Iterator[File[Self]]:
        """Yield files in a folder."""
        # All VPK files use forward slashes.
        folder = folder.replace('\\', '/')
        for file in self._name_to_file.values():
            if file.dir.startswith(folder):
                yield File(self, file.filename, file)

    def open_bin(self, name: Union[str, File[Self]]) -> BinaryIO:
        """Open a file in bytes mode or raise FileNotFoundError."""
        # Extract our VPK info directly.
        if isinstance(name, File):
            file = self._get_data(name)
        else:
            try:
                file = self._name_to_file[name.casefold().replace('\\', '/')]
            except KeyError:
                raise FileNotFoundError(name) from None
        return io.BytesIO(file.read())

    def open_str(
        self,
        name: Union[str, File[Self]],
        encoding: str = 'utf8',
    ) -> TextIO:
        """Open a file in unicode mode or raise FileNotFoundError."""
        # File() calls with the VPK object we need directly.
        if isinstance(name, File):
            file = self._get_data(name)
        else:
            try:
                file = self._name_to_file[name.casefold().replace('\\', '/')]
            except KeyError:
                raise FileNotFoundError(name) from None
        # Wrap the data to treat it as bytes, then
        # wrap that to decode and clean up universal newlines.
        return io.TextIOWrapper(io.BytesIO(file.read()), encoding)

    def _get_cache_key(self, file: File[Self]) -> int:
        """Return the CRC of the VPK file."""
        return self._get_data(file).crc
