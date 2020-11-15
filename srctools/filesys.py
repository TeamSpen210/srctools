"""Implements a consistent interface for accessing files.

This allows accessing raw files, zips and VPKs in the same way.
Files are case-insensitive, and both slashes are converted to '/'.
"""
from zipfile import ZipFile, ZipInfo
import io
import os

from srctools.vpk import VPK, FileInfo as VPKFile
from srctools.property_parser import Property

from typing import (
    TypeVar, Generic, Any,
    Union, Optional, Tuple,
    Iterator, List, Dict, Set,
    TextIO, BinaryIO,
)


__all__ = [
    'File', 'FileSystem', 'get_filesystem',

    'RawFileSystem', 'VPKFileSystem', 'ZipFileSystem',
    'VirtualFileSystem', 'FileSystemChain',
]

FileSysT = TypeVar('FileSysT', bound='FileSystem')
ChildSysT = TypeVar('ChildSysT', bound='FileSystem')

# The following vars should only be used by subclasses:
_SysRefT = TypeVar('_SysRefT')  # the type of FileSystem._ref
_FileDataT = TypeVar('_FileDataT')  # the type of File._data


def get_filesystem(path: str) -> 'FileSystem':
    """Return a filesystem given a path.

    If the path is a directory this returns a RawFileSystem.
    Otherwise it returns a VPK or zip, depending on extension.
    """
    if os.path.isdir(path):
        return RawFileSystem(path)
    ext = path[-4:]
    if ext == '.zip':
        return ZipFileSystem(path)
    if ext == '.vpk':
        return VPKFileSystem(path)
    raise ValueError('Unrecognised filesystem for "{}"'.format(path))


class File(Generic[FileSysT]):
    """Represents a file in a system. Should not be created directly."""
    def __init__(
        self: 'File[FileSystem[_SysRefT, _FileDataT]]',
        system: 'FileSystem[_SysRefT, _FileDataT]',
        path: str,
        data: _FileDataT,
    ) -> None:
        """Create a File.

        system should be the filesystem which matches.
        path is the relative path for the file.
        data is a filesystem-specific data, used to pass to open_bin() and open_str().
        """
        self.sys = system
        self.path = path
        self._data = data

    def __fspath__(self) -> str:
        """This can be interpreted as a path."""
        return self.path

    def open_bin(self: 'File[FileSystem[_SysRefT, _FileDataT]]') -> BinaryIO:
        """Return a file-like object in bytes mode.

        This should be closed when done.
        """
        return self.sys.open_bin(self)

    def open_str(self: 'File[FileSystem[_SysRefT, _FileDataT]]', encoding='utf8') -> TextIO:
        """Return a file-like object in unicode mode.

        This should be closed when done.
        """
        return self.sys.open_str(self, encoding)

    def cache_key(self: 'File[FileSystem[_SysRefT, _FileDataT]]') -> int:
        """Return a checksum or last-modified date suitable for caching.

        This allows preventing re-parsing the file. If not possible, return -1.
        """
        return self.sys._get_cache_key(self)


class FileSystem(Generic[_SysRefT, _FileDataT]):
    """Base class for different systems defining the interface."""
    def __init__(self, path: Union[str, os.PathLike]) -> None:
        self.path = os.fspath(path)
        self._ref: Optional[_SysRefT] = None
        self._ref_count = 0

    def open_ref(self) -> None:
        """Lock open a reference to this system."""
        self._ref_count += 1
        if self._ref is None:
            self._create_ref()

    def close_ref(self) -> None:
        """Reverse self.open_ref() - must be done in pairs."""
        self._ref_count -= 1
        if self._ref_count < 0:
            raise ValueError('Closed too many times!')
        if self._ref_count == 0 and self._ref is not None:
            self._delete_ref()

    def read_prop(self, path: str, encoding='utf8') -> Property:
        """Read a Property file from the filesystem.

        This handles opening and closing files.
        """
        with self, self.open_str(path, encoding) as file:
            return Property.parse(
                file,
                self.path + ':' + path,
            )

    def _check_open(self) -> None:
        """Ensure self._ref is valid."""
        if self._ref is None:
            raise ValueError('The filesystem must have a valid reference!')

    def __eq__(self, other: object) -> bool:
        """Filesystems are equal if they have the same type and same path."""
        if not isinstance(other, type(self)):
            return NotImplemented  # If both ours -> False
        return os.path.normpath(self.path) == os.path.normpath(other.path)

    def __hash__(self) -> int:
        return hash(type(self).__name__ + os.path.normpath(self.path))

    def __enter__(self: FileSysT) -> FileSysT:
        """Temporarily get access to the system's reference.

        This makes it more efficient to access files.
        """
        self.open_ref()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close_ref()

    def __iter__(self: FileSysT) -> Iterator[File[FileSysT]]:
        return self.walk_folder('')

    def __getitem__(self: FileSysT, name: str) -> File[FileSysT]:
        return self._get_file(name)

    def __contains__(self, name: str) -> bool:
        return self._file_exists(name)

    @classmethod
    def _get_data(cls: FileSysT, file: 'File[FileSysT]') -> _FileDataT:
        """Accessor for file._data, to show the relationship to the type
        checker.

        It should only be available to that filesystem, and produces the type.
        """
        return file._data  # type: ignore

    # The following should be overridden:
    def _file_exists(self, name: str) -> bool:
        try:
            self._get_file(name)
            return True
        except FileNotFoundError:
            return False

    def _get_file(self: FileSysT, name: str) -> File[FileSysT]:
        """Return a specific file."""
        raise NotImplementedError

    def walk_folder(self: FileSysT, folder: str) -> Iterator[File[FileSysT]]:
        """Yield files in a folder."""
        raise NotImplementedError

    def _create_ref(self) -> None:
        """Create the _ref object."""
        raise NotImplementedError

    def _delete_ref(self) -> None:
        """Destroy and clean up the _ref object."""
        raise NotImplementedError

    def open_str(self: FileSysT, name: Union[str, File[FileSysT]], encoding='utf8') -> TextIO:
        """Open a file in unicode mode or raise FileNotFoundError.

        This should be closed when done.
        """
        raise NotImplementedError

    def open_bin(self: FileSysT, name: Union[str, File[FileSysT]]) -> BinaryIO:
        """Open a file in bytes mode or raise FileNotFoundError.

        This should be closed when done.
        """
        raise NotImplementedError

    def _get_cache_key(self: FileSysT, file: File[FileSysT]) -> int:
        """Return a checksum or last-modified date suitable for caching.

        This allows preventing re-parsing the file. If not possible, return -2.
        """
        return -1


class FileSystemChain(Generic[ChildSysT], FileSystem[None, File]):
    """Chains several filesystem into one prioritised whole."""

    def __init__(self, *systems: Union[ChildSysT, Tuple[str, ChildSysT]]) -> None:
        super().__init__('')
        self.systems: List[Tuple[ChildSysT, str]] = []
        for sys in systems:
            if isinstance(sys, tuple):
                self.add_sys(*sys)
            else:
                self.add_sys(sys)

    def __repr__(self) -> str:
        return 'FileSystemChain(\n{})'.format(',\n '.join(map(repr, self.systems)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FileSystemChain):
            return NotImplemented
        return self.systems == other.systems

    def __hash__(self) -> int:
        return hash(tuple(self.systems))

    @classmethod
    def get_system(cls, file: 'File[FileSystemChain[ChildSysT]]') -> ChildSysT:
        """Retrieve the system for a File, if it was produced from a FileSystemChain."""
        if not isinstance(file.sys, FileSystemChain):
            raise ValueError('File is not from a FileSystemChain..')
        return cls._get_data(file).sys

    def add_sys(
        self,
        sys: ChildSysT,
        prefix: str='',
        *,
        priority: bool=False,
    ) -> None:
        """Add a filesystem to the list.

        If priority is True, the system will be added to the start.
        """
        if priority:
            self.systems.insert(0, (sys, prefix))
        else:
            self.systems.append((sys, prefix))
        # If we're currently open, apply that to the added systems.
        if self._ref_count > 0:
            sys.open_ref()

    def _get_file(self: 'FileSystemChain[ChildSysT]', name: str) -> File[ChildSysT]:
        """Search for a file on each filesystem in turn."""
        self._check_open()
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

    def open_str(self, name: Union[str, File[ChildSysT]], encoding: str = 'utf8') -> TextIO:
        """Open a file in unicode mode or raise FileNotFoundError.

        This should be closed when done.
        """
        if isinstance(name, File):
            return self._get_data(name).open_str(encoding)
        return self._get_file(name).open_str(encoding)

    def open_bin(self, name: Union[str, File[ChildSysT]]) -> BinaryIO:
        """Open a file in bytes mode or raise FileNotFoundError.

        This should be closed when done.
        """
        if isinstance(name, File):
            return self._get_data(name).open_bin()
        return self._get_file(name).open_bin()

    def walk_folder(self, folder: str) -> Iterator[File]:
        """Walk folders, not repeating files."""
        done: Set[str] = set()
        for file in self.walk_folder_repeat(folder):
            folded = file.path.casefold()
            if folded in done:
                continue
            done.add(folded)
            yield file

    def walk_folder_repeat(self, folder: str='') -> Iterator[File[ChildSysT]]:
        """Walk folders, but allow repeating files.

        If a file is contained in multiple systems, it will be yielded
        for each. The first is the highest-priority.
        """
        for sys, prefix in self.systems:
            full_folder = os.path.join(prefix, folder).replace('\\', '/')
            for file in sys.walk_folder(full_folder):
                yield File(
                    self,
                    os.path.relpath(file.path, prefix).replace('\\', '/'),
                    file,
                )

    def _delete_ref(self) -> None:
        """Creating and deleting refs affects the underlying systems."""
        for sys, prefix in self.systems:
            sys.close_ref()
        self._ref = None

    def _create_ref(self) -> None:
        """Creating and deleting refs affects the underlying systems."""
        for sys, prefix in self.systems:
            sys.open_ref()
        self._ref = True

    def _get_cache_key(self, file: File[ChildSysT]) -> int:
        """Return the last modified time of this file.

        If individual timestamps are not stored, the modification time of the
        filesystem is returned instead."""
        # Delegate to the original File stored in ours.
        if not isinstance(file.sys, FileSystemChain):
            raise ValueError('File is not from a FileSystemChain..')
        return self._get_data(file).cache_key()


class VirtualFileSystem(FileSystem[Any, str]):
    """Access a dict as if it were a filesystem.

    The dict should map file paths to either bytes or strings.
    The encoding arg specifies how text data is presented if open_bin()
    is called.
    """

    def __init__(self, mapping: Dict[str, Union[str, bytes]], encoding='utf8'):
        super().__init__('<virtual>')
        self._mapping = {
            self._clean_path(filename): (filename, data)
            for filename, data in
            dict(mapping).items()
        }
        self.bytes_encoding = encoding

    def __eq__(self, other: object):
        if not isinstance(other, VirtualFileSystem):
            return NotImplemented
        return (
            self.bytes_encoding == other.bytes_encoding and
            self._mapping == other._mapping
        )

    def __hash__(self):
        return hash(self.bytes_encoding) ^ hash(tuple(self._mapping.values()))

    @staticmethod
    def _clean_path(path: str) -> str:
        """Convert paths to one representation."""
        return os.path.normpath(path).replace('\\', '/').casefold()

    def open_bin(self, name: Union[str, File['VirtualFileSystem']]) -> BinaryIO:
        """Return a bytes buffer for a 'file'."""
        # We don't need this, but it should match other filesystems.
        self._check_open()

        try:
            filename, data = self._mapping[self._clean_path(name)]
        except KeyError:
            raise FileNotFoundError(name)
        if isinstance(data, str):
            data = data.encode(self.bytes_encoding)
        return io.BytesIO(data)

    def open_str(
        self,
        name: Union[str, File['VirtualFileSystem']],
        encoding: str = 'utf8',
    ) -> TextIO:
        """Return a string buffer for a 'file'.

        This performs universal newlines conversion.
        The encoding argument is ignored for files which are
        originally text.
        """
        # We don't need this, but it should match other filesystems.
        self._check_open()

        try:
            filename, data = self._mapping[self._clean_path(name)]
        except KeyError:
            raise FileNotFoundError(name)
        if isinstance(data, bytes):
            # Decode on the fly, with universal newlines.
            return io.TextIOWrapper(
                io.BytesIO(data),
                encoding=encoding,
            )
        else:
            # None = universal newlines mode directly.
            # No encoding is needed obviously.
            return io.StringIO(data, newline=None)

    def walk_folder(self, folder: str) -> Iterator[File]:
        # We don't need this, but it should match other filesystems.
        self._check_open()

        for filename, data in self._mapping.values():
            yield File(self, filename, filename)

    def _file_exists(self, name: str) -> bool:
        return self._clean_path(name) in self._mapping

    def _get_file(self, name: str) -> File['VirtualFileSystem']:
        # We don't need this, but it should match other filesystems.
        self._check_open()

        try:
            filename, data = self._mapping[self._clean_path(name)]
        except KeyError:
            raise FileNotFoundError(name)
        return File(self, filename, filename)

    def _delete_ref(self) -> None:
        """The virtual filesystem doesn't need a reference to anything."""
        self._ref = None

    def _create_ref(self) -> None:
        """The virtual filesystem doesn't need a reference to anything."""
        self._ref = True


class RawFileSystem(FileSystem):
    """Accesses files in a real folder.

    This prohibits access to folders above the root.
    """
    def __init__(self, path: Union[str, os.PathLike]):
        super().__init__(os.path.abspath(path))

    def __repr__(self):
        return 'RawFileSystem({!r})'.format(self.path)

    def _resolve_path(self, path: str) -> str:
        """Get the absolute path."""
        abs_path = os.path.abspath(os.path.join(self.path, path))
        if not abs_path.startswith(self.path):
            raise ValueError('Path "{}" escaped "{}"!'.format(path, self.path))
        return abs_path

    def walk_folder(self, folder: str) -> Iterator[File]:
        """Yield files in a folder."""
        path = self._resolve_path(folder)
        for dirpath, dirnames, filenames in os.walk(path):
            for file in filenames:
                rel_path = os.path.relpath(
                    os.path.join(dirpath, file),
                    self.path,
                ).replace('\\', '/')
                yield File(self, rel_path, rel_path)

    def open_str(self, name: Union[str, File['RawFileSystem']], encoding: str = 'utf8') -> TextIO:
        """Open a file in unicode mode or raise FileNotFoundError.

        This should be closed when done.
        """
        # We don't need this, but it should match other filesystems.
        self._check_open()
        if isinstance(name, File):
            name = self._get_data(name)
        return open(self._resolve_path(name), mode='rt', encoding=encoding)

    def open_bin(self, name: Union[str, File['RawFileSystem']]) -> BinaryIO:
        """Open a file in bytes mode or raise FileNotFoundError.

        This should be closed when done.
        """
        # We don't need this, but it should match other filesystems.
        self._check_open()
        if isinstance(name, File):
            name = self._get_data(name)
        return open(self._resolve_path(name), mode='rb')

    def _file_exists(self, name: str) -> bool:
        # We don't need this, but it should match other filesystems.
        self._check_open()

        return os.path.isfile(self._resolve_path(name))

    def _get_file(self, name: str):
        # We don't need this, but it should match other filesystems.
        self._check_open()

        if os.path.isfile(self._resolve_path(name)):
            name = name.replace('\\', '/')
            return File(self, name, name)
        raise FileNotFoundError(name)

    def _delete_ref(self) -> None:
        """The raw filesystem doesn't need a reference to anything."""
        self._ref = None

    def _create_ref(self) -> None:
        """The raw filesystem doesn't need a reference to anything."""
        self._ref = True

    def _get_cache_key(self, file: File['RawFileSystem']) -> int:
        """Our cache key is the last modification time."""
        try:
            return os.stat(self._resolve_path(file.path)).st_mtime_ns
        except FileNotFoundError:
            return -1


class ZipFileSystem(FileSystem[ZipFile, ZipInfo]):
    """Accesses files in a zip file."""
    def __init__(self, path: Union[str, os.PathLike], zipfile: Optional[ZipFile]=None) -> None:
        super().__init__(path)
        self._name_to_info: Dict[str, ZipInfo] = {}

        if zipfile is not None:
            # Use the zipfile directly, and don't close it.
            self._ref_count += 1
            self._ref = zipfile
            self._load_paths()

    def __repr__(self) -> str:
        return 'ZipFileSystem({!r})'.format(self.path)

    def walk_folder(self, folder: str) -> Iterator[File]:
        """Yield files in a folder."""
        self._check_open()
        # \\ is not allowed in zips.
        folder = folder.replace('\\', '/').casefold()
        for filename, fileinfo in self._name_to_info.items():
            if filename.startswith(folder):
                yield File(self, fileinfo.filename, fileinfo)

    def open_bin(self, name: str):
        """Open a file in bytes mode or raise FileNotFoundError.

        The filesystem needs to be open while accessing this.
        """
        self._check_open()

        # We need the zipinfo object.
        if isinstance(name, File):
            info = self._get_data(name)
        else:
            name = name.replace('\\', '/')
            try:
                info = self._name_to_info[name.casefold()]
            except KeyError:
                raise FileNotFoundError('{}:{}'.format(self.path, name)) from None

        return self._ref.open(info)

    def open_str(self, name: str, encoding='utf8'):
        """Open a file in unicode mode or raise FileNotFoundError.

        The filesystem needs to be open while accessing this.
        """
        # Zips only open in binary, so just open that, then wrap to decode.
        return io.TextIOWrapper(self.open_bin(name), encoding)

    def _get_file(self, name: str) -> File:
        name = name.replace('\\', '/')
        self._check_open()
        try:
            info = self._name_to_info[name.casefold()]
        except KeyError:
            raise FileNotFoundError('{}:{}'.format(self.path, name))
        return File(self, name, info)

    def _file_exists(self, name: str) -> bool:
        self._check_open()
        return name.replace('\\', '/').casefold() in self._name_to_info

    def _delete_ref(self) -> None:
        self._ref.close()
        self._name_to_info.clear()
        self._ref = None

    def _create_ref(self) -> None:
        self._ref = ZipFile(self.path)
        self._load_paths()

    def _load_paths(self) -> None:
        self._name_to_info.clear()
        for info in self._ref.infolist():
            # Some zipfiles include entries for the directories too. They have
            # a trailing slash.
            if not info.filename.endswith('/'):
                self._name_to_info[info.filename.casefold()] = info

    def _get_cache_key(self, file: File['ZipFileSystem']) -> int:
        """Return the CRC of the VPK file."""
        return self._get_data(file).CRC


class VPKFileSystem(FileSystem[VPK, VPKFile]):
    """Accesses files in a VPK file."""
    def __init__(self, path: Union[str, os.PathLike]):
        super().__init__(path)
        # Used to enforce case-insensitivity.
        self._name_to_file: Dict[str, VPKFile] = {}

    def __repr__(self) -> str:
        return 'VPKFileSystem({!r})'.format(self.path)

    def _create_ref(self) -> None:
        self._ref = VPK(self.path)

        self._name_to_file.clear()
        for file in self._ref:
            self._name_to_file[
                file.filename.replace('\\', '/').casefold()
            ] = file

    def _delete_ref(self) -> None:
        # We only read from VPKs, so no cleanup needs to be done.
        self._ref = None
        self._name_to_file.clear()

    def _file_exists(self, name: str) -> bool:
        self._check_open()
        return name in self._name_to_file

    def _get_file(self, name: str) -> File['VPKFileSystem']:
        key = name.casefold().replace('\\', '/')
        try:
            file = self._name_to_file[key]
        except KeyError:
            raise FileNotFoundError(name) from None
        return File(self, key, file)

    def walk_folder(self, folder: str) -> Iterator[File['VPKFileSystem']]:
        """Yield files in a folder."""
        # All VPK files use forward slashes.
        folder = folder.replace('\\', '/')
        for file in self._name_to_file.values():
            if file.dir.startswith(folder):
                yield File(self, file.filename, file)

    def open_bin(self, name: Union[str, File['VPKFileSystem']]) -> BinaryIO:
        """Open a file in bytes mode or raise FileNotFoundError."""
        with self:
            # Extract our VPK info directly.
            if isinstance(name, File):
                file = self._get_data(name)
            else:
                try:
                    file = self._name_to_file[name.casefold().replace('\\', '/')]
                except KeyError:
                    raise FileNotFoundError(name)
            return io.BytesIO(file.read())

    def open_str(
        self,
        name: Union[str, File['VPKFileSystem']],
        encoding: str = 'utf8',
    ) -> TextIO:
        """Open a file in unicode mode or raise FileNotFoundError."""
        with self:
            # File() calls with the VPK object we need directly.
            if isinstance(name, File):
                file = self._get_data(name)
            else:
                try:
                    file = self._name_to_file[name.casefold().replace('\\', '/')]
                except KeyError:
                    raise FileNotFoundError(name)
            # Wrap the data to treat it as bytes, then
            # wrap that to decode and clean up universal newlines.
            return io.TextIOWrapper(io.BytesIO(file.read()), encoding)

    def _get_cache_key(self, file: File['VPKFileSystem']) -> int:
        """Return the CRC of the VPK file."""
        return self._get_data(file).crc

