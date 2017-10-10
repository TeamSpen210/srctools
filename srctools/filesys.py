"""Implements a consistent interface for accessing files.

This allows accessing raw files, zips and VPKs in the same way.
Files are case-insensitive, and both slashes are converted to '/'.
"""
from zipfile import ZipFile, ZipInfo
from tarfile import TarFile, TarInfo
import io
import os.path

from srctools.vpk import VPK, FileInfo as VPKFile
from srctools.property_parser import Property

from typing import Iterator, Union, List, Tuple, Dict

__all__ = [
    'File', 'FileSystem', 'get_filesystem',

    'RawFileSystem', 'VPKFileSystem', 'ZipFileSystem',
]


def get_filesystem(path: str) -> 'FileSystem':
    """Return a filesystem given a path.

    If the path is a directory this returns a RawFileSystem.
    Otherwise it returns a VPK or zip, depending on extension.
    """
    if os.path.isdir(path):
        return RawFileSystem(path)
    if path[-4:] == '.zip':
        return ZipFileSystem(path)
    if path[-8:] == '_dir.vpk':
        return VPKFileSystem(path)
    raise ValueError('Unrecognised filesystem for "{}"'.format(path))


class File:
    """Represents a file in a system. Should not be created directly."""
    def __init__(self, system: 'FileSystem', path: str, data=None):
        """Create a File.

        system should be the filesystem which matches.
        path is the relative path for the file.
        data is a filesystem-specific data, used to pass to open_bin() and open_str().
        """
        self.sys = system
        self.path = path
        self._data = path if data is None else data

    def __fspath__(self):
        """This can be interpreted as a path."""
        return self.path

    def open_bin(self):
        """Return a file-like object in bytes mode.

        This should be closed when done.
        """
        return self.sys.open_bin(self._data)

    def open_str(self, encoding='utf8'):
        """Return a file-like object in unicode mode.

        This should be closed when done.
        """
        return self.sys.open_str(self._data, encoding)


class FileSystem:
    """Base class for different systems defining the interface."""
    def __init__(self, path: str):
        self.path = path
        self._ref = None
        self._ref_count = 0
        
    def __repr__(self):
        """Default repr()."""
        return '{}({!r})'.format(self.__class__.__name__, self.path)

    def open_ref(self):
        """Lock open a reference to this system."""
        self._ref_count += 1
        if self._ref is None:
            self._create_ref()
            assert self._ref is not None

    def close_ref(self):
        """Reverse self.open_ref() - must be done in pairs."""
        self._ref_count -= 1
        if self._ref_count < 0:
            raise ValueError('Closed too many times!')
        if self._ref_count == 0 and self._ref is not None:
            self._delete_ref()
            assert self._ref is None

    def read_prop(self, path: str, encoding='utf8') -> Property:
        """Read a Property file from the filesystem.

        This handles opening and closing files.
        """
        with self, self.open_str(path, encoding) as file:
            return Property.parse(
                file,
                self.path + ':' + path,
            )

    def _check_open(self):
        """Ensure self._ref is valid."""
        if self._ref is None:
            raise ValueError(
                'The filesystem must have a valid reference to '
                'execute this method!')

    def __enter__(self):
        """Temporarily get access to the system's reference.

        This makes it possible to access files.
        """
        self.open_ref()
        assert self._ref is not None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_ref()
        assert self._ref is None

    def __iter__(self) -> Iterator[File]:
        return self.walk_folder('')

    def __getitem__(self, name: str):
        try:
            return self._get_file(name)
        except FileNotFoundError:
            raise KeyError

    def __contains__(self, name: str):
        return self._file_exists(name)

    def _file_exists(self, name: str) -> bool:
        try:
            self._get_file(name)
            return True
        except FileNotFoundError:
            return False

    def _get_file(self, name: str) -> File:
        """Return a specific file."""
        raise NotImplementedError

    def walk_folder(self, folder: str) -> Iterator[File]:
        """Yield files in a folder."""
        raise NotImplementedError

    def _create_ref(self) -> None:
        """Create the _ref object. 
        
        After this is called, _ref should not be None.
        """
        raise NotImplementedError

    def _delete_ref(self) -> None:
        """Destroy and clean up the _ref object.
        
        After this is called, _ref should be None.
        """
        raise NotImplementedError

    def open_str(self, name: str, encoding='utf8'):
        """Open a file in unicode mode or raise FileNotFoundError.

        This should be closed when done.
        """
        raise NotImplementedError

    def open_bin(self, name: str):
        """Open a file in bytes mode or raise FileNotFoundError.

        This should be closed when done.
        """
        raise NotImplementedError


class FileSystemChain(FileSystem):
    """Chains several filesystem into one prioritised whole."""

    def __init__(self, *systems: Union[FileSystem, Tuple[str, FileSystem]]):
        super().__init__('')
        self.systems = []  # type: List[Tuple[FileSystem, str]]
        for sys in systems:
            if isinstance(sys, tuple):
                prefix, sys = sys
                self.add_sys(sys, prefix)
            self.add_sys(sys)

    def __repr__(self):
        return 'FileSystemChain({!r})'.format(', '.join(map(repr, self.systems)))

    @staticmethod
    def get_system(file: File) -> FileSystem:
        """Retrieve the system for a File, if it was produced from a FileSystemChain."""
        if not isinstance(file.sys, FileSystemChain):
            raise ValueError('File is not from a FileSystemChain..')
        return file._data.sys

    def add_sys(self, sys: FileSystem, prefix=''):
        """Add a filesystem to the list."""
        self.systems.append((sys, prefix))
        # If we're currently open, apply that to the added systems.
        if self._ref_count > 0:
            sys.open_ref()

    def _get_file(self, name: str) -> File:
        """Search for a file on each filesystem in turn."""
        for sys, prefix in self.systems:
            full_name = os.path.join(prefix, name).replace('\\', '/')
            try:
                file_info = sys._get_file(full_name)
            except FileNotFoundError:
                pass
            else:
                # Pass the original file instance, so we can open
                # from the original system.
                return File(self, full_name, file_info)
        raise FileNotFoundError(name)

    def open_str(self, name: str, encoding='utf8'):
        """Open a file in unicode mode or raise FileNotFoundError.

        This should be closed when done.
        """
        if isinstance(name, File):
            return name.open_str(encoding)
        return self._get_file(name).open_str(encoding)

    def open_bin(self, name: str):
        """Open a file in bytes mode or raise FileNotFoundError.

        This should be closed when done.
        """
        if isinstance(name, File):
            return name.open_bin()
        return self._get_file(name).open_bin()

    def walk_folder(self, folder: str):
        """Walk folders, not repeating files."""
        done = set()
        for file in self.walk_folder_repeat(folder):
            folded = file.path.casefold()
            if folded in done:
                continue
            done.add(folded)
            yield file

    def walk_folder_repeat(self, folder: str=''):
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


class VirtualFileSystem(FileSystem):
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
        
    def __repr__(self):
        return '<VirtualFileSystem: {} files>'.format(len(self._mapping))

    @staticmethod
    def _clean_path(path: str) -> str:
        """Convert paths to one representation."""
        return os.path.normpath(path).replace('\\', '/').casefold()

    def open_bin(self, name: str):
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

    def open_str(self, name: str, encoding='utf8'):
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
            yield File(self, filename)

    def _file_exists(self, name: str) -> bool:
        return self._clean_path(name) in self._mapping

    def _get_file(self, name: str) -> File:
        # We don't need this, but it should match other filesystems.
        self._check_open()

        try:
            filename, data = self._mapping[self._clean_path(name)]
        except KeyError:
            raise FileNotFoundError(name)
        return File(self, filename)

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
    def __init__(self, path: str):
        super().__init__(os.path.abspath(path))     

    def _resolve_path(self, path: str) -> str:
        """Get the absolute path."""
        abs_path = os.path.abspath(os.path.join(self.path, path))
        if not abs_path.startswith(self.path):
            raise ValueError('Path "{}" escaped "{}"!'.format(path, self.path))
        return abs_path

    def walk_folder(self, folder: str):
        """Yield files in a folder."""
        path = self._resolve_path(folder)
        for dirpath, dirnames, filenames in os.walk(path):
            for file in filenames:
                rel_path = os.path.relpath(
                    os.path.join(dirpath, file),
                    self.path,
                )
                yield File(
                    self,
                    rel_path.replace('\\', '/'),
                )

    def open_str(self, name: str, encoding='utf8'):
        """Open a file in unicode mode or raise FileNotFoundError.

        This should be closed when done.
        """
        # We don't need this, but it should match other filesystems.
        self._check_open()

        return open(self._resolve_path(name), mode='rt', encoding=encoding)

    def open_bin(self, name: str):
        """Open a file in bytes mode or raise FileNotFoundError.

        This should be closed when done.
        """
        # We don't need this, but it should match other filesystems.
        self._check_open()

        return open(self._resolve_path(name), mode='rb')

    def _file_exists(self, name: str) -> bool:
        # We don't need this, but it should match other filesystems.
        self._check_open()

        return os.path.isfile(self._resolve_path(name))

    def _get_file(self, name: str):
        # We don't need this, but it should match other filesystems.
        self._check_open()

        if os.path.isfile(self._resolve_path(name)):
            return File(self, name.replace('\\', '/'))
        raise FileNotFoundError(name)

    def _delete_ref(self) -> None:
        """The raw filesystem doesn't need a reference to anything."""
        self._ref = None

    def _create_ref(self) -> None:
        """The raw filesystem doesn't need a reference to anything."""
        self._ref = True
        
class _ArchFileSys(FileSystem):
    """Common code for Zip and Tar systems."""
    
    # *Info class, for open_bin().
    _info_class = object
    
    def __init__(self, path: str):
        self._ref = None  # type: ZipFile
        self._name_to_info = {}
        super().__init__(path)
        
    def walk_folder(self, folder: str):
        """Yield files in a folder."""
        self._check_open()
        # \\ is not allowed in zips.
        folder = folder.replace('\\', '/').casefold()
        for filename, fileinfo in self._name_to_info.items():
            if filename.startswith(folder):
                yield File(self, fileinfo.filename, fileinfo)
        
    def open_bin(self, name: Union[str, TarInfo]):
        """Open a file in bytes mode or raise FileNotFoundError.

        The filesystem needs to be open while accessing this.
        """
        self._check_open()

        # We need the zipinfo object.
        if isinstance(name, self._info_class):
            info = name
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
        # Zip and Tar only open in binary, so just open that,
        # then wrap to decode.
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

class ZipFileSystem(_ArchFileSys):
    """Accesses files in a zip file."""
    _info_class = ZipInfo
    
    def __init__(self, path: str):
        self._ref = None  # type: ZipFile
        self._name_to_info = {}
        super().__init__(path)

    def _create_ref(self) -> None:
        self._ref = zipfile = ZipFile(self.path)
        self._name_to_info.clear()
        for info in zipfile.infolist():
            self._name_to_info[info.filename.casefold()] = info
            
TAR_COMP_NONE = ''
TAR_COMP_GZIP = 'gz'
TAR_COMP_BZIP2 = 'bz2'
TAR_COMP_LZMA = 'xz'
TAR_COMP_GUESS = '*'

_tar_comp_modes = {
    val for name, val in 
    globals().items() 
    if 'TAR_COMP' in name
}
_tar_comp_modes_str = ', '.join(map(repr, sorted(_tar_comp_modes)))
            
class TarFileSystem(_ArchFileSys):
    """Accesses files in a .tar file, compressed or uncompressed."""
    _info_class = TarInfo
    
    def __init__(self, path: str, compression: str=TAR_COMP_GUESS):
        self._ref = None  # type: ZipFile
        self._name_to_info = {}
        
        self._compression = compression.casefold()
        if self._compression not in _tar_comp_modes:
            raise ValueError('Invalid mode {!}! Valid modes: {}'.format(
                compression, 
                _tar_comp_modes_str,
            ))
            
        super().__init__(path)
    
    def _create_ref(self) -> None:
        self._ref = tarfile = ZipFile(self.path)
        self._name_to_info.clear()
        for info in tarfile.infolist():
            if info.isfile(): # Ignore folders, hardlinks, etc.
                self._name_to_info[info.filename.casefold()] = info


class VPKFileSystem(FileSystem):
    """Accesses files in a VPK file."""
    def __init__(self, path: str):
        self._ref = None  # type: VPK
        super().__init__(path)

    def __repr__(self):
        return 'VPKFileSystem({!r})'.format(self.path)

    def _create_ref(self):
        self._ref = VPK(self.path)

    def _delete_ref(self):
        # We only read from VPKs, so no cleanup needs to be done.
        self._ref = None

    def _file_exists(self, name: str):
        if self._ref is None:
            return name in self._ref

    def _get_file(self, name: str):
        try:
            file = self._ref[name]
        except KeyError:
            raise FileNotFoundError(name) from None
        return File(self, name.replace('\\', '/'), file)

    def walk_folder(self, folder: str):
        """Yield files in a folder."""
        # All VPK files use forward slashes.
        folder = folder.replace('\\', '/')
        for file in self._ref:
            if file.dir.startswith(folder):
                yield File(self, file.filename, file)

    def open_bin(self, name: str):
        """Open a file in bytes mode or raise FileNotFoundError.

        The return value is a BytesIO in-memory buffer.
        """
        with self:
            # File() calls with the VPK object we need directly.
            if isinstance(name, VPKFile):
                file = name
            else:
                try:
                    file = self._ref[name]
                except KeyError:
                    raise FileNotFoundError(name)
            return io.BytesIO(file.read())

    def open_str(self, name: str, encoding='utf8'):
        """Open a file in unicode mode or raise FileNotFoundError.

        The return value is a StringIO in-memory buffer.
        """
        with self:
            # File() calls with the VPK object we need directly.
            if isinstance(name, VPKFile):
                file = name
            else:
                try:
                    file = self._ref[name]
                except KeyError:
                    raise FileNotFoundError(name)
            # Wrap the data to treat it as bytes, then
            # wrap that to decode and clean up universal newlines.
            return io.TextIOWrapper(io.BytesIO(file.read()), encoding)