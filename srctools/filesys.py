"""Implements a consistent interface for accessing files.

This allows accessing raw files, zips and VPKs in the same way.
"""
from zipfile import ZipFile
import io
import os.path

from srctools.vpk import VPK, FileInfo as VPKFile
from srctools.property_parser import Property

from typing import Iterator

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
    def __init__(self, system: 'FileSystem', name: str, data=None):
        """Create a File.

        system should be the filesystem which matches.
        name is the filename for the file.
        data is a filesystem-specific data, used to pass to open_bin() and open_str().
        """
        self.sys = system
        self.name = name
        self._data = name if data is None else data

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

    def open_ref(self):
        """Lock open a reference to this system."""
        self._ref_count += 1
        if self._ref is None:
            self._create_ref()

    def close_ref(self):
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

    def _check_open(self):
        """Ensure self._ref is valid."""
        if self._ref is None:
            raise ValueError('The filesystem must have a valid reference!')

    def __enter__(self):
        """Temporarily get access to the system's reference.

        This makes it more efficient to access files.
        """
        self.open_ref()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_ref()

    def __iter__(self) -> Iterator[File]:
        return self.walk_folder('')

    def _file_exists(self, name: str) -> bool:
        """Check that a file exists."""
        raise NotImplementedError

    def walk_folder(self, folder: str):
        """Yield files in a folder."""
        raise NotImplementedError

    def _create_ref(self) -> None:
        """Create the _ref object."""
        raise NotImplementedError

    def _delete_ref(self) -> None:
        """Destroy and clean up the _ref object."""
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
                yield File(
                    self,
                    os.path.join(dirpath, file),
                )

    def open_str(self, name: str, encoding='utf8'):
        """Open a file in unicode mode or raise FileNotFoundError.

        This should be closed when done.
        """
        return open(self._resolve_path(name), mode='rt', encoding=encoding)

    def open_bin(self, name: str):
        """Open a file in bytes mode or raise FileNotFoundError.

        This should be closed when done.
        """
        return open(self._resolve_path(name), mode='rb')

    def _file_exists(self, name: str) -> bool:
        return os.path.exists(self._resolve_path(name))

    def _delete_ref(self) -> None:
        """The raw filesystem doesn't need a reference to anything."""
        self._ref = None

    def _create_ref(self) -> None:
        """The raw filesystem doesn't need a reference to anything."""
        self._ref = True


class ZipFileSystem(FileSystem):
    """Accesses files in a zip file."""
    def __init__(self, path: str):
        self._ref = None  # type: ZipFile
        super().__init__(path)

    def walk_folder(self, folder: str):
        """Yield files in a folder."""
        self._check_open()
        folder = folder.replace('\\', '/')
        for fileinfo in self._ref.infolist():
            tidy_path = fileinfo.filename.replace('\\', '/')
            if tidy_path.startswith(folder):
                yield File(self, fileinfo.filename, fileinfo)

    def open_bin(self, name: str):
        """Open a file in bytes mode or raise FileNotFoundError.

        The filesystem needs to be open while accessing this.
        """
        self._check_open()
        try:
            return self._ref.open(name)
        except KeyError:
            raise FileNotFoundError(name) from None

    def open_str(self, name: str, encoding='utf8'):
        """Open a file in unicode mode or raise FileNotFoundError.

        The filesystem needs to be open while accessing this.
        """
        self._check_open()
        try:
            return io.TextIOWrapper(self._ref.open(name), encoding)
        except KeyError:
            raise FileNotFoundError(name) from None

    def _file_exists(self, name: str) -> bool:
        self._check_open()
        try:
            self._ref.getinfo(name)
            return True
        except KeyError:
            return False

    def _delete_ref(self) -> None:
        self._ref.close()
        self._ref = None

    def _create_ref(self) -> None:
        self._ref = ZipFile(self.path)


class VPKFileSystem(FileSystem):
    """Accesses files in a VPK file."""
    def __init__(self, path: str):
        self._ref = None  # type: VPK
        super().__init__(path)

    def _create_ref(self):
        self._ref = VPK(self.path)

    def _delete_ref(self):
        # We only read from VPKs, so no cleanup needs to be done.
        self._ref = None

    def _file_exists(self, name: str):
        if self._ref is None:
            return name in self._ref

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
            return io.StringIO(file.read().decode(encoding))
