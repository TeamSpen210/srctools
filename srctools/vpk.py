"""Classes for reading and writing Valve's VPK format, version 1."""
import os
import struct
import operator
from enum import Enum
from typing import Union, Dict, Optional, List, Tuple, Iterator, BinaryIO, IO

from srctools.binformat import checksum, EMPTY_CHECKSUM, struct_read


VPK_SIG = 0x55aa1234  # First byte of the file..
DIR_ARCH_INDEX = 0x7fff  # File index used for the _dir file.

FileName = Union[str, Tuple[str, str], Tuple[str, str, str]]


class OpenModes(Enum):
    """Modes for opening VPK files."""
    READ = 'r'
    WRITE = 'w'
    APPEND = 'a'

    @property
    def writable(self):
        return self.value in 'wa'


def iter_nullstr(file: BinaryIO) -> Iterator[str]:
    """Read a null-terminated ASCII string from the file.
    
    This continuously yields strings, with empty strings 
    indicting the end of a section.
    """
    chars = bytearray()
    while True:
        char = file.read(1)
        if char == b'\x00':
            string = chars.decode('ascii')
            chars.clear()
            
            if string == ' ':  # Blank strings are saved as ' '
                yield ''
            elif string == '':
                return  # Actual blanks end the array.
            else:
                yield string
        elif char == b'':
            raise Exception('Reached EOF without null-terminator in {}!'.format(bytes(chars)))
        else:
            chars.extend(char)


def _write_nullstring(file: IO[bytes], string: str) -> None:
    """Write a null-terminated ASCII string back to the file."""
    if string:
        file.write(string.encode('ascii') + b'\x00')
    else:
        # Empty strings are written as a space.
        file.write(b' \x00')


def get_arch_filename(prefix='pak01', index: int=None):
    """Generate the name for a VPK file.
    
    Prefix is the name of the file, usually 'pak01'.
    index is the index of the data file, or None for the directory.
    """
    if index is None:
        return prefix + '_dir.vpk'
    else:
        return '{}_{!s:>03}.vpk'.format(prefix, index)


def _get_file_parts(value: FileName, relative_to='') -> Tuple[str, str, str]:
        """Get folder, name, ext parts from a string/tuple.
        
        Possible arguments:
            'fold/ers/name.ext'
            ('fold/ers', 'name.ext')
            ('fold/ers', 'name', 'ext')
        """
        if isinstance(value, str):
            path, filename = os.path.split(value)
            ext = ''
        elif len(value) == 2:
            path, filename = value
            ext = ''
        else:
            path, filename, ext = value
        
        if not ext and '.' in filename:
            filename, ext = filename.rsplit('.', 1)

        if relative_to:
            path = os.path.relpath(path, relative_to)

        # Strip '/' off the end, and './' from the beginning.
        path = os.path.normpath(path).replace('\\', '/').rstrip('/')

        # Special case - empty path gets returned as '.'...
        if path == '.':
            path = ''
            
        return path, filename, ext


def _join_file_parts(path, filename, ext):
    """Join together path components to the full path.
    
    Any of the segments can be blank, to skip them.
    """
    return (path + '/' if path else '') + filename + ('.' + ext if ext else '')


class FileInfo:
    """Represents a file stored inside a VPK."""

    __slots__ = (
        'vpk',
        'dir', '_filename', 'ext',
        'crc',
        'arch_index',
        'start_data',
        'offset',
        'arch_len',
    )

    def __init__(
        self,
        vpk: 'VPK',
        directory: str,
        file: str,
        ext: str,
        crc: int,
        start_data: bytes=b'',
        offset: int=0,
        arch_len: int=0,
        arch_index: int=None,
    ) -> None:
        """This should only be called by VPK() internally."""

        # Assert the path is ASCII.
        try:
            directory.encode('ascii')
            file.encode('ascii')
            ext.encode('ascii')
        except UnicodeError:
            raise ValueError(
                'VPK filenames are required to be in ASCII format!'
            )

        self.vpk = vpk
        self.dir = directory
        self._filename = file
        self.ext = ext
        self.crc = crc
        self.arch_index = arch_index  # pack_01_000.vpk file to use, or None for _dir.
        self.offset = offset  # Offset into the archive file, including directory data if in _dir
        self.arch_len = arch_len  # Number of bytes in archive files
        self.start_data = start_data  # The bytes saved into the directory header

    @property
    def filename(self) -> str:
        """The full filename for this file."""
        return _join_file_parts(self.dir, self._filename, self.ext)
    name = filename
        
    def __repr__(self):
        return '<VPK File: "{}">'.format(
            _join_file_parts(self.dir, self._filename, self.ext),
        )

    @property
    def size(self) -> int:
        """The total size of this file."""
        return self.arch_len + len(self.start_data)
        
    def read(self):
        """Return the contents for this file."""
        if self.arch_len:
            if self.arch_index is None:
                return self.start_data + self.vpk.footer_data[self.offset: self.offset + self.arch_len]
            else:
                arch_file = get_arch_filename(self.vpk.file_prefix, self.arch_index)
                with open(os.path.join(self.vpk.folder, arch_file), 'rb') as data:
                    data.seek(self.offset)
                    return self.start_data + data.read(self.arch_len)
        else:
            return self.start_data
            
    def verify(self):
        """Check this file matches the checksum."""
        chk = checksum(self.start_data)
        if self.arch_len:
            if self.arch_index is None:
                chk = checksum(
                    self.vpk.footer_data[self.offset: self.offset + self.arch_len],
                    chk
                 )
            else:
                arch_file = get_arch_filename(self.vpk.file_prefix, self.arch_index)
                with open(os.path.join(self.vpk.folder, arch_file), 'rb') as data:
                    data.seek(self.offset)
                    chk = checksum(
                        data.read(self.arch_len),
                        chk,
                    )
        return chk == self.crc
           
    def write(self, data: bytes, arch_index=None):
        """Replace this file with the given byte data.
        
        arch_index is the pak_01_000 file to put data into (or None for _dir).
        If this file already exists in the VPK, the old data is not removed. 
        For this reason VPK writes should be done once per file if possible.
        """
        self.vpk._check_writable()
        # Split the file based on a certain limit.

        new_checksum = checksum(data)

        if new_checksum == self.crc:
            return  # Same data, don't do anything.
        
        self.crc = new_checksum
        
        self.start_data = data[:self.vpk.dir_limit]
        arch_data = data[self.vpk.dir_limit:]
        
        self.arch_len = len(arch_data)
        
        if self.arch_len:
            self.arch_index = arch_index
            arch_file = get_arch_filename(self.vpk.file_prefix, arch_index)
            with open(os.path.join(self.vpk.folder, arch_file), 'ab') as file:
                self.offset = file.seek(0, os.SEEK_END)
                file.write(arch_data)
        else:
            # Only stored in the main index
            self.arch_index = None
            self.offset = 0


class VPK:
    """Represents a VPK file set in a directory."""
    def __init__(
        self,
        dir_file,
        *,
        mode: Union[OpenModes, str]='r',
        dir_data_limit: Optional[int]=1024,
        version: int=1,
    ) -> None:
        """Create a VPK file.
        
        Parameters:
            dir_file: The path to the directory file. This must end in '_dir.vpk'.
            mode: Open in (r)ead, (w)rite or (a)ppend mode.
               In read mode, the file will not be modified and it must exist. 
               Write mode will create the directory if needed.
               Append mode will also create the directory, but not wipe the file.
            dir_data_limit: The maximum amount of data for files saved to the dir file.
               None = no limit, and 0=save all to a data file.
            version: The desired version if the file is not read.
        """
        if version not in (1, 2):
            raise ValueError("Invalid version ({}) - must be 1 or 2!".format(version))

        self.folder = self.file_prefix = ''
        self.path = dir_file

        # fileinfo[extension][directory][filename]
        self._fileinfo = {}  # type: Dict[str, Dict[str, Dict[str, FileInfo]]]
        
        self.mode = OpenModes(mode)
        self.dir_limit = dir_data_limit
        
        self.footer_data = b''

        self.version = version
        self.header_len = 0
        
        self.load_dirfile()
        
    def _check_writable(self):
        """Verify that this is writable."""
        if not self.mode.writable:
            raise ValueError("Can't write with this mode!")

    @property
    def path(self):
        """Return the location of the directory VPK file."""
        return os.path.join(self.folder, self.file_prefix + '_dir.vpk')

    @path.setter
    def path(self, path: str):
        """Set the location and folder from the directory VPK file."""
        folder, filename = os.path.split(path)

        if not filename.endswith('_dir.vpk'):
            raise Exception('Must create with a _dir VPK file!')

        self.folder = folder
        self.file_prefix = filename[:-8]

    def load_dirfile(self):
        """Read in the directory file to get all filenames.
        
        This erases all changes in the file.
        """
        if self.mode is OpenModes.WRITE:
            # Erase the directory file, we ignore current contents.
            open(self.path, 'wb').close()
            self.version = 1
            return

        try:
            dirfile = open(self.path, 'rb')
        except FileNotFoundError:
            if self.mode is OpenModes.APPEND:
                # No directory file - generate a blank file.
                open(self.path, 'wb').close()
                self.version = 1
                return
            else:
                raise  # In read mode, don't overwrite and error when reading.

        with dirfile:
            vpk_sig, version, tree_length = struct_read('<III', dirfile)
            
            if vpk_sig != VPK_SIG:
                raise ValueError('Bad VPK directory signature!')
            
            if version not in (1, 2):
                raise ValueError("Bad VPK version {}!".format(self.version))

            self.version = version

            if version >= 2:
                (
                    data_size,
                    ext_md5_size,
                    dir_md5_size,
                    sig_size,
                ) = struct_read('<4I', dirfile)
                
            self.header_len = dirfile.tell() + tree_length
            
            self._fileinfo.clear()
            
            # Read directory contents
            # These are in a tree of extension, directory, file. '' terminates a part.
            for ext in iter_nullstr(dirfile):
                ext_dict = self._fileinfo.setdefault(ext, {})
                for directory in iter_nullstr(dirfile):
                    dir_dict = ext_dict.setdefault(directory, {})
                    for file in iter_nullstr(dirfile):
                        crc, index_len, arch_ind, offset, arch_len, end = struct_read('<IHHIIH', dirfile)
                        if arch_ind == DIR_ARCH_INDEX:
                            arch_ind = None
                            
                        if arch_len == 0:
                            offset = 0
                        
                        if end != 0xffff:
                            raise Exception('"{}" has bad terminator! {}'.format(
                                _join_file_parts(directory, file, ext),
                                (crc, index_len, arch_ind, offset, arch_len, end),
                            ))
                        dir_dict[file] = FileInfo(
                            self,
                            directory,
                            file,
                            ext,
                            crc=crc,
                            offset=offset,
                            start_data=dirfile.read(index_len),
                            arch_len=arch_len,
                            arch_index=arch_ind,
                        )
                
                # 1 for the ending b'' section
                if dirfile.tell() + 1 == self.header_len:
                    dirfile.read(1)  # Skip null byte.
                    break

            self.footer_data = dirfile.read()
    

    def write_dirfile(self):
        """Write the directory file with the changes.
        
        This must be performed after writing to the VPK.
        """
        self._check_writable()

        if self.version > 1:
            raise NotImplementedError("Can't write V2 VPKs!")
        
        # We don't know how big the directory section is, so we first write the directory,
        # then come back and overwrite the length value.
        with open(self.path, 'wb') as file:
            file.write(struct.pack('<III', VPK_SIG, self.version, 0))
            header_len = file.tell()
            key_getter = operator.itemgetter(0)

            # Write in sorted order - not required, but this ensures multiple
            # saves are deterministic.
            for ext, folders in sorted(self._fileinfo.items(), key=key_getter):
                _write_nullstring(file, ext)
                for folder, files in sorted(folders.items(), key=key_getter):
                    _write_nullstring(file, folder)
                    for filename, info in sorted(files.items(), key=key_getter):
                        _write_nullstring(file, filename)
                        if info.arch_index is None:
                            arch_ind = DIR_ARCH_INDEX
                        else:
                            arch_ind = info.arch_index
                        file.write(struct.pack(
                            '<IHHIIH',
                            info.crc,
                            len(info.start_data),
                            arch_ind,
                            info.offset,
                            info.arch_len,
                            0xffff,
                        ))
                        file.write(info.start_data)
                        # Each block is terminated by an empty null-terminated
                        # string -> one null byte.
                    file.write(b'\x00')
                file.write(b'\x00')
            file.write(b'\x00')

            # Calculate the length of the header..
            dir_len = file.tell() - header_len
                
            file.write(self.footer_data)
                
            # Write the directory size now we know it.
            file.seek(struct.calcsize('<II'))  # Skip signature and version
            file.write(struct.pack('<I', dir_len))
                
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_trace):
        """When exiting a context sucessfully, the index will be saved."""
        if exc_type is None and self.mode.writable:
            self.write_dirfile()
       
    def __getitem__(self, item: FileName) -> FileInfo:
        """Get the FileInfo object for a file.
        
        Possible arguments:
            vpk['folders/name.ext']
            vpk['folders', 'name.ext']
            vpk['folders', 'name', 'ext']
        """
        path, filename, ext = _get_file_parts(item)
        
        try:
            return self._fileinfo[ext][path][filename]
        except KeyError:
            raise KeyError(
                'No file "{}"!'.format(
                    _join_file_parts(path, filename, ext)
                )) from None
                
    def __delitem__(self, item: FileName):
        """Delete a file.
        
        Possible arguments:
            del vpk['folders/name.ext']
            del vpk['folders', 'name.ext']
            del vpk['folders', 'name', 'ext']
        """
        self._check_writable()
        
        path, filename, ext = _get_file_parts(item)
        
        try:
            self._fileinfo[ext][path].pop(filename)
        except KeyError:
            raise KeyError(
                'No file "{}"!'.format(
                    _join_file_parts(path, filename, ext)
                )) from None
                
    def __iter__(self) -> Iterator[FileInfo]:
        """Yield all FileInfo objects."""
        for ext, folders in self._fileinfo.items():
            for folder, files in folders.items():
                for file, info in files.items():
                    yield info
                    
    def filenames(self) -> Iterator[str]:
        """Yield all filenames in this VPK."""
        for ext, folders in self._fileinfo.items():
            for folder, files in folders.items():
                for file, info in files.items():
                    yield info.filename

    def __len__(self) -> int:
        """Returns the number of files we have."""
        count = 0
        for folders in self._fileinfo.values():
            for files in folders.values():
                count += len(files)
        return count

    def __contains__(self, item: FileName) -> bool:
        path, filename, ext = _get_file_parts(item)

        try:
            return filename in self._fileinfo[ext][path]
        except KeyError:
            return False

    def extract_all(self, dest_dir: str) -> None:
        """Extract the contents of this VPK to a directory."""
        for ext, folders in self._fileinfo.items():
            for folder, files in folders.items():
                os.makedirs(os.path.join(dest_dir, folder), exist_ok=True)
                for file, info in files.items():
                    with open(os.path.join(dest_dir, info.filename), 'wb') as f:
                        f.write(info.read())

    def new_file(self, filename: FileName, root: str=None) -> FileInfo:
        """Create the given file, making it empty by default.
        
        If root is set, files are treated as relative to there,
        otherwise the filename must be relative.
        
        FileExistsError will be raised if the file is already present.
        """
        self._check_writable()
        
        path, name, ext = _get_file_parts(filename, root)

        try:
            ext_infos = self._fileinfo[ext]
        except KeyError:
            ext_infos = self._fileinfo[ext] = {}
        try:
            dir_infos = ext_infos[path]
        except KeyError:
            dir_infos = ext_infos[path] = {}
            
        if name in dir_infos:
            raise FileExistsError(
                'Filename already exists! ({!r})'.format(_join_file_parts(path, name, ext))
            )
        
        dir_infos[name] = info = FileInfo(
            self, 
            path,
            name,
            ext,
            EMPTY_CHECKSUM,
        )
        
        return info

    def add_file(self, filename: FileName, data: bytes, root=None, arch_index=0):
        """Add the given data to the VPK. 
        
        If root is set, files are treated as relative to there,
        otherwise the filename must be relative.
        arch_index is the pak01_xxx file to copy this to, if the length
        is larger than self.dir_limit.
        
        FileExistsError will be raised if the file is already present.
        """
        self.new_file(filename, root).write(data, arch_index)

    def add_folder(self, folder: str, prefix: str='') -> None:
        """Write all files in a folder to the VPK. 
        
        If prefix is set, the folders will be written to that subfolder.
        """
        self._check_writable()

        if prefix:
            prefix = prefix.replace('\\', '/')
        
        for subfolder, _, filenames, in os.walk(folder):
            # Prefix + subfolder relative to the folder.
            # normpath removes '.' and similar values from the beginning
            vpk_path = os.path.normpath(
                os.path.join(
                    prefix,
                    os.path.relpath(subfolder, folder)
                )
            )
            for filename in filenames:
                with open(os.path.join(subfolder, filename), 'rb') as f:
                    self.add_file((vpk_path, filename), f.read())
                    
    def verify_all(self) -> bool:
        """Check all files have a correct checksum."""
        return all(file.verify() for file in self)


def script_write(args: List[str]) -> None:
    """Create a VPK archive."""
    if len(args) not in (1, 2):
        raise ValueError("Usage: make_vpk.py [max_arch_mb] <folder>")
    
    folder = args[-1]
    
    vpk_name_base = folder.rstrip('\\/_dir')
    
    if len(args) > 1:
        arch_len = int(args[0]) * 1024 * 1024
    else:
        arch_len = 100 * 1024 * 1024
        
    current_arch = 1
    
    vpk_folder, vpk_name = os.path.split(vpk_name_base)
    for filename in os.listdir(vpk_folder):
        if filename.startswith(vpk_name + '_'):
            print(f'removing existing "{filename}"')
            os.remove(os.path.join(vpk_folder, filename))
    
    with VPK(vpk_name_base + '_dir.vpk', mode='w') as vpk:
        arch_filename = get_arch_filename(vpk_name_base, current_arch)
    
        for subfolder, _, filenames, in os.walk(folder):
            # normpath removes '.' and similar values from the beginning
            vpk_path = os.path.normpath(os.path.relpath(subfolder, folder))
            print(vpk_path + '/')
            for filename in filenames:
                print('\t' + filename)
                with open(os.path.join(subfolder, filename), 'rb') as f:
                    vpk.add_file(
                        (vpk_path, filename), 
                        f.read(), 
                        arch_index=current_arch,
                    )
                if os.path.exists(arch_filename) and os.stat(arch_filename).st_size > arch_len:
                    current_arch += 1
                    arch_filename = get_arch_filename(vpk_name_base, current_arch)
                
                
if __name__ == '__main__':
    import sys
    script_write(sys.argv[1:])
