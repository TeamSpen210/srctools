"""Parses Source models, to extract metadata."""
from srctools.filesys import FileSystem, File
from srctools import Vec
from struct import pack, unpack, Struct, calcsize


def str_read(format, file):
    """Read a structure from the file."""
    return unpack(format, file.read(calcsize(format)))


def read_nullstr(file):
    """Read a null-terminated string from the file."""
    text = []
    while True:
        char = file.read(1)
        if char == b'\0':
            return b''.join(text).decode('ascii')
        if not char:
            raise ValueError('Fell off end of file!')
        text.append(char)
    
ST_VEC = Struct('fff')


def str_readvec(file):
    """Read a vector from a file."""
    return Vec(ST_VEC.unpack(file.read(ST_VEC.size)))


class Model:
    def __init__(self, filesystem: FileSystem, file: File):
        """Parse a model from a file."""
        self._file = file
        self._sys = filesystem
        self.version = 49
        with self._sys, self._file.open_bin() as f:
            self._load(f)
    
    def _load(self, f):
        """Read data from the MDL file."""
        if f.read(4) != b'IDST':
            raise ValueError('Not a model!')
        (
            self.version,
            name,
            file_len,
            # 4 bytes are unknown...
        ) = str_read('i 4x 64s i', f)
        self.name = name.rstrip(b'\0').decode('ascii')
        self.eye_pos = str_readvec(f)
        self.illum_pos = str_readvec(f)
        # Approx dimensions
        self.hull_min = str_readvec(f)
        self.hull_max = str_readvec(f)
        
        self.view_min = str_readvec(f)
        self.view_max = str_readvec(f)
        
        (
            flags,
            
            bone_count,
            bone_off,
            
            bone_controller_count, bone_controller_off,
            
            hitbox_count, hitbox_off,
            anim_count, anim_off,
            sequence_count, sequence_off,

            activitylistversion, eventsindexed,

            texture_count, texture_offset,
            cdmat_count, cdmat_offset,
            
            skinref_count, skinref_ind, skinfamily_count,
            
            bodypart_count, bodypart_offset,
            attachment_count, attachment_offset,
        ) = str_read('24i', f)

        # Build CDMaterials data
        f.seek(cdmat_offset)
        cdmat_offsets = str_read(str(cdmat_count) + 'i', f)
        self.cdmaterials = [None] * cdmat_count
        
        for ind, off in enumerate(cdmat_offsets):
            if off > file_len or off == 0:
                continue
            f.seek(off)
            self.cdmaterials[ind] = read_nullstr(f)
        
        # Build texture data
        f.seek(texture_offset)
        self.textures = [None] * texture_count
        tex_temp = [None] * texture_count
        for tex_ind in range(texture_count):
            tex_temp[tex_ind] = (
                f.tell(),
                str_read('iii 4x ii 40x', f)
            )
        for tex_ind, (offset, data) in enumerate(tex_temp):
            name_offset, flags, used, material, client_material = data
            f.seek(offset + name_offset)
            self.textures[tex_ind] = (
                read_nullstr(f),
                flags,
                used,
                material,
                client_material
            )
