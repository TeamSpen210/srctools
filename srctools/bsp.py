"""Read and write lumps in Source BSP files.

"""
import contextlib

from enum import Enum
from io import BytesIO
import itertools

from zipfile import ZipFile

from srctools import AtomicWriter, Vec, conv_int
from srctools.vmf import VMF, Entity, Output
from srctools.property_parser import Property
import struct

from typing import List, Dict, Iterator, Union


__all__ = [
    'BSP_LUMPS', 'VERSIONS',
    'BSP', 'Lump', 'StaticProp',
]

BSP_MAGIC = b'VBSP'  # All BSP files start with this


def get_struct(file, format):
    """Get a structure from a file."""
    length = struct.calcsize(format)
    data = file.read(length)
    return struct.unpack_from(format, data)


class VERSIONS(Enum):
    """The BSP version numbers for various games."""
    VER_17 = 17
    VER_18 = 18
    VER_19 = 19
    VER_20 = 20
    VER_21 = 21
    VER_22 = 22
    VER_29 = 29

    HL2 = 19
    CS_SOURCE = 19
    DOF_SOURCE = 19
    HL2_EP1 = 20
    HL2_EP2 = 20
    HL2_LC = 20

    GARYS_MOD = 20
    TF2 = 20
    PORTAL = 20
    L4D = 20
    ZENO_CLASH = 20
    DARK_MESSIAH = 20
    VINDICTUS = 20
    THE_SHIP = 20

    BLOODY_GOOD_TIME = 20
    L4D2 = 21
    ALIEN_SWARM = 21
    PORTAL_2 = 21
    CS_GO = 21
    DEAR_ESTHER = 21
    STANLEY_PARABLE = 21
    DOTA2 = 22
    CONTAGION = 23


class BSP_LUMPS(Enum):
    """All the lumps in a BSP file.

    The values represent the order lumps appear in the index.
    Some indexes were reused, so they have aliases.
    """
    ENTITIES = 0
    PLANES = 1
    TEXDATA = 2
    VERTEXES = 3
    VISIBILITY = 4
    NODES = 5
    TEXINFO = 6
    FACES = 7
    LIGHTING = 8
    OCCLUSION = 9
    LEAFS = 10
    FACEIDS = 11
    EDGES = 12
    SURFEDGES = 13
    MODELS = 14
    WORLDLIGHTS = 15
    LEAFFACES = 16
    LEAFBRUSHES = 17
    BRUSHES = 18
    BRUSHSIDES = 19
    AREAS = 20
    AREAPORTALS = 21

    PORTALS = 22
    UNUSED0 = 22
    PROPCOLLISION = 22

    CLUSTERS = 23
    UNUSED1 = 23
    PROPHULLS = 23

    PORTALVERTS = 24
    UNUSED2 = 24
    PROPHULLVERTS = 24

    CLUSTERPORTALS = 25
    UNUSED3 = 25
    PROPTRIS = 25

    DISPINFO = 26
    ORIGINALFACES = 27
    PHYSDISP = 28
    PHYSCOLLIDE = 29
    VERTNORMALS = 30
    VERTNORMALINDICES = 31
    DISP_LIGHTMAP_ALPHAS = 32
    DISP_VERTS = 33
    DISP_LIGHTMAP_SAMPLE_POSITIONS = 34
    GAME_LUMP = 35
    LEAFWATERDATA = 36
    PRIMITIVES = 37
    PRIMVERTS = 38
    PRIMINDICES = 39
    PAKFILE = 40
    CLIPPORTALVERTS = 41
    CUBEMAPS = 42
    TEXDATA_STRING_DATA = 43
    TEXDATA_STRING_TABLE = 44
    OVERLAYS = 45
    LEAFMINDISTTOWATER = 46
    FACE_MACRO_TEXTURE_INFO = 47
    DISP_TRIS = 48
    PHYSCOLLIDESURFACE = 49
    PROP_BLOB = 49
    WATEROVERLAYS = 50

    LIGHTMAPPAGES = 51
    LEAF_AMBIENT_INDEX_HDR = 51

    LIGHTMAPPAGEINFOS = 52
    LEAF_AMBIENT_INDEX = 52
    LIGHTING_HDR = 53
    WORLDLIGHTS_HDR = 54
    LEAF_AMBIENT_LIGHTING_HDR = 55
    LEAF_AMBIENT_LIGHTING = 56
    XZIPPAKFILE = 57
    FACES_HDR = 58
    MAP_FLAGS = 59
    OVERLAY_FADES = 60
    OVERLAY_SYSTEM_LEVELS = 61
    PHYSLEVEL = 62
    DISP_MULTIBLEND = 63

LUMP_COUNT = max(lump.value for lump in BSP_LUMPS) + 1  # 64 normally


class BSP:
    """A BSP file."""
    def __init__(self, filename: str, version: VERSIONS=VERSIONS.PORTAL_2):
        self.filename = filename
        self.map_revision = -1  # The map's revision count
        self.lumps = {}  # type: Dict[BSP_LUMPS, Lump]
        self.game_lumps = {}
        self.header_off = 0
        self.version = version

    def read_header(self):
        """Read through the BSP header to find the lumps.

        This allows locating any data in the BSP.
        """
        with open(self.filename, mode='br') as file:
            # BSP files start with 'VBSP', then a version number.
            magic_name, bsp_version = get_struct(file, '4si')
            assert magic_name == BSP_MAGIC, 'Not a BSP file!'

            assert bsp_version == self.version.value, 'Different BSP version!'

            # Read the index describing each BSP lump.
            for index in range(LUMP_COUNT):
                lump = Lump.from_bytes(index, file)
                self.lumps[lump.type] = lump

            # Remember how big this is, so we can remake it later when needed.
            self.header_off = file.tell()

    def get_lump(self, lump: Union[BSP_LUMPS, 'Lump']):
        """Read a lump from the BSP."""
        if isinstance(lump, BSP_LUMPS):
            lump = self.lumps[lump]
        with open(self.filename, 'rb') as file:
            file.seek(lump.offset)
            return file.read(lump.length)

    def replace_lump(self, new_name: str, lump: Union[BSP_LUMPS, 'Lump'], new_data: bytes):
        """Write out the BSP file, replacing a lump with the given bytes.

        """
        if isinstance(lump, BSP_LUMPS):
            lump = self.lumps[lump]  # type: Lump
        with open(self.filename, 'rb') as file:
            data = file.read()

        before_lump = data[self.header_off:lump.offset]
        after_lump = data[lump.offset + lump.length:]
        del data  # This contains the entire file, we don't want to keep
        # this memory around for long.

        # Adjust the length to match the new data block.
        len_change = len(new_data) - lump.length
        lump.length = len(new_data)

        # Find all lumps after this one, and adjust offsets.
        # The order of headers doesn't need to match data order!
        for other_lump in self.lumps.values():
            # Not >=, that would adjust us too!
            if other_lump.offset > lump.offset:
                other_lump.offset += len_change

        with AtomicWriter(new_name, is_bytes=True) as file:
            self.write_header(file)
            file.write(before_lump)
            file.write(new_data)
            file.write(after_lump)

            # Game lumps need their data to apply offsets.
            # We're not adding/removing headers, so we can rewrite in-place.
            game_lump = self.lumps[BSP_LUMPS.GAME_LUMP]
            if game_lump.offset > lump.offset:
                file.seek(game_lump.offset)
                file.write(struct.pack('i', len(self.game_lumps)))
                for lump_id, (
                    flags,
                    version,
                    file_off,
                    file_len,
                ) in self.game_lumps.items():
                    self.game_lumps[lump_id] = (
                        flags,
                        version,
                        file_off + len_change,
                        file_len,
                    )
                    file.write(struct.pack(
                        '<4s HH ii',
                        lump_id[::-1],
                        flags,
                        version,
                        file_off + len_change,
                        file_len,
                    ))

    def write_header(self, file) -> None:
        """Write the BSP file header into the given file."""
        file.write(BSP_MAGIC)
        file.write(struct.pack('i', self.version.value))
        for lump_name in BSP_LUMPS:
            # Write each header
            lump = self.lumps[lump_name]
            file.write(lump.as_bytes())
        # The map revision would follow, but we never change that value!

    def read_game_lumps(self) -> None:
        """Read in the game-lump's header, so we can get those values."""
        game_lump = BytesIO(self.get_lump(BSP_LUMPS.GAME_LUMP))

        self.game_lumps.clear()
        lump_count = get_struct(game_lump, 'i')[0]

        for _ in range(lump_count):
            (
                lump_id,
                flags,
                version,
                file_off,
                file_len,
            ) = get_struct(game_lump, '<4s HH ii')
            # The lump ID is backward..
            self.game_lumps[lump_id[::-1]] = (flags, version, file_off, file_len)

    def get_game_lump(self, lump_id: bytes) -> bytes:
        """Get a given game-lump, given the 4-character byte ID."""
        try:
            flags, version, file_off, file_len = self.game_lumps[lump_id]
        except KeyError:
            raise ValueError('{} not in {}'.format(lump_id, list(self.game_lumps)))
        with open(self.filename, 'rb') as file:
            file.seek(file_off)
            return file.read(file_len)

    # Lump-specific commands:

    def read_texture_names(self) -> Iterator[str]:
        """Iterate through all brush textures in the map."""
        tex_data = self.get_lump(BSP_LUMPS.TEXDATA_STRING_DATA)
        tex_table = self.get_lump(BSP_LUMPS.TEXDATA_STRING_TABLE)
        # tex_table is an array of int offsets into tex_data. tex_data is a
        # null-terminated block of strings.

        table_offsets = struct.unpack(
            # The number of ints + i, for the repetitions in the struct.
            str(len(tex_table) // struct.calcsize('i')) + 'i',
            tex_table,
        )

        for off in table_offsets:
            # Look for the NULL at the end - strings are limited to 128 chars.
            str_off = 0
            for str_off in range(off, off + 128):
                if tex_data[str_off] == 0:
                    yield tex_data[off: str_off].decode('ascii')
                    break
            else:
                # Reached the 128 char limit without finding a null.
                raise ValueError('Bad string at', off, 'in BSP! ("{}")'.format(
                    tex_data[off:str_off]
                ))

    @contextlib.contextmanager
    def packfile(self):
        """A context manager to allow editing the packed content.

        When successfully exited, the zip will be rewritten to the BSP file.
        """
        data_file = BytesIO(self.get_lump(BSP_LUMPS.PAKFILE))
        data_file.seek(0)
        zip_file = ZipFile(data_file, mode='a')
        # If exception, abort, so we don't need try: or with:
        yield zip_file
        # Explicitly close to finalise the footer.
        zip_file.close()
        self.replace_lump(self.filename, BSP_LUMPS.PAKFILE, data_file.getvalue())

    def read_ent_data(self) -> VMF:
        """Parse in entity data.
        
        This returns a VMF object, with entities mirroring that in the BSP. 
        No brushes are read.
        """
        ent_data = self.get_lump(BSP_LUMPS.ENTITIES)
        vmf = VMF()
        cur_ent = None  # None when between brackets.
        seen_spawn = False  # The first entity is worldspawn.
        
        # This code performs the same thing as property_parser, but simpler
        # since there's no nesting, comments, or whitespace, except between
        # key and value. We also operate directly on the (ASCII) binary.
        for line in ent_data.splitlines():
            if line == b'{':
                if cur_ent is not None:
                    raise ValueError(
                        '2 levels of nesting after {} ents'.format(
                            len(vmf.entities)
                        )
                    )
                if not seen_spawn:
                    cur_ent = vmf.spawn
                    seen_spawn = True
                else:
                    cur_ent = Entity(vmf)
            elif line == b'}':
                if cur_ent is None:
                    raise ValueError(
                        'Too many closing brackets after {} ents'.format(
                            len(vmf.entities)
                        )
                    )
                if cur_ent is vmf.spawn:
                    if cur_ent['classname'] != 'worldspawn':
                        raise ValueError('No worldspawn entity!')
                else:
                    # The spawn ent is stored in the attribute, not in the ent
                    # list.
                    vmf.add_ent(cur_ent)
                cur_ent = None
            elif line == b'\x00':  # Null byte at end of lump.
                if cur_ent is not None:
                    raise ValueError("Last entity didn't end!")
                return vmf
            else:
                # Line is of the form <"key" "val">
                key, value = line.split(b'" "')
                decoded_key = key[1:].decode('ascii')
                decoded_val = value[:-1].decode('ascii')
                if 27 in value:
                    # All outputs use the comma_sep, so we can ID them.
                    cur_ent.add_out(Output.parse(Property(decoded_key, decoded_val)))
                else:
                    # Normal keyvalue.
                    cur_ent[decoded_key] = decoded_val
                    
        # This keyvalue needs to be stored in the VMF object too.
        # The one in the entity is ignored.
        vmf.map_ver = conv_int(vmf.spawn['mapversion'], vmf.map_ver)

        return vmf

    @staticmethod
    def write_ent_data(vmf: VMF):
        """Generate the entity data lump.
        
        This accepts a VMF file like that returned from read_ent_data(). 
        Brushes are ignored, so the VMF must use *xx model references.
        """
        out = BytesIO()
        for ent in itertools.chain([vmf.spawn], vmf.entities):
            out.write(b'{\n')
            for key, value in ent.keys.items():
                out.write('"{}" "{}"\n'.format(key, value).encode('ascii'))
            for output in ent.outputs:
                out.write(output._get_text().encode('ascii'))
            out.write(b'}\n')
        out.write(b'\x00')

        return out.getvalue()

    def static_prop_models(self) -> Iterator[str]:
        """Yield all model filenames used in static props."""
        static_lump = BytesIO(self.get_game_lump(b'sprp'))
        return self._read_static_props_models(static_lump)

    @staticmethod
    def _read_static_props_models(static_lump: BytesIO):
        """Read the static prop dictionary from the lump."""
        dict_num = get_struct(static_lump, 'i')[0]
        for _ in range(dict_num):
            padded_name = get_struct(static_lump, '128s')[0]
            # Strip null chars off the end, and convert to a str.
            yield padded_name.rstrip(b'\x00').decode('ascii')

    def static_props(self) -> Iterator['StaticProp']:
        """Read in the Static Props lump."""
        # The version of the static prop format - different features.
        version = self.game_lumps[b'sprp'][1]
        if version > 9:
            raise ValueError('Unknown version "{}"!'.format(version))

        static_lump = BytesIO(self.get_game_lump(b'sprp'))

        # Array of model filenames.
        model_dict = list(self._read_static_props_models(static_lump))

        visleaf_count = get_struct(static_lump, 'i')[0]
        visleaf_list = list(get_struct(static_lump, 'H' * visleaf_count))

        prop_count = get_struct(static_lump, 'i')[0]

        pos = static_lump.tell()
        data = static_lump.read()
        static_lump.seek(pos)
        for i in range(12, 200, 12):
            vals = Vec(struct.unpack_from('fff', data, i))
            # if vals: and vals == round(vals):
            print(i, repr(vals))

        print(flush=True)
        for i in range(prop_count):
            origin = Vec(get_struct(static_lump, 'fff'))
            angles = Vec(get_struct(static_lump, 'fff'))
            (
                model_ind,
                first_leaf,
                leaf_count,
                solidity,
                flags,
                skin,
                min_fade,
                max_fade,
            ) = get_struct(static_lump, 'HHHBBiff')

            model_name = model_dict[model_ind]

            visleafs = visleaf_list[first_leaf:first_leaf + leaf_count]
            lighting_origin = Vec(get_struct(static_lump, 'fff'))

            if version >= 5:
                fade_scale = get_struct(static_lump, 'f')[0]
            else:
                fade_scale = 1  # default

            if version in (6, 7):
                min_dx_level, max_dx_level = get_struct(static_lump, 'HH')
            else:
                # Replaced by GPU & CPU in later versions.
                min_dx_level = max_dx_level = 0  # None

            if version >= 8:
                (
                    min_cpu_level,
                    max_cpu_level,
                    min_gpu_level,
                    max_gpu_level,
                ) = get_struct(static_lump, 'BBBB')
            else:
                # None
                min_cpu_level = max_cpu_level = min_gpu_level = max_gpu_level = 0

            if version >= 7:
                r, g, b, a = get_struct(static_lump, 'BBBB')
                # Alpha isn't used.
                tint = Vec(r, g, b)
            else:
                # No tint.
                tint = Vec(255, 255, 255)
            if version >= 9:
                disable_on_xbox = get_struct(static_lump, '?')[0]
            else:
                disable_on_xbox = False

            # Unknown padding...
            static_lump.read(3)

            yield StaticProp(
                model_name,
                origin,
                angles,
                visleafs,
                solidity,
                flags,
                skin,
                min_fade,
                max_fade,
                lighting_origin,
                fade_scale,
                min_dx_level,
                max_dx_level,
                min_cpu_level,
                max_cpu_level,
                min_gpu_level,
                max_gpu_level,
                tint,
                disable_on_xbox,
            )



class Lump:
    """Represents a lump header in a BSP file.

    These indicate the location and size of each component.
    """
    def __init__(self, index, offset, length, version, ident):
        self.type = BSP_LUMPS(index)
        self.offset = offset
        self.length = length
        self.version = version
        self.ident = [int(x) for x in ident]

    @classmethod
    def from_bytes(cls, index, file):
        """Decode this header from the file."""
        offset, length, version, ident = get_struct(
            file,
            # 3 ints and a 4-long char array
            '<3i4s',
        )
        return cls(
            index=index,
            offset=offset,
            length=length,
            version=version,
            ident=ident,
        )

    def as_bytes(self):
        """Get the binary version of this lump header."""
        return struct.pack(
            '<3i4s',
            self.offset,
            self.length,
            self.version,
            bytes(self.ident),
        )

    def __len__(self):
        return self.length

    def __repr__(self):
        return (
            'Lump({s.type}, {s.offset}, '
            '{s.length}, {s.version}, {s.ident})'.format(
                s=self
            )
        )


class StaticProp:
    """Represents a prop_static in the BSP.

    Different features were added in different versions.
    v5+ allows fade_scale.
    v6 and v7 allow min/max DXLevel.
    v8+ allows min/max GPU and CPU levels.
    v7+ allows model tinting.
    v9+ allows disabling on XBox 360.
    """
    def __init__(
        self,
        model: str,
        origin: Vec,
        angles: Vec,
        visleafs: List[int],
        solidity: int,
        flags: int,
        skin: int,
        min_fade: float,
        max_fade: float,
        lighting_origin: Vec,
        fade_scale: float,
        min_dx_level: int,
        max_dx_level: int,
        min_cpu_level: int,
        max_cpu_level: int,
        min_gpu_level: int,
        max_gpu_level: int,
        tint: Vec,  # Rendercolor
        disable_on_xbox: bool,
    ):
        self.model = model
        self.origin = origin
        self.angles = angles
        self.visleafs = visleafs
        self.solidity = solidity
        self.flags = flags
        self.skin = skin
        self.min_fade = min_fade
        self.max_fade = max_fade
        self.lighting = lighting_origin
        self.fade_scale = fade_scale
        self.min_dx_level = min_dx_level
        self.max_dx_level = max_dx_level
        self.min_cpu_level = min_cpu_level
        self.max_cpu_level = max_cpu_level
        self.min_gpu_level = min_gpu_level
        self.max_gpu_level = max_gpu_level
        self.tint = tint
        self.disable_on_xbox = disable_on_xbox
