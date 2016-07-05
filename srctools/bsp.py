"""Read and write lumps in Source BSP files.

"""
from enum import Enum
from io import BytesIO
from srctools import AtomicWriter
import struct

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
    def __init__(self, filename, version=VERSIONS.PORTAL_2):
        self.filename = filename
        self.map_revision = -1  # The map's revision count
        self.lumps = {}
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

    def get_lump(self, lump):
        """Read a lump from the BSP."""
        if isinstance(lump, BSP_LUMPS):
            lump = self.lumps[lump]
        with open(self.filename, 'rb') as file:
            file.seek(lump.offset)
            return file.read(lump.length)

    def replace_lump(self, new_name, lump, new_data: bytes):
        """Write out the BSP file, replacing a lump with the given bytes.

        """
        if isinstance(lump, BSP_LUMPS):
            lump = self.lumps[lump]
        with open(self.filename, 'rb') as file:
            data = file.read()

        before_lump = data[self.header_off:lump.offset]
        after_lump = data[lump.offset + lump.length:]
        del data  # This contains the entire file, we don't want to keep
        # this memory around for long.

        # Adjust the length to match the new data block.
        lump.length = len(new_data)

        with AtomicWriter(new_name, is_bytes=True) as file:
            self.write_header(file)
            file.write(before_lump)
            file.write(new_data)
            file.write(after_lump)

    def write_header(self, file):
        """Write the BSP file header into the given file."""
        file.write(BSP_MAGIC)
        file.write(struct.pack('i', self.version.value))
        for lump_name in BSP_LUMPS:
            # Write each header
            lump = self.lumps[lump_name]
            file.write(lump.as_bytes())
        # The map revision would follow, but we never change that value!

    def read_game_lumps(self):
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
            self.game_lumps[lump_id] = (flags, version, file_off, file_len)

    def get_game_lump(self, lump_id):
        """Get a given game-lump, given the 4-character byte ID."""
        flags, version, file_off, file_len = self.game_lumps[lump_id]
        with open(self.filename, 'rb') as file:
            file.seek(file_off)
            return file.read(file_len)

    # Lump-specific commands:

    def read_texture_names(self):
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

    def read_ent_data(self):
        """Iterate through the entities in a map.

        This yields a series of keyvalue dictionaries. The first is WorldSpawn.
        """
        ent_data = self.get_lump(BSP_LUMPS.ENTITIES).decode('ascii')
        cur_dict = None  # None = waiting for '{'

        # This code is similar to property_parser, but simpler since there's
        # no nesting, comments, or whitespace, except between key and value.
        for line in ent_data.splitlines():
            if line == '{':
                cur_dict = {}
            elif line == '}':
                yield cur_dict
                cur_dict = None
            elif line == '\x00':
                return
            else:
                # Line is of the form <"key" "val">
                key, value = line.split('" "')
                cur_dict[key[1:]] = value[:-1]

    @staticmethod
    def write_ent_data(ent_dicts):
        """Generate the entity data lump, given a list of dictionaries."""
        out = BytesIO()
        for keyvals in ent_dicts:
            out.write(b'{\n')
            for key, value in keyvals.items():
                out.write('"{}" "{}"'.format(key, value).encode('ascii'))
            out.write(b'}\n')
        out.write(b'\x00')

        return out.getvalue()

    def read_static_prop_models(self):
        """Get a list of all model names used by static props."""
        static_lump = BytesIO(self.get_game_lump(b'prps'))
        dict_num = get_struct(static_lump, 'i')[0]

        model_dict = []
        for _ in range(dict_num):
            padded_name = get_struct(static_lump, '128s')[0]
            # Strip null chars off the end, and convert to a str.
            model_dict.append(
                padded_name.rstrip(b'\x00').decode('ascii')
            )

        return model_dict


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