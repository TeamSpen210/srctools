"""Read and write lumps in Source BSP files.

"""
import contextlib

from io import BytesIO
import itertools

from zipfile import ZipFile

from srctools import AtomicWriter, Vec, conv_int
from srctools.vmf import VMF, Entity, Output
from srctools.property_parser import Property
import struct

from typing import List, Dict, Iterator, Union, Optional, Tuple, BinaryIO


try:
    from enum import Enum, Flag
except ImportError:
    from aenum import Enum, Flag  # type: ignore

__all__ = [
    'BSP_LUMPS', 'VERSIONS',
    'BSP', 'Lump',
    'StaticProp', 'StaticPropFlags',
]

BSP_MAGIC = b'VBSP'  # All BSP files start with this
HEADER_1 = '<4si'  # Header section before the lump list.
HEADER_LUMP = '<3i4s'  # Header section for each lump.
HEADER_2 = '<i'  # Header section after the lumps.

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

    DESOLATION = 42

    def __eq__(self, other: object):
        """Versions are equal to their integer value."""
        return self.value == other


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

# Special-case the packfile lump, put it at the end.
# This way the BSP can be opened by generic zip programs.
LUMP_WRITE_ORDER = list(BSP_LUMPS)
LUMP_WRITE_ORDER.remove(BSP_LUMPS.PAKFILE)
LUMP_WRITE_ORDER.append(BSP_LUMPS.PAKFILE)


class StaticPropFlags(Flag):
    """Bitflags specified for static props."""
    NONE = 0

    DOES_FADE = 0x01  # Is the fade distances set?
    HAS_LIGHTING_ORIGIN = 0x02  # info_lighting entity used.
    DISABLE_DRAW = 0x04  # Changes at runtime.
    IGNORE_NORMALS = 0x08
    NO_SHADOW = 0x10
    SCREEN_SPACE_FADE = 0x20  # Use screen space fading. Obsolete since at least ASW.
    NO_PER_VERTEX_LIGHTING = 0x40
    NO_SELF_SHADOWING = 0x80

    # These are set in the secondary flags section.
    NO_FLASHLIGHT = 0x100  # Disable projected texture lighting.
    BOUNCED_LIGHTING = 0x0400  # Bounce lighting off the prop.

    @property
    def value_prim(self) -> int:
        """Return the data for the original flag byte."""
        return self.value & 0xFF

    @property
    def value_sec(self) -> int:
        """Return the data for the secondary flag byte."""
        return self.value >> 8

class BSP:
    """A BSP file."""
    def __init__(self, filename: str, version: VERSIONS=None):
        self.filename = filename
        self.map_revision = -1  # The map's revision count
        self.lumps = {}  # type: Dict[BSP_LUMPS, Lump]
        self.game_lumps = {}  # type: Dict[bytes, GameLump]
        self.header_off = 0
        self.version = version  # type: Optional[Union[VERSIONS, int]]

        self.read()

    def read(self) -> None:
        """Load all data."""
        self.lumps.clear()
        self.game_lumps.clear()

        with open(self.filename, mode='br') as file:
            # BSP files start with 'VBSP', then a version number.
            magic_name, bsp_version = get_struct(file, HEADER_1)
            assert magic_name == BSP_MAGIC, 'Not a BSP file!'

            if self.version is None:
                try:
                    self.version = VERSIONS(bsp_version)
                except ValueError:
                    self.version = bsp_version
            else:
                assert bsp_version == self.version, 'Different BSP version!'

            lump_offsets = {}

            # Read the index describing each BSP lump.
            for index in range(LUMP_COUNT):
                offset, length, version, ident = get_struct(file, HEADER_LUMP)
                lump_id = BSP_LUMPS(index)
                self.lumps[lump_id] = Lump(
                    lump_id,
                    version,
                    ident,
                )
                lump_offsets[lump_id] = offset, length

            [self.map_revision] = get_struct(file, HEADER_2)

            for lump in self.lumps.values():
                # Now read in each lump.
                offset, length = lump_offsets[lump.type]
                file.seek(offset)
                lump.data = file.read(length)

            game_lump = self.lumps[BSP_LUMPS.GAME_LUMP]

            self.game_lumps.clear()

            [lump_count] = struct.unpack_from('<i', game_lump.data)
            lump_offset = 4

            for _ in range(lump_count):
                (
                    game_lump_id,
                    flags,
                    glump_version,
                    file_off,
                    file_len,
                ) = GameLump.ST.unpack_from(game_lump.data, lump_offset)  # type: bytes, int, int, int, int
                lump_offset += GameLump.ST.size

                file.seek(file_off)

                # The lump ID is backward..
                game_lump_id = game_lump_id[::-1]

                self.game_lumps[game_lump_id] = GameLump(
                    game_lump_id,
                    flags,
                    glump_version,
                    file.read(file_len),
                )
            # This is not valid any longer.
            game_lump.data = b''

    def save(self, filename=None) -> None:
        """Write the BSP back into the given file."""
        # This gets difficult. The offsets need to be written before we know
        # what they are. So write empty bytes, record that location then go
        # back to fill them in after we actually determine where they are.
        # We use either BSP_LUMPS enums or game-lump byte IDs for dict keys.

        # Location of the header field.
        fixup_loc = {}  # type: Dict[Union[BSP_LUMPS, bytes], int]
        # The data to write.
        fixup_data = {}  # type: Dict[Union[BSP_LUMPS, bytes], bytes]

        game_lumps = list(self.game_lumps.values())  # Lock iteration order.

        with AtomicWriter(filename or self.filename, is_bytes=True) as file:  # type: BinaryIO
            if isinstance(self.version, VERSIONS):
                version = self.version.value
            else:
                version = self.version

            file.write(struct.pack(HEADER_1, BSP_MAGIC, version))

            # Write headers.
            for lump_name in BSP_LUMPS:
                lump = self.lumps[lump_name]
                fixup_loc[lump_name] = file.tell()
                file.write(struct.pack(
                    HEADER_LUMP,
                    0,  # offset
                    0,  # length
                    lump.version,
                    bytes(lump.ident),
                ))

            # After lump headers, the map revision...
            file.write(struct.pack(HEADER_2, self.map_revision))

            # Then each lump.
            for lump_name in LUMP_WRITE_ORDER:
                # Write out the actual data.
                lump = self.lumps[lump_name]
                if lump_name is BSP_LUMPS.GAME_LUMP:
                    # Construct this right here.
                    lump_start = file.tell()
                    file.write(struct.pack('<i', len(game_lumps)))
                    for game_lump in game_lumps:
                        file.write(struct.pack(
                            '<4s HH',
                            game_lump.id[::-1],
                            game_lump.flags,
                            game_lump.version,
                        ))
                        fixup_loc[game_lump.id] = file.tell()  # Offset goes here.
                        file.write(struct.pack('<4xi', len(game_lump.data)))

                    # Now write data.
                    for game_lump in game_lumps:
                        fixup_data[game_lump.id] = struct.pack('<i', file.tell())
                        file.write(game_lump.data)
                    # Length of the game lump is current - start.
                    fixup_data[lump_name] = struct.pack(
                        '<ii',
                        lump_start,
                        file.tell() - lump_start,
                    )
                else:
                    # Normal lump.
                    fixup_data[lump_name] = struct.pack(
                        '<ii',
                        file.tell(),
                        len(lump.data),
                    )
                    file.write(lump.data)

            # Now apply all the fixups we deferred.
            for fixup_key in fixup_loc:
                file.seek(fixup_loc[fixup_key])
                file.write(fixup_data[fixup_key])

    def read_header(self) -> None:
        """No longer used."""

    def read_game_lumps(self) -> None:
        """No longer used."""

    def replace_lump(
        self,
        new_name: str,
        lump: Union[BSP_LUMPS, 'Lump'],
        new_data: bytes
    ) -> None:
        """Write out the BSP file, replacing a lump with the given bytes.

        This is deprecated, simply assign to the .data attribute of the lump.
        """
        if isinstance(lump, BSP_LUMPS):
            lump = self.lumps[lump]

        lump.data = new_data

        self.save(new_name)

    def get_lump(self, lump: BSP_LUMPS) -> bytes:
        """Return the contents of the given lump."""
        return self.lumps[lump].data

    def get_game_lump(self, lump_id: bytes) -> bytes:
        """Get a given game-lump, given the 4-character byte ID."""
        try:
            lump = self.game_lumps[lump_id]
        except KeyError:
            raise ValueError('{} not in {}'.format(lump_id, list(self.game_lumps)))
        return lump.data

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
    def packfile(self) -> Iterator[ZipFile]:
        """A context manager to allow editing the packed content.

        When successfully exited, the zip will be rewritten to the BSP file.
        """
        pak_lump = self.lumps[BSP_LUMPS.PAKFILE]
        data_file = BytesIO(pak_lump.data)

        zip_file = ZipFile(data_file, mode='a')
        # If exception, abort, so we don't need try: or with:.
        # Because this is purely in memory, there are no actual resources here
        # and we don't actually care about not closing the zip file or BytesIO.
        yield zip_file
        # Explicitly close to finalise the footer.
        zip_file.close()
        # Note: because data is bytes, CPython won't end up doing any copying
        # here.
        pak_lump.data = data_file.getvalue()

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
                if cur_ent is None:
                    raise ValueError("Keyvalue outside brackets!")

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
    def write_ent_data(vmf: VMF) -> bytes:
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
    def _read_static_props_models(static_lump: BytesIO) -> Iterator[str]:
        """Read the static prop dictionary from the lump."""
        [dict_num] = get_struct(static_lump, '<i')
        for _ in range(dict_num):
            [padded_name] = get_struct(static_lump, '<128s')
            # Strip null chars off the end, and convert to a str.
            yield padded_name.rstrip(b'\x00').decode('ascii')

    def static_props(self) -> Iterator['StaticProp']:
        """Read in the Static Props lump."""
        # The version of the static prop format - different features.
        try:
            version = self.game_lumps[b'sprp'].version
        except KeyError:
            raise ValueError('No static prop lump!') from None

        if version > 11:
            raise ValueError('Unknown version ({})!'.format(version))
        if version < 4:
            # Predates HL2...
            raise ValueError('Static prop version {} is too old!')

        static_lump = BytesIO(self.game_lumps[b'sprp'].data)

        # Array of model filenames.
        model_dict = list(self._read_static_props_models(static_lump))

        [visleaf_count] = get_struct(static_lump, '<i')
        visleaf_list = list(get_struct(static_lump, 'H' * visleaf_count))

        [prop_count] = get_struct(static_lump, '<i')

        for i in range(prop_count):
            origin = Vec(get_struct(static_lump, 'fff'))
            angles = Vec(get_struct(static_lump, 'fff'))

            [model_ind] = get_struct(static_lump, '<H')

            (
                first_leaf,
                leaf_count,
                solidity,
                flags,
                skin,
                min_fade,
                max_fade,
            ) = get_struct(static_lump, '<HHBBiff')

            model_name = model_dict[model_ind]

            visleafs = visleaf_list[first_leaf:first_leaf + leaf_count]
            lighting_origin = Vec(get_struct(static_lump, '<fff'))

            if version >= 5:
                fade_scale = get_struct(static_lump, '<f')[0]
            else:
                fade_scale = 1  # default

            if version in (6, 7):
                min_dx_level, max_dx_level = get_struct(static_lump, '<HH')
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
                min_cpu_level = max_cpu_level = 0
                min_gpu_level = max_gpu_level = 0

            if version >= 7:
                r, g, b, renderfx = get_struct(static_lump, 'BBBB')
                # Alpha isn't used.
                tint = Vec(r, g, b)
            else:
                # No tint.
                tint = Vec(255, 255, 255)
                renderfx = 255

            if version >= 11:
                # Unknown data, though it's float-like.
                unknown_1 = get_struct(static_lump, '<i')

            if version >= 10:
                # Extra flags, post-CSGO.
                flags |= get_struct(static_lump, '<I')[0] << 8

            flags = StaticPropFlags(flags)

            scaling = 1.0
            disable_on_xbox = False

            if version >= 11:
                # XBox support was removed. Instead this is the scaling factor.
                [scaling] = get_struct(static_lump, "<f")
            elif version >= 9:
                # The single boolean byte also produces 3 pad bytes.
                [disable_on_xbox] = get_struct(static_lump, '<?xxx')

            yield StaticProp(
                model_name,
                origin,
                angles,
                scaling,
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
                renderfx,
                disable_on_xbox,
            )

    def write_static_props(self, props: List['StaticProp']) -> None:
        """Remake the static prop lump."""

        # First generate the visleaf and model-names block.
        # Unfortunately it seems reusing visleaf parts isn't possible.
        leaf_array = []  # type: List[int]
        leaf_offsets = []  # type: List[int]

        models = set()

        for prop in props:
            leaf_offsets.append(len(leaf_array))
            leaf_array.extend(prop.visleafs)
            models.add(prop.model)

        # Lock down the order of the names.
        model_list = list(models)
        model_ind = {
            mdl: i
            for i, mdl in enumerate(model_list)
        }

        game_lump = self.game_lumps[b'sprp']

        # Now write out the sections.
        prop_lump = BytesIO()
        prop_lump.write(struct.pack('<i', len(model_list)))
        for name in model_list:
            prop_lump.write(struct.pack('<128s', name.encode('ascii')))

        prop_lump.write(struct.pack('<i', len(leaf_array)))
        prop_lump.write(struct.pack('<{}H'.format(len(leaf_array)), *leaf_array))

        prop_lump.write(struct.pack('<i', len(props)))
        for leaf_off, prop in zip(leaf_offsets, props):
            prop_lump.write(struct.pack(
                '<6fH',
                prop.origin.x,
                prop.origin.y,
                prop.origin.z,
                prop.angles.x,
                prop.angles.y,
                prop.angles.z,
                model_ind[prop.model],
            ))

            prop_lump.write(struct.pack(
                '<HHBBifffff',
                leaf_off,
                len(prop.visleafs),
                prop.solidity,
                prop.flags.value_prim,
                prop.skin,
                prop.min_fade,
                prop.max_fade,
                prop.lighting.x,
                prop.lighting.y,
                prop.lighting.z,
            ))
            if game_lump.version >= 5:
                prop_lump.write(struct.pack('<f', prop.fade_scale))

            if game_lump.version in (6, 7):
                prop_lump.write(struct.pack(
                    '<HH',
                    prop.min_dx_level,
                    prop.max_dx_level,
                ))

            if game_lump.version >= 8:
                prop_lump.write(struct.pack(
                    '<BBBB',
                    prop.min_cpu_level,
                    prop.max_cpu_level,
                    prop.min_gpu_level,
                    prop.max_gpu_level
                ))

            if game_lump.version >= 7:
                prop_lump.write(struct.pack(
                    '<BBBB',
                    int(prop.tint.x),
                    int(prop.tint.y),
                    int(prop.tint.z),
                    prop.renderfx,
                ))

            if game_lump.version >= 10:
                prop_lump.write(struct.pack('<I', prop.flags.value_sec))

            if game_lump.version >= 11:
                # Unknown padding/data, though it's always zero.

                prop_lump.write(struct.pack('<xxxxf', prop.scaling))
            elif game_lump.version >= 9:
                # The 1-byte bool gets expanded to the full 4-byte size.
                prop_lump.write(struct.pack('<?xxx', prop.disable_on_xbox))

        game_lump.data = prop_lump.getvalue()


class Lump:
    """Represents a lump header in a BSP file.

    """
    def __init__(
        self,
        typ: BSP_LUMPS,
        version: int,
        ident: bytes,
    ) -> None:
        """This should not be constructed outside a BSP."""
        self.type = typ
        self.version = version
        self.ident = [int(x) for x in ident]
        self.data = b''

    def __repr__(self) -> str:
        return '<BSP Lump "{}", v{}, ident={}, {} bytes>'.format(
            self.type.name,
            self.version,
            bytes(self.ident),
            len(self.data)
        )


class GameLump:
    """Represents a game lump.

    These are designed to be game-specific.
    """
    __slots__ = [
        'id',
        'flags',
        'version',
        'data',
    ]

    ST = struct.Struct('<4s HH ii')

    def __init__(
        self,
        lump_id: bytes,
        flags: int,
        version: int,
        data: bytes,
    ) -> None:
        """This should not be constructed outside a BSP."""
        self.id = lump_id
        self.flags = flags
        self.version = version
        self.data = data

    def __repr__(self) -> str:
        return '<GameLump {}, flags={}, v{}, {} bytes>'.format(
            repr(self.id)[1:],
            self.flags,
            self.version,
            len(self.data),
        )


class StaticProp:
    """Represents a prop_static in the BSP.

    Different features were added in different versions.
    v5+ allows fade_scale.
    v6 and v7 allow min/max DXLevel.
    v8+ allows min/max GPU and CPU levels.
    v7+ allows model tinting, and renderfx.
    v9+ allows disabling on XBox 360.
    v10+ adds 4 unknown bytes (float?), and an expanded flags section.
    v11+ adds uniform scaling and removes XBox disabling.
    """
    def __init__(
        self,
        model: str,
        origin: Vec,
        angles: Vec,
        scaling: float,
        visleafs: List[int],
        solidity: int,
        flags: StaticPropFlags=StaticPropFlags.NONE,
        skin: int=0,
        min_fade: float=0,
        max_fade: float=0,
        lighting_origin: Vec=None,
        fade_scale: float=-1,
        min_dx_level: int=0,
        max_dx_level: int=0,
        min_cpu_level: int=0,
        max_cpu_level: int=0,
        min_gpu_level: int=0,
        max_gpu_level: int=0,
        tint: Vec=Vec(255, 255, 255),  # Rendercolor
        renderfx: int=255,
        disable_on_xbox: bool=False,
    ) -> None:
        self.model = model
        self.origin = origin
        self.angles = angles
        self.scaling = scaling
        self.visleafs = visleafs
        self.solidity = solidity
        self.flags = flags
        self.skin = skin
        self.min_fade = min_fade
        self.max_fade = max_fade

        if lighting_origin is None:
            self.lighting = Vec(origin)
        else:
            self.lighting = lighting_origin

        self.fade_scale = fade_scale
        self.min_dx_level = min_dx_level
        self.max_dx_level = max_dx_level
        self.min_cpu_level = min_cpu_level
        self.max_cpu_level = max_cpu_level
        self.min_gpu_level = min_gpu_level
        self.max_gpu_level = max_gpu_level
        self.tint = Vec(tint)
        self.renderfx = renderfx
        self.disable_on_xbox = disable_on_xbox

    def __repr__(self) -> str:
        return '<Prop "{}#{}" @ {} rot {}>'.format(
            self.model,
            self.skin,
            self.origin,
            self.angles,
        )
