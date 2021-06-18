"""Read and write parts of Source BSP files.

Data from a read BSP is lazily parsed when each section is accessed.
"""
from typing import (
    overload, TypeVar, Any, Generic, Union, Optional, ClassVar, Type,
    List, Iterator, BinaryIO, Tuple, Callable, Dict, Set,
)
from io import BytesIO
from enum import Enum, Flag
from zipfile import ZipFile
import itertools
import struct
import inspect
import contextlib
import warnings
import attr

from srctools import AtomicWriter, conv_int
from srctools.math import Vec, Angle
from srctools.filesys import FileSystem
from srctools.vtf import VTF
from srctools.vmt import Material
from srctools.vmf import VMF, Entity, Output
from srctools.tokenizer import escape_text
from srctools.binformat import struct_read, DeferredWrites
from srctools.property_parser import Property

__all__ = [
    'BSP_LUMPS', 'VERSIONS',
    'BSP', 'Lump',
    'StaticProp', 'StaticPropFlags',
    'TexData', 'TexInfo',
    'Cubemap', 'Overlay',
    'VisTree', 'VisLeaf',
    'BModel', 'Plane', 'PlaneType',
    'Brush', 'BrushSide', 'BrushContents',
]


BSP_MAGIC = b'VBSP'  # All BSP files start with this
HEADER_1 = '<4si'  # Header section before the lump list.
HEADER_LUMP = '<3i4s'  # Header section for each lump.
HEADER_2 = '<i'  # Header section after the lumps.

T = TypeVar('T')
S = TypeVar('S')
Edge = Tuple[Vec, Vec]


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
    PROP_BLOB = 49
    PHYSCOLLIDESURFACE = 49
    WATEROVERLAYS = 50

    LEAF_AMBIENT_INDEX_HDR = 51
    LEAF_AMBIENT_INDEX = 52

    LIGHTMAPPAGES = 51
    LIGHTMAPPAGEINFOS = 52

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

# When remaking the lumps from trees of objects,
# they need to be done in the correct order so stuff referring
# to other trees can add their data.
LUMP_REBUILD_ORDER = [
    BSP_LUMPS.PAKFILE,
    BSP_LUMPS.ENTITIES,  # References brushmodels, overlays, potentially many others.
    BSP_LUMPS.CUBEMAPS,

    BSP_LUMPS.MODELS,  # Brushmodels reference their vis tree, and faces
    BSP_LUMPS.NODES,  # References planes, faces, visleafs.
    BSP_LUMPS.LEAFS,  # References brushes, faces

    BSP_LUMPS.BRUSHES,  # also brushsides, references texinfo.

    BSP_LUMPS.FACES,  # References their original face, surfedges, texinfo.
    BSP_LUMPS.FACES_HDR,  # References their original face, surfedges, texinfo.
    BSP_LUMPS.ORIGINALFACES,  # references surfedges & texinfo.

    BSP_LUMPS.SURFEDGES,  # surfedges references vertexes.
    BSP_LUMPS.PLANES,
    BSP_LUMPS.VERTEXES,

    BSP_LUMPS.OVERLAYS,  # Adds texinfo entries.

    BSP_LUMPS.TEXINFO,  # Adds texdata -> texdata_string_data entries.
    BSP_LUMPS.TEXDATA_STRING_DATA,
]


class PlaneType(Enum):
    """The orientation of a plane."""
    X = 0  # Exactly in the X axis.
    Y = 1  # Exactly in the Y axis.
    Z = 2  # Exactly in the Z axis.
    ANY_X = 3  # Pointing mostly in the X axis
    ANY_Y = 4  # Pointing mostly in the Y axis.
    ANY_Z = 5  # Pointing mostly in the Z axis.


class SurfFlags(Flag):
    """The various SURF_ flags, indicating different attributes for faces."""
    NONE = 0
    LIGHT = 0x1  # The face has lighting info.
    SKYBOX_2D = 0x2  # Nodraw, but when visible 2D skybox should be rendered.
    SKYBOX_3D = 0x4  # Nodraw, but when visible 2D and 3D skybox should be rendered.
    WATER_WARP = 0x8  # 'turbulent water warp'
    TRANSLUCENT = 0x10  # Translucent material.
    NOPORTAL = 0x20  # Portalgun blocking material.
    TRIGGER = 0x40  # XBox only - is a trigger surface.
    NODRAW = 0x80  # Texture isn't used, it's invisible.
    HINT = 0x100  # A hint brush.
    SKIP = 0x200  # Skip brush, removed from map.
    NOLIGHT = 0x400  # No light needs to be calculated.
    BUMPLIGHT = 0x800  # Needs three lightmaps for bumpmapping.
    NO_SHADOWS = 0x1000  # Doesn't receive shadows.
    NO_DECALS = 0x2000  # Rejects decals.
    NO_SUBDIVIDE = 0x4000  # Not allowed to split up the brush face.
    HITBOX = 0x8000  # 'Part of a hitbox'


class BrushContents(Flag):
    """The various CONTENTS_ flags, indicating different attributes for an entire brush."""
    EMPTY = 0
    SOLID = 0x1  # Player camera is not valid inside here.
    WINDOW = 0x2  # Translucent glass.
    AUX = 0x4
    GRATE = 0x8  # Grating, bullets/LOS pass, objects do not.
    SLIME = 0x10  # Slime-style liquid.
    WATER = 0x20  # Is a water brush
    MIST = 0x40
    OPAQUE = 0x80  # Blocks LOS
    TEST_FOG_VOLUME = 0x100  # May be non-solid, but cannot be seen through.
    TEAM1 = 0x800  # Special team-only clips.
    TEAM2 = 0x1000  # Special team-only clips.
    IGNORE_NODRAW_OPAQUE = 0x2000  # ignore CONTENTS_OPAQUE on surfaces that have SURF_NODRAW
    MOVABLE = 0x4000

    AREAPORTAL = 0x8000  # Is an areaportal brush.
    PLAYER_CLIP = 0x10000  # Is tools/toolsplayerclip.
    NPC_CLIP = 0x20000  # Is tools/toolsclip.
    # Specifies water currents, can be mixed.
    CURRENT_0 = 0x40000
    CURRENT_90 = 0x80000
    CURRENT_180 = 0x100000
    CURRENT_270 = 0x200000
    CURRENT_UP = 0x400000
    CURRENT_DOWN = 0x800000
    ORIGIN = 0x1000000  # tools/toolsorigin brush, used to set origin.
    NPC = 0x2000000  # Shouldn't be on brushes, for NPCs.
    DEBRIS = 0x4000000
    DETAIL = 0x8000000  # Is func_detail.
    TRANSLUCENT = 0x10000000  # Brush is $translucent/$alphatest/$alpha/etc
    LADDER = 0x20000000
    HITBOX = 0x40000000

    UNUSED_1 = 0x200
    UNUSED_2 = 0x400


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


class VisLeafFlags(Flag):
    """Visleaf flags."""
    NONE = 0x0
    SKY_3D = 0x01  # The 3D skybox is visible from here.
    SKY_2D = 0x04  # The 2D skybox is visible from here.
    RADIAL = 0x02  # Has culled portals, due to far-z fog limits.
    HAS_DETAIL_OBJECTS = 0x08  # Contains detail props.

    # Undocumented flags, still in maps though?
    # Looks like uninitialised members.
    _BIT_4 = 1 << 3
    _BIT_5 = 1 << 4
    _BIT_6 = 1 << 5
    _BIT_7 = 1 << 6


def identity(x: T) -> T:
    """Identity function."""
    return x


def _find_or_insert(item_list: List[T], key_func: Callable[[T], S]=id) -> Callable[[T], int]:
    """Create a function for inserting items in a list if not found.

    This is used to build up the structure arrays which other lumps refer
    to by index.
    If the provided argument to the callable is already in the list,
    the index is returned. Otherwise it is appended and the new index returned.
    The key function is used to match existing items.

    """
    by_index: dict[S, int] = {key_func(item): i for i, item in enumerate(item_list)}

    def finder(item: T) -> int:
        """Find or append the item."""
        key = key_func(item)
        try:
            return by_index[key]
        except KeyError:
            ind = by_index[key] = len(item_list)
            item_list.append(item)
            return ind
    return finder


def _find_or_extend(item_list: List[T], key_func: Callable[[T], S]=id) -> Callable[[List[T]], int]:
    """Create a function for positioning a sublist inside the larger list, adding it if required.

    This is used to build up structure arrays where othe lumps access subsections of it.
    """
    # We expect repeated items to be fairly uncommon, so we can skip to all
    # occurrences of the first index to speed up the search.
    by_index: dict[S, List[int]] = {}
    for k, item in enumerate(item_list):
        by_index.setdefault(key_func(item), []).append(k)

    def finder(items: List[T]) -> int:
        """Find or append the items."""
        if not items:
            # Array is empty, so the index doesn't matter, it'll never be
            # dereferenced.
            return 0
        for i in by_index.get(key_func(items[0]), ()):
            if item_list[i:i + len(items)] == items:
                return i
        # Not found, append to the end.
        i = len(item_list)
        item_list.extend(items)
        # Update the index.
        for j, item2 in enumerate(items):
            by_index.setdefault(key_func(item2), []).append(i + j)
        return i

    return finder


class ParsedLump(Generic[T]):
    """Allows access to parsed versions of lumps.
    
    When accessed, the corresponding lump is parsed into an object tree.
    The lump is then cleared of data.
    When the BSP is saved, the lump data is then constructed.
    """

    def __init__(self, lump: BSP_LUMPS, *extra: BSP_LUMPS) -> None:
        self.lump = lump
        self.to_clear = (lump, ) + extra
        self.__name__ = ''
        # May also be a Generator[X] if T = List[X]
        self._read: Optional[Callable[[BSP, bytes], T]] = None
        self._check: Optional[Callable[[BSP, T], None]] = None
        assert self.lump in LUMP_REBUILD_ORDER, self.lump

    def __set_name__(self, owner: Type['BSP'], name: str) -> None:
        self.__name__ = name
        self.__objclass__ = owner
        self._read = getattr(owner, '_lmp_read_' + name)
        self._check = getattr(owner, '_lmp_check_' + name, None)
        # noinspection PyProtectedMember
        owner._save_funcs[self.lump] = getattr(owner, '_lmp_write_' + name)

    def __repr__(self) -> str:
        return f'<srctools.BSP.{self.__name__} member>'
        
    @overload
    def __get__(self, instance: None, owner=None) -> 'ParsedLump[T]': ...
    @overload
    def __get__(self, instance: 'BSP', owner=None) -> T: ...
        
    def __get__(self, instance: Optional['BSP'], owner=None) -> Union['ParsedLump', T]:
        """Read the lump, then discard."""
        if instance is None:  # Accessed on the class.
            return self
        try:
            return instance._parsed_lumps[self.lump]  # noqa
        except KeyError:
            pass
        if self._read is None:
            raise TypeError('ParsedLump.__set_name__ was never called!')
        
        data = instance.lumps[self.lump].data
        result = self._read(instance, data)
        if inspect.isgenerator(result):  # Convenience, yield to accumulate into a list.
            result = list(result)  # type: ignore
        instance._parsed_lumps[self.lump] = result # noqa
        for lump in self.to_clear:
            instance.lumps[lump].data = b''
        return result

    def __set__(self, instance: Optional['BSP'], value: T) -> None:
        """Discard lump data, then store."""

        if instance is None:
            raise TypeError('Cannot assign directly to lump descriptor!')
        if self._check is not None:
            # Allow raising if an invalid value.
            self._check(instance, value)
        for lump in self.to_clear:
            instance.lumps[lump].data = b''
        instance._parsed_lumps[self.lump] = value  # noqa


# noinspection PyMethodMayBeStatic
class BSP:
    """A BSP file."""
    # Parsed lump -> func which remakes the raw data. Any = ParsedLump's T, but
    # that can't bind here.
    _save_funcs: ClassVar[Dict[BSP_LUMPS, Callable[['BSP', Any], bytes]]] = {}
    def __init__(self, filename: str, version: VERSIONS=None):
        self.filename = filename
        self.map_revision = -1  # The map's revision count
        self.lumps = {}  # type: dict[BSP_LUMPS, Lump]
        self._parsed_lumps = {}  # type: dict[BSP_LUMPS, Any]
        self.game_lumps = {}  # type: dict[bytes, GameLump]
        self.header_off = 0
        self.version = version  # type: Optional[Union[VERSIONS, int]]
        # Tracks if the ent lump is using the new x1D output separators,
        # or the old comma separators. If no outputs are present there's no
        # way to determine this.
        self.out_comma_sep: Optional[bool] = None
        # This internally stores the texdata values texinfo refers to. Users
        # don't interact directly, instead they use the create_texinfo / texinfo.set()
        # methods that create the data as required.
        self._texdata: Dict[str, TexData] = {}

        self.read()

    pakfile: ParsedLump[ZipFile] = ParsedLump(BSP_LUMPS.PAKFILE)
    ents: ParsedLump[VMF] = ParsedLump(BSP_LUMPS.ENTITIES)
    textures: ParsedLump[List[str]] = ParsedLump(BSP_LUMPS.TEXDATA_STRING_DATA, BSP_LUMPS.TEXDATA_STRING_TABLE)
    texinfo: ParsedLump[List['TexInfo']] = ParsedLump(BSP_LUMPS.TEXINFO, BSP_LUMPS.TEXDATA)
    cubemaps: ParsedLump[List['Cubemap']] = ParsedLump(BSP_LUMPS.CUBEMAPS)
    overlays: ParsedLump[List['Overlay']] = ParsedLump(BSP_LUMPS.OVERLAYS)

    bmodels: ParsedLump[List['BModel']] = ParsedLump(BSP_LUMPS.MODELS)
    brushes: ParsedLump[List['Brush']] = ParsedLump(BSP_LUMPS.BRUSHES, BSP_LUMPS.BRUSHSIDES)
    visleafs: ParsedLump[List['VisLeaf']] = ParsedLump(BSP_LUMPS.LEAFS, BSP_LUMPS.LEAFFACES, BSP_LUMPS.LEAFBRUSHES)
    nodes: ParsedLump[List['VisTree']] = ParsedLump(BSP_LUMPS.NODES)

    vertexes: ParsedLump[List[Vec]] = ParsedLump(BSP_LUMPS.VERTEXES)
    surfedges: ParsedLump[List[Edge]] = ParsedLump(BSP_LUMPS.SURFEDGES, BSP_LUMPS.EDGES)
    planes: ParsedLump[List['Plane']] = ParsedLump(BSP_LUMPS.PLANES)
    faces: ParsedLump[List['Face']] = ParsedLump(BSP_LUMPS.FACES)
    orig_faces: ParsedLump[List['Face']] = ParsedLump(BSP_LUMPS.ORIGINALFACES)
    hdr_faces: ParsedLump[List['Face']] = ParsedLump(BSP_LUMPS.FACES_HDR)

    def read(self) -> None:
        """Load all data."""
        self.lumps.clear()
        self.game_lumps.clear()
        self._parsed_lumps.clear()
        self._texdata.clear()

        with open(self.filename, mode='br') as file:
            # BSP files start with 'VBSP', then a version number.
            magic_name, bsp_version = struct_read(HEADER_1, file)
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
                offset, length, version, ident = struct_read(HEADER_LUMP, file)
                lump_id = BSP_LUMPS(index)
                self.lumps[lump_id] = Lump(
                    lump_id,
                    version,
                    ident,
                )
                lump_offsets[lump_id] = offset, length

            [self.map_revision] = struct_read(HEADER_2, file)

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
        # First, go through lumps the user has accessed, and rebuild their data.
        for lump_name in LUMP_REBUILD_ORDER:
            try:
                data = self._parsed_lumps.pop(lump_name)
            except KeyError:
                pass
            else:
                self.lumps[lump_name].data = self._save_funcs[lump_name](self, data)
        game_lumps = list(self.game_lumps.values())  # Lock iteration order.

        with AtomicWriter(filename or self.filename, is_bytes=True) as file:  # type: BinaryIO
            # Needed to allow writing out the header before we know the position
            # data will be.
            defer = DeferredWrites(file)

            if isinstance(self.version, VERSIONS):
                version = self.version.value
            else:
                version = self.version

            file.write(struct.pack(HEADER_1, BSP_MAGIC, version))

            # Write headers.
            for lump_name in BSP_LUMPS:
                lump = self.lumps[lump_name]
                defer.defer(lump_name, '<ii')
                file.write(struct.pack(
                    HEADER_LUMP,
                    0,  # offset
                    0,  # length
                    lump.version,
                    lump.ident,
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
                        defer.defer(game_lump.id, '<i', write=True)
                        file.write(struct.pack('<i', len(game_lump.data)))

                    # Now write data.
                    for game_lump in game_lumps:
                        defer.set_data(game_lump.id, file.tell())
                        file.write(game_lump.data)
                    # Length of the game lump is current - start.
                    defer.set_data(
                        lump_name,
                        lump_start,
                        file.tell() - lump_start,
                    )
                else:
                    # Normal lump.
                    defer.set_data(lump_name, file.tell(), len(lump.data))
                    file.write(lump.data)
            # Apply all the deferred writes.
            defer.write()

    def read_header(self) -> None:
        """No longer used."""
        warnings.warn('Does nothing.', DeprecationWarning, stacklevel=2)

    def read_game_lumps(self) -> None:
        """No longer used."""
        warnings.warn('Does nothing.', DeprecationWarning, stacklevel=2)

    def replace_lump(
        self,
        new_name: str,
        lump: Union[BSP_LUMPS, 'Lump'],
        new_data: bytes
    ) -> None:
        """Write out the BSP file, replacing a lump with the given bytes.

        This is deprecated, simply assign to the .data attribute of the lump.
        """
        warnings.warn(
            'This is deprecated, use the appropriate property, '
            'or the .data attribute of the lump.',
            DeprecationWarning, stacklevel=2,
        )
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
            raise ValueError('{!r} not in {}'.format(lump_id, list(self.game_lumps)))
        return lump.data

    @overload
    def create_texinfo(self, mat: str, *, copy_from: 'TexInfo', fsys: FileSystem) -> 'TexInfo':
        """Copy from texinfo using filesystem."""
    @overload
    def create_texinfo(
        self, mat: str, *, copy_from: 'TexInfo',
        reflectivity: Vec, width: int, height: int,
    ) -> 'TexInfo':
        """Copy from texinfo and explicit texdata."""

    @overload
    def create_texinfo(
        self, mat: str,
        s_off: Vec=Vec(),
        s_shift: float=-99999.0,
        t_off: Vec=Vec(),
        t_shift: float=-99999.0,
        lightmap_s_off: Vec=Vec(),
        lightmap_s_shift: float=-99999.0,
        lightmap_t_off: Vec=Vec(),
        lightmap_t_shift: float=-99999.0,
        flags: SurfFlags = SurfFlags.NONE,
        *, fsys: FileSystem,
    ) -> 'TexInfo':
        """Construct from filesystem."""
    @overload
    def create_texinfo(
        self, mat: str,
        s_off: Vec=Vec(),
        s_shift: float=-99999.0,
        t_off: Vec=Vec(),
        t_shift: float=-99999.0,
        lightmap_s_off: Vec=Vec(),
        lightmap_s_shift: float=-99999.0,
        lightmap_t_off: Vec=Vec(),
        lightmap_t_shift: float=-99999.0,
        flags: SurfFlags = SurfFlags.NONE,
        *,
        reflectivity: Vec, width: int, height: int,
    ) -> 'TexInfo':
        """Construct with explicit texdata."""

    def create_texinfo(
        self, mat: str,
        s_off: Vec=Vec(),
        s_shift: float=-99999.0,
        t_off: Vec=Vec(),
        t_shift: float=-99999.0,
        lightmap_s_off: Vec=Vec(),
        lightmap_s_shift: float=-99999.0,
        lightmap_t_off: Vec=Vec(),
        lightmap_t_shift: float=-99999.0,
        flags: SurfFlags = SurfFlags.NONE,
        *,
        copy_from: 'TexInfo' = None,
        reflectivity: Vec=None, width: int=0, height: int=0,
        fsys: FileSystem=None,
    ) -> 'TexInfo':
        """Create or find a texinfo entry with the specified values.

        The s/t offset and shift values control the texture positioning. The
        defaults are those used for overlays, but for brushes all must be
        specified. Alternatively copy_from can be provided an existing texinfo
        to copy from, if a texture is being swapped out.

        In the BSP each material also stores its texture size and reflectivity.
        If the material has not been used yet, these must either be specified
        manually or a filesystem provided for the VMT and VTFs to be read from.
        """
        if copy_from is not None:
            s_off = copy_from.s_off
            s_shift = copy_from.s_shift
            t_off = copy_from.t_off
            t_shift = copy_from.t_shift
            lightmap_s_off = copy_from.lightmap_s_off
            lightmap_s_shift = copy_from.lightmap_s_shift
            lightmap_t_off = copy_from.lightmap_t_off
            lightmap_t_shift = copy_from.lightmap_t_shift

        try:
            data = self._texdata[mat.casefold()]
            search = True
        except KeyError:
            search = False
            if fsys is None:
                if reflectivity is None or not width or not height:
                    raise TypeError(
                        'Either valid data must be provided or a filesystem '
                        'to read them from!')
                data = TexData(mat, reflectivity.copy(), width, height)
            else:
                data = TexData.from_material(fsys, mat)
            self._texdata[mat.casefold()] = data

        new_texinfo = TexInfo(
            Vec(s_off), s_shift,
            Vec(t_off), t_shift,
            Vec(lightmap_s_off), lightmap_s_shift,
            Vec(lightmap_t_off), lightmap_t_shift,
            flags, data,
        )
        # Texdata is present, look for matching texinfos?
        if search:
            for orig_texinfo in self.texinfo:
                if orig_texinfo == new_texinfo:
                    return orig_texinfo
        self.texinfo.append(new_texinfo)
        return new_texinfo

    # Lump-specific commands:
    def _lmp_read_planes(self, data: bytes) -> List['Plane']:
        for x, y, z, dist, typ in struct.iter_unpack('<ffffi', data):
            yield Plane(Vec(x, y, z), dist, PlaneType(typ))

    def _lmp_write_planes(self, planes: List['Plane']) -> bytes:
        return b''.join([
            struct.pack(
                '<ffffi',
                plane.normal.x, plane.normal.y, plane.normal.z,
                plane.dist,
                plane.type.value,
            )
            for plane in planes
        ])

    def _lmp_read_vertexes(self, vertexes: bytes) -> List[Vec]:
        return list(map(Vec, struct.iter_unpack('<fff', vertexes)))

    def _lmp_write_vertexes(self, vertexes: List[Vec]) -> bytes:
        return b''.join([struct.pack('<fff', pos.x, pos.y, pos.z) for pos in vertexes])

    def _lmp_read_surfedges(self, vertexes: bytes) -> Iterator[Edge]:
        verts: list[Vec] = self.vertexes
        edges = [
            (verts[a], verts[b])
            for a, b in struct.iter_unpack('<HH', self.lumps[BSP_LUMPS.EDGES].data)
        ]
        for [ind] in struct.iter_unpack('i', vertexes):
            if ind < 0:  # If negative, the vertexes are reversed order.
                yield edges[-ind][::-1]
            else:
                yield edges[ind]

    def _lmp_write_surfedges(self, surf_edges: List[Edge]) -> bytes:
        """Reconstruct the surfedges and edges lumps."""
        edge_buf = BytesIO()
        surf_buf = BytesIO()

        # (a, b) -> edge
        edge_to_ind: dict[tuple[float, float, float, float, float, float], int] = {}
        add_vert = _find_or_insert(self.vertexes, Vec.as_tuple)

        for a, b in surf_edges:
            # Check to see if this edge is already defined.
            # positive indexes are in forward order, negative
            # allows us to refer to a reversed definition.
            try:
                ind = edge_to_ind[a.x, a.y, a.z, b.x, b.y, b.z]
            except KeyError:
                pass
            else:
                surf_buf.write(struct.pack('i', ind))
                continue

            try:
                ind = -edge_to_ind[b.x, b.y, b.z, a.x, a.y, a.z]
                if ind == 0:  # Edge case, -0 = +0
                    raise KeyError
            except KeyError:
                pass
            else:
                surf_buf.write(struct.pack('i', ind))
                continue

            # No luck, we need to add an edge definition.
            ind = len(edge_to_ind)
            edge_to_ind[a.x, a.y, a.z, b.x, b.y, b.z] = ind

            edge_buf.write(struct.pack('<HH',  add_vert(a), add_vert(b)))
            surf_buf.write(struct.pack('i', ind))

        self.lumps[BSP_LUMPS.EDGES].data = edge_buf.getvalue()
        return surf_buf.getvalue()

    def _lmp_read_orig_faces(self, data: bytes, _orig_faces: List['Face'] = None) -> Iterator['Face']:
        """Read one of the faces arrays.

        For ORIG_FACES, _orig_faces is None and that entry is ignored.
        For the others, that is the parsed orig faces lump, which each face
        may reference.
        """
        for (
            plane_num,
            side,
            on_node,
            first_edge, num_edges,
            texinfo_ind,
            dispinfo,
            surf_fog_vol_id,
            lightstyles,
            light_offset,
            area,
            lightmap_mins_x, lightmap_mins_y,
            lightmap_size_x, lightmap_size_y,
            orig_face_ind,
            prim_num,
            prim_first,
            smoothing_group,
        ) in struct.iter_unpack('<H??i4h4sif5iHHI', data):
            # If orig faces is provided, that is the original face
            # we were created from. Additionally, it seems the original
            # face data has invalid texinfo, so copy ours on top of it.
            if _orig_faces is not None:
                orig_face = _orig_faces[orig_face_ind]
                orig_face.texinfo = texinfo = self.texinfo[texinfo_ind]
            else:
                orig_face = texinfo = None
            yield Face(
                self.planes[plane_num],
                side, on_node,
                self.surfedges[first_edge:first_edge+num_edges],
                texinfo,
                dispinfo,
                surf_fog_vol_id,
                lightstyles,
                light_offset,
                area,
                (lightmap_mins_x, lightmap_mins_y),
                (lightmap_size_x, lightmap_size_y),
                orig_face,
                prim_num, prim_first,
                smoothing_group,
            )

    def _lmp_write_orig_faces(self, faces: List['Face'], get_orig_face: Callable[['Face'], int]=None) -> bytes:
        """Reconstruct one of the faces arrays.

        If this isn't the orig faces array, get_orig_face should be
        _find_or_insert(self.orig_faces).
        """
        face_buf = BytesIO()
        add_texinfo = _find_or_insert(self.texinfo)
        add_plane = _find_or_insert(self.planes)
        add_edges = _find_or_extend(
            self.surfedges,
            lambda edges: edges[0].as_tuple() + edges[1].as_tuple(),
        )
        for face in faces:
            if face.orig_face is not None and get_orig_face is not None:
                orig_ind = get_orig_face(face.orig_face)
            else:
                orig_ind = -1
            if face.texinfo is not None:
                texinfo = add_texinfo(face.texinfo)
            else:
                texinfo = -1

            # noinspection PyProtectedMember
            face_buf.write(struct.pack(
                '<H??i4h4sif5iHHI',
                add_plane(face.plane),
                face.same_dir_as_plane,
                face.on_node,
                add_edges(face.edges), len(face.edges),
                texinfo,
                face._dispinfo_ind,
                face.surf_fog_volume_id,
                face.light_styles,
                face._lightmap_off,
                face.area,
                *face.lightmap_mins, *face.lightmap_size,
                orig_ind,
                face._prim_count, face._first_prim_id,
                face.smoothing_groups,
            ))
        return face_buf.getvalue()

    def _lmp_read_faces(self, data: bytes) -> Iterator['Face']:
        """Parse the main split faces lump."""
        return self._lmp_read_orig_faces(data, self.orig_faces)

    def _lmp_write_faces(self, faces: List['Face']) -> bytes:
        return self._lmp_write_orig_faces(faces, _find_or_insert(self.orig_faces))

    _lmp_read_hdr_faces = _lmp_read_faces
    _lmp_write_hdr_faces = _lmp_write_faces

    def _lmp_read_brushes(self, data: bytes) -> Iterator['Brush']:
        """Parse brush definitions, along with the sides."""
        # The bevel param should be a bool, but randomly has other bits set?
        sides = [
            BrushSide(self.planes[plane_num], self.texinfo[texinfo], dispinfo, bool(bevel & 1), bevel & ~1)
            for (plane_num, texinfo, dispinfo, bevel)
            in struct.iter_unpack('<Hhhh', self.lumps[BSP_LUMPS.BRUSHSIDES].data)
        ]
        for first_side, side_count, contents in struct.iter_unpack('<iii', data):
            yield Brush(BrushContents(contents), sides[first_side:first_side+side_count])

    def _lmp_write_brushes(self, brushes: List['Brush']) -> bytes:
        sides: list[BrushSide] = []
        add_plane = _find_or_insert(self.planes)
        add_texinfo = _find_or_insert(self.texinfo)
        add_sides = _find_or_extend(sides)

        brush_buf = BytesIO()
        sides_buf = BytesIO()
        for brush in brushes:
            brush_buf.write(struct.pack(
                '<iii',
                add_sides(brush.sides), len(brush.sides),
                brush.contents.value,
            ))
        for side in sides:
            sides_buf.write(struct.pack(
                '<Hhhh',
                add_plane(side.plane),
                add_texinfo(side.texinfo),
                side._dispinfo,
                side.is_bevel_plane | side._unknown_bevel_bits,
            ))
        self.lumps[BSP_LUMPS.BRUSHSIDES].data = sides_buf.getvalue()
        return brush_buf.getvalue()

    def _lmp_read_visleafs(self, data: bytes) -> List['VisLeaf']:
        """Parse the leafs of the visleaf/bsp tree."""
        # There's an indirection through these index arrays.
        # starmap() to unpack the 1-tuple struct result, then index with that.
        leaf_brushes = list(itertools.starmap(
            self.brushes.__getitem__,
            struct.iter_unpack('<H', self.lumps[BSP_LUMPS.LEAFBRUSHES].data),
        ))
        leaf_faces = list(itertools.starmap(
            self.faces.__getitem__,
            struct.iter_unpack('<H', self.lumps[BSP_LUMPS.LEAFFACES].data),
        ))

        leaf_fmt = '<ihh6h4Hh2x'
        # Some extra ambient light data.
        if self.version.value <= 19:
            leaf_fmt += '26x'

        for (
            contents,
            cluster_ind, area_and_flags,
            min_x, min_y, min_z,
            max_x, max_y, max_z,
            first_face, num_faces,
            first_brush, num_brushes,
            water_ind
        ) in struct.iter_unpack(leaf_fmt, data):
            area = area_and_flags >> 7
            flags = area_and_flags & 0b1111111
            yield VisLeaf(
                BrushContents(contents), cluster_ind, area, VisLeafFlags(flags),
                Vec(min_x, min_y, min_z),
                Vec(max_x, max_y, max_z),
                leaf_faces[first_face:first_face+num_faces],
                leaf_brushes[first_brush:first_brush+num_brushes],
                water_ind,
            )

    def _lmp_read_nodes(self, data: bytes) -> List['VisTree']:
        """Parse the main visleaf/bsp trees."""
        # First parse all the nodes, then link them up.
        nodes: List[Tuple[VisTree, int, int]] = []

        for (
            plane_ind, neg_ind, pos_ind,
            min_x, min_y, min_z,
            max_x, max_y, max_z,
            first_face, face_count, area_ind,
        ) in struct.iter_unpack('<iii6hHHh2x', data):
            nodes.append((VisTree(
                self.planes[plane_ind],
                Vec(min_x, min_y, min_z),
                Vec(max_x, max_y, max_z),
                self.faces[first_face:first_face+face_count],
                area_ind,
            ), neg_ind, pos_ind))

        for node, neg_ind, pos_ind in nodes:
            if neg_ind < 0:
                node.child_neg = self.visleafs[-1 - neg_ind]
            else:
                node.child_neg = nodes[neg_ind][0]
            if pos_ind < 0:
                node.child_pos = self.visleafs[-1 - pos_ind]
            else:
                node.child_pos = nodes[pos_ind][0]
        return [node for node, i, j in nodes]

    def _lmp_write_nodes(self, nodes: List['VisTree']) -> bytes:
        """Reconstruct the visleaf/bsp tree data."""
        add_node = _find_or_insert(nodes)
        add_plane = _find_or_insert(self.planes)
        add_leaf = _find_or_insert(self.visleafs)
        add_faces = _find_or_extend(self.faces)

        buf = BytesIO()

        node: VisTree
        for node in nodes:
            if isinstance(node.child_pos, VisLeaf):
                pos_ind = -(add_leaf(node.child_pos) + 1)
            else:
                pos_ind = add_node(node.child_pos)
            if isinstance(node.child_neg, VisLeaf):
                neg_ind = -(add_leaf(node.child_neg) + 1)
            else:
                neg_ind = add_node(node.child_neg)

            buf.write(struct.pack(
                '<iii6hHHh2x',
                add_plane(node.plane), neg_ind, pos_ind,
                int(node.mins.x), int(node.mins.y), int(node.mins.z),
                int(node.maxes.x), int(node.maxes.y), int(node.maxes.z),
                add_faces(node.faces), len(node.faces), node.area_ind,
            ))

        return buf.getvalue()

    def _lmp_write_visleafs(self, visleafs: List['VisLeaf']) -> bytes:
        """Reconstruct the leafs of the visleaf/bsp tree."""
        leaf_faces: list[int] = []
        leaf_brushes: list[int] = []

        add_face = _find_or_insert(self.faces)
        add_brush = _find_or_insert(self.brushes)
        add_faces = _find_or_extend(leaf_faces, identity)
        add_brushes = _find_or_extend(leaf_brushes, identity)

        buf = BytesIO()

        leaf_fmt = '<ihh6h4Hh2x'
        # Some extra ambient light data. TODO: handle this?
        if self.version.value <= 19:
            leaf_fmt += '26x'

        for leaf in visleafs:
            face_ind = add_faces([add_face(face) for face in leaf.faces])
            brush_ind = add_brushes([add_brush(brush) for brush in leaf.brushes])

            buf.write(struct.pack(
                leaf_fmt,
                leaf.contents.value, leaf.cluster_id,
                (leaf.area << 7 | leaf.flags.value),
                int(leaf.mins.x), int(leaf.mins.y), int(leaf.mins.z),
                int(leaf.maxes.x), int(leaf.maxes.y), int(leaf.maxes.z),
                face_ind, len(leaf.faces),
                brush_ind, len(leaf.brushes),
                leaf.water_id,
            ))

        self.lumps[BSP_LUMPS.LEAFFACES].data = struct.pack(f'<{len(leaf_faces)}H', *leaf_faces)
        self.lumps[BSP_LUMPS.LEAFBRUSHES].data = struct.pack(f'<{len(leaf_brushes)}H', *leaf_brushes)
        return buf.getvalue()

    def read_texture_names(self) -> Iterator[str]:
        """Iterate through all brush textures in the map."""
        warnings.warn('Access bsp.textures', DeprecationWarning, stacklevel=2)
        return iter(self.textures)

    def _lmp_read_textures(self, tex_data: bytes) -> Iterator[str]:
        tex_table = self.lumps[BSP_LUMPS.TEXDATA_STRING_TABLE].data
        # tex_table is an array of int offsets into tex_data. tex_data is a
        # null-terminated block of strings.

        table_offsets = struct.unpack(
            # The number of ints + i, for the repetitions in the struct.
            '<' + str(len(tex_table) // struct.calcsize('i')) + 'i',
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
                raise ValueError(f'Bad string at {off} in BSP! ({tex_data[off:str_off]!r})')

    def _lmp_write_textures(self, textures: List[str]) -> bytes:
        table = BytesIO()
        data = bytearray()
        for tex in textures:
            if len(tex) >= 128:
                raise OverflowError(f'Texture "{tex}" exceeds 128 character limit')
            string = tex.encode('ascii') + b'\0'
            ind = data.find(string)
            if ind == -1:
                ind = len(data)
                data.extend(string)
            table.write(struct.pack('<i', ind))
        self.lumps[BSP_LUMPS.TEXDATA_STRING_TABLE].data = table.getvalue()
        return bytes(data)

    def _lmp_read_texinfo(self, data: bytes) -> Iterator['TexInfo']:
        """Read the texture info lump, providing positioning information."""
        texdata_list = []
        for (
            ref_x, ref_y, ref_z, ind,
            w, h, vw, vh,
        ) in struct.iter_unpack('<3f5i', self.lumps[BSP_LUMPS.TEXDATA].data):
            mat = self.textures[ind]
            # The view width/height is unused stuff, identical to regular
            # width/height.
            assert vw == w and vh == h, f'{mat}: {w}x{h} != view {vw}x{vh}'
            texdata = TexData(mat, Vec(ref_x, ref_y, ref_z), w, h)
            texdata_list.append(texdata)
            self._texdata[mat.casefold()] = texdata
        self.lumps[BSP_LUMPS.TEXDATA].data = b''

        for (
            sx, sy, sz, so, tx, ty, tz, to,
            l_sx, l_sy, l_sz, l_so, l_tx, l_ty, l_tz, l_to,
            flags, texdata_ind,
        ) in struct.iter_unpack('<16fii', data):
            yield TexInfo(
                Vec(sx, sy, sz), so,
                Vec(tx, ty, tz), to,
                Vec(l_sx, l_sy, l_sz), l_so,
                Vec(l_tx, l_ty, l_tz), l_to,
                SurfFlags(flags),
                texdata_list[texdata_ind],
            )

    def _lmp_write_texinfo(self, texinfos: List['TexInfo']) -> bytes:
        """Rebuild the texinfo and texdata lump."""
        find_or_add_texture = _find_or_insert(self.textures, str.casefold)
        texdata_ind: dict[TexData, int] = {}

        texdata_list: list[bytes] = []
        texinfo_result: list[bytes] = []

        for info in texinfos:
            # noinspection PyProtectedMember
            tdat = info._info
            # Try and find an existing reference to this texdata.
            # If not, the index is at the end of the list, where we write and
            # insert it.
            # That's then done again for the texture name itself.
            try:
                ind = texdata_ind[tdat]
            except KeyError:
                ind = texdata_ind[tdat] = len(texdata_list)
                texdata_list.append(struct.pack(
                    '<3f5i',
                    tdat.reflectivity.x, tdat.reflectivity.y, tdat.reflectivity.z,
                    find_or_add_texture(tdat.mat),
                    tdat.width, tdat.height, tdat.width, tdat.height,
                ))
            texinfo_result.append(struct.pack(
                '<16fii',
                *info.s_off, info.s_shift,
                *info.t_off, info.t_shift,
                *info.lightmap_s_off, info.lightmap_s_shift,
                *info.lightmap_t_off, info.lightmap_t_shift,
                info.flags.value,
                ind,
            ))
        self.lumps[BSP_LUMPS.TEXDATA].data = b''.join(texdata_list)
        return b''.join(texinfo_result)

    def _lmp_read_bmodels(self, data: bytes) -> Iterator['BModel']:
        """Parse the brush model definitions."""
        for (
            min_x, min_y, min_z, max_x, max_y, max_z,
            pos_x, pos_y, pos_z,
            headnode,
            first_face, num_face,
        ) in struct.iter_unpack('<9fiii', data):
            yield BModel(
                Vec(min_x, min_y, min_z), Vec(max_x, max_y, max_z),
                Vec(pos_x, pos_y, pos_z),
                self.nodes[headnode],
                self.faces[first_face:first_face+num_face],
            )

    def _lmp_write_bmodels(self, bmodels: List['BModel']) -> bytes:
        """Write the brush model definitions."""
        add_node = _find_or_insert(self.nodes)
        add_faces = _find_or_extend(self.faces)
        buf = BytesIO()
        for model in bmodels:
            # noinspection PyProtectedMember
            buf.write(struct.pack(
                '<9fiii',
                model.mins.x, model.mins.y, model.mins.z,
                model.maxes.x, model.maxes.y, model.maxes.z,
                model.origin.x, model.origin.y, model.origin.z,
                add_node(model.node),
                add_faces(model.faces), len(model.faces),
            ))
        return buf.getvalue()

    def _lmp_read_pakfile(self, data: bytes) -> ZipFile:
        """Read the raw binary as writable zip archive."""
        zipfile = ZipFile(BytesIO(data), mode='a')
        zipfile.filename = self.filename
        return zipfile

    def _lmp_write_pakfile(self, file: ZipFile) -> bytes:
        """Extract the final zip data from the zipfile."""
        # Explicitly close the zip file, so the footer is done.
        buf = file.fp
        file.close()
        if isinstance(buf, BytesIO):
            return buf.getvalue()
        else:
            buf.seek(0)
            return buf.read()

    def _lmp_check_pakfile(self, file: ZipFile) -> None:
        if not isinstance(file.fp, BytesIO):
            raise ValueError('Zipfiles must be constructed with a BytesIO buffer.')

    def _lmp_read_cubemaps(self, data: bytes) -> List['Cubemap']:
        """Read the cubemaps lump."""
        return [
            Cubemap(Vec(x, y, z), size)
            for (x, y, z, size) in struct.iter_unpack('<iiii', data)
        ]

    def _lmp_write_cubemaps(self, cubemaps: List['Cubemap']) -> bytes:
        """Write out the cubemaps lump."""
        return b''.join([
            struct.pack(
                '<iiii',
                int(round(cube.origin.x)),
                int(round(cube.origin.y)),
                int(round(cube.origin.z)),
                cube.size,
            )
            for cube in cubemaps
        ])

    def _lmp_read_overlays(self, data: bytes) -> Iterator['Overlay']:
        """Read the overlays lump."""
        for block in struct.iter_unpack(
            '<ihH'  # id, texinfo, face-and-render-order
            '64i'  # face array.
            '4f'  # UV min/max
            '18f',  # 4 handle points, origin, normal
            data,
        ):
            over_id, texinfo, face_ro = block[:3]
            face_count = face_ro & ((1 << 14) - 1)
            render_order = face_ro >> 14
            if face_count > 64:
                raise ValueError(f'{face_ro} exceeds OVERLAY_BSP_FACE_COUNT (64)!')
            faces = list(block[3: 3 + face_count])
            u_min, u_max, v_min, v_max = block[67:71]
            uv1 = Vec(block[71:74])
            uv2 = Vec(block[74:77])
            uv3 = Vec(block[77:80])
            uv4 = Vec(block[80:83])
            origin = Vec(block[83:86])
            normal = Vec(block[86:89])
            assert len(block) == 89

            yield Overlay(
                over_id, origin, normal,
                self.texinfo[texinfo], face_count,
                faces, render_order,
                u_min, u_max,
                v_min, v_max,
                uv1, uv2, uv3, uv4,
            )

    def _lmp_write_overlays(self, overlays: List['Overlay']) -> bytes:
        """Write out all overlays."""
        add_texinfo = _find_or_insert(self.texinfo)
        buf = BytesIO()
        for over in overlays:
            face_cnt = len(over.faces)
            if face_cnt > 64:
                raise ValueError(f'{over.faces} exceeds OVERLAY_BSP_FACE_COUNT (64)!')
            buf.write(struct.pack(
                '<ihH',
                over.id,
                add_texinfo(over.texture),
                (over.render_order << 14 | face_cnt),
            ))
            # Build the array, then zero fill the remaining space.
            buf.write(struct.pack(f'<{face_cnt}i {4*(64-face_cnt)}x', *over.faces))
            buf.write(struct.pack('<4f', over.u_min, over.u_max, over.v_min, over.v_max))
            buf.write(struct.pack(
                '<18f',
                *over.uv1, *over.uv2, *over.uv3, *over.uv4,
                *over.origin, *over.normal,
            ))

        return buf.getvalue()

    @contextlib.contextmanager
    def packfile(self) -> Iterator[ZipFile]:
        """A context manager to allow editing the packed content.

        When successfully exited, the zip will be rewritten to the BSP file.
        """
        warnings.warn('Use BSP.pakfile to access the cached archive.', DeprecationWarning, stacklevel=2)
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
        """Deprecated function to parse the entdata lump.

        Use BSP.ents directly.
        """
        warnings.warn('Use BSP.ents directly.', DeprecationWarning, stacklevel=2)
        return self._lmp_read_ents(self.get_lump(BSP_LUMPS.ENTITIES))

    def _lmp_read_ents(self, ent_data: bytes) -> VMF:
        """Parse in entity data.

        This returns a VMF object, with entities mirroring that in the BSP.
        No brushes are read.
        """
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
                continue
            elif line == b'}':
                if cur_ent is None:
                    raise ValueError(
                        f'Too many closing brackets after'
                        f' {len(vmf.entities)} ents!'
                    )
                if cur_ent is vmf.spawn:
                    if cur_ent['classname'] != 'worldspawn':
                        raise ValueError('No worldspawn entity!')
                else:
                    # The spawn ent is stored in the attribute, not in the ent
                    # list.
                    vmf.add_ent(cur_ent)
                cur_ent = None
                continue
            elif line == b'\x00':  # Null byte at end of lump.
                if cur_ent is not None:
                    raise ValueError("Last entity didn't end!")
                return vmf

            if cur_ent is None:
                raise ValueError("Keyvalue outside brackets!")

            # Line is of the form <"key" "val">, but handle escaped quotes
            # in the value. Valve's parser doesn't allow that, but we might
            # as well be better...
            key, value = line.split(b'" "', 2)
            decoded_key = key[1:].decode('ascii')
            decoded_value = value[:-1].replace(br'\"', b'"').decode('ascii')

            # Now, we need to figure out if this is a keyvalue,
            # or connection.
            # If we're L4D+, this is easy - they use 0x1D as separator.
            # Before, it's a comma which is common in keyvalues.
            # Assume it's an output if it has exactly 4 commas, and the last two
            # successfully parse as numbers.
            if 27 in value:
                # All outputs use the comma_sep, so we can ID them.
                cur_ent.add_out(Output.parse(Property(decoded_key, decoded_value)))
                if self.out_comma_sep is None:
                    self.out_comma_sep = False
            elif value.count(b',') == 4:
                try:
                    cur_ent.add_out(Output.parse(Property(decoded_key, decoded_value)))
                except ValueError:
                    cur_ent[decoded_key] = decoded_value
                if self.out_comma_sep is None:
                    self.out_comma_sep = True
            else:
                # Normal keyvalue.
                cur_ent[decoded_key] = decoded_value

        # This keyvalue needs to be stored in the VMF object too.
        # The one in the entity is ignored.
        vmf.map_ver = conv_int(vmf.spawn['mapversion'], vmf.map_ver)

        return vmf

    def _lmp_write_ents(self, vmf: VMF) -> bytes:
        return self.write_ent_data(vmf, self.out_comma_sep, _show_dep=False)

    @staticmethod
    def write_ent_data(vmf: VMF, use_comma_sep: Optional[bool]=None, *, _show_dep=True) -> bytes:
        """Generate the entity data lump.
        
        This accepts a VMF file like that returned from read_ent_data(). 
        Brushes are ignored, so the VMF must use *xx model references.

        use_comma_sep can be used to force using either commas, or 0x1D in I/O.
        """
        if _show_dep:
            warnings.warn('Modify BSP.ents instead', DeprecationWarning, stacklevel=2)
        out = BytesIO()
        for ent in itertools.chain([vmf.spawn], vmf.entities):
            out.write(b'{\n')
            for key, value in ent.keys.items():
                out.write('"{}" "{}"\n'.format(key, escape_text(value)).encode('ascii'))
            for output in ent.outputs:
                if use_comma_sep is not None:
                    output.comma_sep = use_comma_sep
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
        [dict_num] = struct_read('<i', static_lump)
        for _ in range(dict_num):
            [padded_name] = struct_read('<128s', static_lump)
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

        [visleaf_count] = struct_read('<i', static_lump)
        visleaf_list = list(map(
            self.visleafs.__getitem__,
            struct_read('H' * visleaf_count, static_lump),
        ))

        [prop_count] = struct_read('<i', static_lump)

        for i in range(prop_count):
            origin = Vec(struct_read('fff', static_lump))
            angles = Angle(struct_read('fff', static_lump))

            [model_ind] = struct_read('<H', static_lump)

            (
                first_leaf,
                leaf_count,
                solidity,
                flags,
                skin,
                min_fade,
                max_fade,
            ) = struct_read('<HHBBiff', static_lump)

            model_name = model_dict[model_ind]

            visleafs = set(visleaf_list[first_leaf:first_leaf + leaf_count])
            lighting_origin = Vec(struct_read('<fff', static_lump))

            if version >= 5:
                fade_scale = struct_read('<f', static_lump)[0]
            else:
                fade_scale = 1  # default

            if version in (6, 7):
                min_dx_level, max_dx_level = struct_read('<HH', static_lump)
            else:
                # Replaced by GPU & CPU in later versions.
                min_dx_level = max_dx_level = 0  # None

            if version >= 8:
                (
                    min_cpu_level,
                    max_cpu_level,
                    min_gpu_level,
                    max_gpu_level,
                ) = struct_read('BBBB', static_lump)
            else:
                # None
                min_cpu_level = max_cpu_level = 0
                min_gpu_level = max_gpu_level = 0

            if version >= 7:
                r, g, b, renderfx = struct_read('BBBB', static_lump)
                # Alpha isn't used.
                tint = Vec(r, g, b)
            else:
                # No tint.
                tint = Vec(255, 255, 255)
                renderfx = 255

            if version >= 11:
                # Unknown data, though it's float-like.
                unknown_1 = struct_read('<i', static_lump)

            if version >= 10:
                # Extra flags, post-CSGO.
                flags |= struct_read('<I', static_lump)[0] << 8

            flags = StaticPropFlags(flags)

            scaling = 1.0
            disable_on_xbox = False

            if version >= 11:
                # XBox support was removed. Instead this is the scaling factor.
                [scaling] = struct_read("<f", static_lump)
            elif version >= 9:
                # The single boolean byte also produces 3 pad bytes.
                [disable_on_xbox] = struct_read('<?xxx', static_lump)

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
        leaf_array: list[int] = []
        model_list: list[str] = []
        add_model = _find_or_insert(model_list, identity)
        add_leaf = _find_or_insert(self.visleafs)

        indexes: list[tuple[int, int]] = []
        for prop in props:
            indexes.append((len(leaf_array), add_model(prop.model)))
            leaf_array.extend(sorted([add_leaf(leaf) for leaf in prop.visleafs]))

        game_lump = self.game_lumps[b'sprp']

        # Now write out the sections.
        prop_lump = BytesIO()
        prop_lump.write(struct.pack('<i', len(model_list)))
        for name in model_list:
            prop_lump.write(struct.pack('<128s', name.encode('ascii')))

        prop_lump.write(struct.pack('<i', len(leaf_array)))
        prop_lump.write(struct.pack('<{}H'.format(len(leaf_array)), *leaf_array))

        prop_lump.write(struct.pack('<i', len(props)))
        for (leaf_off, model_ind), prop in zip(indexes, props):
            prop_lump.write(struct.pack(
                '<6fH',
                prop.origin.x,
                prop.origin.y,
                prop.origin.z,
                prop.angles.pitch,
                prop.angles.yaw,
                prop.angles.roll,
                model_ind,
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

    def vis_tree(self) -> 'VisTree':
        """Parse the visleaf data, and return the root node."""
        # First node is the top of the tree.
        return self.nodes[0]


@attr.define(eq=False)
class Lump:
    """Represents a lump header in a BSP file.

    """
    type: BSP_LUMPS
    version: int
    ident: bytes
    data: bytes = b''

    def __repr__(self) -> str:
        return '<BSP Lump {!r}, v{}, ident={!r}, {} bytes>'.format(
            self.type.name,
            self.version,
            self.ident,
            len(self.data),
        )


@attr.define(eq=False)
class GameLump:
    """Represents a game lump.

    These are designed to be game-specific.
    """
    id: bytes
    flags: int
    version: int
    data: bytes = b''

    ST: ClassVar[struct.Struct] = struct.Struct('<4s HH ii')

    def __repr__(self) -> str:
        return '<GameLump {}, flags={}, v{}, {} bytes>'.format(
            repr(self.id)[1:],
            self.flags,
            self.version,
            len(self.data),
        )


@attr.define(eq=False)
class TexData:
    """Represents some additional infomation for textures.

    Do not construct directly, use BSP.create_texinfo() or TexInfo.set().
    """
    mat: str
    reflectivity: Vec
    width: int
    height: int
    # TexData has a 'view' width/height too, but it isn't used at all and is
    # always the same as regular width/height.

    @classmethod
    def from_material(cls, fsys: FileSystem, mat_name: str) -> 'TexData':
        """Given a filesystem, parse the specified material and compute the texture values."""
        orig_mat = mat_name
        mat_folded = mat_name.casefold()
        if not mat_folded.replace('\\', '/').startswith('materials/'):
            mat_name = 'materials/' + mat_name

        mat_fname = mat_name
        if mat_folded[-4] != '.':
            mat_fname += '.vmt'

        with fsys, fsys[mat_fname].open_str() as tfile:
            mat = Material.parse(tfile, mat_name)
            mat.apply_patches(fsys)

        # Figure out the matching texture. If no texture, we look for one
        # with the same name as the material.
        tex_name = mat.get('$basetexture', mat_name)
        if not tex_name.casefold().replace('\\', '/').startswith('materials/'):
            tex_name = 'materials/' + tex_name
        if tex_name[-4] != '.':
            tex_name += '.vtf'

        try:
            with fsys, fsys[tex_name].open_bin() as bfile:
                vtf = VTF.read(bfile)
                reflect = vtf.reflectivity.copy()
                width, height = vtf.width, vtf.height
        except FileNotFoundError:
            width = height = 0
            reflect = Vec(0.2, 0.2, 0.2)  # CMaterial:CMaterial()

        try:
            reflect = Vec.from_str(mat['$reflectivity'])
            print(reflect, repr(mat['$reflectivity']))
        except KeyError:
            pass
        return TexData(orig_mat, reflect, width, height)


@attr.define(eq=True)
class TexInfo:
    """Represents texture positioning / scaling info."""
    s_off: Vec
    s_shift: float
    t_off: Vec
    t_shift: float
    lightmap_s_off: Vec
    lightmap_s_shift: float
    lightmap_t_off: Vec
    lightmap_t_shift: float
    flags: SurfFlags
    _info: TexData

    @property
    def mat(self) -> str:
        """The material used for this texinfo."""
        return self._info.mat

    @property
    def reflectivity(self) -> Vec:
        """The reflectivity of the texture."""
        return self._info.reflectivity.copy()

    @property
    def tex_size(self) -> Tuple[int, int]:
        """The size of the texture."""
        return self._info.width, self._info.height

    @overload
    def set(self, bsp: BSP, mat: str, *, fsys: FileSystem) -> None: ...
    @overload
    def set(self, bsp: BSP, mat: str, reflectivity: Vec, width: int, height: int) -> None: ...

    def set(
        self,
        bsp: BSP, mat: str,
        reflectivity: Vec=None, width: int=0, height: int=0,
        fsys: FileSystem=None,
    ) -> None:
        """Set the material used for this texinfo.

        If it is not already used in the BSP, some additional info is required.
        This can either be parsed from the VMT and VTF, or provided directly.
        """
        # noinspection PyProtectedMember
        try:
            data = bsp._texdata[mat.casefold()]
        except KeyError:
            # Need to create.
            if fsys is None:
                if reflectivity is None or not width or not height:
                    raise TypeError('Either valid data must be provided or a filesystem to read them from!')
                data = TexData(mat, reflectivity.copy(), width, height)
            else:
                data = TexData.from_material(fsys, mat)
            bsp._texdata[mat.casefold()] = data
        self._info = data


@attr.define(eq=False)
class Plane:
    """A plane."""
    normal: Vec
    dist: float
    type: PlaneType = attr.ib()

    @type.default
    def compute_type(self) -> 'PlaneType':
        """Compute the plane type parameter."""
        x = abs(self.normal.x)
        y = abs(self.normal.y)
        z = abs(self.normal.z)
        if x > 0.99:
            return PlaneType.X
        if y > 0.99:
            return PlaneType.Y
        if z > 0.99:
            return PlaneType.Z
        if x > y and x > z:
            return PlaneType.ANY_X
        if y > x and y > z:
            return PlaneType.ANY_Y
        return PlaneType.ANY_Z


@attr.define(eq=False)
class Face:
    """A brush face definition."""
    plane: Plane
    same_dir_as_plane: bool
    on_node: bool
    edges: List[Edge]
    texinfo: TexInfo
    _dispinfo_ind: int  # TODO
    surf_fog_volume_id: int
    light_styles: bytes
    _lightmap_off: int  # TODO
    area: float
    lightmap_mins: Tuple[int, int]
    lightmap_size: Tuple[int, int]
    orig_face: Optional['Face']
    _prim_count: int  # TODO, parse
    _first_prim_id: int
    smoothing_groups: int


@attr.define(eq=False)
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
    model: str
    origin: Vec
    angles: Angle = attr.ib(factory=Angle)
    scaling: float = 1.0
    visleafs: Set['VisLeaf'] = attr.ib(factory=set)
    solidity: int = 6
    flags: StaticPropFlags = StaticPropFlags.NONE
    skin: int = 0
    min_fade: float = 0.0
    max_fade: float = 0.0
    # If not provided, uses origin.
    lighting: Vec = attr.ib(default=attr.Factory(lambda prp: prp.origin.copy(), takes_self=True))
    fade_scale: float = -1.0
    min_dx_level: int = 0
    max_dx_level: int = 0
    min_cpu_level: int = 0
    max_cpu_level: int = 0
    min_gpu_level: int = 0
    max_gpu_level: int = 0
    tint: Vec = attr.ib(factory=lambda: Vec(255, 255, 255))
    renderfx: int = 255
    disable_on_xbox: bool = False

    def __repr__(self) -> str:
        return '<Prop "{}#{}" @ {} rot {}>'.format(
            self.model,
            self.skin,
            self.origin,
            self.angles,
        )


@attr.define(eq=False)
class Cubemap:
    """A env_cubemap positioned in the map.

    The position is integral, and the size can be zero for the default
    or a positive number for different powers of 2.
    """
    origin: Vec  # Always integer coordinates
    size: int = 0

    @property
    def resolution(self) -> int:
        """Return the actual image size."""
        if self.size == 0:
            return 32
        return 2**(self.size-1)


@attr.define(eq=False)
class Overlay:
    """An overlay embedded in the map."""
    id: int = attr.ib(eq=True)
    origin: Vec
    normal: Vec
    texture: TexInfo
    face_count: int
    faces: List[int] = attr.ib(factory=list, validator=attr.validators.deep_iterable(
        attr.validators.instance_of(int),
        attr.validators.instance_of(list),
    ))
    render_order: int = attr.ib(default=0, validator=attr.validators.in_(range(4)))
    u_min: float = 0.0
    u_max: float = 1.0
    v_min: float = 0.0
    v_max: float = 1.0
    # Four corner handles of the overlay.
    uv1: Vec = attr.ib(factory=lambda: Vec(-16, -16))
    uv2: Vec = attr.ib(factory=lambda: Vec(-16, +16))
    uv3: Vec = attr.ib(factory=lambda: Vec(+16, +16))
    uv4: Vec = attr.ib(factory=lambda: Vec(+16, -16))


@attr.define(eq=False)
class BModel:
    """A brush model definition, used for the world entity along with all other brush ents."""
    mins: Vec
    maxes: Vec
    origin: Vec
    node: 'VisTree'
    faces: List[Face]


@attr.define(eq=False)
class BrushSide:
    """A side of the original brush geometry which the map is constructed from.

    This matches the original VMF.
    """
    plane: Plane
    texinfo: TexInfo
    _dispinfo: int  # TODO
    is_bevel_plane: bool
    # The bevel member should be bool, but it has other bits set randomly.
    _unknown_bevel_bits: int = 0


@attr.define(eq=False)
class Brush:
    """A brush definition."""
    contents: BrushContents
    sides: List[BrushSide]


@attr.define(eq=False)
class VisLeaf:
    """A leaf in the visleaf/BSP data.

    The bounds is defined implicitly by the parent node planes.
    """
    contents: BrushContents
    cluster_id: int
    area: int
    flags: VisLeafFlags
    mins: Vec
    maxes: Vec
    faces: List[Face]
    brushes: List[Brush]
    water_id: int

    def test_point(self, point: Vec) -> Optional['VisLeaf']:
        """Test the given point against us, returning ourself or None."""
        return self if point.in_bbox(self.mins, self.maxes) else None


@attr.define(eq=False)
class VisTree:
    """A tree node in the visleaf/BSP data.

    Each of these is a plane splitting the map in two, which then has a child
    tree or visleaf on either side.
    """
    plane: Plane
    mins: Vec
    maxes: Vec
    faces: List[Face]
    area_ind: int
    # Initialised empty, set during loading.
    child_neg: Union['VisTree', VisLeaf] = attr.ib(default=None)
    child_pos: Union['VisTree', VisLeaf] = attr.ib(default=None)

    @property
    def plane_norm(self) -> Vec:
        """Deprecated alias for tree.plane.normal."""
        warnings.warn('Use tree.plane.normal', DeprecationWarning, stacklevel=2)
        return self.plane.normal

    @property
    def plane_dist(self) -> float:
        """Deprecated alias for tree.plane.dist."""
        warnings.warn('Use tree.plane.dist', DeprecationWarning, stacklevel=2)
        return self.plane.dist

    def test_point(self, point: Vec) -> Optional[VisLeaf]:
        """Test the given point against us, returning the hit leaf or None."""
        # Quick early out.
        if not point.in_bbox(self.mins, self.maxes):
            return None
        dist = self.plane.dist - point.dot(self.plane.normal)
        if dist > -1e-6:
            res = self.child_pos.test_point(point)
            if res is not None:
                return res
        elif dist < 1e-6:
            res = self.child_neg.test_point(point)
            if res is not None:
                return res
        return None

    def iter_leafs(self) -> Iterator[VisLeaf]:
        """Iterate over all child leafs, recursively."""
        checked: set[int] = set()  # Guard against recursion.
        nodes = [self]
        while nodes:
            node = nodes.pop()
            if id(node) in checked:
                continue
            checked.add(id(node))
            for child in [node.child_neg, node.child_pos]:
                if isinstance(child, VisLeaf):
                    yield child
                else:
                    nodes.append(child)
