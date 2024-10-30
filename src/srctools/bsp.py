"""Read and write parts of Source BSP files.

Data from a read BSP is lazily parsed when each section is accessed.
"""
from typing import (
    Any, Callable, ClassVar, Dict, Generator, Generic, Iterator, List, Mapping, Optional,
    Sequence, Set, Tuple, Type, TypeVar, Union, overload,
)
from typing_extensions import TypedDict, deprecated
from enum import Enum, Flag
from io import BytesIO
from weakref import WeakKeyDictionary
from zipfile import ZipFile
import contextlib
import inspect
import itertools
import math
import os
import struct
import warnings

import attrs

from srctools import AtomicWriter, StringPath, conv_int, logger
from srctools.binformat import (
    DeferredWrites, compress_lzma, decompress_lzma, find_or_extend, find_or_insert,
    read_array, struct_read, write_array,
)
from srctools.const import BSPContents as BrushContents, SurfFlags, add_unknown
from srctools.filesys import FileSystem
from srctools.keyvalues import Keyvalues
from srctools.math import Angle, AnyVec, FrozenVec, Vec
from srctools.tokenizer import Token, Tokenizer, escape_text
from srctools.vmf import VMF, Entity, Output
from srctools.vmt import Material
from srctools.vtf import VTF


__all__ = [
    'BSP_LUMPS', 'VERSIONS', 'GameVersion',
    'BSP', 'Lump', 'GameLump',
    'StaticProp', 'StaticPropFlags',
    'DetailProp', 'DetailPropModel', 'DetailPropOrientation', 'DetailPropShape', 'DetailPropSprite',
    'TexData', 'TexInfo',
    'Cubemap', 'Overlay',
    'VisTree', 'VisLeaf', 'VisLeafFlags', 'LeafWaterInfo',
    'Visibility',
    'BModel', 'Plane', 'PlaneType',
    'Primitive', 'Face', 'Edge', 'RevEdge',
    'Brush', 'BrushSide', 'BrushContents',
]

# Various constants, some exposed to allow handling unreleased formats.
BSP_MAGIC = b'VBSP'  # All BSP files start with this
VITAMIN_MAGIC = b'FART'  # Desolation's branch of Source.
HEADER_1 = '<4si'  # Header section before the lump list.
HEADER_LUMP = '<4i'  # Header section for each lump.
HEADER_2 = '<i'  # Header section after the lumps.
OVERLAY_FACE_COUNT = 64  # Max number of overlay faces.
TEXINFO_IND_TYPE = 'h'  # The type used to index into texinfo (i or h).

T = TypeVar('T')
KeyT = TypeVar('KeyT')  # Needs to be hashable, typecheckers currently don't handle that.

# Game lump IDs
LMP_ID_STATIC_PROPS = b'sprp'
LMP_ID_DETAIL_PROPS = b'dprp'

LOGGER = logger.get_logger(__name__)


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
    BLACK_MESA = 20

    BLOODY_GOOD_TIME = 20
    L4D2 = 21
    ALIEN_SWARM = 21
    PORTAL_2 = 21
    CS_GO = 21
    DEAR_ESTHER = 21
    STANLEY_PARABLE = 21

    INFRA = 22
    DOTA2 = 22
    CONTAGION = 23
    CHAOSSOURCE = 25  # Chaos' limit increased BSPs.

    DESOLATION_OLD = 42  # Old version.
    VITAMINSOURCE = 43  # Desolation's expanded map format.

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        """Versions are equal to their integer value."""
        return self.value == other

    def __ne__(self, other: object) -> bool:
        """Versions are equal to their integer value."""
        return self.value != other

    def __lt__(self, other: int) -> bool:
        """Versions are comparable to their integer value."""
        return self.value < other

    def __gt__(self, other: int) -> bool:
        """Versions are comparable to their integer value."""
        return self.value > other

    def __le__(self, other: int) -> bool:
        """Versions are comparable to their integer value."""
        return self.value <= other

    def __ge__(self, other: int) -> bool:
        """Versions are comparable to their integer value."""
        return self.value >= other


class GameVersion(Enum):
    """Identifies specific games which we need to detect and specially handle."""
    NORMAL = 'normal'  # Anything else.
    L4D2 = 'l4d2'  # L4D2 has some weirdness with the lump struct.
    VITAMINSOURCE = 'vitaminsource'  # Desolation's map format, changes many sections.

_GAMEVER_TO_REG = {
    GameVersion.L4D2: VERSIONS.L4D2,
    GameVersion.VITAMINSOURCE: VERSIONS.VITAMINSOURCE
}


class BSP_LUMPS(Enum):
    """All the lumps in a BSP file.

    The values represent the order lumps appear in the index.
    Some indexes were reused, so they have aliases.
    """
    ENTITIES = 0  #: self.ents
    PLANES = 1  #: self.planes
    TEXDATA = 2  # Inside self.texinfo
    VERTEXES = 3  #: self.vertexes
    VISIBILITY = 4
    NODES = 5  #: self.nodes
    TEXINFO = 6  #: self.texinfo
    FACES = 7  #: self.faces
    LIGHTING = 8
    OCCLUSION = 9
    LEAFS = 10  #: self.leafs
    FACEIDS = 11
    EDGES = 12  # Inside self.surfedges
    SURFEDGES = 13  #: self.surfedges
    MODELS = 14  #: self.bmodels
    WORLDLIGHTS = 15
    LEAFFACES = 16  # Inside self.leafs
    LEAFBRUSHES = 17  # Inside self.leafs
    BRUSHES = 18  #: self.brushes
    BRUSHSIDES = 19  # Inside self.brushes
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
    ORIGINALFACES = 27  #: self.orig_faces
    PHYSDISP = 28
    PHYSCOLLIDE = 29  # Inside self.bmodels
    VERTNORMALS = 30
    VERTNORMALINDICES = 31
    DISP_LIGHTMAP_ALPHAS = 32
    DISP_VERTS = 33
    DISP_LIGHTMAP_SAMPLE_POSITIONS = 34
    GAME_LUMP = 35
    LEAFWATERDATA = 36  #: self.water_leaf_info
    PRIMITIVES = 37  #: self.primitives
    PRIMVERTS = 38  # Inside self.primitives
    PRIMINDICES = 39  # Inside self.primitives
    PAKFILE = 40  #: self.pakfile
    CLIPPORTALVERTS = 41
    CUBEMAPS = 42  #: self.cubemaps
    TEXDATA_STRING_DATA = 43  # Inside self.textures
    TEXDATA_STRING_TABLE = 44  #: self.textures
    OVERLAYS = 45  #: self.overlays
    LEAFMINDISTTOWATER = 46  # Inside self.leafs
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
    FACES_HDR = 58  #: self.hdr_faces
    MAP_FLAGS = 59
    OVERLAY_FADES = 60
    OVERLAY_SYSTEM_LEVELS = 61
    PHYSLEVEL = 62
    DISP_MULTIBLEND = 63


class LumpDataLayout(TypedDict):
    """Container dictionary for lump layouts."""
    FACE: struct.Struct
    FACEID: struct.Struct
    EDGE: struct.Struct
    PRIMITIVE: struct.Struct
    PRIMINDEX: struct.Struct
    NODE: struct.Struct
    LEAF: struct.Struct
    LEAFFACE: struct.Struct
    LEAFBRUSH: struct.Struct
    LEAF_AREA_OFFSET: int
    LEAFWATERDATA: struct.Struct
    BRUSHSIDE: struct.Struct
    STATICPROPLEAF: struct.Struct


# Version specific lump data layout description
LUMP_LAYOUT_STANDARD: LumpDataLayout = {
    "FACE":             struct.Struct('<H??i4h4sif5iHHI'),
    "FACEID":           struct.Struct('<H'),
    "EDGE":             struct.Struct('<HH'),
    "PRIMITIVE":        struct.Struct('<HHHHH'),
    "PRIMINDEX":        struct.Struct('<H'),
    "NODE":             struct.Struct('<iii6hHHh2x'),
    "LEAF":             struct.Struct('<ihh6h4Hh2x'),  # Version 1
    "LEAFFACE":         struct.Struct('<H'),
    "LEAFBRUSH":        struct.Struct('<H'),
    "LEAF_AREA_OFFSET": 7,
    "LEAFWATERDATA":    struct.Struct('<ffH2x'),
    "BRUSHSIDE":        struct.Struct('<HhhH'),
    "STATICPROPLEAF":   struct.Struct('<H'),
}


LUMP_LAYOUT_V19: LumpDataLayout = {
    **LUMP_LAYOUT_STANDARD,
    "LEAF": struct.Struct('<ihh6h4Hh24s2x'),  # Version 0
}

LUMP_LAYOUT_INFRA: LumpDataLayout = {
    **LUMP_LAYOUT_STANDARD,
    # INFRA seems to have a different lump. It's 16 bytes, it seems to be:
    # char type;
    # int first_ind, ind_count;
    # short vert_ind, vert_count;
    # Then the type is promoted to int for structure alignment.
    "PRIMITIVE": struct.Struct('<IIIHH'),
}

LUMP_LAYOUT_VITAMIN: LumpDataLayout = {
    **LUMP_LAYOUT_STANDARD,
    "LEAF": struct.Struct('<ihh6I4HhBx'),
    "FACE": struct.Struct('<5i4iB3x'),
    "BRUSHSIDE": struct.Struct('<IIhBB'),
    "NODE": struct.Struct('<iii6iHHh2x'),
}

# https://chaosinitiative.github.io/Wiki/docs/Reference/bsp-v25/
LUMP_LAYOUT_CHAOS: LumpDataLayout = {
    **LUMP_LAYOUT_STANDARD,
    "FACE":             struct.Struct('<I??xx5i4sif5i3I'),
    "FACEID":           struct.Struct('<I'),
    "EDGE":             struct.Struct('<II'),
    "PRIMITIVE":        struct.Struct('<IIIII'),
    "PRIMINDEX":        struct.Struct('<I'),
    "NODE":             struct.Struct('<iii6fIIhxx'),
    "LEAF":             struct.Struct('<iii6f4Ii'),  # Version 2
    "LEAFFACE":         struct.Struct('<I'),
    "LEAFBRUSH":        struct.Struct('<I'),
    "LEAF_AREA_OFFSET": 17,
    "LEAFWATERDATA":    struct.Struct('<ffI'),
    "BRUSHSIDE":        struct.Struct('<IiiHxx'),
    "STATICPROPLEAF":   struct.Struct('<I'),
}


LUMP_COUNT = max(lump.value for lump in BSP_LUMPS) + 1  # 64 normally

# Special-case the packfile lump, put it at the end.
# This way the BSP can be opened by generic zip programs.
LUMP_WRITE_ORDER = list(BSP_LUMPS)
LUMP_WRITE_ORDER.remove(BSP_LUMPS.PAKFILE)
LUMP_WRITE_ORDER.append(BSP_LUMPS.PAKFILE)

# When remaking the lumps from trees of objects,
# they need to be done in the correct order so stuff referring
# to other trees can add their data.
LUMP_REBUILD_ORDER: List[Union[bytes, BSP_LUMPS]] = [
    BSP_LUMPS.PAKFILE,
    BSP_LUMPS.CUBEMAPS,
    LMP_ID_STATIC_PROPS,  # References visleafs.
    LMP_ID_DETAIL_PROPS,

    BSP_LUMPS.MODELS,  # Brushmodels reference their vis tree, faces, and the entity they're tied to.
    BSP_LUMPS.ENTITIES,  # References brushmodels, overlays, potentially many others.
    BSP_LUMPS.NODES,  # References planes, faces, visleafs.
    BSP_LUMPS.LEAFS,  # References brushes, faces, leaf water info, visibility.
    BSP_LUMPS.LEAFWATERDATA,  # References texinfo

    BSP_LUMPS.BRUSHES,  # also brushsides, references texinfo.

    BSP_LUMPS.FACES,  # References their original face, surfedges, texinfo, primitives.
    BSP_LUMPS.FACES_HDR,  # References their original face, surfedges, texinfo, primitives.
    BSP_LUMPS.ORIGINALFACES,  # references surfedges & texinfo, primitives.

    BSP_LUMPS.PRIMITIVES,
    BSP_LUMPS.SURFEDGES,  # surfedges references vertexes.
    BSP_LUMPS.PLANES,
    BSP_LUMPS.VERTEXES,
    BSP_LUMPS.VISIBILITY,

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

    @classmethod
    def from_normal(cls, normal: Vec) -> 'PlaneType':
        """Compute the correct orientation for a normal."""
        x = abs(normal.x)
        y = abs(normal.y)
        z = abs(normal.z)
        if x > 0.99:
            return cls.X
        if y > 0.99:
            return cls.Y
        if z > 0.99:
            return cls.Z
        if x > y and x > z:
            return cls.ANY_X
        if y > x and y > z:
            return cls.ANY_Y
        return cls.ANY_Z


class StaticPropVersion(Enum):
    """The detected version for static props.

    Despite the format having version numbers, several engine branches
    use the same number but have various changes.
    Thanks to BSPSource for this information.
    We record the version in the file, and the size of the structure.

    Name is set to prevent aliasing special variants that can't lookup.
    """
    def __init__(self, ver: int, size: int, name: str = '') -> None:
        self.version = ver
        self.size = size
        self.variant = name

    # V4 and V5 are used in original HL2 maps.
    V4 = (4, 56)
    V5 = (5, 60)  # adds forcedFadeScale
    V6 = (6, 64)  # Some TF2 maps, adds min/max DX level
    V7 = (7, 68)  # Old L4D maps, adds rendercolor
    V8 = (8, 68)  # Main L4D, removes min/max DX, adds min/max GPU and CPU
    V9 = (9, 72)  # L4D2, adds disableX360.
    V10 = (10, 76)  # Old CSGO, adds new flags integer
    V11 = (11, 80)  # New CSGO, with uniform prop scaling.

    # Source 2013, also appears with version 7 but is identical.
    # Based on v6, adds lightmapped props.
    # Despite the actual versions, more like v6.
    V_LIGHTMAP_v7 = (7, 72)
    V_LIGHTMAP_v10 = (10, 72)
    V_LIGHTMAP_MESA = (11, 80, 'Mesa')  # Adds rendercolor to V10

    V_CHAOS_V12 = (12, 80)  # Changes the leaf list from uint16 to uint32
    V_CHAOS_V13 = (13, 88)  # Changes scale from one float to three for non-uniform scaling

    # V6_WNAME = (5, 188)  # adds targetname, used by The Ship and Bloody Good Time.
    UNKNOWN = (0, 0, 'unknown')  # Before prop is read.
    # All games should recognise this, so switch to this if set to unknown.
    DEFAULT = V5

    @property
    def is_lightmap(self) -> bool:
        """Check if this has lightmaps version."""
        return self.name.startswith('V_LIGHTMAP')

    @property
    def is_sdk_2013(self) -> bool:
        """Check if this is either Source 2013 version, not including Mesa's modified one."""
        return self.name.startswith('V_LIGHTMAP_v')


_STATIC_PROP_VERSIONS: Mapping[Tuple[int, int], StaticPropVersion] = {
    (ver.version, ver.size): ver
    for ver in StaticPropVersion
    if not ver.variant
}


class StaticPropFlags(Flag):
    """Bitflags specified for static props."""
    NONE = 0

    DOES_FADE = 0x01  # Is the fade distances set?
    HAS_LIGHTING_ORIGIN = 0x02  # info_lighting entity used.
    #: This was nodraw in earlier versions, but it now prevents projected textures from affecting the prop.
    NO_FLASHLIGHT = DISABLE_DRAW = 0x04
    IGNORE_NORMALS = 0x08
    NO_SHADOW = 0x10
    SCREEN_SPACE_FADE = 0x20  #: Use screen space fading. Obsolete since at least ASW.
    NO_PER_VERTEX_LIGHTING = 0x40
    NO_SELF_SHADOWING = 0x80

    # These are set in the secondary flags section.
    #: Disable affecting projected texture lighting.
    #: In games supporting lightmapped props (TF2), this instead, disables per-luxel lighting.
    NO_SHADOW_DEPTH = 0x100
    NO_LIGHTMAP = 0x100
    BOUNCED_LIGHTING = 0x0400  #: Bounce lighting off the prop.

    # Add _BIT_XX members, so any bit combo can be preserved.
    add_unknown(locals(), long=True)

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
    HAS_DETAIL_OBJECTS = 0x08  # Contains detail props - ingame only, not set in BSP.

    # Undocumented flags, still in maps though?
    # Looks like uninitialised members.
    _BIT_3 = 1 << 3
    _BIT_4 = 1 << 4
    _BIT_5 = 1 << 5
    _BIT_6 = 1 << 6


class DetailPropOrientation(Enum):
    """The kind of orientation for detail props."""
    NORMAL = 0
    SCREEN_ALIGNED = 1
    SCREEN_ALIGNED_VERTICAL = 2


@attrs.define(eq=False, repr=False)
class Lump:
    """Represents a lump header in a BSP file.

    """
    type: BSP_LUMPS
    version: int
    data: bytes = b''
    # If true, this is LZMA compressed.
    is_compressed: bool = False

    def __repr__(self) -> str:
        return f'<BSP Lump {self.type.name!r}, v{self.version}, {len(self.data)} bytes>'


@attrs.define(eq=False, repr=False)
class GameLump:
    """Represents a game lump.

    These are designed to be game-specific.
    """
    id: bytes
    flags: int
    version: int
    data: bytes = b''

    ST: ClassVar[struct.Struct] = struct.Struct('<4s HH ii')

    @property
    def is_compressed(self) -> bool:
        """This flag indicates if the lump was compressed."""
        return self.flags & 0x1 != 0

    @is_compressed.setter
    def is_compressed(self, compressed: bool) -> None:
        """Change if the lump will be compressed when saved."""
        if compressed:
            self.flags |= 0x1
        else:
            self.flags &= ~0x1

    def __repr__(self) -> str:
        return (
            f'<GameLump {repr(self.id)[1:]}, flags={self.flags}, '
            f'v{self.version}, {len(self.data)} bytes>'
        )


@attrs.define(eq=False)
class TexData:
    """Represents some additional infomation for textures.

    Usually does not need to be constructed directly. Use :py:meth:`BSP.create_texinfo()` or
    :py:meth:`TexInfo.set()` to create this along with the texinfo.
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

        with fsys[mat_fname].open_str() as tfile:
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
            with fsys[tex_name].open_bin() as bfile:
                vtf = VTF.read(bfile)
                reflect = vtf.reflectivity.copy()
                width, height = vtf.width, vtf.height
        except FileNotFoundError:
            width = height = 0
            reflect = Vec(0.2, 0.2, 0.2)  # CMaterial:CMaterial()

        try:
            reflect = Vec.from_str(mat['$reflectivity'])
        except KeyError:
            pass
        return TexData(orig_mat, reflect, width, height)


@attrs.define(eq=True, repr=False)
class TexInfo:
    """Represents texture positioning / scaling info.

    Overlays don't use the offset/shifts, setting them to ``(0, 0, 0)`` and ``-99999.0`` respectively.
    """
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

    def __repr__(self) -> str:
        """For overlays, all the shift data is not important."""
        res = []
        if self.s_off or self.s_shift != -99999.0:
            res.append(f's_off={self.s_off}, s_shift={self.s_shift}')
        if self.t_off or self.t_shift != -99999.0:
            res.append(f't_off={self.t_off}, t_shift={self.t_shift}')
        if self.lightmap_s_off or self.lightmap_s_shift != -99999.0:
            res.append(f'lightmap_s_off={self.lightmap_s_off}, lightmap_s_shift={self.lightmap_s_shift}')
        if self.lightmap_t_off or self.lightmap_t_shift != -99999.0:
            res.append(f'lightmap_t_off={self.lightmap_t_off}, lightmap_t_shift={self.lightmap_t_shift}')
        res.append(f'flags={self.flags}, _info={self._info}')
        return f'TexInfo({", ".join(res)})'

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
    def set(self, bsp: 'BSP', mat: str, *, fsys: FileSystem) -> None: ...
    @overload
    def set(self, bsp: 'BSP', mat: str, reflectivity: Vec, width: int, height: int) -> None: ...

    def set(
        self,
        bsp: 'BSP', mat: str,
        reflectivity: Optional[Vec] = None,
        width: int = 0, height: int = 0,
        fsys: Optional[FileSystem] = None,
    ) -> None:
        """Set the material used for this texinfo.

        If it is not already used in the BSP, some additional info is required.
        This can either be parsed from the VMT and VTF, or provided directly.
        """
        try:
            self._info = bsp._texdata[mat.casefold()]
            return
        except KeyError:
            pass
            # Need to create.
        if fsys is None:
            if reflectivity is None or not width or not height:
                raise TypeError('Either valid data must be provided or a filesystem to read them from!')
            data = TexData(mat, reflectivity.copy(), width, height)
        else:
            data = TexData.from_material(fsys, mat)
        bsp._texdata[mat.casefold()] = data
        self._info = data


@attrs.define(eq=False)
class Plane:
    """A plane."""
    def _normal_setattr(self, _: 'attrs.Attribute[Vec]', value: Vec) -> Vec:
        """Recompute the plane type whenever the normal is changed."""
        value = Vec(value)
        self.type = PlaneType.from_normal(value)
        return value

    def _type_default(self) -> 'PlaneType':
        """Compute the plane type parameter if not provided."""
        return PlaneType.from_normal(self.normal)

    normal: Vec = attrs.field(on_setattr=_normal_setattr)
    dist: float = attrs.field(converter=float, validator=attrs.validators.instance_of(float))
    type: PlaneType = attrs.Factory(_type_default, takes_self=True)

    del _normal_setattr, _type_default


@attrs.define(eq=False)
class Primitive:
    """A 'primitive' surface (AKA t-junction, waterverts).

    These are generated to stitch together T-junction faces.
    """
    is_tristrip: bool
    indexed_verts: List[int]
    verts: List[Vec]


class Edge:
    """A pair of vertexes defining an edge of a face.

    The face on the other side of the edge has a RevEdge instead, which shares these vectors.
    """
    opposite: 'Edge'

    def __init__(self, a: Vec, b: Vec) -> None:
        self._a = a
        self._b = b
        self.opposite = RevEdge(self)

    @property
    def a(self) -> Vec:
        return self._a

    @a.setter
    def a(self, value: Vec) -> None:
        self._a = value

    @property
    def b(self) -> Vec:
        return self._b

    @b.setter
    def b(self, value: Vec) -> None:
        self._b = value

    def __repr__(self) -> str:
        return f'Edge({self.a!r}, {self.b!r})'

    def key(self) -> Tuple[object, ...]:
        """A key to match the edge with."""
        a, b = self.a, self.b
        return (a.x, a.y, a.z, b.x, b.y, b.z)


class RevEdge(Edge):
    """The edge on the opposite side from the original.

    This is implicitly created when an :class:`Edge` is.
    """
    # noinspection PyMissingConstructor
    def __init__(self, ed: Edge) -> None:
        # Deliberately not calling super to set a and b.
        self.opposite = ed

    def __repr__(self) -> str:
        return f'RevEdge({self.a!r}, {self.b!r})'

    @property
    def a(self) -> Vec:
        """This is a proxy for our opposite's B vec."""
        return self.opposite.b

    @a.setter
    def a(self, value: Vec) -> None:
        self.opposite.b = value

    @property
    def b(self) -> Vec:
        """This is a proxy for our opposite's A vec."""
        return self.opposite.a

    @b.setter
    def b(self, value: Vec) -> None:
        self.opposite.a = value


@attrs.define(eq=False)
class Face:
    """A brush face definition."""
    plane: Plane
    same_dir_as_plane: bool
    on_node: bool
    edges: List[Edge]
    texinfo: Optional[TexInfo]
    _dispinfo_ind: int  # TODO
    surf_fog_volume_id: int
    light_styles: bytes
    _lightmap_off: int  # TODO
    area: float
    lightmap_mins: Tuple[int, int]
    lightmap_size: Tuple[int, int]
    orig_face: Optional['Face']
    primitives: List[Primitive]
    dynamic_shadows: bool
    smoothing_groups: int
    hammer_id: Optional[int]  #: The original ID of the Hammer face.
    vitamin_flags: int  #: VitaminSource-specific flags.


@attrs.define(eq=False)
class DetailProp:
    """A detail prop, automatically placed on surfaces.

    This is a base class, use one of the subclasses only.
    """
    origin: Vec
    angles: Angle
    orientation: DetailPropOrientation
    leaf: int
    lighting: Tuple[int, int, int, int]
    _light_styles: Tuple[int, int]  # TODO: generate List[int]
    sway_amount: int

    def __attrs_pre_init__(self) -> None:
        """Only allow instantiation of subclasses."""
        if type(self) is DetailProp:
            raise TypeError('Cannot instantiate base DetailProp directly, use subclasses!')


@attrs.define(eq=False)
class DetailPropModel(DetailProp):
    """A MDL detail prop."""
    model: str


@attrs.define(eq=False)
class DetailPropSprite(DetailProp):
    """A sprite-type detail prop."""
    sprite_scale: float
    dims_upper_left: Tuple[float, float]
    dims_lower_right: Tuple[float, float]
    texcoord_upper_left: Tuple[float, float]
    texcoord_lower_right: Tuple[float, float]


@attrs.define(eq=False)
class DetailPropShape(DetailPropSprite):
    """A shape-type detail prop, rendered as a triangle or cross shape."""
    is_cross: bool
    shape_angle: int
    shape_size: int


@attrs.define(eq=False)
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
        res = 2 ** (self.size - 1)
        assert isinstance(res, int), self.size
        return res

# Pyright infers the default as Literal below, causing invariance issues in the validators.
_ZERO: int = int('0')


@attrs.define(eq=False)
class Overlay:
    """An overlay embedded in the map."""
    id: int = attrs.field(eq=True)
    origin: Vec
    normal: Vec
    texture: TexInfo
    face_count: int
    faces: List[int] = attrs.field(factory=list, validator=attrs.validators.deep_iterable(
        attrs.validators.instance_of(int),
        attrs.validators.instance_of(list),
    ))
    render_order: int = attrs.field(default=_ZERO, validator=attrs.validators.in_(range(4)))
    u_min: float = 0.0
    u_max: float = 1.0
    v_min: float = 0.0
    v_max: float = 1.0
    # Four corner handles of the overlay.
    uv1: Vec = attrs.field(factory=lambda: Vec(-16, -16))
    uv2: Vec = attrs.field(factory=lambda: Vec(-16, +16))
    uv3: Vec = attrs.field(factory=lambda: Vec(+16, +16))
    uv4: Vec = attrs.field(factory=lambda: Vec(+16, -16))

    fade_min_sq: float = -1.0
    fade_max_sq: float = 0.0

    # If system exceeds these limits, the overlay is skipped. Each is a single byte.
    min_cpu: int = attrs.field(default=_ZERO, validator=attrs.validators.in_(range(255)))
    max_cpu: int = attrs.field(default=_ZERO, validator=attrs.validators.in_(range(255)))
    min_gpu: int = attrs.field(default=_ZERO, validator=attrs.validators.in_(range(255)))
    max_gpu: int = attrs.field(default=_ZERO, validator=attrs.validators.in_(range(255)))


@attrs.define(eq=False)
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


@attrs.define(eq=False)
class Brush:
    """A brush definition."""
    contents: BrushContents
    sides: List[BrushSide]


@attrs.define(eq=False)
class VisLeaf:
    """A leaf in the visleaf/BSP data.

    The bounds are defined implicitly by the parent node planes.
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
    _ambient: bytes = bytes(24)
    # This is LEAF_MIN_DIST_TO_WATER
    min_water_dist: int = 65535

    def test_point(self, point: Vec) -> Optional['VisLeaf']:
        """Test the given point against us, returning ourselves or None."""
        return self if point.in_bbox(self.mins, self.maxes) else None


@attrs.define(eq=False)
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
    child_neg: Union['VisTree', VisLeaf] = attrs.field(default=None)
    child_pos: Union['VisTree', VisLeaf] = attrs.field(default=None)

    @property
    @deprecated('Use tree.plane.normal')
    def plane_norm(self) -> Vec:
        """Deprecated alias for tree.plane.normal."""
        return self.plane.normal

    @property
    @deprecated('Use tree.plane.dist')
    def plane_dist(self) -> float:
        """Deprecated alias for tree.plane.dist."""
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
        checked: Set[int] = set()  # Guard against recursion.
        nodes: List[VisTree] = [self]
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


@attrs.define(eq=False)
class LeafWaterInfo:
    """Additional data about water volumes."""
    surface_z: float
    min_z: float
    surface_texinfo: TexInfo


@attrs.define(eq=False)
class Visibility:
    """The visibility data produced by VVIS.

    Visleafs each have a "cluster" ID. For every pair of cluster IDs, this indicates if the first
    can see the second, and whether they can hear each other.
    """
    potentially_visible: List[bytearray]
    potentially_audible: List[bytearray]


@attrs.define(eq=False)
class BModel:
    """A brush model definition, used for the world entity along with all other brush ents."""
    mins: Vec
    maxes: Vec
    origin: Vec
    node: VisTree
    faces: List[Face]

    # If solid, the .phy file-like physics data.
    # This is a text section, and a list of blocks.
    phys_keyvalues: Optional[Keyvalues] = None
    _phys_solids: List[bytes] = attrs.field(factory=list)

    def clear_physics(self) -> None:
        """Delete the physics data for this brush model, and set the visleafs to non-solid.

        This is useful to optimise if the entity is known to not be solid to physics objects.
        """
        self._phys_solids.clear()
        self.phys_keyvalues = None
        for leaf in self.node.iter_leafs():
            leaf.contents = BrushContents.EMPTY


def _staticprop_lighting_default(self: 'StaticProp') -> Vec:
    """If not provided, lighting defaults to the origin."""
    return self.origin.copy()


@attrs.define(eq=False, repr=False)
class StaticProp:
    """Represents a ``prop_static`` in the BSP.

    Different features were added in different versions.

    * ``v5+`` allows fade_scale.
    * ``v6`` and ``v7`` allow min/max DXLevel.
    * ``v8+`` allows min/max GPU and CPU levels.
    * ``v7+`` allows model tinting, and renderfx.
    * ``v9+`` allows disabling on XBox 360.
    * ``v10+`` adds 4 unknown bytes (float?), and an expanded flags section.
    * ``v11+`` adds uniform scaling.
    """
    model: str
    origin: Vec
    angles: Angle = attrs.field(factory=Angle)
    scaling: Union[Vec, float] = attrs.field(factory=lambda: Vec(1.0, 1.0, 1.0))
    visleafs: Set[VisLeaf] = attrs.field(factory=set, repr=False)
    solidity: int = 6
    flags: StaticPropFlags = StaticPropFlags.NONE
    skin: int = 0
    min_fade: float = 0.0
    max_fade: float = 0.0
    lighting: Vec = attrs.Factory(_staticprop_lighting_default, takes_self=True)
    fade_scale: float = -1.0

    min_dx_level: int = 0
    max_dx_level: int = 0
    min_cpu_level: int = 0
    max_cpu_level: int = 0
    min_gpu_level: int = 0
    max_gpu_level: int = 0

    tint: Vec = attrs.field(factory=lambda: Vec(255, 255, 255))
    renderfx: int = 255
    disable_on_xbox: bool = False

    lightmap_x: int = 32
    lightmap_y: int = 32

    def __repr__(self) -> str:
        return f'<Prop "{self.model}#{self.skin}" @ {self.origin} rot {self.angles}>'


del _staticprop_lighting_default


def identity(x: T) -> T:
    """Identity function."""
    return x


def runlength_decode(
    data: Union[bytes, bytearray],
    start: int = 0, max_clusters: int = -1,
) -> bytearray:
    """Decode the run-length encoded viscluster flags in the visibility lump."""
    result = bytearray()
    pos = start
    size = len(data)
    # Use a memoryview, so we can copy right from the original -> dest.
    view = memoryview(data)
    if max_clusters == -1:
        ret_bytes = 1 << 128
    else:
        ret_bytes = math.ceil(max_clusters / 8)
    while pos < size and len(result) < ret_bytes:
        try:
            zero_ind = data.index(0x00, pos)
        except ValueError:
            # No more zeros.
            result += view[pos:]
            break
        # Copy the data from here to there.
        result += view[pos:zero_ind]
        # The byte afterward is how many zeros to insert.
        zeros = data[zero_ind + 1]
        result += bytes(zeros)
        # Advance past that.
        pos = zero_ind + 2

    # Trim down, if the last bulk copy went past the bytes we need.
    return result[:ret_bytes]


def runlength_encode(data: Union[bytes, bytearray]) -> bytearray:
    """Re-compress the run-length encoded viscluster flags in the visibility lump."""
    result = bytearray()
    pos = 0
    size = len(data)
    view = memoryview(data)
    while pos < size:
        try:
            zero_ind = data.index(0x00, pos)
        except ValueError:
            # No more zeros.
            result += view[pos:]
            break
        # Copy the data from here to there.
        result += view[pos:zero_ind]
        # Find the end of the zeros section.
        zero_end = zero_ind
        while zero_end < size and data[zero_end] == 0x00:
            zero_end += 1
        # If the distance is > 255, we need multiple zero sections.
        dist = zero_end - zero_ind
        while dist > 0:
            result.append(0x00)
            result.append(min(255, dist))
            dist -= 255
        pos = zero_end

    return result


class ParsedLump(Generic[T]):
    """Allows access to parsed versions of lumps.

    When accessed, the corresponding lump is parsed into an object tree.
    The lump is then cleared of data.
    When the BSP is saved, the lump data is then constructed.

    If the lump name is bytes, it's a game lump identifier.
    """
    lump: Union[bytes, BSP_LUMPS]
    to_clear: Sequence[Union[bytes, BSP_LUMPS]]
    __name__: str

    def __init__(self, lump: Union[bytes, BSP_LUMPS], *extra: Union[bytes, BSP_LUMPS]) -> None:
        self.lump = lump
        self.to_clear = (lump, *extra)
        self.__name__ = ''
        # May also be a Generator[X] if T = List[X]
        # Args are (BSP, version, data) if game lump, else (BSP, data).
        self._read: Optional[Callable[..., T]] = None
        self._check: Optional[Callable[[BSP, T], None]] = None
        assert self.lump in LUMP_REBUILD_ORDER, self.lump

    def __set_name__(self, owner: Type['BSP'], name: str) -> None:
        func_suffix = name.lstrip('_')  # Don't have us do blah__name if private.
        self.__name__ = name
        self.__objclass__ = owner
        self._read = getattr(owner, '_lmp_read_' + func_suffix)
        self._check = getattr(owner, '_lmp_check_' + func_suffix, None)
        # noinspection PyProtectedMember
        owner._save_funcs[self.lump] = getattr(owner, '_lmp_write_' + func_suffix)

    def __repr__(self) -> str:
        return f'<srctools.BSP.{self.__name__} member>'

    @overload
    def __get__(self, instance: None, owner: Optional[type] = None) -> 'ParsedLump[T]': ...
    @overload
    def __get__(self, instance: 'BSP', owner: Optional[type] = None) -> T: ...

    def __get__(self, instance: Optional['BSP'], owner: Optional[type] = None) -> Union['ParsedLump[T]', T]:
        """Read the lump, then discard."""
        if instance is None:  # Accessed on the class.
            return self
        result: T
        try:
            # noinspection PyProtectedMember
            result = instance._parsed_lumps[self.lump]
            return result
        except KeyError:
            pass
        if self._read is None:
            raise TypeError('ParsedLump.__set_name__ was never called!')
        if isinstance(self.lump, BSP_LUMPS):
            data = instance.lumps[self.lump].data
            LOGGER.debug('Load game lump {} ({} bytes)', self.lump, len(data))
            result = self._read(instance, data)
        else:  # Game lump
            gm_lump = instance.game_lumps[self.lump]
            LOGGER.debug('Load game lump {} v{} ({} bytes)', self.lump, gm_lump.version, len(gm_lump.data))
            result = self._read(instance, gm_lump.version, gm_lump.data)
        if inspect.isgenerator(result):  # Convenience, yield to accumulate into a list.
            result = list(result)  # type: ignore

        instance._parsed_lumps[self.lump] = result # noqa
        for lump in self.to_clear:
            if isinstance(lump, BSP_LUMPS):
                instance.lumps[lump].data = b''
            else:
                instance.game_lumps[lump].data = b''
        return result

    def __set__(self, instance: Optional['BSP'], value: T) -> None:
        """Discard lump data, then store."""
        if instance is None:
            raise TypeError('Cannot assign directly to lump descriptor!')
        if self._check is not None:
            # Allow raising if an invalid value.
            self._check(instance, value)
        for lump in self.to_clear:
            if isinstance(lump, BSP_LUMPS):
                instance.lumps[lump].data = b''
            else:
                instance.game_lumps[lump].data = b''
        instance._parsed_lumps[self.lump] = value  # noqa


# noinspection PyMethodMayBeStatic
class BSP:
    """A BSP file."""
    # Parsed lump -> func which remakes the raw data. Any = ParsedLump's T, but
    # that can't bind here.
    _save_funcs: ClassVar[Dict[
        Union[bytes, BSP_LUMPS],
        Callable[['BSP', Any], Union[bytes, Generator[bytes, None, None]]]
    ]] = {}
    #: The version ID in the file.
    version: Union[VERSIONS, int]
    #: A srctools-specific version to identify some games with unique handling.
    game_ver: GameVersion
    lump_layout: LumpDataLayout
    map_revision: int

    def __init__(
        self,
        filename: StringPath,
        version: Union[VERSIONS, GameVersion, None] = None,
    ) -> None:
        self.filename = filename
        self.map_revision = -1  # The map's revision count
        self.lumps: Dict[BSP_LUMPS, Lump] = {}
        self._parsed_lumps: Dict[Union[bytes, BSP_LUMPS], Any] = {}
        self.game_lumps: Dict[bytes, GameLump] = {}
        self.header_off = 0
        # Tracks if the ent lump is using the new x1D output separators,
        # or the old comma separators. If no outputs are present there's no
        # way to determine this.
        self.out_comma_sep: Optional[bool] = None
        self.static_prop_version: StaticPropVersion = StaticPropVersion.UNKNOWN
        # This internally stores the texdata values texinfo refers to. Users
        # don't interact directly, instead they use the create_texinfo / texinfo.set()
        # methods that create the data as required.
        self._texdata: Dict[str, TexData] = {}

        # lump_layout holds version specific struct formats for lumps
        self.lump_layout = LUMP_LAYOUT_STANDARD
        self.version = VERSIONS.HL2

        self.read(version)

    # The first lump is the main one this reads/writes to, any additional are simpler lumps it
    # reads and includes all the data for.
    pakfile: ParsedLump[ZipFile] = ParsedLump(BSP_LUMPS.PAKFILE)
    ents: ParsedLump[VMF] = ParsedLump(BSP_LUMPS.ENTITIES)
    textures: ParsedLump[List[str]] = ParsedLump(
        BSP_LUMPS.TEXDATA_STRING_DATA,
        BSP_LUMPS.TEXDATA_STRING_TABLE,
    )
    texinfo: ParsedLump[List[TexInfo]] = ParsedLump(BSP_LUMPS.TEXINFO, BSP_LUMPS.TEXDATA)
    cubemaps: ParsedLump[List[Cubemap]] = ParsedLump(BSP_LUMPS.CUBEMAPS)
    overlays: ParsedLump[List[Overlay]] = ParsedLump(
        BSP_LUMPS.OVERLAYS,
        BSP_LUMPS.OVERLAY_FADES, BSP_LUMPS.OVERLAY_SYSTEM_LEVELS,
    )

    bmodels: ParsedLump['WeakKeyDictionary[Entity, BModel]'] = ParsedLump(
        BSP_LUMPS.MODELS,
        BSP_LUMPS.PHYSCOLLIDE,
    )
    brushes: ParsedLump[List[Brush]] = ParsedLump(BSP_LUMPS.BRUSHES, BSP_LUMPS.BRUSHSIDES)
    visleafs: ParsedLump[List[VisLeaf]] = ParsedLump(
        BSP_LUMPS.LEAFS,
        BSP_LUMPS.LEAFFACES, BSP_LUMPS.LEAFBRUSHES, BSP_LUMPS.LEAFMINDISTTOWATER,
    )
    water_leaf_info: ParsedLump[List[LeafWaterInfo]] = ParsedLump(BSP_LUMPS.LEAFWATERDATA)
    nodes: ParsedLump[List[VisTree]] = ParsedLump(BSP_LUMPS.NODES)
    # This is None if VVIS has not been run.
    visibility: ParsedLump[Optional[Visibility]] = ParsedLump(BSP_LUMPS.VISIBILITY)

    vertexes: ParsedLump[List[Vec]] = ParsedLump(BSP_LUMPS.VERTEXES)
    surfedges: ParsedLump[List[Edge]] = ParsedLump(BSP_LUMPS.SURFEDGES, BSP_LUMPS.EDGES)
    planes: ParsedLump[List[Plane]] = ParsedLump(BSP_LUMPS.PLANES)
    faces: ParsedLump[List[Face]] = ParsedLump(BSP_LUMPS.FACES)
    orig_faces: ParsedLump[List[Face]] = ParsedLump(BSP_LUMPS.ORIGINALFACES)
    hdr_faces: ParsedLump[List[Face]] = ParsedLump(BSP_LUMPS.FACES_HDR)
    primitives: ParsedLump[List[Primitive]] = ParsedLump(
        BSP_LUMPS.PRIMITIVES,
        BSP_LUMPS.PRIMINDICES, BSP_LUMPS.PRIMVERTS,
    )

    # Game lumps
    props: ParsedLump[List['StaticProp']] = ParsedLump(LMP_ID_STATIC_PROPS)
    detail_props: ParsedLump[List['DetailProp']] = ParsedLump(LMP_ID_DETAIL_PROPS)

    @property
    def is_vitamin(self) -> bool:
        """Vitaminsource has a lot of structure changes."""
        return self.game_ver is GameVersion.VITAMINSOURCE

    def read(self, expected_version: Union[VERSIONS, GameVersion, None] = None) -> None:
        """Load all data."""
        self.lumps.clear()
        self.game_lumps.clear()
        self._parsed_lumps.clear()
        self._texdata.clear()

        if isinstance(expected_version, GameVersion):
            self.game_ver = expected_version
            if expected_version is GameVersion.NORMAL:
                expected_version = None
            else:
                expected_version = _GAMEVER_TO_REG[expected_version]
        else:
            self.game_ver = GameVersion.NORMAL

        with open(self.filename, mode='br') as file:
            # BSP files start with 'VBSP', then a version number.
            magic_name, bsp_version = struct_read(HEADER_1, file)
            if magic_name != BSP_MAGIC and magic_name != VITAMIN_MAGIC:
                raise ValueError('File is not a BSP file!')

            if expected_version is None:
                try:
                    self.version = VERSIONS(bsp_version)
                except ValueError:
                    self.version = bsp_version
                    LOGGER.warning(f'Unknown BSP Version \"{bsp_version}\"!')
            else:
                if bsp_version != expected_version:
                    raise ValueError(
                        f'Unexpected BSP version {bsp_version!r}, expected {expected_version!r}!'
                    )
                self.version = expected_version

            if magic_name == VITAMIN_MAGIC and self.version is not VERSIONS.VITAMINSOURCE:
                raise ValueError('VitaminSource uses a different version number.')

            if self.version is VERSIONS.CHAOSSOURCE:
                # Change the expected structure for lumps to fit chaos' increased limits
                self.lump_layout = LUMP_LAYOUT_CHAOS
            elif self.version is VERSIONS.VITAMINSOURCE:
                # Change the expected structure for lumps to fit vitamin's rad new format
                self.lump_layout = LUMP_LAYOUT_VITAMIN
                self.game_ver = GameVersion.VITAMINSOURCE
            elif self.version is VERSIONS.INFRA:
                self.lump_layout = LUMP_LAYOUT_INFRA
            elif self.version <= 19:
                self.lump_layout = LUMP_LAYOUT_V19

            lump_offsets: Dict[BSP_LUMPS, Tuple[int, int, int]] = {}
            offset: int
            length: int
            version: int
            uncomp_size: int

            if self.game_ver is GameVersion.NORMAL and self.version is VERSIONS.L4D2:
                # L4D2 is weird. See BSPSource here:
                # https://github.com/ata4/bspsrc/blob/678ab441f40361efa9d4ebf994989ed2b8e7ffce/src/main/java/info/ata4/bsplib/BspFile.java#L146
                offset = file.tell()
                file.seek(8)
                if file.read(4) == b'\0\0\0\0':
                    self.game_ver = GameVersion.L4D2
                file.seek(offset)

            # Read the index describing each BSP lump.
            for index in range(LUMP_COUNT):
                # The 4th value here is originally the fourCC identity, but is
                # instead used to indicate the unpacked size if compressed.
                offset, length, version, uncomp_size = struct_read(HEADER_LUMP, file)
                lump_id = BSP_LUMPS(index)
                if self.game_ver is GameVersion.L4D2:
                    # Order is slightly different.
                    version, offset, length = offset, length, version

                self.lumps[lump_id] = Lump(
                    lump_id,
                    version,
                )
                lump_offsets[lump_id] = offset, length, uncomp_size

            [self.map_revision] = struct_read(HEADER_2, file)

            for lump in self.lumps.values():
                # Now read in each lump.
                offset, length, uncomp_size = lump_offsets[lump.type]
                file.seek(offset)
                lump_data = file.read(length)
                if uncomp_size > 0:
                    lump.is_compressed = True
                    lump_data = decompress_lzma(lump_data)
                else:
                    lump.is_compressed = False
                lump.data = lump_data

            game_lump = self.lumps[BSP_LUMPS.GAME_LUMP]

            self.game_lumps.clear()

            [lump_count] = struct.unpack_from('<i', game_lump.data)
            lump_offset = 4
            gm_lump_offsets: Dict[bytes, Tuple[int, int]] = {}
            gm_lump_sizes: Dict[bytes, int] = {}
            prev_game_lump = b''
            prev_lump_offset = 0
            for _ in range(lump_count):
                game_lump_id: bytes
                flags: int
                glump_version: int
                file_off: int
                (
                    game_lump_id,
                    flags,
                    glump_version,
                    file_off,
                    uncomp_size,
                ) = GameLump.ST.unpack_from(game_lump.data, lump_offset)
                lump_offset += GameLump.ST.size
                # The lump ID is backward..
                game_lump_id = game_lump_id[::-1]

                gm_lump_offsets[game_lump_id] = file_off, uncomp_size

                # Determine the size of the lump by comparing offsets. This is necessary for
                # compressed lumps, but still works normally. To allow this BSPs have an extra
                # dummy entry with this ID which should have an offset value, but it's also just
                # zero and therefore useless. Skip that and use the size of the lump itself.
                if game_lump_id == b'\x00\x00\x00\x00':
                    continue

                if prev_game_lump:
                    gm_lump_sizes[prev_game_lump] = file_off - prev_lump_offset - 1
                prev_game_lump = game_lump_id
                prev_lump_offset = file_off

                self.game_lumps[game_lump_id] = GameLump(
                    game_lump_id,
                    flags,
                    glump_version,
                )
            # Handle the last game lump offset, by comparing to the end of the
            # whole lump.
            if prev_game_lump:
                gm_lump_start, gm_lump_length, _ = lump_offsets[BSP_LUMPS.GAME_LUMP]
                gm_lump_sizes[prev_game_lump] = gm_lump_start + gm_lump_length - prev_lump_offset

            for gm_lump in self.game_lumps.values():
                file_off, uncomp_size = gm_lump_offsets[gm_lump.id]
                file.seek(file_off)
                if gm_lump.is_compressed:
                    lump_data = file.read(gm_lump_sizes[gm_lump.id])
                    gm_lump.data = decompress_lzma(lump_data)
                else:
                    gm_lump.data = file.read(uncomp_size)

            # This is not valid any longer.
            game_lump.data = b''

    def save(self, filename: Optional[str] = None) -> None:
        """Write the BSP back into the given file."""
        # First, go through lumps the user has accessed, and rebuild their data.
        for lump_or_game in LUMP_REBUILD_ORDER:
            try:
                data = self._parsed_lumps.pop(lump_or_game)
            except KeyError:
                pass
            else:
                lump_result = self._save_funcs[lump_or_game](self, data)
                # Convenience, yield to accumulate into bytes.
                if inspect.isgenerator(lump_result):
                    buf = BytesIO()
                    for chunk in lump_result:
                        buf.write(chunk)
                    result = buf.getvalue()
                elif isinstance(lump_result, bytes):
                    result = lump_result
                else:
                    raise ValueError(lump_result)
                if isinstance(lump_or_game, BSP_LUMPS):
                    self.lumps[lump_or_game].data = result
                else:
                    self.game_lumps[lump_or_game].data = result
        game_lumps = list(self.game_lumps.values())  # Lock iteration order.

        with AtomicWriter(filename or self.filename, is_bytes=True) as file:
            # Needed to allow writing out the header before we know the position
            # data will be.
            defer = DeferredWrites(file)

            if self.version is None:
                raise ValueError('No version specified for BSP!')
            elif isinstance(self.version, VERSIONS):
                version = self.version.value
            else:
                version = self.version

            file.write(struct.pack(
                HEADER_1,
                VITAMIN_MAGIC if self.is_vitamin else BSP_MAGIC,
                version,
            ))

            # Write dummy values for the headers.
            for lump_name in BSP_LUMPS:
                defer.defer(lump_name, HEADER_LUMP, write=True)

            # After lump headers, the map revision...
            file.write(struct.pack(HEADER_2, self.map_revision))

            # Then each lump.
            for lump_name in LUMP_WRITE_ORDER:
                # Write out the actual data.
                lump = self.lumps[lump_name]
                if lump_name is BSP_LUMPS.GAME_LUMP:
                    # Construct this right here.
                    lump_start = file.tell()

                    # The size of each compressed segment is determined
                    # by checking the offset of the next part. So if
                    # the last is compressed, we need to add a dummy
                    # segment to supply the offset.
                    dummy_segment = 1 if (game_lumps and game_lumps[-1].is_compressed) else 0

                    file.write(struct.pack('<i', len(game_lumps) + dummy_segment))
                    for game_lump in game_lumps:
                        file.write(struct.pack(
                            '<4s HH',
                            game_lump.id[::-1],
                            game_lump.flags,
                            game_lump.version,
                        ))
                        # Offset, then data length. But we have to compress if
                        # required.
                        defer.defer(game_lump.id, '<ii', write=True)
                    if dummy_segment:
                        defer.defer(GameLump, '<8xi4x', write=True)

                    # Now write data.
                    for game_lump in game_lumps:
                        if game_lump.is_compressed:
                            print('Compress: ', game_lump.id)
                            lump_data = compress_lzma(game_lump.data)
                        else:
                            lump_data = game_lump.data
                        defer.set_data(game_lump.id, file.tell(), len(game_lump.data))
                        file.write(lump_data)
                        # If compressed, this is a big buffer, so discard asap.
                        del lump_data
                        # Put a null byte between each, for reasons.
                        if game_lump is not game_lumps[-1]:
                            file.write(b'\0')
                    if dummy_segment:
                        # As described above, dummy segment with zero length
                        # but valid ID.
                        defer.set_data(GameLump, file.tell())
                    # Length of the game lump is current - start.
                    # It is never compressed, and is always version 0.
                    length = file.tell() - lump_start
                    # However, L4D2 has a different order.
                    if self.game_ver is GameVersion.L4D2:
                        defer.set_data(
                            lump_name,
                            0, lump_start, length, 0,
                        )
                    else:
                        defer.set_data(
                            lump_name,
                            lump_start, length, 0, 0,
                        )

                else:
                    # Normal lump, pakfiles can't be compressed.
                    if lump.is_compressed and lump_name is not BSP_LUMPS.PAKFILE:
                        lump_fourcc = len(lump.data)
                        print('Compress: ', lump.type)
                        lump_data = compress_lzma(lump.data)
                    else:
                        lump_data = lump.data
                        lump_fourcc = 0
                    # However, L4D2 has a different order.
                    if self.game_ver is GameVersion.L4D2:
                        defer.set_data(lump_name, lump.version, file.tell(), len(lump_data), lump_fourcc)
                    else:
                        defer.set_data(lump_name, file.tell(), len(lump_data), lump.version, lump_fourcc)
                    file.write(lump_data)
                    # If compressed, this is a big buffer, so discard asap.
                    del lump_data

            # Apply all the deferred writes.
            defer.write()

    @deprecated("No longer has functionality")
    def read_header(self) -> None:
        """No longer used."""

    @deprecated("No longer has functionality")
    def read_game_lumps(self) -> None:
        """No longer used."""

    @deprecated(
        'This is deprecated, use the appropriate property, '
        'or the .data attribute of the lump.',
    )
    def replace_lump(
        self,
        new_name: str,
        lump: Union[BSP_LUMPS, 'Lump'],
        new_data: bytes,
    ) -> None:
        """Write out the BSP file, replacing a lump with the given bytes.

        :deprecated: simply assign to the ``.data`` attribute of the lump.
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
            raise ValueError(f'{lump_id!r} not in {list(self.game_lumps)}') from None
        return lump.data

    @overload
    def create_texinfo(self, mat: str, *, copy_from: 'TexInfo', fsys: FileSystem) -> 'TexInfo':
        """Copy from texinfo using filesystem."""
    @overload
    def create_texinfo(
        self, mat: str, *, copy_from: 'TexInfo',
        reflectivity: AnyVec, width: int, height: int,
    ) -> 'TexInfo':
        """Copy from texinfo and explicit texdata."""

    @overload
    def create_texinfo(
        self, mat: str,
        s_off: AnyVec = FrozenVec(),
        s_shift: float = -99999.0,
        t_off: AnyVec = FrozenVec(),
        t_shift: float = -99999.0,
        lightmap_s_off: AnyVec = FrozenVec(),
        lightmap_s_shift: float = -99999.0,
        lightmap_t_off: AnyVec = FrozenVec(),
        lightmap_t_shift: float = -99999.0,
        flags: SurfFlags = SurfFlags.NONE,
        *, fsys: FileSystem,
    ) -> 'TexInfo':
        """Construct from filesystem."""
    @overload
    def create_texinfo(
        self, mat: str,
        s_off: AnyVec = FrozenVec(),
        s_shift: float = -99999.0,
        t_off: AnyVec = FrozenVec(),
        t_shift: float = -99999.0,
        lightmap_s_off: AnyVec = FrozenVec(),
        lightmap_s_shift: float = -99999.0,
        lightmap_t_off: AnyVec = FrozenVec(),
        lightmap_t_shift: float = -99999.0,
        flags: SurfFlags = SurfFlags.NONE,
        *,
        reflectivity: AnyVec, width: int, height: int,
    ) -> 'TexInfo':
        """Construct with explicit texdata."""

    def create_texinfo(
        self, mat: str,
        s_off: AnyVec = FrozenVec(),
        s_shift: float = -99999.0,
        t_off: AnyVec = FrozenVec(),
        t_shift: float = -99999.0,
        lightmap_s_off: AnyVec = FrozenVec(),
        lightmap_s_shift: float = -99999.0,
        lightmap_t_off: AnyVec = FrozenVec(),
        lightmap_t_shift: float = -99999.0,
        flags: SurfFlags = SurfFlags.NONE,
        *,
        copy_from: Optional['TexInfo'] = None,
        reflectivity: Optional[AnyVec] = None,
        width: int = 0, height: int = 0,
        fsys: Optional[FileSystem] = None,
    ) -> 'TexInfo':
        """Create or find a texinfo entry with the specified values.

        The s/t offset and shift values control the texture positioning. The
        defaults are those used for overlays, but for brushes all must be
        specified. Alternatively ``copy_from`` can be provided an existing texinfo
        to copy from, if a texture is being swapped out.

        In the BSP each material also stores its texture size and reflectivity.
        If the material has not been used yet, these must either be specified
        manually or a filesystem provided for the VMT and VTFs to be read from.
        """
        if copy_from is not None:
            s_off = copy_from.s_off.copy()
            s_shift = copy_from.s_shift
            t_off = copy_from.t_off.copy()
            t_shift = copy_from.t_shift
            lightmap_s_off = copy_from.lightmap_s_off.copy()
            lightmap_s_shift = copy_from.lightmap_s_shift
            lightmap_t_off = copy_from.lightmap_t_off.copy()
            lightmap_t_shift = copy_from.lightmap_t_shift
            flags = copy_from.flags
        try:
            data = self._texdata[mat.casefold()]
            search = True
        except KeyError:
            search = False
            if fsys is None:
                if reflectivity is None or not width or not height:
                    raise TypeError(
                        'Either valid data must be provided or a filesystem '
                        'to read them from!') from None
                data = TexData(mat, Vec(reflectivity), width, height)
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

    def is_cordoned_heuristic(self) -> bool:
        """Guess to see if the map uses cordons.

        There's no definite flag, but we can guess based on the shape of the geometry. Cordoning
        causes a brush almost the map size units to be created, then the cordon regions carved out
        of it. So the overall map size will be very close to the max range.

        This isn't certain, since users could manually create brushes this large, but that's not
        too likely.
        """
        tree = self.nodes[0]
        # Round to avoid precision issues. 32776 is the size in Desolation, which has a bigger
        # map size.
        return all(
            round(abs(val)) in {16392, 32776}
            for vec in [tree.mins, tree.maxes]
            for val in vec
        )

    def is_potentially_visible(self, leaf1: VisLeaf, leaf2: VisLeaf) -> Tuple[bool, bool]:
        """Check if the first leaf can potentially see and hear the second.

        Always returns :obj:`True` if visibility data has not been computed (``self.visibility is None``).
        """
        vis: Optional[Visibility] = self.visibility
        if vis is None:
            return True, True
        byte_ind, bit_ind = divmod(leaf2.cluster_id, 8)
        pvs = vis.potentially_visible[leaf1.cluster_id][byte_ind]
        pas = vis.potentially_audible[leaf1.cluster_id][byte_ind]
        bits = 1 << bit_ind
        return pvs & bits != 0, pas & bits != 0

    def set_potentially_visible(
        self, leaf1: VisLeaf, leaf2: VisLeaf,
        visible: Optional[bool] = None,
        audible: Optional[bool] = None,
    ) -> None:
        """Override whether the first leaf can see/hear the second.

        If either :obj:`bool` is :obj:`None` that value is left unaltered.
        """
        vis: Optional[Visibility] = self.visibility
        if (visible is None and audible is None) or vis is None:
            return  # Nothing to do.
        byte_ind, bit_ind = divmod(leaf2.cluster_id, 8)
        bits = 1 << bit_ind
        if visible is not None:
            pvs = vis.potentially_visible[leaf1.cluster_id][byte_ind]
            pvs = pvs | bits if visible else pvs & ~bits
            vis.potentially_visible[leaf1.cluster_id][byte_ind] = pvs
            if visible:
                audible = True
        if audible is not None:
            pas = vis.potentially_audible[leaf1.cluster_id][byte_ind]
            pas = pas | bits if visible else pas & ~bits
            vis.potentially_audible[leaf1.cluster_id][byte_ind] = pas

    # Lump reading and writing code:
    def _lmp_read_planes(self, data: bytes) -> Iterator['Plane']:
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

    def _lmp_read_surfedges(self, edge_inds: bytes) -> Iterator[Edge]:
        verts: List[Vec] = self.vertexes
        edges = [
            Edge(verts[a], verts[b])
            for a, b in self.lump_layout['EDGE'].iter_unpack(self.lumps[BSP_LUMPS.EDGES].data)
        ]
        for [ind] in struct.iter_unpack('i', edge_inds):
            if ind < 0:  # If negative, the vertexes are reversed order.
                yield edges[-ind].opposite
            else:
                yield edges[ind]

    def _lmp_write_surfedges(self, surf_edges: List[Edge]) -> bytes:
        """Reconstruct the surfedges and edges lumps."""
        edge_buf = BytesIO()
        surf_buf = BytesIO()

        # The first edge is never actually used, since -0 = 0. Set it to be 0 0 0, adding that if
        # not present.
        try:
            first_vert = self.vertexes[self.vertexes.index(Vec())]
        except (IndexError, ValueError):
            first_vert = Vec()
            self.vertexes.append(first_vert)
        edges: List[Edge] = [Edge(first_vert, first_vert)]

        # We cannot share vertexes or edges, it breaks VRAD!
        add_edge = find_or_insert(edges)
        add_vert = find_or_insert(self.vertexes)

        for edge in surf_edges:
            # Check to see if this edge is already defined.
            # positive indexes are in forward order, negative
            # allows us to refer to a reversed definition.
            if isinstance(edge, RevEdge):
                ind = -add_edge(edge.opposite)
            else:
                ind = add_edge(edge)
            surf_buf.write(struct.pack('i', ind))

        for edge in edges:
            assert not isinstance(edge, RevEdge), edge
            edge_buf.write(self.lump_layout['EDGE'].pack(add_vert(edge.a), add_vert(edge.b)))

        self.lumps[BSP_LUMPS.EDGES].data = edge_buf.getvalue()
        return surf_buf.getvalue()

    def _lmp_read_primitives(self, data: bytes) -> Iterator['Primitive']:
        """Parse the primitives lumps."""
        if self.is_vitamin:
            # VitaminSource no longer uses primitives.
            return

        verts = list(map(Vec, struct.iter_unpack('<fff', self.lumps[BSP_LUMPS.PRIMVERTS].data)))
        indices = read_array(self.lump_layout['PRIMINDEX'], self.lumps[BSP_LUMPS.PRIMINDICES].data)

        fmt = self.lump_layout['PRIMITIVE']
        for (
            prim_type,
            first_ind, ind_count,
            first_vert, vert_count,
        ) in fmt.iter_unpack(data):
            yield Primitive(
                prim_type,
                indices[first_ind: first_ind + ind_count],
                verts[first_vert: first_vert + vert_count],
            )

    def _lmp_write_primitives(self, prims: List['Primitive']) -> Iterator[bytes]:
        if self.is_vitamin:
            # VitaminSource no longer uses primitives.
            return

        verts: List[bytes] = []
        indices: List[int] = []

        fmt = self.lump_layout['PRIMITIVE']
        for prim in prims:
            vert_loc = len(verts)
            index_loc = len(indices)
            verts += [struct.pack('<fff', pos.x, pos.y, pos.z) for pos in prim.verts]
            indices.extend(prim.indexed_verts)
            yield fmt.pack(
                prim.is_tristrip,
                index_loc, len(prim.indexed_verts),
                vert_loc, len(prim.verts),
            )
        self.lumps[BSP_LUMPS.PRIMINDICES].data = write_array(self.lump_layout['PRIMINDEX'], indices)
        self.lumps[BSP_LUMPS.PRIMVERTS].data = b''.join(verts)

    def _read_faces_common(self, data: bytes, orig_faces: Optional[List['Face']]) -> Iterator['Face']:
        """Read one of the faces arrays.

        For ORIG_FACES, _orig_faces is None and that entry is ignored.
        For the others, that is the parsed orig faces lump, which each face
        may reference.
        In VitaminSource, we only have the regular faces lump.
        """
        is_vitamin = self.is_vitamin

        # The non-original faces have the Hammer ID value, which is an array
        # in the same order. But some versions don't define it as anything...
        if orig_faces is not None or is_vitamin:
            hammer_ids = read_array(self.lump_layout['FACEID'], self.lumps[BSP_LUMPS.FACEIDS].data)
        else:
            hammer_ids = []

        hammer_id: Optional[int]

        for i, face_data in enumerate(
            self.lump_layout['FACE'].iter_unpack(data),
        ):
            if is_vitamin:
                (
                    plane_num,
                    texinfo_ind,
                    dispinfo,
                    first_edge, num_edges,
                    lightmap_mins_x, lightmap_mins_y,
                    lightmap_size_x, lightmap_size_y,
                    vitamin_flags,
                ) = face_data
                texinfo = self.texinfo[texinfo_ind]

                # All these values are unused.
                side = False
                surf_fog_vol_id = 0
                light_offset = 0
                lightstyles = b'\0\0\0\0'
                on_node = False
                orig_face = None
                area = 0
                primitives = []
                no_dynamic_shadows = False
                smoothing_group = 0
                hammer_id = None
            else:
                (
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
                ) = face_data
                primitives = self.primitives[prim_first:prim_first + (prim_num & 0x7fff)]
                no_dynamic_shadows = not (prim_num & 0x8000)
                vitamin_flags = 0

                # If orig faces is provided, that is the original face
                # we were created from. Additionally, it seems the original
                # face data has invalid texinfo, so copy ours on top of it.
                if orig_faces is not None:
                    orig_face = orig_faces[orig_face_ind]
                    orig_face.texinfo = texinfo = self.texinfo[texinfo_ind]
                    try:
                        orig_face.hammer_id = hammer_id = hammer_ids[i]
                    except IndexError:
                        hammer_id = None
                else:
                    orig_face = texinfo = None
                    hammer_id = None
            yield Face(
                self.planes[plane_num],
                side, on_node,
                self.surfedges[first_edge:first_edge+num_edges],
                # (first_edge, num_edges),
                texinfo,
                dispinfo,
                surf_fog_vol_id,
                lightstyles,
                light_offset,
                area,
                (lightmap_mins_x, lightmap_mins_y),
                (lightmap_size_x, lightmap_size_y),
                orig_face,
                primitives, no_dynamic_shadows,
                smoothing_group,
                hammer_id,
                vitamin_flags,
            )

    def _write_faces_common(
        self, faces: List['Face'],
        get_orig_face: Optional[Callable[['Face'], int]],
    ) -> bytes:
        """Reconstruct one of the faces arrays.

        If this isn't the orig faces array, get_orig_face should be
        _find_or_insert(self.orig_faces).
        """
        face_buf = BytesIO()
        add_texinfo = find_or_insert(self.texinfo)
        add_plane = find_or_insert(self.planes)
        add_edges = find_or_extend(self.surfedges)
        add_prims = find_or_extend(self.primitives)
        hammer_ids = []

        if self.is_vitamin:
            for face in faces:
                if face.texinfo is not None:
                    texinfo = add_texinfo(face.texinfo)
                else:
                    texinfo = -1
                # noinspection PyProtectedMember
                face_buf.write(struct.pack(
                    '<5i4iB3x',
                    add_plane(face.plane),
                    texinfo,
                    face._dispinfo_ind,
                    add_edges(face.edges), len(face.edges),
                    *face.lightmap_mins, *face.lightmap_size,
                    face.vitamin_flags,
                ))
        else:
            for face in faces:
                if face.orig_face is not None and get_orig_face is not None:
                    orig_ind = get_orig_face(face.orig_face)
                    hammer_ids.append(face.hammer_id or 0)  # Dummy value if not set.
                else:
                    orig_ind = -1
                if face.texinfo is not None:
                    texinfo = add_texinfo(face.texinfo)
                else:
                    texinfo = -1
                prim_count = len(face.primitives)
                if prim_count > 0x7fff:
                    raise ValueError(f'Too many primitives: {prim_count} in {orig_ind}')
                if not face.dynamic_shadows:
                    prim_count |= 0x8000

                # noinspection PyProtectedMember
                face_buf.write(self.lump_layout['FACE'].pack(
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
                    prim_count,
                    add_prims(face.primitives),
                    face.smoothing_groups,
                ))
            if hammer_ids:
                self.lumps[BSP_LUMPS.FACEIDS].data = write_array(self.lump_layout['FACEID'], hammer_ids)
        return face_buf.getvalue()

    def _lmp_read_orig_faces(self, data: bytes) -> Iterator['Face']:
        """Parse the unsplit faces lump."""
        if self.is_vitamin:
            return iter(())  # Unused
        else:
            return self._read_faces_common(data, None)

    def _lmp_write_orig_faces(self, faces: List['Face']) -> bytes:
        """Write the unsplit faces lump."""
        if self.is_vitamin:
            return b''  # Unused.
        else:
            return self._write_faces_common(faces, None)

    def _lmp_read_faces(self, data: bytes) -> Iterator['Face']:
        """Parse the main split faces lump."""
        if self.is_vitamin:
            return self._read_faces_common(data, None)  # No orig faces.
        else:
            return self._read_faces_common(data, self.orig_faces)

    def _lmp_write_faces(self, faces: List['Face']) -> bytes:
        """Write the main split faces lump."""
        if self.is_vitamin:
            return self._write_faces_common(faces, None)  # No orig faces.
        else:
            return self._write_faces_common(faces, find_or_insert(self.orig_faces))

    def _lmp_read_hdr_faces(self, data: bytes) -> Iterator['Face']:
        """Parse the HDR-specific split faces lump."""
        if self.is_vitamin:
            return iter(())  # Unused
        else:
            return self._read_faces_common(data, self.orig_faces)

    def _lmp_write_hdr_faces(self, faces: List['Face']) -> bytes:
        """Write the HDR-specific split faces lump."""
        if self.is_vitamin:
            return b''  # Unused.
        else:
            return self._write_faces_common(faces, find_or_insert(self.orig_faces))

    def _lmp_read_brushes(self, data: bytes) -> Iterator['Brush']:
        """Parse brush definitions, along with the sides."""
        # The bevel param should be a bool, but randomly has other bits set?
        if self.is_vitamin:
            sides = [
                BrushSide(self.planes[plane_num], self.texinfo[texinfo], dispinfo, bevel, extra)
                for (plane_num, texinfo, dispinfo, bevel, extra)
                in self.lump_layout['BRUSHSIDE'].iter_unpack(self.lumps[BSP_LUMPS.BRUSHSIDES].data)
            ]
        else:
            sides = [
                BrushSide(self.planes[plane_num], self.texinfo[texinfo], dispinfo, bool(bevel & 1), bevel & ~1)
                for (plane_num, texinfo, dispinfo, bevel)
                in self.lump_layout['BRUSHSIDE'].iter_unpack(self.lumps[BSP_LUMPS.BRUSHSIDES].data)
            ]
        for first_side, side_count, contents in struct.iter_unpack('<iii', data):
            yield Brush(BrushContents(contents), sides[first_side:first_side+side_count])

    def _lmp_write_brushes(self, brushes: List['Brush']) -> bytes:
        sides: List[BrushSide] = []
        add_plane = find_or_insert(self.planes)
        add_texinfo = find_or_insert(self.texinfo)
        add_sides = find_or_extend(sides)

        brush_buf = BytesIO()
        sides_buf = BytesIO()
        for brush in brushes:
            brush_buf.write(struct.pack(
                '<iii',
                add_sides(brush.sides), len(brush.sides),
                brush.contents.value,
            ))

        side_struct = self.lump_layout['BRUSHSIDE']
        if self.is_vitamin:
            for side in sides:
                sides_buf.write(side_struct.pack(
                    add_plane(side.plane),
                    add_texinfo(side.texinfo),
                    side._dispinfo,
                    side.is_bevel_plane,
                    side._unknown_bevel_bits,
                ))
        else:
            for side in sides:
                sides_buf.write(side_struct.pack(
                    add_plane(side.plane),
                    add_texinfo(side.texinfo),
                    side._dispinfo,
                    side.is_bevel_plane | side._unknown_bevel_bits,
                ))
        self.lumps[BSP_LUMPS.BRUSHSIDES].data = sides_buf.getvalue()
        return brush_buf.getvalue()

    def _lmp_read_water_leaf_info(self, data: bytes) -> Iterator[LeafWaterInfo]:
        """Parse data associated with visleafs containing water."""
        texinfo = self.texinfo
        for surf_z, min_z, texinfo_ind in self.lump_layout['LEAFWATERDATA'].iter_unpack(data):
            yield LeafWaterInfo(surf_z, min_z, texinfo[texinfo_ind])

    def _lmp_write_water_leaf_info(self, data: List[LeafWaterInfo]) -> Iterator[bytes]:
        """Write data associated with visleafs containing water."""
        add_texinfo = find_or_insert(self.texinfo)
        for info in self.water_leaf_info:
            yield self.lump_layout['LEAFWATERDATA'].pack(info.surface_z, info.min_z, add_texinfo(info.surface_texinfo))

    def _lmp_read_visleafs(self, data: bytes) -> Iterator[VisLeaf]:
        """Parse the leafs of the visleaf/bsp tree."""
        # There's an indirection through these index arrays.
        leaf_brushes = list(map(
            self.brushes.__getitem__,
            read_array(self.lump_layout['LEAFBRUSH'], self.lumps[BSP_LUMPS.LEAFBRUSHES].data),
        ))
        leaf_faces = list(map(
            self.faces.__getitem__,
            read_array(self.lump_layout['LEAFFACE'], self.lumps[BSP_LUMPS.LEAFFACES].data),
        ))
        # Another lump which is just an array of ints - no point being separate.
        dist_to_water = read_array('<H', self.lumps[BSP_LUMPS.LEAFMINDISTTOWATER].data)

        is_vitamin = self.is_vitamin
        leaf_fmt = self.lump_layout['LEAF']
        # Some extra ambient light data.
        has_ambient = not is_vitamin and self.version <= 19

        for leaf_data, water_dist in zip(leaf_fmt.iter_unpack(data), dist_to_water):
            if is_vitamin:
                # VitaminSource moves the flags into its own block.
                (
                    contents,
                    cluster_ind, area,
                    min_x, min_y, min_z,
                    max_x, max_y, max_z,
                    first_face, num_faces,
                    first_brush, num_brushes,
                    water_ind, flags,
                ) = leaf_data
                ambient = b'\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0'
            else:
                if has_ambient:
                    (
                        contents,
                        cluster_ind, area_and_flags,
                        min_x, min_y, min_z,
                        max_x, max_y, max_z,
                        first_face, num_faces,
                        first_brush, num_brushes,
                        water_ind, ambient,
                    ) = leaf_data
                else:
                    (
                        contents,
                        cluster_ind, area_and_flags,
                        min_x, min_y, min_z,
                        max_x, max_y, max_z,
                        first_face, num_faces,
                        first_brush, num_brushes,
                        water_ind,
                    ) = leaf_data
                    # bytes(24), but can constant fold.
                    ambient = b'\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0'
                area = area_and_flags >> self.lump_layout['LEAF_AREA_OFFSET']
                flags = area_and_flags & ((1 << self.lump_layout['LEAF_AREA_OFFSET']) - 1)
            yield VisLeaf(
                BrushContents(contents), cluster_ind, area, VisLeafFlags(flags),
                Vec(min_x, min_y, min_z),
                Vec(max_x, max_y, max_z),
                leaf_faces[first_face:first_face+num_faces],
                leaf_brushes[first_brush:first_brush+num_brushes],
                water_ind, ambient, water_dist,
            )

    def _lmp_read_nodes(self, data: bytes) -> List['VisTree']:
        """Parse the main visleaf/bsp trees (dnode_t)."""
        # First parse all the nodes, then link them up.
        nodes: List[Tuple[VisTree, int, int]] = []

        for (
            plane_ind, neg_ind, pos_ind,
            min_x, min_y, min_z,
            max_x, max_y, max_z,
            first_face, face_count, area_ind,
        ) in self.lump_layout['NODE'].iter_unpack(data):
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
        add_node = find_or_insert(nodes)
        add_plane = find_or_insert(self.planes)
        add_leaf = find_or_insert(self.visleafs)
        add_faces = find_or_extend(self.faces)

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

            buf.write(self.lump_layout['NODE'].pack(
                add_plane(node.plane), neg_ind, pos_ind,
                int(node.mins.x), int(node.mins.y), int(node.mins.z),
                int(node.maxes.x), int(node.maxes.y), int(node.maxes.z),
                add_faces(node.faces), len(node.faces), node.area_ind,
            ))

        return buf.getvalue()

    def _lmp_write_visleafs(self, visleafs: List['VisLeaf']) -> bytes:
        """Reconstruct the leafs of the visleaf/bsp tree."""
        leaf_faces: List[int] = []
        leaf_brushes: List[int] = []
        min_water_dists: List[int] = []

        add_face = find_or_insert(self.faces)
        add_brush = find_or_insert(self.brushes)

        buf = BytesIO()
        is_vitamin = self.is_vitamin

        # Some extra ambient light data.
        has_ambient = self.version <= 19

        for leaf in visleafs:
            # Do not deduplicate these, engine assumes they aren't when allocating memory.
            face_ind = len(leaf_faces)
            brush_ind = len(leaf_brushes)
            leaf_faces.extend(map(add_face, leaf.faces))
            leaf_brushes.extend(map(add_brush, leaf.brushes))
            min_water_dists.append(leaf.min_water_dist)

            if is_vitamin:
                buf.write(self.lump_layout['LEAF'].pack(
                    leaf.contents.value, leaf.cluster_id, leaf.area,
                    int(leaf.mins.x), int(leaf.mins.y), int(leaf.mins.z),
                    int(leaf.maxes.x), int(leaf.maxes.y), int(leaf.maxes.z),
                    face_ind, len(leaf.faces),
                    brush_ind, len(leaf.brushes),
                    leaf.water_id, leaf.flags.value,
                ))
            else:
                leafdata: Tuple[Union[int, bytes], ...] = (
                    leaf.contents.value, leaf.cluster_id,
                    (leaf.area << self.lump_layout['LEAF_AREA_OFFSET'] | leaf.flags.value),
                    int(leaf.mins.x), int(leaf.mins.y), int(leaf.mins.z),
                    int(leaf.maxes.x), int(leaf.maxes.y), int(leaf.maxes.z),
                    face_ind, len(leaf.faces),
                    brush_ind, len(leaf.brushes),
                    leaf.water_id)

                # Older leaf lumps include some ambient light data at the end.
                if has_ambient:
                    leafdata = (*leafdata, leaf._ambient)

                buf.write(self.lump_layout['LEAF'].pack(*leafdata))

        self.lumps[BSP_LUMPS.LEAFFACES].data = write_array(self.lump_layout['LEAFFACE'], leaf_faces)
        self.lumps[BSP_LUMPS.LEAFBRUSHES].data = write_array(self.lump_layout['LEAFBRUSH'], leaf_brushes)
        self.lumps[BSP_LUMPS.LEAFMINDISTTOWATER].data = write_array('<H', min_water_dists)
        return buf.getvalue()

    def _lmp_read_visibility(self, data: bytes) -> Optional[Visibility]:
        """Read VVIS data."""
        if not data:
            return None  # VVIS hasn't run.
        [cluster_count] = struct.unpack_from('i', data, 0)
        offset = struct.calcsize('i')
        two_ints = struct.Struct('ii')
        vis = Visibility([], [])
        for _ in range(cluster_count):
            [pvs_offset, pas_offset] = two_ints.unpack_from(data, offset)
            offset += two_ints.size
            vis.potentially_visible.append(runlength_decode(data, pvs_offset, cluster_count))
            vis.potentially_audible.append(runlength_decode(data, pas_offset, cluster_count))
        return vis

    def _lmp_write_visibility(self, vis: Optional[Visibility]) -> bytes:
        """Reconstruct the VVIS data."""
        if vis is None:
            return b''  # VVIS hasn't run.
        cluster_count = len(vis.potentially_visible)
        if cluster_count != len(vis.potentially_audible):
            raise ValueError('Inconsistent PVS/PAS lengths!')

        data = BytesIO()
        writes = DeferredWrites(data)
        data.write(struct.pack('i', cluster_count))
        for ind in range(cluster_count):
            writes.defer(ind, 'ii', True)

        for ind, (pvs, pas) in enumerate(zip(vis.potentially_visible, vis.potentially_audible)):
            pvs_off = data.tell()
            data.write(runlength_encode(pvs))
            pas_off = data.tell()
            data.write(runlength_encode(pas))
            writes.set_data(ind, pvs_off, pas_off)
        writes.write()
        return data.getvalue()

    @deprecated('Access bsp.textures')
    def read_texture_names(self) -> Iterator[str]:
        """Iterate through all brush textures in the map."""
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
            # Look for the NULL at the end of the string. They're limited to 128 chars long.
            try:
                str_off = tex_data.index(b'\0', off, off + 128)
            except ValueError:
                # Reached the 128 char limit without finding a null.
                raise ValueError(f'Bad string at {off} in BSP! ({tex_data[off:off + 128]!r})') from None
            else:
                yield tex_data[off:str_off].decode('ascii', 'surrogateescape')

    def _lmp_write_textures(self, textures: List[str]) -> bytes:
        table = BytesIO()
        data = bytearray()
        for tex in textures:
            if len(tex) >= 128:
                raise OverflowError(f'Texture "{tex}" exceeds 128 character limit')
            string = tex.encode('ascii', 'surrogateescape') + b'\0'
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
        # The view width/height is unused stuff, identical to regular
        # width/height. VitaminSource elides this useless info.
        if self.is_vitamin:
            for ref_x, ref_y, ref_z, ind, w, h in struct.iter_unpack('<3f3i', self.lumps[BSP_LUMPS.TEXDATA].data):
                mat = self.textures[ind]
                texdata = TexData(mat, Vec(ref_x, ref_y, ref_z), w, h)
                texdata_list.append(texdata)
                self._texdata[mat.casefold()] = texdata
        else:
            for (
                ref_x, ref_y, ref_z, ind,
                w, h, vw, vh,
            ) in struct.iter_unpack('<3f5i', self.lumps[BSP_LUMPS.TEXDATA].data):
                mat = self.textures[ind]
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
        find_or_add_texture = find_or_insert(self.textures, str.casefold)
        texdata_ind: Dict[TexData, int] = {}

        texdata_list: List[bytes] = []
        texinfo_result: List[bytes] = []
        next_ind = 0

        for info in texinfos:
            # noinspection PyProtectedMember
            tdat = info._info
            # Try and find an existing reference to this texdata.
            # If not, put it at the end of the list. Since we're rebuilding it, we can just
            # increment an index each time to track that. We can't use the list len because
            # we append twice for non-Desolation games which have an extra bit of data.
            try:
                ind = texdata_ind[tdat]
            except KeyError:
                ind = texdata_ind[tdat] = next_ind
                next_ind += 1
                texdata_list.append(struct.pack(
                    '<3f3i',
                    tdat.reflectivity.x, tdat.reflectivity.y, tdat.reflectivity.z,
                    find_or_add_texture(tdat.mat),
                    tdat.width, tdat.height,
                ))
                # A second set of  'view' dimensions, which is always the same.
                if not self.is_vitamin:
                    texdata_list.append(struct.pack('<2i', tdat.width, tdat.height))
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

    def _lmp_read_bmodels(self, data: bytes) -> 'WeakKeyDictionary[Entity, BModel]':
        """Parse the brush model definitions."""
        bmodel_list = []
        for (
            min_x, min_y, min_z, max_x, max_y, max_z,
            pos_x, pos_y, pos_z,
            headnode,
            first_face, num_face,
        ) in struct.iter_unpack('<9fiii', data):
            bmodel_list.append(BModel(
                Vec(min_x, min_y, min_z), Vec(max_x, max_y, max_z),
                Vec(pos_x, pos_y, pos_z),
                self.nodes[headnode],
                self.faces[first_face:first_face+num_face],
            ))

        # Parse the physics lump.
        phys_buf = BytesIO(self.lumps[BSP_LUMPS.PHYSCOLLIDE].data)
        while True:
            (mdl_ind, data_size, kv_size, solid_count) = struct_read('<iiii', phys_buf)
            if mdl_ind == -1:
                break  # end of segment.
            mdl = bmodel_list[mdl_ind]
            # Possible in the format, but is broken - you can't merge the
            # definitions really, and our object layout doesn't allow it.
            if mdl._phys_solids or mdl.phys_keyvalues is not None:
                raise ValueError(f'Two physics definitions for bmodel #{mdl_ind}!')
            mdl._phys_solids = [
                phys_buf.read(struct_read('<i', phys_buf)[0])
                for _ in range(solid_count)
            ]
            kvs = phys_buf.read(kv_size).rstrip(b'\x00').decode('ascii')
            mdl.phys_keyvalues = Keyvalues.parse(
                kvs,
                f'bmodel[{mdl_ind}].keyvalues',
            )

        # Loop over entities, map to their brush model.
        brush_ents: WeakKeyDictionary[Entity, BModel] = WeakKeyDictionary()
        vmf: VMF = self.ents
        brush_ents[vmf.spawn] = bmodel_list[0]
        for ent in vmf.entities:
            if ent['model'].startswith('*'):
                mdl_ind = int(ent.pop('model')[1:])
                brush_ents[ent] = bmodel_list[mdl_ind]

        return brush_ents

    def _lmp_write_bmodels(self, bmodels: 'WeakKeyDictionary[Entity, BModel]') -> Iterator[bytes]:
        """Write the brush model definitions."""
        add_node = find_or_insert(self.nodes)
        add_faces = find_or_extend(self.faces)

        phys_buf = BytesIO()

        # The models in order, worldspawn must be zero.
        worldspawn = self.ents.spawn
        try:
            model_list = [bmodels[worldspawn]]
        except KeyError:
            raise ValueError('Worldspawn has no brush model!') from None
        add_model = find_or_insert(model_list)

        for ent, model in bmodels.items():
            # Apply the brush model to the entity. Worldspawn doesn't actually
            # need the key though.
            if ent is not worldspawn:
                ent['model'] = f'*{add_model(model)}'

        for i, model in enumerate(model_list):
            yield struct.pack(
                '<9fiii',
                model.mins.x, model.mins.y, model.mins.z,
                model.maxes.x, model.maxes.y, model.maxes.z,
                model.origin.x, model.origin.y, model.origin.z,
                add_node(model.node),
                add_faces(model.faces), len(model.faces),
            )
            if model.phys_keyvalues is not None:
                kvs = '\n'.join(model.phys_keyvalues.serialise()).encode('ascii') + b'\x00'
            else:
                kvs = b'\x00'
                if not model._phys_solids:
                    continue  # No physics info.
            phys_size = sum(map(len, model._phys_solids)) + 4 * len(model._phys_solids)
            phys_buf.write(struct.pack(
                '<iiii',
                i, phys_size,
                len(kvs),
                len(model._phys_solids),
            ))
            for solid in model._phys_solids:
                phys_buf.write(struct.pack('<i', len(solid)))
                phys_buf.write(solid)
            phys_buf.write(kvs)

        # Sentinel physics definition.
        phys_buf.write(struct.pack('<iiii', -1, 0, 0, 0))
        self.lumps[BSP_LUMPS.PHYSCOLLIDE].data = phys_buf.getvalue()

    def _lmp_read_pakfile(self, data: bytes) -> ZipFile:
        """Read the raw binary as writable zip archive."""
        zipfile = ZipFile(BytesIO(data), mode='a')
        if self.filename is not None:
            zipfile.filename = os.fspath(self.filename)
        return zipfile

    def _lmp_write_pakfile(self, file: ZipFile) -> bytes:
        """Extract the final zip data from the zipfile."""
        # Explicitly close the zip file, so the footer is done.
        buf = file.fp
        file.close()
        if isinstance(buf, BytesIO):
            return buf.getvalue()
        elif buf is None:
            raise ValueError('Zipfile has no buffer?')
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

    def _lmp_write_cubemaps(self, cubemaps: List['Cubemap']) -> Iterator[bytes]:
        """Write out the cubemaps lump."""
        for cube in cubemaps:
            yield struct.pack(
                '<iiii',
                int(round(cube.origin.x)),
                int(round(cube.origin.y)),
                int(round(cube.origin.z)),
                cube.size,
            )

    def _lmp_read_overlays(self, data: bytes) -> Iterator[Overlay]:
        """Read the overlays lump."""
        # Use zip longest, so we handle cases where these newer auxiliary lumps
        # are empty.
        for block, fades, sys_levels in itertools.zip_longest(
            struct.iter_unpack(
                '<i'  # ID
                # texinfo, face-and-render-order
                f'{"iHxx" if TEXINFO_IND_TYPE == "i" else "hH"}'
                f'{OVERLAY_FACE_COUNT}i'  # face array.
                '4f'  # UV min/max
                '18f',  # 4 handle points, origin, normal
                data,
            ),
            struct.iter_unpack('<ff', self.lumps[BSP_LUMPS.OVERLAY_FADES].data),
            struct.iter_unpack('<4B', self.lumps[BSP_LUMPS.OVERLAY_SYSTEM_LEVELS].data),
        ):
            if block is None:
                # Too many of either aux lump, ignore.
                break
            over_id, texinfo, face_ro = block[:3]
            face_count = face_ro & ((1 << 14) - 1)
            render_order = face_ro >> 14
            if face_count > OVERLAY_FACE_COUNT:
                raise ValueError(f'{face_ro} exceeds OVERLAY_BSP_FACE_COUNT ({OVERLAY_FACE_COUNT})!')
            faces = list(block[3: 3 + face_count])
            u_min, u_max, v_min, v_max = block[-22:-18]
            uv1 = Vec(block[-18:-15])
            uv2 = Vec(block[-15:-12])
            uv3 = Vec(block[-12:-9])
            uv4 = Vec(block[-9:-6])
            origin = Vec(block[-6:-3])
            normal = Vec(block[-3:])
            assert len(block) == 25 + OVERLAY_FACE_COUNT

            if fades is not None:
                fade_min, fade_max = fades
            else:
                fade_min = -1.0
                fade_max = 0.0
            if sys_levels is not None:
                min_cpu, max_cpu, min_gpu, max_gpu = sys_levels
            else:
                min_cpu = min_gpu = 0
                max_cpu = max_gpu = 0

            yield Overlay(
                over_id, origin, normal,
                self.texinfo[texinfo], face_count,
                faces, render_order,
                u_min, u_max,
                v_min, v_max,
                uv1, uv2, uv3, uv4,
                fade_min, fade_max,
                min_cpu, max_cpu,
                min_gpu, max_gpu
            )

    def _lmp_write_overlays(self, overlays: List[Overlay]) -> Iterator[bytes]:
        """Write out all overlays."""
        add_texinfo = find_or_insert(self.texinfo)
        fade_buf = BytesIO()
        levels_buf = BytesIO()
        for over in overlays:
            face_cnt = len(over.faces)
            if face_cnt > OVERLAY_FACE_COUNT:
                raise ValueError(f'{over.faces} exceeds OVERLAY_BSP_FACE_COUNT ({OVERLAY_FACE_COUNT})!')
            fade_buf.write(struct.pack('<ff', over.fade_min_sq, over.fade_max_sq))
            levels_buf.write(struct.pack('<4B', over.min_cpu, over.max_cpu, over.min_gpu, over.max_gpu))
            yield struct.pack(
                # texinfo, face-and-render-order
                '<iiHxx' if TEXINFO_IND_TYPE == "i" else '<ihH',
                over.id,
                add_texinfo(over.texture),
                (over.render_order << 14 | face_cnt),
            )
            # Build the array, then zero fill the remaining space.
            yield struct.pack(f'<{face_cnt}i {4*(OVERLAY_FACE_COUNT-face_cnt)}x', *over.faces)
            yield struct.pack('<4f', over.u_min, over.u_max, over.v_min, over.v_max)
            yield struct.pack(
                '<18f',
                *over.uv1, *over.uv2, *over.uv3, *over.uv4,
                *over.origin, *over.normal,
            )
        self.lumps[BSP_LUMPS.OVERLAY_FADES].data = fade_buf.getvalue()
        self.lumps[BSP_LUMPS.OVERLAY_SYSTEM_LEVELS].data = levels_buf.getvalue()

    @contextlib.contextmanager
    @deprecated('Use BSP.pakfile to access the cached archive.')
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

    @deprecated('Use BSP.ents directly.')
    def read_ent_data(self) -> VMF:
        """Deprecated function to parse the entdata lump.

        Use BSP.ents directly.
        """
        return self._lmp_read_ents(self.get_lump(BSP_LUMPS.ENTITIES))

    def _lmp_read_ents(self, ent_data: bytes) -> VMF:
        """Parse in entity data.

        This returns a VMF object, with entities mirroring that in the BSP.
        No brushes are read.
        """
        vmf = VMF()
        cur_ent: Optional[Entity] = None  # None when between brackets.
        seen_spawn = False  # The first entity is worldspawn.

        # We have to use the tokenizer to handle newlines inside quotes.
        # Use surrogate-escape, to preserve bytes > 127 - VMFs don't have a clear encoding.
        tok = Tokenizer(ent_data.decode('ascii', 'surrogateescape'), allow_escapes=True)
        for tok_typ, tok_value in tok:
            if tok_typ is Token.BRACE_OPEN:
                if cur_ent is not None:
                    raise tok.error('2 levels of nesting after {} ents', len(vmf.entities))
                # The first entity updates worldspawn.
                if not seen_spawn:
                    cur_ent = vmf.spawn
                    seen_spawn = True
                else:
                    cur_ent = Entity(vmf)
                continue
            elif tok_typ is Token.BRACE_CLOSE:
                if cur_ent is None:
                    raise tok.error('Too many closing brackets after {} ents!', len(vmf.entities))
                if cur_ent is vmf.spawn:
                    if cur_ent['classname'] != 'worldspawn':
                        raise tok.error('First entity must be worldspawn, not "{}"!', cur_ent["classname"])
                else:
                    # The spawn ent is stored in the attribute, not in the ent list.
                    vmf.add_ent(cur_ent)
                cur_ent = None
                continue
            elif tok_typ is Token.NEWLINE:
                continue
            elif tok_typ is Token.EOF:
                if cur_ent is not None:
                    raise ValueError("Last entity didn't end!")
                return vmf
            elif tok_typ is not Token.STRING:
                raise tok.error(tok_typ, tok_value)

            # Null byte at end of lump.
            if tok_value == '\x00':
                tok.expect(Token.EOF)  # If in the middle, raise error.
                if cur_ent is not None:
                    raise ValueError("Last entity didn't end!")
                break

            if cur_ent is None:
                raise tok.error("Keyvalue outside brackets: {}({!r})", tok_typ, tok_value)

            # Line is of the form <"key" "val">, but handle escaped quotes
            # in the value. Valve's parser doesn't allow that, but we might
            # as well be better...
            key = tok_value
            value = tok.expect(Token.STRING)

            # Now, we need to figure out if this is a keyvalue,
            # or connection.
            # If we're L4D+, this is easy - they use 0x1B as separator.
            # Before, it's a comma which is common in keyvalues.
            # Assume it's an output if it has exactly 4 commas, and the last two
            # successfully parse as numbers.
            if '\x1B' in value:
                # All outputs use the comma_sep, so we can ID them.
                try:
                    cur_ent.add_out(Output.parse(Keyvalues(key, value)))
                except ValueError as exc:
                    raise ValueError(f'Failed to parse output in {key!r} {value!r}') from exc
                if self.out_comma_sep is None:
                    self.out_comma_sep = False
            elif value.count(',') == 4:
                try:
                    cur_ent.add_out(Output.parse(Keyvalues(key, value)))
                except ValueError:
                    cur_ent[key] = value
                if self.out_comma_sep is None:
                    self.out_comma_sep = True
            else:
                # Normal keyvalue.
                cur_ent[key] = value

        # This keyvalue needs to be stored in the VMF object too.
        # The one in the entity is ignored.
        vmf.map_ver = conv_int(vmf.spawn['mapversion'], vmf.map_ver)

        return vmf

    def _lmp_write_ents(self, vmf: VMF) -> bytes:
        return self.write_ent_data(vmf, self.out_comma_sep, _show_dep=False)

    @staticmethod
    @deprecated('Modify bsp.ents instead', category=None)
    def write_ent_data(vmf: VMF, use_comma_sep: Optional[bool] = None, *, _show_dep: bool = True) -> bytes:
        """Generate the entity data lump.

        :deprecated: Read and write :py:attr:`BSP.ents` instead.
        :param vmf: This accepts a VMF file like that returned from read_ent_data().
            Brushes are ignored, so the VMF must use ``*xx`` model references.
        :param use_comma_sep: This is used to force using either commas, or ``0x1D`` in I/O.
        """
        if _show_dep:
            warnings.warn('Modify bsp.ents instead', DeprecationWarning, stacklevel=2)
        out = BytesIO()
        for ent in itertools.chain([vmf.spawn], vmf.entities):
            out.write(b'{\n')
            for key, value in ent.items():
                out.write(f'"{key}" "{escape_text(value)}"\n'.encode('ascii', 'surrogateescape'))
            for output in ent.outputs:
                if use_comma_sep is not None:
                    output.comma_sep = use_comma_sep
                out.write(output.as_keyvalue().encode('ascii', 'surrogateescape'))
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
            yield padded_name.rstrip(b'\x00').decode('ascii', 'surrogateescape')

    @deprecated('Access bsp.props instead')
    def static_props(self) -> Iterator['StaticProp']:
        """Read in the Static Props lump.

        Deprecated, use ``bsp.props``.
        """
        return iter(self.props)

    @deprecated('Assign to bsp.props instead')
    def write_static_props(self, props: List['StaticProp']) -> None:
        """Remake the static prop lump.

        Deprecated, ``bsp.props`` is stored and resaved.
        """
        self.props = props

    def _lmp_read_props(self, vers_num: int, data: bytes) -> Iterator['StaticProp']:
        # The version of the static prop format - different features.
        if vers_num > 13:
            raise ValueError(f'Unknown version ({vers_num})!')
        if vers_num < 4:
            # Predates HL2, no game produces these.
            raise ValueError(f'Static prop version {vers_num} is too old!')

        static_lump = BytesIO(data)

        # Array of model filenames.
        model_dict = list(self._read_static_props_models(static_lump))

        [visleaf_count] = struct_read('<i', static_lump)
        visleaf_list = list(map(
            self.visleafs.__getitem__,
            struct_read(self.lump_layout['STATICPROPLEAF'].format[1] * visleaf_count, static_lump),
        ))

        [prop_count] = struct_read('<i', static_lump)

        if prop_count == 0:
            # No props, following code will divide by zero, also no point anyway.
            # Use the 'standard' version for the given version number.
            if self.static_prop_version is StaticPropVersion.UNKNOWN:
                for vers in StaticPropVersion:
                    if vers.version == vers_num:
                        self.static_prop_version = vers
            return
        struct_size = (len(data) - static_lump.tell()) / prop_count
        # print(f'Static prop: {prop_count} * {struct_size} bytes')

        # The prop data itself changes drastically, depending on version.
        # Some numbers are reused, so add the size of the struct to guess the
        # version.
        if self.static_prop_version is StaticPropVersion.UNKNOWN:
            try:
                version = _STATIC_PROP_VERSIONS[vers_num, struct_size]
            except KeyError:
                raise ValueError(
                    "Don't know a static prop "
                    f"version={vers_num} with a size of {struct_size} bytes!"
                ) from None
            if version is StaticPropVersion.V11 and self.version is VERSIONS.BLACK_MESA:
                # Black Mesa uses a different version.
                version = StaticPropVersion.V_LIGHTMAP_MESA
            self.static_prop_version = version
        else:
            # It was manually specified, believe whatever was passed.
            version = self.static_prop_version

        # The 2013 versions have versions 7, 10 and 11 (mesa). They're more similar to V7 though.
        if version.is_lightmap:
            vers_num = 7

        # print('Decoded version: ', version, vers_num)

        for i in range(prop_count):
            start = static_lump.tell()
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

            if vers_num >= 5:
                [fade_scale] = struct_read('<f', static_lump)
            else:
                fade_scale = 1  # default

            if vers_num in (6, 7):
                min_dx_level, max_dx_level = struct_read('<HH', static_lump)
            else:
                # Replaced by GPU & CPU in later versions.
                min_dx_level = max_dx_level = 0  # None

            if vers_num >= 8:
                (
                    min_cpu_level,
                    max_cpu_level,
                    min_gpu_level,
                    max_gpu_level,
                ) = struct_read('<BBBB', static_lump)
            else:
                # None
                min_cpu_level = max_cpu_level = 0
                min_gpu_level = max_gpu_level = 0

            if version.is_lightmap:
                # Regular flags byte above is totally ignored!
                [flags, lightmap_x, lightmap_y] = struct_read('<IHH', static_lump)
            else:
                # FGD default.
                lightmap_x = lightmap_y = 32

            # The 2013 SDK doesn't have rendercolour, but Mesa does.
            if vers_num >= 7 and not version.is_sdk_2013:
                r, g, b, renderfx = struct_read('<BBBB', static_lump)
                # Alpha isn't used.
                tint = Vec(r, g, b)
            else:
                # No tint.
                tint = Vec(255, 255, 255)
                renderfx = 255

            disable_on_xbox = False
            if vers_num >= 9 and not version.is_lightmap:
                # The single boolean byte also produces 3 pad bytes.
                [disable_on_xbox] = struct_read('<?xxx', static_lump)

            if vers_num >= 10 or version is StaticPropVersion.V_LIGHTMAP_MESA:
                # Extra flags, post-CSGO, also in Black Mesa.
                flags |= struct_read('<I', static_lump)[0] << 8

            flags = StaticPropFlags(flags)

            scaling = Vec(1.0, 1.0, 1.0)
            if version is StaticPropVersion.V_CHAOS_V13:
                # Three floats for non-uniform scaling
                [scaling.x, scaling.y, scaling.z] = struct_read("<fff", static_lump)
            elif vers_num >= 11:
                # One float for uniform scaling
                [scaling.x] = struct_read("<f", static_lump)
                scaling.z = scaling.y = scaling.x

            real_size = static_lump.tell() - start
            if struct_size != real_size:
                raise ValueError(f'Expected {struct_size} for {version}, got {real_size}!')

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
                lightmap_x, lightmap_y,
            )

    def _lmp_write_props(self, props: List['StaticProp']) -> bytes:
        # First generate the visleaf and model-names block.
        # Unfortunately it seems reusing visleaf parts isn't possible.
        leaf_array: List[int] = []
        model_list: List[str] = []
        add_model = find_or_insert(model_list, identity)
        add_leaf = find_or_insert(self.visleafs)

        indexes: List[Tuple[int, int]] = []
        for prop in props:
            indexes.append((len(leaf_array), add_model(prop.model)))
            leaf_array.extend(sorted([add_leaf(leaf) for leaf in prop.visleafs]))

        if self.static_prop_version is StaticPropVersion.UNKNOWN:
            self.static_prop_version = StaticPropVersion.DEFAULT

        version = self.static_prop_version
        vers_num = self.static_prop_version.version
        if version.is_lightmap:
            vers_num = 7

        # Now write out the sections.
        prop_lump = BytesIO()
        prop_lump.write(struct.pack('<i', len(model_list)))
        for name in model_list:
            prop_lump.write(struct.pack('<128s', name.encode('ascii', 'surrogateescape')))

        prop_lump.write(struct.pack('<i', len(leaf_array)))
        prop_lump.write(write_array(self.lump_layout['STATICPROPLEAF'], leaf_array))

        prop_lump.write(struct.pack('<i', len(props)))
        for (leaf_off, model_ind), prop in zip(indexes, props):
            start = prop_lump.tell()
            prop_lump.write(struct.pack(
                '<3f3fH',
                prop.origin.x,
                prop.origin.y,
                prop.origin.z,
                prop.angles.pitch,
                prop.angles.yaw,
                prop.angles.roll,
                model_ind,
            ))

            prop_lump.write(struct.pack(
                '<HHBBiff3f',
                leaf_off,
                len(prop.visleafs),
                prop.solidity,
                # TF2 doesn't use this, it's random.
                0 if version.is_lightmap else prop.flags.value_prim,
                prop.skin,
                prop.min_fade,
                prop.max_fade,
                prop.lighting.x,
                prop.lighting.y,
                prop.lighting.z,
            ))
            if vers_num >= 5:
                prop_lump.write(struct.pack('<f', prop.fade_scale))

            if vers_num in (6, 7):
                prop_lump.write(struct.pack(
                    '<HH',
                    prop.min_dx_level,
                    prop.max_dx_level,
                ))

            if vers_num >= 8:
                prop_lump.write(struct.pack(
                    '<BBBB',
                    prop.min_cpu_level,
                    prop.max_cpu_level,
                    prop.min_gpu_level,
                    prop.max_gpu_level
                ))

            if version.is_lightmap:
                prop_lump.write(struct.pack(
                    '<IHH',
                    prop.flags.value,
                    prop.lightmap_x, prop.lightmap_y,
                ))

            if vers_num >= 7 and not version.is_sdk_2013:
                prop_lump.write(struct.pack(
                    '<BBBB',
                    int(prop.tint.x),
                    int(prop.tint.y),
                    int(prop.tint.z),
                    prop.renderfx,
                ))

            if vers_num >= 9 and not version.is_lightmap:
                # The 1-byte bool gets expanded to the full 4-byte size.
                prop_lump.write(struct.pack('<?xxx', prop.disable_on_xbox))

            if vers_num >= 10 or version is StaticPropVersion.V_LIGHTMAP_MESA:
                prop_lump.write(struct.pack('<I', prop.flags.value_sec))

            if version is StaticPropVersion.V_CHAOS_V13:
                # Three floats for non-uniform scaling
                if isinstance(prop.scaling, Vec):
                    scaling_3 = prop.scaling
                else:
                    scaling_3 = Vec(prop.scaling, prop.scaling, prop.scaling)

                prop_lump.write(struct.pack(
                    '<fff',
                    scaling_3.x,
                    scaling_3.y,
                    scaling_3.z,
                ))
            elif vers_num >= 11:
                # One float for uniform scaling
                scaling_1 = prop.scaling.x if isinstance(prop.scaling, Vec) else prop.scaling

                prop_lump.write(struct.pack('<f', scaling_1))

            real_size = prop_lump.tell() - start
            if version.size != real_size:
                raise ValueError(f'Expected {version.size} for {version}, got {real_size}!')

        return prop_lump.getvalue()

    def _lmp_read_detail_props(self, version: int, data: bytes) -> Iterator['DetailProp']:
        """Parse detail props."""
        buf = BytesIO(data)
        # First read the model dictionary.
        models = list(self._read_static_props_models(buf))

        [sprite_count] = struct_read('<i', buf)
        detail_sprites = []
        for _ in range(sprite_count):
            (
                dim_ul_x, dim_ul_y,
                dim_lr_x, dim_lr_y,
                tex_ul_x, tex_ul_y,
                tex_lr_x, tex_lr_y,
            ) = struct_read('<8f', buf)
            detail_sprites.append((
                (dim_ul_x, dim_ul_y),
                (dim_lr_x, dim_lr_y),
                (tex_ul_x, tex_ul_y),
                (tex_lr_x, tex_lr_y),
            ))

        [detail_count] = struct_read('<i', buf)
        for _ in range(detail_count):
            (
                x, y, z,
                pit, yaw, rol,
                mdl,
                leaf,
                r, g, b, a,
                styles,
                style_count,
                sway_amt,
                shape_ang,
                shape_size,
                orient,
                detail_type,
                spr_scale,
            ) = struct_read('<3f3fHH4BI5B3xB3xf', buf)
            if detail_type == 0:  # Model
                yield DetailPropModel(
                    Vec(x, y, z),
                    Angle(pit, yaw, rol),
                    DetailPropOrientation(orient),
                    leaf,
                    (r, g, b, a),
                    (styles, style_count),
                    sway_amt,
                    models[mdl],
                )
            elif detail_type == 1:  # Sprite
                (
                    dim_ul, dim_lr,
                    tex_ul, tex_lr,
                ) = detail_sprites[mdl]
                yield DetailPropSprite(
                    Vec(x, y, z),
                    Angle(pit, yaw, rol),
                    DetailPropOrientation(orient),
                    leaf,
                    (r, g, b, a),
                    (styles, style_count),
                    sway_amt,
                    spr_scale,
                    dim_ul, dim_lr,
                    tex_ul, tex_lr,
                )
            elif detail_type in (2, 3):  # Shapes.
                (
                    dim_ul, dim_lr,
                    tex_ul, tex_lr,
                ) = detail_sprites[mdl]
                yield DetailPropShape(
                    Vec(x, y, z),
                    Angle(pit, yaw, rol),
                    DetailPropOrientation(orient),
                    leaf,
                    (r, g, b, a),
                    (styles, style_count),
                    sway_amt,
                    spr_scale,
                    dim_ul, dim_lr,
                    tex_ul, tex_lr,
                    detail_type == 3,
                    shape_ang,
                    shape_size,
                )
            else:
                raise ValueError(f'Unknown detail prop type {detail_type}!')

    def _lmp_write_detail_props(self, props: List['DetailProp']) -> Iterator[bytes]:
        """Reconstruct the detail props lump."""
        sprites: List[Tuple[
            float, float, float, float,
            float, float, float, float,
        ]] = []
        models: List[str] = []

        add_sprite = find_or_insert(sprites, identity)
        add_model = find_or_insert(models, identity)
        buf = BytesIO()

        for prop in props:
            if isinstance(prop, DetailPropModel):
                mdl_ind = add_model(prop.model)
                detail_type = 0
                spr_scale = 1.0
                shape_ang = 0
                shape_size = 1
            elif isinstance(prop, DetailPropSprite):
                mdl_ind = add_sprite(
                    prop.dims_upper_left + prop.dims_lower_right +
                    prop.texcoord_upper_left + prop.texcoord_lower_right
                )
                detail_type = 1
                spr_scale = prop.sprite_scale
                shape_ang = 0
                shape_size = 1
            elif isinstance(prop, DetailPropShape):
                mdl_ind = add_sprite(
                    prop.dims_upper_left + prop.dims_lower_right +
                    prop.texcoord_upper_left + prop.texcoord_lower_right
                )
                detail_type = 3 if prop.is_cross else 2
                spr_scale = prop.sprite_scale
                shape_ang = prop.shape_angle
                shape_size = prop.shape_size
            else:
                raise TypeError(f'Unknown detail prop type {prop}!')

            buf.write(struct.pack(
                '<3f3fHH4BI5B3xB3xf',
                prop.origin.x, prop.origin.y, prop.origin.z,
                prop.angles.pitch, prop.angles.yaw, prop.angles.roll,
                mdl_ind,
                prop.leaf,
                *prop.lighting,
                *prop._light_styles,
                prop.sway_amount,
                shape_ang,
                shape_size,
                prop.orientation.value,
                detail_type,
                spr_scale,
            ))

        # Now build the complete lump.
        yield struct.pack('<i', len(models))
        for name in models:
            yield struct.pack('<128s', name.encode('ascii', 'surrogateescape'))
        yield struct.pack('<i', len(sprites))
        spr_format = struct.Struct('<8f')
        for spr in sprites:
            yield spr_format.pack(*spr)
        yield struct.pack('<i', len(props))
        yield buf.getvalue()

    def vis_tree(self) -> 'VisTree':
        """Parse the visleaf data, and return the root node."""
        # First node is the top of the tree.
        return self.nodes[0]
