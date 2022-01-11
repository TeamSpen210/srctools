"""Read and write parts of Source BSP files.

Data from a read BSP is lazily parsed when each section is accessed.
"""
from typing import (
    overload, TypeVar, Any, Generic, Union, Optional, ClassVar, Type,
    List, Iterator, BinaryIO, Tuple, Callable, Dict, Set, Hashable, Generator, cast,
)
from io import BytesIO
from enum import Enum, Flag
from weakref import WeakKeyDictionary
from zipfile import ZipFile
import itertools
import struct
import inspect
import contextlib
import warnings
import attr
import os

from srctools import AtomicWriter, conv_int
from srctools.math import Vec, Angle
from srctools.filesys import FileSystem
from srctools.vtf import VTF
from srctools.vmt import Material
from srctools.vmf import VMF, Entity, Output
from srctools.tokenizer import escape_text
from srctools.binformat import (
    DeferredWrites,
    struct_read,
    read_array, write_array,
)
from srctools.property_parser import Property
from srctools.const import SurfFlags, BSPContents as BrushContents, add_unknown


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
KeyT = TypeVar('KeyT')  # Needs to be hashable, typecheckers don't work for that.

# Game lump IDs
LMP_ID_STATIC_PROPS = b'sprp'
LMP_ID_DETAIL_PROPS = b'dprp'


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
LUMP_REBUILD_ORDER: List[Union[bytes, BSP_LUMPS]] = [
    BSP_LUMPS.PAKFILE,
    BSP_LUMPS.CUBEMAPS,
    LMP_ID_STATIC_PROPS,  # References visleafs.
    LMP_ID_DETAIL_PROPS,

    BSP_LUMPS.MODELS,  # Brushmodels reference their vis tree, faces, and the entity they're tied to.
    BSP_LUMPS.ENTITIES,  # References brushmodels, overlays, potentially many others.
    BSP_LUMPS.NODES,  # References planes, faces, visleafs.
    BSP_LUMPS.LEAFS,  # References brushes, faces

    BSP_LUMPS.BRUSHES,  # also brushsides, references texinfo.

    BSP_LUMPS.FACES,  # References their original face, surfedges, texinfo, primitives.
    BSP_LUMPS.FACES_HDR,  # References their original face, surfedges, texinfo, primitives.
    BSP_LUMPS.ORIGINALFACES,  # references surfedges & texinfo, primitives.

    BSP_LUMPS.PRIMITIVES,
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
    _BIT_4 = 1 << 3
    _BIT_5 = 1 << 4
    _BIT_6 = 1 << 5
    _BIT_7 = 1 << 6


class DetailPropOrientation(Enum):
    """The kind of orientation for detail props."""
    NORMAL = 0
    SCREEN_ALIGNED = 1
    SCREEN_ALIGNED_VERTICAL = 2


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
        reflectivity: Vec=None, width: int=0, height: int=0,
        fsys: FileSystem=None,
    ) -> None:
        """Set the material used for this texinfo.

        If it is not already used in the BSP, some additional info is required.
        This can either be parsed from the VMT and VTF, or provided directly.
        """
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
    def _normal_setattr(self, _: attr.Attribute, value: Vec) -> Vec:
        """Recompute the plane type whenever the normal is changed."""
        value = Vec(value)
        self.type = PlaneType.from_normal(value)
        return value

    def _type_default(self) -> 'PlaneType':
        """Compute the plane type parameter if not provided."""
        return PlaneType.from_normal(self.normal)

    normal: Vec = attr.ib(on_setattr=_normal_setattr)
    dist: float = attr.ib(converter=float, validator=attr.validators.instance_of(float))
    type: PlaneType = attr.Factory(_type_default, takes_self=True)

    del _normal_setattr, _type_default


@attr.define(eq=False)
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

    def key(self) -> tuple:
        """A key to match the edge with."""
        a, b = self.a, self.b
        return (a.x, a.y, a.z, b.x, b.y, b.z)


class RevEdge(Edge):
    """The edge on the opposite side from the original."""
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


@attr.define(eq=False)
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
    smoothing_groups: int
    hammer_id: Optional[int]  # The original ID of the Hammer face.


@attr.define(eq=False)
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


@attr.define(eq=False)
class DetailPropModel(DetailProp):
    """A MDL detail prop."""
    model: str


@attr.define(eq=False)
class DetailPropSprite(DetailProp):
    """A sprite-type detail prop."""
    sprite_scale: float
    dims_upper_left: Tuple[float, float]
    dims_lower_right: Tuple[float, float]
    texcoord_upper_left: Tuple[float, float]
    texcoord_lower_right: Tuple[float, float]


@attr.define(eq=False)
class DetailPropShape(DetailPropSprite):
    """A shape-type detail prop, rendered as a triangle or cross shape."""
    is_cross: bool
    shape_angle: int
    shape_size: int


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

    fade_min_sq: float = -1.0
    fade_max_sq: float = 0.0

    # If system exceeds these limits, the overlay is skipped.
    min_cpu: int = -1
    max_cpu: int = -1
    min_gpu: int = -1
    max_gpu: int = -1


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
    _ambient: bytes = bytes(24)

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
        checked: Set[int] = set()  # Guard against recursion.
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


@attr.define(eq=False)
class BModel:
    """A brush model definition, used for the world entity along with all other brush ents."""
    mins: Vec
    maxes: Vec
    origin: Vec
    node: VisTree
    faces: List[Face]

    # If solid, the .phy file-like physics data.
    # This is a text section, and a list of blocks.
    phys_keyvalues: Optional[Property] = None
    _phys_solids: List[bytes] = []


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
    visleafs: Set[VisLeaf] = attr.ib(factory=set)
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


def identity(x: T) -> T:
    """Identity function."""
    return x


def _find_or_insert(item_list: List[T], key_func: Callable[[T], Hashable]=id) -> Callable[[T], int]:
    """Create a function for inserting items in a list if not found.

    This is used to build up the structure arrays which other lumps refer
    to by index.
    If the provided argument to the callable is already in the list,
    the index is returned. Otherwise it is appended and the new index returned.
    The key function is used to match existing items.

    """
    by_index: Dict[Hashable, int] = {key_func(item): i for i, item in enumerate(item_list)}

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


def _find_or_extend(item_list: List[T], key_func: Callable[[T], Hashable]=id) -> Callable[[List[T]], int]:
    """Create a function for positioning a sublist inside the larger list, adding it if required.

    This is used to build up structure arrays where other lumps access subsections of it.
    """
    # We expect repeated items to be fairly uncommon, so we can skip to all
    # occurrences of the first index to speed up the search.
    by_index: Dict[Hashable, List[int]] = {}
    for k, item in enumerate(item_list):
        by_index.setdefault(key_func(item), []).append(k)

    def finder(items: List[T]) -> int:
        """Find or append the items."""
        if not items:
            # Array is empty, so the index doesn't matter, it'll never be
            # dereferenced.
            return 0
        try:
            indices = by_index[key_func(items[0])]
        except KeyError:
            pass
        else:
            for i in indices:
                if item_list[i:i + len(items)] == items:
                    return i
        # Not found, append to the end.
        i = len(item_list)
        item_list.extend(items)
        assert item_list[i: i + len(items)] == items
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

    If the lump name is bytes, it's a game lump identifier.
    """

    def __init__(self, lump: Union[bytes, BSP_LUMPS], *extra: Union[bytes, BSP_LUMPS]) -> None:
        self.lump = lump
        self.to_clear = (lump, ) + extra
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
        result: T
        if isinstance(self.lump, bytes):  # Game lump
            gm_lump = instance.game_lumps[self.lump]
            result = self._read(instance, gm_lump.version, gm_lump.data)
        else:
            data = instance.lumps[self.lump].data
            result = self._read(instance, data)
        if inspect.isgenerator(result):  # Convenience, yield to accumulate into a list.
            result = list(result)  # type: ignore

        instance._parsed_lumps[self.lump] = result # noqa
        for lump in self.to_clear:
            if isinstance(lump, bytes):
                instance.game_lumps[lump].data = b''
            else:
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
            if isinstance(lump, bytes):
                instance.game_lumps[lump].data = b''
            else:
                instance.lumps[lump].data = b''
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
    version: Union[VERSIONS, int]

    def __init__(self, filename: Union[str, os.PathLike], version: VERSIONS=None):
        self.filename = filename
        self.map_revision = -1  # The map's revision count
        self.lumps: Dict[BSP_LUMPS, Lump] = {}
        self._parsed_lumps: Dict[Union[bytes, BSP_LUMPS], Any] = {}
        self.game_lumps: Dict[bytes, GameLump] = {}
        self.header_off = 0
        self.version = version  # type: ignore  # read() will make it non-none.
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
    overlays: ParsedLump[List['Overlay']] = ParsedLump(BSP_LUMPS.OVERLAYS, BSP_LUMPS.OVERLAY_FADES, BSP_LUMPS.OVERLAY_SYSTEM_LEVELS)

    bmodels: ParsedLump['WeakKeyDictionary[Entity, BModel]'] = ParsedLump(BSP_LUMPS.MODELS, BSP_LUMPS.PHYSCOLLIDE)
    brushes: ParsedLump[List['Brush']] = ParsedLump(BSP_LUMPS.BRUSHES, BSP_LUMPS.BRUSHSIDES)
    visleafs: ParsedLump[List['VisLeaf']] = ParsedLump(BSP_LUMPS.LEAFS, BSP_LUMPS.LEAFFACES, BSP_LUMPS.LEAFBRUSHES)
    nodes: ParsedLump[List['VisTree']] = ParsedLump(BSP_LUMPS.NODES)

    vertexes: ParsedLump[List[Vec]] = ParsedLump(BSP_LUMPS.VERTEXES)
    surfedges: ParsedLump[List[Edge]] = ParsedLump(BSP_LUMPS.SURFEDGES, BSP_LUMPS.EDGES)
    planes: ParsedLump[List['Plane']] = ParsedLump(BSP_LUMPS.PLANES)
    faces: ParsedLump[List['Face']] = ParsedLump(BSP_LUMPS.FACES)
    orig_faces: ParsedLump[List['Face']] = ParsedLump(BSP_LUMPS.ORIGINALFACES)
    hdr_faces: ParsedLump[List['Face']] = ParsedLump(BSP_LUMPS.FACES_HDR)
    primitives: ParsedLump[List['Primitive']] = ParsedLump(BSP_LUMPS.PRIMITIVES, BSP_LUMPS.PRIMINDICES, BSP_LUMPS.PRIMVERTS)

    # Game lumps
    props: ParsedLump[List['StaticProp']] = ParsedLump(LMP_ID_STATIC_PROPS)
    detail_props: ParsedLump[List['DetailProp']] = ParsedLump(LMP_ID_DETAIL_PROPS)

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
                game_lump_id: bytes
                flags: int
                glump_version: int
                file_off: int
                file_len: int
                (
                    game_lump_id,
                    flags,
                    glump_version,
                    file_off,
                    file_len,
                ) = GameLump.ST.unpack_from(game_lump.data, lump_offset)
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
        for lump_or_game in LUMP_REBUILD_ORDER:
            try:
                data = self._parsed_lumps.pop(lump_or_game)
            except KeyError:
                pass
            else:
                lump_result: bytes = cast(bytes, self._save_funcs[lump_or_game](self, data))
                # Convenience, yield to accumulate into bytes.
                if inspect.isgenerator(lump_result):
                    buf = BytesIO()
                    for chunk in lump_result:
                        buf.write(chunk)  # type: ignore
                    lump_result = buf.getvalue()
                if isinstance(lump_or_game, bytes):
                    self.game_lumps[lump_or_game].data = lump_result
                else:
                    self.lumps[lump_or_game].data = lump_result
        game_lumps = list(self.game_lumps.values())  # Lock iteration order.

        file: BinaryIO
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

    def _lmp_read_surfedges(self, vertexes: bytes) -> Iterator[Edge]:
        verts: List[Vec] = self.vertexes
        edges = [
            Edge(verts[a], verts[b])
            for a, b in struct.iter_unpack('<HH', self.lumps[BSP_LUMPS.EDGES].data)
        ]
        for [ind] in struct.iter_unpack('i', vertexes):
            if ind < 0:  # If negative, the vertexes are reversed order.
                yield edges[-ind].opposite
            else:
                yield edges[ind]

    def _lmp_write_surfedges(self, surf_edges: List[Edge]) -> bytes:
        """Reconstruct the surfedges and edges lumps."""
        edge_buf = BytesIO()
        surf_buf = BytesIO()

        edges: List[Edge] = []
        # We cannot share vertexes or edges, it breaks VRAD!
        add_edge = _find_or_insert(edges)
        add_vert = _find_or_insert(self.vertexes)

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
            edge_buf.write(struct.pack('<HH', add_vert(edge.a), add_vert(edge.b)))

        self.lumps[BSP_LUMPS.EDGES].data = edge_buf.getvalue()
        return surf_buf.getvalue()

    def _lmp_read_primitives(self, data: bytes) -> Iterator['Primitive']:
        """Parse the primitives lumps."""
        verts = list(map(Vec, struct.iter_unpack('<fff', self.lumps[BSP_LUMPS.PRIMVERTS].data)))
        indices = read_array('<H', self.lumps[BSP_LUMPS.PRIMINDICES].data)
        for (
            prim_type,
            first_ind, ind_count,
            first_vert, vert_count,
        ) in struct.iter_unpack('<HHHHH', data):
            yield Primitive(
                prim_type,
                indices[first_ind: first_ind + ind_count],
                verts[first_vert: first_vert + vert_count],
            )

    def _lmp_write_primitives(self, prims: List['Primitive']) -> Iterator[bytes]:
        verts: List[bytes] = []
        indices: List[int] = []

        for prim in prims:
            vert_loc = len(verts)
            index_loc = len(indices)
            verts += [struct.pack('<fff', pos.x, pos.y, pos.z) for pos in prim.verts]
            indices.extend(prim.indexed_verts)
            yield struct.pack(
                '<HHHHH',
                prim.is_tristrip,
                index_loc, len(prim.indexed_verts),
                vert_loc, len(prim.verts),
            )
        self.lumps[BSP_LUMPS.PRIMINDICES].data = write_array('<H', indices)
        self.lumps[BSP_LUMPS.PRIMVERTS].data = b''.join(verts)

    def _lmp_read_orig_faces(self, data: bytes, _orig_faces: List['Face'] = None) -> Iterator['Face']:
        """Read one of the faces arrays.

        For ORIG_FACES, _orig_faces is None and that entry is ignored.
        For the others, that is the parsed orig faces lump, which each face
        may reference.
        """
        # The non-original faces have the Hammer ID value, which is an array
        # in the same order. But some versions don't define it as anything...
        if _orig_faces is not None:
            hammer_ids = read_array('H', self.lumps[BSP_LUMPS.FACEIDS].data)
        else:
            hammer_ids = []

        for i, (
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
        ) in enumerate(struct.iter_unpack('<H??i4h4sif5iHHI', data)):
            # If orig faces is provided, that is the original face
            # we were created from. Additionally, it seems the original
            # face data has invalid texinfo, so copy ours on top of it.
            hammer_id: Optional[int]
            if _orig_faces is not None:
                orig_face = _orig_faces[orig_face_ind]
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
                self.primitives[prim_first:prim_first + prim_num],
                smoothing_group,
                hammer_id,
            )

    def _lmp_write_orig_faces(self, faces: List['Face'], get_orig_face: Callable[['Face'], int]=None) -> bytes:
        """Reconstruct one of the faces arrays.

        If this isn't the orig faces array, get_orig_face should be
        _find_or_insert(self.orig_faces).
        """
        face_buf = BytesIO()
        add_texinfo = _find_or_insert(self.texinfo)
        add_plane = _find_or_insert(self.planes)
        add_edges = _find_or_extend(self.surfedges)
        add_prims = _find_or_extend(self.primitives)
        hammer_ids = []

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
                len(face.primitives), add_prims(face.primitives),
                face.smoothing_groups,
            ))
        if hammer_ids:
            self.lumps[BSP_LUMPS.FACEIDS].data = write_array('<H', hammer_ids)
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
        sides: List[BrushSide] = []
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

    def _lmp_read_visleafs(self, data: bytes) -> Iterator['VisLeaf']:
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

        leaf_fmt = '<ihh6h4Hh'
        has_ambient = False
        # Some extra ambient light data.
        if self.version <= 19:
            has_ambient = True
            leaf_fmt += '24s'
        leaf_fmt += '2x'

        for (
            contents,
            cluster_ind, area_and_flags,
            min_x, min_y, min_z,
            max_x, max_y, max_z,
            first_face, num_faces,
            first_brush, num_brushes,
            *water_ind_and_ambient
        ) in struct.iter_unpack(leaf_fmt, data):
            area = area_and_flags >> 7
            flags = area_and_flags & 0b1111111
            if has_ambient:
                [water_ind, ambient] = water_ind_and_ambient
            else:
                [water_ind] = water_ind_and_ambient
                # bytes(24), but can constant fold.
                ambient = b'\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0'
            yield VisLeaf(
                BrushContents(contents), cluster_ind, area, VisLeafFlags(flags),
                Vec(min_x, min_y, min_z),
                Vec(max_x, max_y, max_z),
                leaf_faces[first_face:first_face+num_faces],
                leaf_brushes[first_brush:first_brush+num_brushes],
                water_ind, ambient,
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
        leaf_faces: List[int] = []
        leaf_brushes: List[int] = []

        add_face = _find_or_insert(self.faces)
        add_brush = _find_or_insert(self.brushes)

        buf = BytesIO()

        for leaf in visleafs:
            # Do not deduplicate these, engine assumes they aren't when allocating memory.
            face_ind = len(leaf_faces)
            brush_ind = len(leaf_brushes)
            leaf_faces.extend(map(add_face, leaf.faces))
            leaf_brushes.extend(map(add_brush, leaf.brushes))

            buf.write(struct.pack(
                '<ihh6h4Hh',
                leaf.contents.value, leaf.cluster_id,
                (leaf.area << 7 | leaf.flags.value),
                int(leaf.mins.x), int(leaf.mins.y), int(leaf.mins.z),
                int(leaf.maxes.x), int(leaf.maxes.y), int(leaf.maxes.z),
                face_ind, len(leaf.faces),
                brush_ind, len(leaf.brushes),
                leaf.water_id,
            ))
            if self.version <= 19:
                buf.write(leaf._ambient)
            buf.write(b'\x00\x00')  # Padding.

        self.lumps[BSP_LUMPS.LEAFFACES].data = write_array('<H', leaf_faces)
        self.lumps[BSP_LUMPS.LEAFBRUSHES].data = write_array('<H', leaf_brushes)
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
        texdata_ind: Dict[TexData, int] = {}

        texdata_list: List[bytes] = []
        texinfo_result: List[bytes] = []

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
            mdl.phys_keyvalues = Property.parse(
                kvs,
                f'bmodel[{mdl_ind}].keyvalues',
            )

        # Loop over entities, map to their brush model.
        brush_ents: WeakKeyDictionary[Entity, BModel] = WeakKeyDictionary()
        vmf: VMF = self.ents
        brush_ents[vmf.spawn] = bmodel_list[0]
        for ent in vmf.entities:
            if ent['model'].startswith('*'):
                mdl_ind = int(ent.keys.pop('model')[1:])
                brush_ents[ent] = bmodel_list[mdl_ind]

        return brush_ents

    def _lmp_write_bmodels(self, bmodels: 'WeakKeyDictionary[Entity, BModel]') -> Iterator[bytes]:
        """Write the brush model definitions."""
        add_node = _find_or_insert(self.nodes)
        add_faces = _find_or_extend(self.faces)

        phys_buf = BytesIO()

        # The models in order, worldspawn must be zero.
        worldspawn = self.ents.spawn
        try:
            model_list = [bmodels[worldspawn]]
        except KeyError:
            raise ValueError('Worldspawn has no brush model!')
        add_model = _find_or_insert(model_list)

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
                kvs = '\n'.join(model.phys_keyvalues.export()).encode('ascii') + b'\x00'
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

    def _lmp_read_overlays(self, data: bytes) -> Iterator['Overlay']:
        """Read the overlays lump."""
        # Use zip longest, so we handle cases where these newer auxiliary lumps
        # are empty.
        for block, fades, sys_levels in itertools.zip_longest(
            struct.iter_unpack(
                '<ihH'  # id, texinfo, face-and-render-order
                '64i'  # face array.
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

    def _lmp_write_overlays(self, overlays: List['Overlay']) -> Iterator[bytes]:
        """Write out all overlays."""
        add_texinfo = _find_or_insert(self.texinfo)
        fade_buf = BytesIO()
        levels_buf = BytesIO()
        for over in overlays:
            face_cnt = len(over.faces)
            if face_cnt > 64:
                raise ValueError(f'{over.faces} exceeds OVERLAY_BSP_FACE_COUNT (64)!')
            fade_buf.write(struct.pack('<ff', over.fade_min_sq, over.fade_max_sq))
            levels_buf.write(struct.pack('<BBBB', over.min_cpu, over.max_cpu, over.min_gpu, over.max_gpu))
            yield struct.pack(
                '<ihH',
                over.id,
                add_texinfo(over.texture),
                (over.render_order << 14 | face_cnt),
            )
            # Build the array, then zero fill the remaining space.
            yield struct.pack(f'<{face_cnt}i {4*(64-face_cnt)}x', *over.faces)
            yield struct.pack('<4f', over.u_min, over.u_max, over.v_min, over.v_max)
            yield struct.pack(
                '<18f',
                *over.uv1, *over.uv2, *over.uv3, *over.uv4,
                *over.origin, *over.normal,
            )
        self.lumps[BSP_LUMPS.OVERLAY_FADES].data = fade_buf.getvalue()
        self.lumps[BSP_LUMPS.OVERLAY_SYSTEM_LEVELS].data = levels_buf.getvalue()

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
                out.write(output.as_keyvalue().encode('ascii'))
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
        """Read in the Static Props lump.

        Deprecated, use bsp.props.
        """
        warnings.warn('Access BSP.props instead', DeprecationWarning, stacklevel=2)
        return iter(self.props)

    def write_static_props(self, props: List['StaticProp']) -> None:
        """Remake the static prop lump.

        Deprecated, bsp.props is stored and resaved.
        """
        warnings.warn('Assign to BSP.props', DeprecationWarning, stacklevel=2)
        self.props = props

    def _lmp_read_props(self, version: int, data: bytes) -> Iterator['StaticProp']:
        # The version of the static prop format - different features.
        if version > 11:
            raise ValueError('Unknown version ({})!'.format(version))
        if version < 4:
            # Predates HL2...
            raise ValueError('Static prop version {} is too old!')

        static_lump = BytesIO(data)

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

    def _lmp_write_props(self, props: List['StaticProp']) -> bytes:
        # First generate the visleaf and model-names block.
        # Unfortunately it seems reusing visleaf parts isn't possible.
        leaf_array: List[int] = []
        model_list: List[str] = []
        add_model = _find_or_insert(model_list, identity)
        add_leaf = _find_or_insert(self.visleafs)

        indexes: List[Tuple[int, int]] = []
        for prop in props:
            indexes.append((len(leaf_array), add_model(prop.model)))
            leaf_array.extend(sorted([add_leaf(leaf) for leaf in prop.visleafs]))

        version = self.game_lumps[LMP_ID_STATIC_PROPS].version

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
            if version >= 5:
                prop_lump.write(struct.pack('<f', prop.fade_scale))

            if version in (6, 7):
                prop_lump.write(struct.pack(
                    '<HH',
                    prop.min_dx_level,
                    prop.max_dx_level,
                ))

            if version >= 8:
                prop_lump.write(struct.pack(
                    '<BBBB',
                    prop.min_cpu_level,
                    prop.max_cpu_level,
                    prop.min_gpu_level,
                    prop.max_gpu_level
                ))

            if version >= 7:
                prop_lump.write(struct.pack(
                    '<BBBB',
                    int(prop.tint.x),
                    int(prop.tint.y),
                    int(prop.tint.z),
                    prop.renderfx,
                ))

            if version >= 10:
                prop_lump.write(struct.pack('<I', prop.flags.value_sec))

            if version >= 11:
                # Unknown padding/data, though it's always zero.

                prop_lump.write(struct.pack('<xxxxf', prop.scaling))
            elif version >= 9:
                # The 1-byte bool gets expanded to the full 4-byte size.
                prop_lump.write(struct.pack('<?xxx', prop.disable_on_xbox))

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

        add_sprite = _find_or_insert(sprites, identity)
        add_model = _find_or_insert(models, identity)
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
            yield struct.pack('<128s', name.encode('ascii'))
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
