""" VMF Library

Wraps Keyvalues trees in a set of classes which smartly handle
specifics of VMF files.
"""
from typing import (
    IO, TYPE_CHECKING, AbstractSet, Any, Callable, Dict, Final, FrozenSet, ItemsView,
    Iterable, Iterator, KeysView, List, Mapping, Match, MutableMapping, Optional, Pattern,
    Protocol, Set, Tuple, TypeVar, Union, ValuesView, overload,
)
from typing_extensions import Literal, TypeAlias, deprecated
from array import ArrayType as Array
from collections import defaultdict
from enum import Enum, Flag
from sys import intern
import builtins
import io
import operator
import re
import struct
import warnings

import attrs

from srctools import BOOL_LOOKUP, EmptyMapping
from srctools.keyvalues import Keyvalues, escape_text
from srctools.math import (
    Angle, AnyAngle, AnyMatrix, AnyVec, FrozenAngle, FrozenMatrix, FrozenVec, Matrix, Vec,
    format_float, to_matrix,
)
import srctools


__all__ = [
    'CURRENT_HAMMER_BUILD', 'CURRENT_HAMMER_VERSION',
    'conv_kv', 'ValidKVs', 'Axis',
    'overlay_bounds', 'make_overlay', 'localise_overlay',
    'VMF', 'Camera', 'Cordon', 'VisGroup', 'Solid', 'Side', 'Entity', 'EntityGroup',
    'DispFlag', 'TriangleTag', 'DispVertex',
    'PrismFace', 'UVAxis', 'EntityFixup', 'FixupValue',  # For typing, shouldn't be constructed.
    'StrataInstanceVisibility',
    'Output', 'OUTPUT_SEP',
]

# Used to set the defaults for versioninfo
CURRENT_HAMMER_VERSION = 400
CURRENT_HAMMER_BUILD = 5304


#: The character used to separate output values, after L4D. Before then commas (``,``) were used.
#: This is also available as :py:attr:`Output.SEP`.
OUTPUT_SEP: Final = chr(27)


T = TypeVar('T')
# Types we allow for setting keyvalues. These we can stringify into something
# matching Valve's usual encoding.
# Other types are just str()ed, which might produce a bad result.
_ValidKVBasics = Union[str, int, bool, float, Vec, FrozenVec, AnyAngle, AnyMatrix]


class _ValidKVEnum(Protocol):
    @property
    def value(self) -> _ValidKVBasics:
        """Matches an enum with a value which itself can be converted to a keyvalue."""
        return ""


ValidKVs: TypeAlias = Union[_ValidKVBasics, _ValidKVEnum]
Axis: TypeAlias = Literal['x', 'y', 'z']
ValidKV_T = TypeVar('ValidKV_T', bound=ValidKVs)
# We include strings here to catch subclasses.
_KVToString = (Vec, FrozenVec, Angle, FrozenAngle, int, str)


class DispFlag(Flag):
    """Per-displacement flags, configuring collisions.

    Does NOT match the file values, since those are inverted.
    """
    COLL_NONE = 0
    COLL_PHYSICS = 1  # Can physics objects collide?
    COLL_PLAYER_NPC = 2  # QPhysics hull collisions
    COLL_BULLET = 4  # Raytraces IE bullets.
    COLL_ALL = COLL_PHYSICS | COLL_PLAYER_NPC | COLL_BULLET

    SUBDIV = 8  # Is it subdivided?
    SUBDIV_COLL_ALL = SUBDIV | COLL_ALL  # Makes the repr nicer.


# The VMF stores 2/4/8 if the displacement *doesn't* collide.
# Bit 1 stores if the face has a bumpmap, set by VBSP.
_DISP_FLAG_TO_COLL: List[DispFlag] = [
    (
        (DispFlag.COLL_PHYSICS if i & 2 == 0 else DispFlag.COLL_NONE) |
        (DispFlag.COLL_PLAYER_NPC if i & 4 == 0 else DispFlag.COLL_NONE) |
        (DispFlag.COLL_BULLET if i & 8 == 0 else DispFlag.COLL_NONE)
    ) for i in range(16)
]
# Invert, prefer smaller = less bits set.
_DISP_COLL_TO_FLAG: Dict[DispFlag, int] = {
    v: k for (k, v) in
    list(enumerate(_DISP_FLAG_TO_COLL))[::-1]
}


class TriangleTag(Flag):
    """Two flags set on all displacement triangles.

    If walkable, it is shallow enough to stand on.
    If buildable, TF2 Engineers can place buildings.
    """
    STEEP = 0  # Not buildable/walkable
    WALKABLE = 1  # Just walkable
    # 8 by itself is buildable only, that's impossible.
    BUILDABLE = 1 | 8  # Walkable and buildable
    FLAT = BUILDABLE


def conv_kv(val: ValidKVs) -> str:
    """Convert a type into a string matching Valve's syntax.

    The following types are allowed:
    * Strings: Passed unchanged.
    * Booleans: Converted to `1` or `0`.
    * :py:class:`int`: Stringified as normal.
    * :py:class`float`: Strips `.0` if integral.
    * [:py:class:`Frozen<srctools.math.FrozenVec>`] :py:class:`Vec <srctools.math.Vec>`,
      [:py:class:`Frozen<srctools.math.FrozenAngle>`] :py:class:`Angle <srctools.math.Angle>`:
      Uses the standard `1 2 3` form.
    * [:py:class:`Frozen<srctools.math.FrozenMatrix>`] :py:class:`Matrix <srctools.math.Matrix>`:
      Converted to the corresponding angle.
    * Any enum: Allowed if the value is itself convertable.
    """
    if type(val) is str:
        # Can return unchanged. We need to make sure to unwrap subclasses.
        return val
    elif val is True:
        return '1'
    elif val is False:
        return '0'
    elif isinstance(val, Matrix) or isinstance(val, FrozenMatrix):
        return str(val.to_angle())
    elif isinstance(val, float):
        return format_float(val)
    elif isinstance(val, _KVToString):
        return str(val)
    try:
        val = val.value
    except AttributeError:
        return str(val)  # Fallback.
    else:
        # Recursively convert enum values.
        return conv_kv(val)


class IDMan(AbstractSet[int]):
    """Allocate and manage a set of unique IDs.

    This implements some of MutableSet, but the adding methods cannot
    be used since the ID may need to change to ensure uniqueness.
    """
    _used: Set[int]
    search_pos: int

    def __init__(self, existing: Iterable[int] = ()) -> None:
        """Initialise the ID manager."""
        super().__init__()
        self._used = set(existing)
        # This is used to hint where we should start searching from.
        # IDs from 1:search_pos must have been used already.
        # search_pos and above may or may not have been used.

        # The ID space is usually pretty fragmented, so we will tend to
        # find blocks of unused IDs that we can instantly pass out.
        self.search_pos = 1

    def clear(self) -> None:
        """Remove all IDs from the manager."""
        self._used = set()
        self.search_pos = 1

    def get_id(self, desired: int = -1) -> int:
        """Get a valid ID."""
        if desired > 0 and desired not in self._used:
            # The desired ID is available!
            self._used.add(desired)
            return desired

        # Check every ID in order to find a valid one.
        poss_id = self.search_pos
        while True:
            if poss_id not in self:
                self._used.add(poss_id)
                self.search_pos = poss_id + 1
                return poss_id
            poss_id += 1

    def __len__(self) -> int:
        return len(self._used)

    def __iter__(self) -> Iterator[int]:
        return iter(self._used)

    def __contains__(self, item: object) -> bool:
        """Check if the given ID is registered."""
        return item in self._used

    def discard(self, element: int) -> None:
        """Return the specified ID for others to use, or do nothing if already removed."""
        self._used.discard(element)
        if element < self.search_pos:
            self.search_pos = element

    def remove(self, element: int) -> None:
        """Return the specified ID for others to use."""
        self._used.remove(element)
        if element < self.search_pos:
            self.search_pos = element


class NullIDMan(IDMan):
    """An alternate ID manager which allows repeated IDs."""
    def get_id(self, desired: int = -1) -> int:
        """Get a valid ID.

        If no desired one is passed, a unique one will be found.
        If a desired ID is set, it will be passed through unchanged.
        """

        if desired == -1:
            return super().get_id()
        else:
            self._used.add(desired)
            return desired


def overlay_bounds(over: 'Entity') -> Tuple[Vec, Vec]:
    """Compute the bounding box of an overlay."""
    origin = Vec.from_str(over['origin'])
    mat = Matrix.from_angle(Angle.from_str(over['angles']))
    return Vec.bbox(
        (origin + Vec.from_str(over['uv' + str(x)]) @ mat)
        for x in
        range(4)
    )


def make_overlay(
    vmf: 'VMF',
    normal: Vec,
    origin: Vec,
    uax: Vec,
    vax: Vec,
    material: str,
    surfaces: Iterable['Side'],
    u_repeat: float = 1,
    v_repeat: float = 1,
    swap: bool = False,
    render_order: int = 0,
) -> 'Entity':
    """Generate an overlay on an axis-aligned surface.

    :param origin: The center point of the overlay.
    :param uax: The direction and distance for the texture's width (``right``).
    :param vax: The direction and distance for the texture's height (``up``).
    :param normal: The normal of the surface.
    :param material: The material used.
    :param u_repeat: Defines how many times to repeat the texture in the U direction.
    :param v_repeat: Defines how many times to repeat the texture in the V direction.
    :param swap: If true, the texture will be rotated ``90``.
    """
    if swap:
        uax, vax = vax, -uax

    u_dist = uax.mag()/2
    v_dist = vax.mag()/2
    basis_u = uax.norm()
    basis_v = vax.norm()

    return vmf.create_ent(
        classname='info_overlay',
        angles='0 0 0',  # Not actually used by VBSP!
        # Ensure it's not exactly on the edge plane.
        origin=(origin + normal).join(' '),
        basisnormal=normal.join(' '),
        basisorigin=origin.join(' '),
        basisu=basis_u.join(' '),
        basisv=basis_v.join(' '),

        material=material,
        sides=' '.join(str(side.id) for side in surfaces),
        renderorder=render_order,

        startu='0',
        startv='0',
        endu=format(u_repeat, 'g'),
        endv=format(v_repeat, 'g'),

        uv0=f'{-u_dist:g} {-v_dist:g} 0',
        uv1=f'{-u_dist:g} {v_dist:g} 0',
        uv2=f'{u_dist:g} {v_dist:g} 0',
        uv3=f'{u_dist:g} {-v_dist:g} 0',
    )


def localise_overlay(over: 'Entity', origin: AnyVec, angles: Union[AnyAngle, AnyMatrix, None] = None) -> None:
    """Rotate an overlay like what is done in instances."""
    orient = to_matrix(angles)
    if angles is not None:  # Skip if known to be identity matrix.
        for key in ('basisNormal', 'basisU', 'basisV'):
            ang = Vec.from_str(over[key]) @ orient
            over[key] = ang.join(' ')

    for key in ('basisOrigin', 'origin'):
        ang = Vec.from_str(over[key]) @ orient
        ang += origin
        over[key] = ang.join(' ')


class CopySet(Set[T]):
    """Modified version of a Set which allows modification during iteration.

    """
    __slots__ = ()  # No extra vars

    def __iter__(self) -> Iterator[T]:
        cur_items: FrozenSet[T] = frozenset(self)

        yield from cur_items
        # after iterating through ourselves, iterate through any new ents.
        yield from self - cur_items


def _remove_copyset(mapping: MutableMapping[T, CopySet['Entity']], key: T, ent: 'Entity') -> None:
    """Remove the entity from the by_class or by_target mappings.

    We also remove the set if now empty.
    """
    copyset = mapping.get(key, None)
    if copyset is not None:
        copyset.discard(ent)
        if not copyset:
            del mapping[key]


class StrataInstanceVisibility(Enum):
    """Strata Source saves and restores the 'view instances' option."""
    HIDDEN = 0  #: Do not preview instances.
    TINTED = 1  #: Show with a green tint
    NORMAL = 2  #: Show normally.


@attrs.frozen
class PrismFace:
    """Return value for VMF.make_prism().

    This can be inded with an axis-aligned :py:class:`~srctools.Vec` or 3-tuple normal to fetch a side.
    """
    solid: 'Solid'  #: The generated brush.
    top: 'Side'  #: The ``+z`` side of the brush.
    bottom: 'Side'  #: The ``-z`` side of the brush.
    north: 'Side'  #: The ``+y`` side of the brush.
    south: 'Side'  #: The ``-y`` side of the brush.
    east: 'Side'  #: The ``+x`` side of the brush.
    west: 'Side'  #: The ``-x`` side of the brush.

    def __getitem__(self, item: AnyVec) -> 'Side':
        """Given an axis-aligned normal, return the matching side."""
        if item == (1, 0, 0):
            return self.east
        elif item == (-1, 0, 0):
            return self.west
        elif item == (0, 1, 0):
            return self.north
        elif item == (0, -1, 0):
            return self.south
        elif item == (0, 0, 1):
            return self.top
        elif item == (0, 0, -1):
            return self.bottom
        else:
            raise KeyError(item)


@attrs.define
class Strata2DViewport:
    """Represents the position of a 2D viewport in Strata Source.

    In the file this is specified as a single vector, with the planar axis set to ±65536.
    """
    axis: Axis
    u: float
    v: float
    zoom: float

    @classmethod
    def from_vector(cls, pos: Vec, zoom: float = 1.0) -> 'Strata2DViewport':
        """Determine the appropriate axis from the position vector."""
        chosen_axis: Optional[Axis] = None
        axis: Axis
        for axis in ('x', 'y', 'z'):
            if pos[axis] in (0.0, -65536.0, 65536.0):
                if chosen_axis is not None:
                    raise ValueError(f'Multiple axes specified for 2D view position "{pos}"!')
                chosen_axis = axis
        if chosen_axis is None:
            raise ValueError(f'No axis for 2D view position "{pos}"!')
        u, v = Vec.INV_AXIS[chosen_axis]
        return cls(chosen_axis, pos[u], pos[v], zoom)

    def export(self, buffer: IO[str], title: str) -> None:
        """Export the 2D viewport definition."""
        buffer.write(f'\t\t{title}\n')
        buffer.write('\t\t{\n')
        buffer.write('\t\t\t"3d" "0"\n')
        if self.axis == 'x':
            buffer.write(f'\t\t\t"position" "(65536 {format_float(self.u)} {format_float(self.v)})"\n')
        elif self.axis == 'y':
            buffer.write(f'\t\t\t"position" "({format_float(self.u)} -65536 {format_float(self.v)})"\n')
        elif self.axis == 'z':
            buffer.write(f'\t\t\t"position" "({format_float(self.u)} {format_float(self.v)} 65536)"\n')
        buffer.write(f'\t\t\t"zoom" "{format_float(self.zoom)}"\n')
        buffer.write('\t\t}\n')


@attrs.define
class Strata3DViewport:
    """Represents the position of a 3D viewport in Strata Source.

    Changing roll does work, but should be avoided since Hammer doesn't allow control of that
    axis.
    """
    position: Vec
    angle: Angle

    def export(self, buffer: IO[str], title: str) -> None:
        """Export the 3D viewport definition."""
        buffer.write(f'\t\t{title}\n')
        buffer.write('\t\t{\n')
        buffer.write('\t\t\t"3d" "1"\n')
        buffer.write(f'\t\t\t"position" "({self.position})"\n')
        buffer.write(f'\t\t\t"angle" "[{self.angle}]"\n')
        buffer.write('\t\t}\n')


def _parse_strata_viewport(kvs: Keyvalues) -> Optional[List[Union[Strata2DViewport, Strata3DViewport]]]:
    """Look for and parse the Strata viewport definitions."""
    try:
        vp_block = kvs.find_key('views')
    except LookupError:
        return None

    ports: List[Union[Strata2DViewport, Strata3DViewport]] = []
    default_2d: Axis

    for key, default_2d in [  # type: ignore[assignment]  # Doesn't infer literal
        ('v0', 'x'),
        ('v1', 'x'),
        ('v2', 'y'),
        ('v3', 'z'),
    ]:
        sub_kv = vp_block.find_key(key, or_blank=True)
        pos = sub_kv.vec('position')
        # Default the upper-left view to 3D.
        if sub_kv.bool('3d', key == 'v0'):
            ports.append(Strata3DViewport(
                pos,
                Angle.from_str(sub_kv['angle', '[0 0 0]']),
            ))
        else:
            zoom = sub_kv.float('zoom', 1.0)
            if pos:
                ports.append(Strata2DViewport.from_vector(pos, zoom))
            else:
                # All zero, use a default axis to produce the default behaviour.
                ports.append(Strata2DViewport(default_2d, 0, 0, zoom))
    return ports


def _mapinfo_int(
    map_info: Mapping[str, str],
    key: str,
    value: Optional[int],
    default: int,
) -> int:
    """Handle reading a value from the map_info dictionary."""
    if value is not None:
        return value
    try:
        val_str = map_info[key]
    except KeyError:
        return default
    warnings.warn(
        'Passing parameters via the map_info dict is deprecated. '
        'Use keyword arguments instead.',
        DeprecationWarning, stacklevel=2,
    )
    return srctools.conv_int(val_str, default)


def _mapinfo_bool(
    map_info: Mapping[str, str],
    key: str,
    value: Optional[bool],
    default: bool,
) -> bool:
    """Handle reading a value from the map_info dictionary."""
    if value is not None:
        return value
    try:
        val_str = map_info[key]
    except KeyError:
        return default
    warnings.warn(
        'Passing parameters via the map_info dict is deprecated. '
        'Use keyword arguments instead.',
        DeprecationWarning, stacklevel=2,
    )
    return srctools.conv_bool(val_str, default)


class VMF:
    """Represents a VMF file, and holds counters for various IDs used.

    Has functions for searching for specific entities or brushes, and
    converts to/from a property_parser tree.

    The dictionaries by_target and by_class allow quickly getting a set
    of entities with the given class or targetname.
    """
    solid_id: IDMan
    face_id: IDMan
    ent_id: IDMan
    group_id: IDMan
    vis_id: IDMan
    node_id: IDMan

    # Allow quick searching for particular groups, without checking
    # the whole map
    by_target: MutableMapping[Optional[str], CopySet['Entity']]
    by_class: MutableMapping[str, CopySet['Entity']]
    entities: List['Entity']
    brushes: List['Solid']
    cameras: List['Camera']
    cordons: List['Cordon']
    vis_tree: List['VisGroup']
    groups: Dict[int, 'EntityGroup']
    spawn: 'Entity'

    is_prefab: bool
    cordon_enabled: bool
    map_ver: int

    format_ver: int
    hammer_ver: int
    hammer_build: int

    # Various Hammer settings
    show_grid: bool
    show_3d_grid: bool
    snap_grid: bool
    show_logic_grid: bool
    grid_spacing: int
    active_cam: int
    quickhide_count: int
    # If None, these are omitted in the exported file.
    strata_instance_vis: Optional[StrataInstanceVisibility]
    strata_viewports: Optional[List[Union[Strata2DViewport, Strata3DViewport]]]

    # Ignore our own deprecation helper function.
    # noinspection PyDeprecation
    def __init__(
        self,
        map_info: Mapping[str, str] = EmptyMapping,
        preserve_ids: bool = False,
        *,
        format_version: Optional[int] = None,
        hammer_version: Optional[int] = None,
        hammer_build: Optional[int] = None,
        is_prefab: Optional[bool] = None,
        cordon_enabled: Optional[bool] = None,
        map_version: Optional[int] = None,
        show_grid: Optional[bool] = None,
        show_3d_grid: Optional[bool] = None,
        snap_grid: Optional[bool] = None,
        show_logic_grid: Optional[bool] = None,
        grid_spacing: Optional[int] = None,
        active_cam: Optional[int] = None,
        quickhide_count: Optional[int] = None,
        strata_inst_visibility: Optional[StrataInstanceVisibility] = None,
    ) -> None:
        """Create a VMF.

        :param preserve_ids: If False (default), various IDs will be changed to ensure they are unique
          when adding items to the VMF. If True the IDs will be preseved. New items will aquire a
          unique ID.
        :param map_info: Deprecated method for passing along the other parameters.

        :param format_version: The version of the VMF format, always 100.
        :param hammer_version: Version number for Hammer.
        :param hammer_build: This encodes the date Hammer was built.
        :param is_prefab: Indicates if this map was saved to be a prefab.
        :param cordon_enabled: Sets whether corndons are currently being used.
        :param map_version: The map version is incremented whenever the map is saved.
        :param show_grid: Whether the grid is visible in the 2D viewports.
        :param show_3d_grid: Whether the grid is visible in the 3D viewport.
        :param snap_grid: Whether objects snap to the selected grid.
        :param show_logic_grid: Unused option.
        :param grid_spacing: The current size of the grid. Should be a power of two.
        :param active_cam: The ID of the currently active camera.
        :param quickhide_count: The number of quick-hidden objects.
        :param strata_inst_visibility: Strata Source stores the visibility of instances.
        """
        id_man = NullIDMan if preserve_ids else IDMan
        self.solid_id = id_man()  # All occupied solid ids
        self.face_id = id_man()  # Ditto for faces
        self.ent_id = id_man()  # Same for entities
        self.group_id = id_man()  # Group IDs (not visgroups)
        self.vis_id = id_man()  # VisGroup IDs
        self.node_id = id_man()  # Nav node ent IDs.

        # Allow quick searching for particular groups, without checking
        # the whole map
        self.by_target: MutableMapping[Optional[str], CopySet[Entity]] = defaultdict(CopySet)
        self.by_class: MutableMapping[str, CopySet[Entity]] = defaultdict(CopySet)

        self.entities = []
        self.brushes = []
        self.cameras = []
        self.cordons = []
        self.vis_tree = []
        self.groups = {}

        # mapspawn entity, which is the entity world brushes are saved to.
        self.spawn = Entity(self)
        self.spawn.solids = self.brushes  # Shared list!

        # The worldspawn entity should always be worldspawn.
        self.spawn['classname'] = 'worldspawn'
        self.by_target[None].add(self.spawn)

        self.is_prefab = _mapinfo_bool(map_info, 'prefab', is_prefab, False)
        self.cordon_enabled = _mapinfo_bool(map_info, 'cordons_on', cordon_enabled, False)
        self.map_ver = _mapinfo_int(map_info, 'mapversion', map_version, 0)
        self.format_ver = _mapinfo_int(map_info,'formatversion', format_version, 100)
        self.hammer_ver = _mapinfo_int(map_info, 'editorversion', hammer_version, CURRENT_HAMMER_VERSION)
        self.hammer_build = _mapinfo_int(map_info, 'editorbuild', hammer_build, CURRENT_HAMMER_BUILD)

        # Various Hammer settings
        self.show_grid = _mapinfo_bool(map_info, 'showgrid', show_grid, True)
        self.show_3d_grid = _mapinfo_bool(map_info, 'show3dgrid', show_3d_grid, False)
        self.snap_grid = _mapinfo_bool(map_info, 'snaptogrid', snap_grid, True)
        self.show_logic_grid = _mapinfo_bool(map_info, 'showlogicalgrid', show_logic_grid, False)
        self.grid_spacing = _mapinfo_int(map_info, 'gridspacing', grid_spacing, 64)
        self.active_cam = _mapinfo_int(map_info, 'active_cam', active_cam, -1)
        self.quickhide_count = _mapinfo_int(map_info, 'quickhide', quickhide_count, 0)

        self.strata_instance_vis = strata_inst_visibility
        self.strata_viewports = None

    def add_brush(self, item: Union['Solid', PrismFace]) -> None:
        """Add a world brush to this map."""
        if isinstance(item, PrismFace):
            item = item.solid
        self.brushes.append(item)

    def remove_brush(self, brush: 'Solid') -> None:
        """Remove a world brush from this map."""
        try:
            self.brushes.remove(brush)
        except ValueError:
            pass  # Already removed.

    def add_ent(self, item: 'Entity') -> None:
        """Add an entity to the map.

        The entity should have been created with this VMF as a parent.
        """
        self.entities.append(item)
        self.by_class[item['classname', ''].casefold()].add(item)
        self.by_target[item['targetname', ''].casefold() or None].add(item)
        if 'nodeid' in item:
            try:
                node_id = int(item['nodeid'])
            except (TypeError, ValueError):
                pass
            else:
                item['nodeid'] = str(self.node_id.get_id(node_id))

    def remove_ent(self, item: 'Entity') -> None:
        """Remove an entity from the map.

        After this is called, the entity will no longer be exported.
        The object still exists, so it can be reused.
        """
        try:
            self.entities.remove(item)
        except ValueError:
            pass  # Already removed.

        _remove_copyset(self.by_class, item['classname'].casefold(), item)
        _remove_copyset(self.by_target, item['targetname'].casefold() or None, item)
        if 'nodeid' in item:
            try:
                node_id = int(item['nodeid'])
            except (TypeError, ValueError):
                pass
            else:
                self.node_id.discard(node_id)

        self.ent_id.discard(item.id)

    def add_brushes(self, brushes: Iterable['Solid']) -> None:
        """Add multiple brushes to the map."""
        self.brushes.extend(brushes)

    def add_ents(self, ents: Iterable['Entity']) -> None:
        """Add multiple entities to the map."""
        ents = list(ents)
        self.entities.extend(ents)
        for item in ents:
            self.by_class[item['classname'].casefold()].add(item)
            self.by_target[item['targetname', ''].casefold() or None].add(item)
            if 'nodeid' in item:
                try:
                    node_id = int(item['nodeid'])
                except (TypeError, ValueError):
                    pass
                else:
                    item['nodeid'] = str(self.node_id.get_id(node_id))

    def create_ent(self, classname: str, **kargs: ValidKVs) -> 'Entity':
        """Convenience method to allow creating point entities.

        This constructs an entity, adds it to the map, and then returns
        it.
        A classname must be passed!
        """
        kargs['classname'] = classname
        ent = Entity(self, keys=kargs)
        self.add_ent(ent)
        return ent

    def create_visgroup(
        self,
        name: str,
        color: Union[Vec, Tuple[int, int, int]] = (255, 255, 255),
    ) -> 'VisGroup':
        """Convenience method for creating visgroups."""
        vis = VisGroup(self, name, -1, Vec(color))
        self.vis_tree.append(vis)
        return vis

    @staticmethod
    def parse(tree: Union[Keyvalues, str], preserve_ids: bool = False) -> 'VMF':
        """Convert a property_parser tree into VMF classes.
        """
        if not isinstance(tree, Keyvalues):
            # if not a tree, try to read the file
            with open(tree, encoding='cp1251') as file:
                tree = Keyvalues.parse(file)

        ver_info = tree.find_block('versioninfo', or_blank=True)

        format_version = ver_info['formatversion', '100']
        if format_version != '100':
            # If the version is different, we're probably about to fail horribly
            raise Exception(f'Unknown VMF format version "{format_version}"!')

        view_opt = tree.find_block('viewsettings', or_blank=True)
        cordons = tree.find_block('cordons', or_blank=True)
        cam_kv = tree.find_block('cameras', or_blank=True)
        quickhide_kv = tree.find_block('quickhide', or_blank=True)

        try:
            inst_vis = StrataInstanceVisibility(int(view_opt['nInstanceVisibility']))
        except LookupError:
            inst_vis = None
        except ValueError:
            inst_vis = StrataInstanceVisibility.TINTED

        # We have to create an incomplete map before parsing any data.
        # This ensures the ID manager objects have been created, so we can
        # ensure unique IDs in brushes, entities and faces.
        map_obj = VMF(
            preserve_ids=preserve_ids,

            format_version=srctools.conv_int(format_version, 100),
            hammer_version=ver_info.int('editorversion', CURRENT_HAMMER_VERSION),
            hammer_build=ver_info.int('editorbuild', CURRENT_HAMMER_BUILD),
            is_prefab=ver_info.bool('prefab'),
            map_version=ver_info.int('mapversion', 0),

            snap_grid=view_opt.bool('bSnapToGrid', True),
            show_grid=view_opt.bool('bShowGrid', True),
            show_3d_grid=view_opt.bool('bShow3DGrid', False),
            show_logic_grid=view_opt.bool('bShowLogicalGrid', False),
            grid_spacing=view_opt.int('nGridSpacing', 64),

            cordon_enabled=cordons.bool('active'),
            active_cam=cam_kv.int('activecamera', -1),
            quickhide_count=quickhide_kv.int('count'),
            strata_inst_visibility=inst_vis,
        )

        map_obj.strata_viewports = _parse_strata_viewport(view_opt)

        for vis in tree.find_all('visgroups', 'visgroup'):
            map_obj.vis_tree.append(VisGroup.parse(map_obj, vis))

        for c in cam_kv:
            if c.name != 'activecamera':
                Camera.parse(map_obj, c)

        for ent in cordons.find_all('cordon'):
            Cordon.parse(map_obj, ent)

        map_spawn = tree.find_block('world', or_blank=True)
        map_obj.spawn = worldspawn = Entity.parse(map_obj, map_spawn, _worldspawn=True)
        # Ensure the correct classname, which adds to by_class as a side effect. It is possible
        # to name worldspawn, kinda pointless though.
        worldspawn['classname'] = 'worldspawn'
        map_obj.by_target[worldspawn['targetname'].casefold() or None].add(worldspawn)
        # Always a brush entity.
        if worldspawn.solids is None:
            worldspawn.solids = []
        map_obj.brushes = worldspawn.solids

        for ent in tree.find_all('Entity'):
            map_obj.add_ent(
                Entity.parse(map_obj, ent, False)  # hidden=False
            )

        # find hidden entities
        for hidden_ent in tree.find_all('hidden'):
            for ent in hidden_ent:
                map_obj.add_ent(
                    Entity.parse(map_obj, ent, True)  # hidden=True
                )

        return map_obj

    @overload
    def export(
        self, *,
        inc_version: bool = True, minimal: bool = False, disp_multiblend: bool = True,
    ) -> str: ...

    @overload
    def export(
        self, dest_file: IO[str], *,
        inc_version: bool = True, minimal: bool = False, disp_multiblend: bool = True,
    ) -> None: ...

    def export(
        self,
        dest_file: Optional[IO[str]] = None, *,
        inc_version: bool = True,
        minimal: bool = False,
        disp_multiblend: bool = True,
    ) -> Optional[str]:
        """Serialises the object's contents into a VMF file.

        - If no file is given the map will be returned as a string.
        - By default, this will increment the map's version - disable inc_version to suppress this.
        - If minimal is True, several blocks will be skipped
          (Viewsettings, cameras, cordons and visgroups)
        - disp_multiblend controls whether displacements produce their multiblend
          data (added in ASW), or if it is stripped.
        - named_cordons controls if multiple cordons may be used (post L4D), or
          if only a single cordon is allowed.
        """
        if dest_file is None:
            string_buf = io.StringIO()
            dest_file = string_buf
        else:
            string_buf = None

        if inc_version:
            # Increment this to indicate the map was modified
            self.map_ver += 1

        dest_file.write('versioninfo\n{\n')
        dest_file.write(f'\t"editorversion" "{self.hammer_ver}"\n')
        dest_file.write(f'\t"editorbuild" "{self.hammer_build}"\n')
        dest_file.write(f'\t"mapversion" "{self.map_ver}"\n')
        dest_file.write(f'\t"formatversion" "{self.format_ver}"\n')
        dest_file.write('\t"prefab" "' +
                        srctools.bool_as_int(self.is_prefab) + '"\n}\n')

        dest_file.write('visgroups\n{\n')
        for vis in self.vis_tree:
            vis.export(dest_file, ind='\t')
        dest_file.write('}\n')

        if not minimal:
            dest_file.write('viewsettings\n{\n')
            dest_file.write('\t"bSnapToGrid" "' +
                            srctools.bool_as_int(self.snap_grid) + '"\n')
            dest_file.write('\t"bShowGrid" "' +
                            srctools.bool_as_int(self.show_grid) + '"\n')
            dest_file.write('\t"bShowLogicalGrid" "' +
                            srctools.bool_as_int(self.show_logic_grid) + '"\n')
            dest_file.write(f'\t"nGridSpacing" "{self.grid_spacing}"\n')
            dest_file.write('\t"bShow3DGrid" "' +
                            srctools.bool_as_int(self.show_3d_grid) + '"\n')
            if self.strata_instance_vis is not None:
                dest_file.write(f'\t"nInstanceVisibility" "{self.strata_instance_vis.value}"\n')
            if self.strata_viewports is not None:
                dest_file.write('\tviews\n\t{\n')
                for name, view in zip(('v0', 'v1', 'v2', 'v3'), self.strata_viewports):
                    view.export(dest_file, name)
                dest_file.write('\t}\n')
            dest_file.write('}\n')

        # The worldspawn version should always match the global value.
        # Also force the classname, since this will crash if it's different.
        self.spawn['mapversion'] = str(self.map_ver)
        self.spawn['classname'] = 'worldspawn'
        self.spawn.export(dest_file, disp_multiblend=disp_multiblend, _is_worldspawn=True)
        del self.spawn['mapversion']

        for ent in self.entities:
            ent.export(dest_file, disp_multiblend=disp_multiblend)

        if not minimal:
            dest_file.write('cameras\n{\n')
            if len(self.cameras) == 0:
                self.active_cam = -1
            dest_file.write(f'\t"activecamera" "{self.active_cam}"\n')
            for cam in self.cameras:
                cam.export(dest_file, '\t')
            dest_file.write('}\n')

            dest_file.write('cordons\n{\n')
            if len(self.cordons) > 0:
                dest_file.write('\t"active" "' +
                                srctools.bool_as_int(self.cordon_enabled) +
                                '"\n')
                for cord in self.cordons:
                    cord.export(dest_file, '\t')
            else:
                dest_file.write('\t"active" "0"\n')
            dest_file.write('}\n')

        if self.quickhide_count > 0:
            dest_file.write(
                'quickhide\n'
                '{\n'
                f'\t"count" "{self.quickhide_count}"\n'
                '}\n'
            )

        if string_buf is not None:
            return string_buf.getvalue()
        else:
            return None

    def iter_wbrushes(self, world: bool = True, detail: bool = True) -> Iterator['Solid']:
        """Iterate through all world and detail solids in the map."""
        if world:
            yield from self.brushes
        if detail:
            for ent in self.by_class['func_detail']:
                yield from ent.solids

    def iter_wfaces(self, world: bool = True, detail: bool = True) -> Iterator['Side']:
        """Iterate through the faces of world and detail solids."""
        for brush in self.iter_wbrushes(world, detail):
            yield from brush

    def iter_ents(self, **cond: str) -> Iterator['Entity']:
        """Iterate through entities having the given keyvalue values."""
        items = cond.items()
        for ent in self.entities[:]:
            for key, value in items:
                if key not in ent or ent[key] != value:
                    break
            else:
                yield ent

    def iter_ents_tags(
        self,
        vals: Mapping[str, str] = EmptyMapping,
        tags: Mapping[str, str] = EmptyMapping,
    ) -> Iterator['Entity']:
        """Iterate through all entities.

        The returned entities must have exactly the given keyvalue values,
        and have keyvalues containing the tags.
        """
        for ent in self.entities[:]:
            for key, value in vals.items():
                if key not in ent or ent[key] != value:
                    break
            else:  # passed through without breaks
                for key, value in tags.items():
                    if key not in ent or value not in ent[key]:
                        break
                else:
                    yield ent

    def iter_inputs(self, name: str) -> Iterator['Output']:
        """Loop through all Outputs which target the named entity.

        - Allows using * at beginning/end
        """
        wild_start = name[:1] == '*'
        wild_end = name[-1:] == '*'
        if wild_start:
            name = name[1:]
        if wild_end:
            name = name[:-1]
        for ent in self.entities:
            for out in ent.outputs:
                if wild_start:
                    if wild_end:
                        if name in out.target:  # blah-target-blah
                            yield out
                    else:
                        if out.target.endswith(name):  # target-blah
                            yield out
                else:
                    if wild_end:
                        if out.target.startswith(name):  # blah-target
                            yield out
                    else:
                        if out.target == name:  # target
                            yield out

    def search(self, name: str) -> Iterator['Entity']:
        """Yield all entities that fit this search string.

        This can be the exact targetname, end-* matching,
        or the exact classname.
        """
        name = name.casefold()
        if not name:
            return

        if name[-1] == '*':
            name = name[:-1]
            for ent_name, ents in self.by_target.items():
                if ent_name is not None and ent_name.casefold().startswith(name):
                    yield from ents
        else:
            for ent_name, ents in self.by_target.items():
                if ent_name is not None and ent_name.casefold() == name:
                    yield from ents

            if name in self.by_class:
                yield from self.by_class[name]

    def make_prism(
        self,
        p1: Union[Vec, FrozenVec],
        p2: Union[Vec, FrozenVec],
        mat: str = 'tools/toolsnodraw',
        set_points: bool = False,
    ) -> PrismFace:
        """Create an axis-aligned brush connecting the two points.

        A PrismFaces tuple will be returned which contains the six
        faces, as well as the solid.
        All faces will be textured with 'mat'.
        If set_points is defined, explicit vertex info will be included.
        """
        b_min, b_max = Vec.bbox(p1, p2)

        # Sanity check - all dimensions must be different, otherwise we'll
        # have an invalid zero-thick brush.
        if b_min.x == b_max.x or b_min.y == b_max.y or b_min.z == b_max.z:
            raise ValueError(f"Zero volume brush requested! ({b_min}, {b_max})")

        f_bottom = Side(
            self,
            [  # -z side
                Vec(b_min.x, b_min.y, b_min.z),
                Vec(b_max.x, b_min.y, b_min.z),
                Vec(b_max.x, b_max.y, b_min.z),
            ],
            mat=mat,
            uaxis=UVAxis(1, 0, 0),
            vaxis=UVAxis(0, -1, 0),
        )
        if set_points:
            f_bottom.strata_points = [
                Vec(b_min.x, b_min.y, b_min.z),
                Vec(b_max.x, b_min.y, b_min.z),
                Vec(b_max.x, b_max.y, b_min.z),
                Vec(b_min.x, b_max.y, b_min.z),
            ]

        f_top = Side(
            self,
            [  # +z side
                Vec(b_min.x, b_max.y, b_max.z),
                Vec(b_max.x, b_max.y, b_max.z),
                Vec(b_max.x, b_min.y, b_max.z),
            ],
            mat=mat,
            uaxis=UVAxis(1, 0, 0),
            vaxis=UVAxis(0, -1, 0),
        )
        if set_points:
            f_top.strata_points = [
                Vec(b_min.x, b_max.y, b_max.z),
                Vec(b_max.x, b_max.y, b_max.z),
                Vec(b_max.x, b_min.y, b_max.z),
                Vec(b_min.x, b_min.y, b_max.z),
            ]

        f_west = Side(
            self,
            [  # -x side
                Vec(b_min.x, b_max.y, b_max.z),
                Vec(b_min.x, b_min.y, b_max.z),
                Vec(b_min.x, b_min.y, b_min.z),
            ],
            mat=mat,
            uaxis=UVAxis(0, 1, 0),
            vaxis=UVAxis(0, 0, -1),
        )
        if set_points:
            f_west.strata_points = [
                Vec(b_min.x, b_max.y, b_max.z),
                Vec(b_min.x, b_min.y, b_max.z),
                Vec(b_min.x, b_min.y, b_min.z),
                Vec(b_min.x, b_max.y, b_min.z),
            ]

        f_east = Side(
            self,
            planes=[  # +x side
                Vec(b_max.x, b_max.y, b_min.z),
                Vec(b_max.x, b_min.y, b_min.z),
                Vec(b_max.x, b_min.y, b_max.z),
            ],
            mat=mat,
            uaxis=UVAxis(0, 1, 0),
            vaxis=UVAxis(0, 0, -1),
        )
        if set_points:
            f_east.strata_points = [
                Vec(b_max.x, b_max.y, b_min.z),
                Vec(b_max.x, b_min.y, b_min.z),
                Vec(b_max.x, b_min.y, b_max.z),
                Vec(b_max.x, b_max.y, b_max.z),
            ]

        f_south = Side(
            self,
            [  # -y side
                Vec(b_max.x, b_min.y, b_min.z),
                Vec(b_min.x, b_min.y, b_min.z),
                Vec(b_min.x, b_min.y, b_max.z),
            ],
            mat=mat,
            uaxis=UVAxis(1, 0, 0),
            vaxis=UVAxis(0, 0, -1),
        )
        if set_points:
            f_south.strata_points = [
                Vec(b_max.x, b_min.y, b_min.z),
                Vec(b_min.x, b_min.y, b_min.z),
                Vec(b_min.x, b_min.y, b_max.z),
                Vec(b_max.x, b_min.y, b_max.z),
            ]

        f_north = Side(
            self,
            [  # +y side
                Vec(b_min.x, b_max.y, b_min.z),
                Vec(b_max.x, b_max.y, b_min.z),
                Vec(b_max.x, b_max.y, b_max.z),
            ],
            mat=mat,
            uaxis=UVAxis(1, 0, 0),
            vaxis=UVAxis(0, 0, -1),
        )
        if set_points:
            f_north.strata_points = [
                Vec(b_min.x, b_max.y, b_min.z),
                Vec(b_max.x, b_max.y, b_min.z),
                Vec(b_max.x, b_max.y, b_max.z),
                Vec(b_min.x, b_max.y, b_max.z),
            ]

        solid = Solid(
            self,
            sides=[
                f_bottom,
                f_top,
                f_north,
                f_south,
                f_east,
                f_west,
            ],
        )
        return PrismFace(
            solid=solid,
            top=f_top,
            bottom=f_bottom,
            north=f_north,
            south=f_south,
            east=f_east,
            west=f_west,
        )

    def make_hollow(
        self,
        p1: Union[Vec, FrozenVec],
        p2: Union[Vec, FrozenVec],
        thick: float = 16,
        mat: str = 'tools/toolsnodraw',
        inner_mat: str = '',
    ) -> List['Solid']:
        """Create 6 brushes to surround the given region.

        If inner_mat is not specified, it's set to mat.
        """
        if not inner_mat:
            inner_mat = mat
        b_min, b_max = Vec.bbox(p1, p2)

        top = self.make_prism(
            Vec(b_min.x, b_min.y, b_max.z),
            Vec(b_max.x, b_max.y, b_max.z + thick),
            mat,
        )

        bottom = self.make_prism(
            Vec(b_min.x, b_min.y, b_min.z),
            Vec(b_max.x, b_max.y, b_min.z - thick),
            mat,
        )

        west = self.make_prism(
            Vec(b_min.x - thick, b_min.y, b_min.z),
            Vec(b_min.x, b_max.y, b_max.z),
            mat,
        )

        east = self.make_prism(
            Vec(b_max.x, b_min.y, b_min.z),
            Vec(b_max.x + thick, b_max.y, b_max.z),
            mat
        )

        north = self.make_prism(
            Vec(b_min.x, b_max.y, b_min.z),
            Vec(b_max.x, b_max.y + thick, b_max.z),
            mat,
        )

        south = self.make_prism(
            Vec(b_min.x, b_min.y - thick, b_min.z),
            Vec(b_max.x, b_min.y, b_max.z),
            mat,
        )

        top.bottom.mat = bottom.top.mat = inner_mat
        east.west.mat = west.east.mat = inner_mat
        north.south.mat = south.north.mat = inner_mat

        return [
            north.solid, south.solid,
            east.solid, west.solid,
            top.solid, bottom.solid,
        ]


class Camera:
    """Represents one of several cameras which can be swapped between."""
    pos: Vec
    target: Vec
    map: VMF
    def __init__(self, vmf_file: VMF, pos: Vec, targ: Vec) -> None:
        self.pos = pos
        self.target = targ
        self.map = vmf_file
        vmf_file.cameras.append(self)

    def targ_ent(self, ent: 'Entity') -> None:
        """Point the camera at an entity."""
        if ent['origin']:
            self.target = Vec.from_str(ent['origin'])

    def is_active(self) -> bool:
        """Is this camera in use?"""
        return self.map.active_cam == self.map.cameras.index(self) + 1

    def set_active(self) -> None:
        """Set this to be the map's active camera"""
        self.map.active_cam = self.map.cameras.index(self) + 1

    def set_inactive_all(self) -> None:
        """Disable all cameras in this map."""
        self.map.active_cam = -1

    @classmethod
    def parse(cls, vmf_file: VMF, tree: Keyvalues) -> 'Camera':
        """Read a camera from a property_parser tree."""
        pos = tree.vec('position')
        targ = tree.vec('look', 0.0, 64.0, 0.0)
        return cls(vmf_file, pos, targ)

    def copy(self) -> 'Camera':
        """Duplicate this camera object."""
        return Camera(self.map, self.pos.copy(), self.target.copy())

    def remove(self) -> None:
        """Delete this camera from the map."""
        self.map.cameras.remove(self)
        if self.is_active():
            self.set_inactive_all()

    def export(self, buffer: IO[str], ind: str = '') -> None:
        """Export the camera to the VMF file."""
        buffer.write(ind + 'camera\n')
        buffer.write(ind + '{\n')
        buffer.write(f'{ind}\t"position" "[{self.pos}]"\n')
        buffer.write(f'{ind}\t"look" "[{self.target}]"\n')
        buffer.write(ind + '}\n')


class Cordon:
    """Represents one cordon volume."""
    def __init__(
        self,
        vmf_file: VMF,
        min_: Vec,
        max_: Vec,
        is_active: bool = True,
        name: str = 'Cordon',
    ) -> None:
        self.map = vmf_file
        self.name = name
        self.bounds_min = min_
        self.bounds_max = max_
        self.active = is_active
        vmf_file.cordons.append(self)

    @classmethod
    def parse(cls, vmf_file: VMF, tree: Keyvalues) -> 'Cordon':
        """Parse a cordon from the VMF file."""
        name = tree['name', 'cordon']
        is_active = tree.bool('active', False)
        bounds = tree.find_block('box', or_blank=True)
        min_ = bounds.vec('mins', 0, 0, 0)
        max_ = bounds.vec('maxs', 128, 128, 128)
        return Cordon(vmf_file, min_, max_, is_active, name)

    def export(self, buffer: IO[str], ind: str = '') -> None:
        """Write the cordon into the VMF."""
        buffer.write(f'{ind}cordon\n')
        buffer.write(f'{ind}{{\n')
        buffer.write(f'{ind}\t\"name\" \"{escape_text(self.name)}\"\n')
        buffer.write(f'{ind}\t\"active\" \"{srctools.bool_as_int(self.active)}\"\n')
        buffer.write(f'{ind}\tbox\n')
        buffer.write(f'{ind}\t{{\n')
        buffer.write(f'{ind}\t\t\"mins\" \"({self.bounds_min})\"\n')
        buffer.write(f'{ind}\t\t\"maxs\" \"({self.bounds_max})\"\n')
        buffer.write(f'{ind}\t}}\n')
        buffer.write(f'{ind}}}\n')

    def copy(self) -> 'Cordon':
        """Duplicate this cordon."""
        return Cordon(
            self.map,
            self.bounds_min.copy(),
            self.bounds_max.copy(),
            self.active,
            self.name,
        )

    def remove(self) -> None:
        """Remove this cordon from the map."""
        self.map.cordons.remove(self)


@attrs.define(auto_attribs=True, hash=False, eq=False, order=False, getstate_setstate=True)
class VisGroup:
    """Defines one visgroup."""
    vmf: VMF
    name: str
    id: int = attrs.field(default=-1)
    color: Vec = attrs.field(factory=lambda: Vec(255, 255, 255))
    child_groups: List['VisGroup'] = attrs.field(factory=list)

    def __attrs_post_init__(self) -> None:
        self.id = self.vmf.vis_id.get_id(self.id)

    @classmethod
    def parse(cls, vmf: VMF, props: Keyvalues) -> 'VisGroup':
        """Parse a visgroup from the VMF file."""
        vis_id = props.int('visgroupid', -1)
        name = props['name', f'VisGroup_{vis_id}']
        color = props.vec('color', 255, 255, 255)

        children = [
            cls.parse(vmf, child)
            for child in
            props.find_all('visgroup')
        ]

        return cls(
            vmf,
            name,
            vis_id,
            color,
            children,
        )

    def export(self, buffer: IO[str], ind: str = '') -> None:
        """Write out the VMF text for a visgroup"""
        buffer.write(
            f'{ind}visgroup\n'
            f'{ind}{{\n'
            f'{ind}\t"name" "{escape_text(self.name)}"\n'
            f'{ind}\t"visgroupid" "{self.id}"\n'
            f'{ind}\t"color" "{self.color}"\n'
        )
        for child in self.child_groups:
            child.export(buffer, ind + '\t')
        buffer.write(ind + '}\n')

    def set_visible(self, target: bool) -> None:
        """Find all objects with this ID, and set them to the given visibility."""
        hidden = not target
        for ent in self.child_ents():
            ent.vis_shown = target
            ent.hidden = hidden
            for solid in ent.solids:
                solid.vis_shown = target
                solid.hidden = hidden

        for solid in self.child_solids():
            solid.vis_shown = solid.hidden = target
            solid.hidden = hidden

    def child_ents(self) -> Iterator['Entity']:
        """Yields Entities in this visgroup."""
        for ent in self.vmf.entities:
            if self.id in ent.visgroup_ids:
                yield ent

    def child_solids(self) -> Iterator['Solid']:
        """Yields Solids in this visgroup."""
        for solid in self.vmf.brushes:
            if self.id in solid.visgroup_ids:
                yield solid

    def copy(
        self,
        vmf: Optional[VMF] = None,
        group_mapping: MutableMapping[int, int] = EmptyMapping,
        des_id: int = -1,
    ) -> 'VisGroup':
        """Duplicate this visgroup and all children."""
        newgroup = VisGroup(
            vmf or self.vmf,
            self.name,
            des_id,
            self.color.copy(),
            [
                child.copy(vmf, group_mapping)
                for child in self.child_groups
            ],
        )
        group_mapping[self.id] = newgroup.id
        return newgroup


# Workaround MyPy not evaluating generics on converters.
if TYPE_CHECKING:
    def _conv_visgroups(x: Iterable[int]) -> Set[int]: return set(x)
else:
    _conv_visgroups = set


@attrs.define(auto_attribs=True, hash=False, eq=False, order=False, getstate_setstate=True)
class Solid:
    """A single brush, serving as both world brushes and brush entities."""
    map: VMF
    #: A unique ID assigned to this solid. Will be picked automatically when the brush is created.
    id: int = attrs.field(default=-1)
    #: The faces for this brush.
    sides: List['Side'] = attrs.field(factory=list)
    #: The set of IDs of user-authored visgroups this brush belongs to.
    visgroup_ids: Set[int] = attrs.field(default=(), converter=_conv_visgroups)  # pyright: ignore
    #: If set, this brush has been hidden in some way (quick hide, cordons, visgroups). It will
    #: not be compiled into the final map.
    hidden: bool = False
    #: If set, this brush is grouped in the group with this ID.
    group_id: Optional[int] = None
    #: Determines whether this brush has been hidden via a user-authored visgroup.
    vis_shown: bool = True
    #: Determines whether this brush has been hidden via a builtin "Auto" visgroup.
    vis_auto_shown: bool = True
    #: If set, this brush has been generated by Hammer to seal cordoned areas. These are deleted
    #: when loading the
    is_cordon: bool = False
    #: The RGB colour this brush appears as in 2D views. Randomly assigned when the brush is
    #: created, but then set to the colour of the tied entity or visgroup.
    editor_color: Vec = attrs.field(factory=lambda: Vec(255, 255, 255))

    def __attrs_post_init__(self) -> None:
        self.id = self.map.solid_id.get_id(self.id)

    def copy(
        self,
        des_id: int = -1,
        vmf_file: Optional[VMF] = None,
        side_mapping: MutableMapping[int, int] = EmptyMapping,
        keep_vis: bool = True,
    ) -> 'Solid':
        """Duplicate this brush."""
        sides = [
            s.copy(-1, vmf_file, side_mapping)
            for s in
            self.sides
        ]

        return Solid(
            vmf_file or self.map,
            des_id,
            sides,
            self.visgroup_ids if keep_vis else set(),
            self.hidden if keep_vis else False,
            self.group_id,
            self.vis_shown if keep_vis else True,
            self.vis_auto_shown if keep_vis else True,
            self.is_cordon,
            self.editor_color,
        )

    @classmethod
    def parse(cls, vmf_file: VMF, tree: Keyvalues, hidden: bool = False) -> 'Solid':
        """Parse a Property tree into a Solid object."""
        solid_id = tree.int('id', -1)
        sides = []
        for side in tree.find_all("side"):
            sides.append(Side.parse(vmf_file, side))

        visgroups = set()
        group_id = None
        vis_shown = vis_auto_shown = True
        is_cordon = False
        editor_color = Vec(255, 255, 255)

        for v in tree.find_children("editor"):
            if v.name == "visgroupshown":
                vis_shown = srctools.conv_bool(v.value, default=True)
            elif v.name == "visgroupautoshown":
                vis_auto_shown = srctools.conv_bool(v.value, default=True)
            elif v.name == "cordonsolid":
                # This is a little odd, any value is treated as 'true', despite being a
                # boolean.
                is_cordon = True
            elif v.name == 'color':
                editor_color = Vec.from_str(v.value, 255, 255, 255)
            elif v.name == 'group':
                group_id = int(v.value)
            elif v.name == 'visgroupid':
                try:
                    visgroups.add(int(v.value))
                except (ValueError, TypeError):
                    pass

        return cls(
            vmf_file,
            solid_id,
            sides,
            visgroups,
            hidden,
            group_id,
            vis_shown,
            vis_auto_shown,
            is_cordon,
            editor_color,
        )

    def export(
        self,
        buffer: IO[str],
        ind: str = '',
        disp_multiblend: bool = True,
        include_groups: bool = True,
    ) -> None:
        """Generate the strings needed to define this brush.

        - disp_multiblend controls whether displacements produce their multiblend
          data (added in ASW), or if it is stripped.
        - include_groups specifies if visgroup/group values are included. This is not allowed
          inside brush entities.
        """
        if self.hidden:
            buffer.write(f'{ind}hidden\n{ind}{{\n')
            ind += '\t'
        buffer.write(f'{ind}solid\n')
        buffer.write(f'{ind}{{\n')
        buffer.write(f'{ind}\t"id" "{self.id}"\n')
        for s in self.sides:
            s.export(buffer, f'{ind}\t', disp_multiblend)

        buffer.write(f'{ind}\teditor\n')
        buffer.write(f'{ind}\t{{\n')
        buffer.write(f'{ind}\t\t"color" "{self.editor_color}"\n')
        if include_groups:
            if self.group_id is not None:
                buffer.write(f'{ind}\t\t"groupid" "{self.group_id}"\n')

            for group in self.visgroup_ids:
                buffer.write(f'{ind}\t\t"visgroupid" "{group}"\n')

        buffer.write(f'{ind}\t\t"visgroupshown" "{"1" if self.vis_shown else "0"}"\n')
        buffer.write(f'{ind}\t\t"visgroupautoshown" "{"1" if self.vis_auto_shown else "0"}"\n')
        if self.is_cordon:
            # If present this is always treated as true.
            buffer.write(f'{ind}\t\t"cordonsolid" "1"\n')

        buffer.write(f'{ind}\t}}\n')

        buffer.write(f'{ind}}}\n')
        if self.hidden:
            buffer.write(f'{ind[:-1]}}}\n')

    def __str__(self) -> str:
        """Return a user-friendly description of our data."""
        return ''.join([f'<solid:{self.id}>\n{{', *map(str, self.sides), '}'])

    def __iter__(self) -> Iterator['Side']:
        return iter(self.sides)

    def __del__(self) -> None:
        """Forget this solid's ID when the object is destroyed."""
        self.map.solid_id.discard(self.id)

    def remove(self) -> None:
        """Remove this brush from the map."""
        self.map.remove_brush(self)

    def get_bbox(self) -> Tuple[Vec, Vec]:
        """Get two vectors representing the space this brush takes up."""
        bbox_min, bbox_max = self.sides[0].get_bbox()
        for s in self.sides[1:]:
            side_min, side_max = s.get_bbox()
            bbox_max.max(side_max)
            bbox_min.min(side_min)
        return bbox_min, bbox_max

    def get_origin(self, bbox_min: Optional[Vec] = None, bbox_max: Optional[Vec] = None) -> Vec:
        """Calculates a vector representing the exact center of this brush."""
        if bbox_min is None or bbox_max is None:
            bbox_min, bbox_max = self.get_bbox()
        return (bbox_min + bbox_max) / 2

    def translate(self, diff: AnyVec) -> None:
        """Move this solid by the specified vector."""
        for s in self.sides:
            s.translate(diff)

    def localise(self, origin: AnyVec, angles: Union[AnyAngle, AnyMatrix, None] = None) -> None:
        """Shift this brush by the given origin/angles."""
        angles = to_matrix(angles)  # Only do this once.
        for s in self.sides:
            s.localise(origin, angles)

    def point_inside(self, point: AnyVec, threshold: float = 1e-6) -> bool:
        """Check if the specified point is inside the brush.

        The threshold controls tolerance - a point is still counted if it is this far away from
        the surface.
        """
        for side in self.sides:
            offset = side.normal().dot(side.planes[1] - point)
            if offset > threshold:
                return False
        return True

    @property
    @deprecated('Use is_cordon instead', category=DeprecationWarning)
    def cordon_solid(self) -> Optional[int]:
        """Deprecated old version of :py:attr:`is_cordon`."""
        return 1 if self.is_cordon else None

    # noinspection PyDeprecation
    @cordon_solid.setter
    @deprecated('Use is_cordon instead', category=DeprecationWarning)
    def cordon_solid(self, value: Optional[int]) -> None:
        """Deprected old version of :py:attr:`is_cordon`."""
        self.is_cordon = bool(value)


@attrs.define(frozen=True, hash=True, order=True)
class Vec4:
    """Defines a 4-dimensional vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 0.0

    def __str__(self) -> str:
        return f'{self.x:g} {self.y:g} {self.z:g} {self.w:g}'

    def __bool__(self) -> bool:
        return bool(self.x or self.y or self.z or self.w)


def _list4_validator(_: object, attrib: 'attrs.Attribute[List[Vec]]', value: object) -> None:
    """Validate the value is a list of 4 elements."""
    if not isinstance(value, list):
        raise TypeError(attrib.name + ' should be a list!')
    if len(value) != 4:
        raise ValueError(attrib.name + ' must have 4 values!')


@attrs.define(frozen=False)
class DispVertex:
    """A vertex in dislacements."""
    x: int  # Position of the vertex in the displacement.
    y: int

    normal: Vec = attrs.field(factory=Vec, validator=attrs.validators.instance_of(Vec))
    distance: float = 0
    offset: Vec = attrs.field(factory=Vec, validator=attrs.validators.instance_of(Vec))
    offset_norm: Vec = attrs.field(factory=Vec, validator=attrs.validators.instance_of(Vec))
    alpha: float = 0.0
    # The pair of triangle tags for the quad in the +ve direction
    # from us. This means the last row/column's triangles are ignored.
    triangle_a: TriangleTag = TriangleTag.FLAT
    triangle_b: TriangleTag = TriangleTag.FLAT

    # These are for multiblend displacements, added in ASW+.
    multi_blend: Vec4 = Vec4()
    multi_alpha: Vec4 = Vec4()
    multi_colors: Optional[List[Vec]] = attrs.field(default=None, validator=attrs.validators.optional(
        attrs.validators.deep_iterable(attrs.validators.instance_of(Vec), _list4_validator)
    ))


@attrs.define(eq=False)
class UVAxis:
    """Values saved into Side.uaxis and Side.vaxis.

    These define the alignment of textures on a face.
    """
    x: float = attrs.field(repr=format_float)
    y: float = attrs.field(repr=format_float)
    z: float = attrs.field(repr=format_float)
    offset: float = attrs.field(repr=format_float, default=0.0)
    scale: float = attrs.field(repr=format_float, default=0.25)

    @classmethod
    def parse(cls, value: str) -> 'UVAxis':
        """Parse a UV axis from a string."""
        vals = value.split()
        return cls(
            x=float(vals[0].lstrip('[')),
            y=float(vals[1]),
            z=float(vals[2]),
            offset=float(vals[3].rstrip(']')),
            scale=float(vals[4]),
        )

    def copy(self) -> 'UVAxis':
        """Return a duplicate of this axis."""
        return UVAxis(
            x=self.x,
            y=self.y,
            z=self.z,
            offset=self.offset,
            scale=self.scale,
        )

    def vec(self) -> Vec:
        """Return the axis as a vector."""
        return Vec(self.x, self.y, self.z)

    def rotate(self, angles: Union[AnyAngle, AnyMatrix]) -> 'UVAxis':
        """Rotate the axis by some orientation.

        This doesn't alter texture offsets.
        """
        vec = self.vec() @ angles
        return UVAxis(
            vec.x,
            vec.y,
            vec.z,
            self.offset,
            self.scale,
        )

    def localise(self, origin: AnyVec, angles: Union[AnyAngle, AnyMatrix]) -> 'UVAxis':
        """Rotate and translate the texture coordinates."""
        vec = self.vec() @ angles

        # Fix offset - see source-sdk: utils/vbsp/map.cpp line 2237
        offset = self.offset - vec.dot(origin) / self.scale

        return UVAxis(
            vec.x,
            vec.y,
            vec.z,
            offset,
            self.scale,
        )

    def __str__(self) -> str:
        """Generate the text form for this UV data."""
        return (
            f'[{format_float(self.x)} {format_float(self.y)} {format_float(self.z)} '
            f'{format_float(self.offset)}] {format_float(self.scale)}'
        )


DispPower: TypeAlias = Literal[0, 1, 2, 3, 4]


class _FloatSetter:
    """Helps type checkers know that UVAxis.offset/scale are settable only.

    TODO: Can be removed if/when generic properties are available, perhaps.
    """
    def __set__(self, instance: object, value: float) -> None:
        ...


class Side:
    """A brush face."""
    __slots__ = [
        'map',
        'planes',
        'id',
        'lightmap',
        'smooth',
        'mat',
        'ham_rot',
        'uaxis',
        'vaxis',
        'strata_points',
        'disp_power',
        'disp_pos',
        'disp_elevation',
        'disp_flags',
        'disp_allowed_vert',
        '_disp_verts',
    ]
    map: VMF
    planes: List[Vec]
    id: int
    lightmap: int
    smooth: int
    mat: str
    ham_rot: float
    uaxis: UVAxis
    vaxis: UVAxis
    strata_points: Optional[List[Vec]]

    disp_power: DispPower
    disp_pos: Optional[Vec]
    disp_elevation: float
    disp_flags: DispFlag
    disp_allowed_vert: Optional['Array[int]']
    _disp_verts: Optional[List[DispVertex]]

    def __init__(
        self,
        vmf_file: VMF,
        planes: List[Vec],
        des_id: int = -1,
        lightmap: int = 16,
        smoothing: int = 0,
        mat: str = 'tools/toolsnodraw',
        rotation: float = 0,
        uaxis: Optional[UVAxis] = None,
        vaxis: Optional[UVAxis] = None,
        disp_power: DispPower = 0,
    ) -> None:
        """Planes must be a list of 3 Vecs or 3-tuples."""
        self.map = vmf_file
        if len(planes) != 3:
            raise ValueError('Must have only 3 planes!')
        self.planes = planes
        self.id = vmf_file.face_id.get_id(des_id)
        self.lightmap = lightmap
        self.smooth = smoothing
        self.mat = mat
        self.ham_rot = rotation
        self.uaxis = uaxis or UVAxis(0, 1, 0)
        self.vaxis = vaxis or UVAxis(0, 0, -1)
        self.strata_points = None

        self.disp_power = disp_power
        self.disp_flags = DispFlag.COLL_ALL
        self.disp_elevation = 0.0
        if disp_power > 0:
            self._disp_verts = [
                DispVertex(x, y)
                for y in range(self.disp_size)
                for x in range(self.disp_size)
            ]
            self.disp_pos = Vec()
            self.disp_allowed_vert = Array('i', (-1, ) * 10)
        else:
            self._disp_verts = self.disp_pos = self.disp_allowed_vert = None

    @property
    def is_disp(self) -> bool:
        """Returns whether this is a displacement or not."""
        return self.disp_power > 0

    @property
    def disp_size(self) -> int:
        """Return the number of vertexes in each direction of a displacement."""
        if self.disp_power == 0:
            return 0
        return 2 ** self.disp_power + 1

    @classmethod
    def parse(cls, vmf_file: VMF, tree: Keyvalues) -> 'Side':
        """Parse the property tree into a Side object."""
        # planes = "(x1 y1 z1) (x2 y2 z2) (x3 y3 z3)"
        verts = tree["plane", "(0 0 0) (0 0 0) (0 0 0)"][1:-1].split(") (")
        if len(verts) != 3:
            raise ValueError(f'Wrong number of solid planes in "{tree["plane", ""]}"')
        planes = [
            Vec.from_str(verts[0]),
            Vec.from_str(verts[1]),
            Vec.from_str(verts[2]),
        ]

        side: Side = cls(
            vmf_file,
            planes,
            tree.int('id', -1),
            tree.int('lightmapscale', 16),
            tree.int('smoothing_groups'),
            tree['material', ''],
            tree.float('rotation'),
            UVAxis.parse(tree['uaxis', '[0 1 0 0] 0.25']),
            UVAxis.parse(tree['vaxis', '[0 0 -1 0] 0.25']),
        )

        try:
            disp_tree = tree.find_key('dispinfo')
        except LookupError:  # Not a displacement.
            pass
        else:
            side._parse_displacement_data(disp_tree)

        try:
            points_block = tree.find_block('point_data')
        except LookupError:
            pass
        else:
            side._parse_strata_points(points_block)
        return side

    def _parse_displacement_data(self, disp_tree: Keyvalues) -> None:
        """Parse displacement data. This is less common, split into its own method."""

        # Deal with displacements.
        disp_power = disp_tree.int('power', 4)
        if disp_power in (0, 1, 2, 3, 4):
            self.disp_power = disp_power  # type: ignore
        else:
            raise ValueError(f'Invalid displacement power {disp_power}!')
        self.disp_pos = disp_tree.vec('startposition')
        self.disp_elevation = disp_tree.float('elevation')
        disp_flag_ind = disp_tree.int('flags')
        if 0 <= disp_flag_ind <= 16:
            self.disp_flags = _DISP_FLAG_TO_COLL[disp_flag_ind]
        else:
            raise ValueError(f'Invalid displacement flags {disp_flag_ind} in side {self.id}!')
        if disp_tree.bool('subdiv'):
            self.disp_flags |= DispFlag.SUBDIV

        # This is 10 int32 numbers. Strata Source alternately uses 5 int64s.
        vert_key = disp_tree.find_key('allowed_verts')
        allowed_vert = Array('i')
        if '10' in vert_key:
            allowed_vert.extend(map(int, vert_key['10'].split()))
        elif '5' in vert_key:
            # Pull out the two 32-bit ints from a 64-bit int.
            allowed_vert.extend(
                val
                for num in vert_key['5'].split()
                for val in struct.unpack('<ii', struct.pack('Q', int(num)))
            )
        if len(allowed_vert) != 10:
            raise ValueError(
                f'Displacement allowed_verts in side {self.id} '
                f'must be 10 long!'
            )
        self.disp_allowed_vert = allowed_vert

        size = self.disp_size
        self._disp_verts = [
            DispVertex(x, y)
            for y in range(size)
            for x in range(size)
        ]
        # Parse all the rows...
        self._parse_disp_vecrow(disp_tree, 'normals', _disprow_set_norm)
        self._parse_disp_vecrow(disp_tree, 'offsets', _disprow_set_off)
        self._parse_disp_vecrow(disp_tree, 'offset_normals', _disprow_set_off_norm)

        for y, row in self._iter_disp_row(disp_tree, 'alphas', size):
            try:
                for x, alpha in enumerate(row):
                    self._disp_verts[y * size + x].alpha = float(alpha)
            except ValueError as exc:
                raise ValueError(
                    f'Displacement array for alpha in side {self.id}, '
                    f'row {y} had invalid number: {exc.args[0]}'
                ) from None

        for y, row in self._iter_disp_row(disp_tree, 'distances', size):
            try:
                for x, alpha in enumerate(row):
                    self._disp_verts[y * size + x].distance = float(alpha)
            except ValueError as exc:
                raise ValueError(
                    f'Displacement array for distances in side {self.id}, '
                    f'row {y} had invalid number: {exc.args[0]}'
                ) from None

        # Not the same, 1 less row and column since it's per-quad.
        tri_tags_count = 2 ** disp_power
        for y, row in self._iter_disp_row(disp_tree, 'triangle_tags', 2 * tri_tags_count):
            try:
                for x in range(tri_tags_count):
                    vert = self._disp_verts[y * size + x]
                    vert.triangle_a = TriangleTag(int(row[2 * x]))
                    vert.triangle_b = TriangleTag(int(row[2 * x + 1]))
            except ValueError as exc:
                raise ValueError(
                    f'Displacement array for triangle tags in side {self.id}, '
                    f'row {y} had invalid number: {exc.args[0]}'
                ) from None

        if 'multiblend' not in disp_tree:
            return
        # Else: Parse multiblend too.
        # First initialise this list.
        for vert in self._disp_verts:
            vert.multi_colors = [
                Vec(1, 1, 1), Vec(1, 1, 1),
                Vec(1, 1, 1), Vec(1, 1, 1),
            ]
        for name, setter in _disprow_multiblend:
            self._parse_disp_vecrow(disp_tree, name, setter)

        for y, split in self._iter_disp_row(disp_tree, 'multiblend', 4 * size):
            try:
                for x in range(size):
                    self._disp_verts[y * size + x].multi_blend = Vec4(
                        float(split[4*x]),
                        float(split[4*x + 1]),
                        float(split[4*x + 2]),
                        float(split[4*x + 3]),
                    )
            except ValueError as exc:
                raise ValueError(
                    f'Displacement array for multiblend in side {self.id}, '
                    f'row {y} had invalid number: {exc.args[0]}'
                ) from None

        for y, split in self._iter_disp_row(disp_tree, 'alphablend', 4 * size):
            try:
                for x in range(size):
                    self._disp_verts[y * size + x].multi_alpha = Vec4(
                        float(split[4*x]),
                        float(split[4*x + 1]),
                        float(split[4*x + 2]),
                        float(split[4*x + 3]),
                    )
            except ValueError as exc:
                raise ValueError(
                    f'Displacement array for multiblend in side {self.id}, '
                    f'row {y} had invalid number: {exc.args[0]}'
                ) from None

    def _iter_disp_row(self, tree: Keyvalues, name: str, size: int) -> Iterator[Tuple[int, List[str]]]:
        """Return y, row pairs of values from a displacement array row.

        It verifies the row is `size` long.
        """
        for row_prop in tree.find_children(name):
            if row_prop.name.startswith('row'):
                y = int(row_prop.name[3:])
            else:
                continue  # Ignore unknown keys.
            split = row_prop.value.split()
            if len(split) != size:
                raise ValueError(
                    f'Displacement array for {name} in side {self.id}, '
                    f'row {y} must have a length of '
                    f'{size}, not {len(split)}!'
                )
            yield y, split

    def _parse_disp_vecrow(self, tree: Keyvalues, name: str, setter: Callable[[DispVertex, Vec], None]) -> None:
        """Parse one of the very similar per-vert sections.

        If member is a string, it is an attribute to set on the DispVertex.
        Otherwise, it's an int specifying the channel for multi_colors (which must
        have been initialised beforehand).
        """
        assert self._disp_verts is not None
        size = self.disp_size
        for y, split in self._iter_disp_row(tree, name, 3 * size):
            try:
                for x in range(size):
                    res = Vec(float(split[3 * x]), float(split[3 * x + 1]), float(split[3 * x + 2]))
                    setter(self._disp_verts[y * size + x], res)
            except ValueError as exc:
                raise ValueError(
                    f'Displacement array for {name} in side {self.id}, '
                    f'row {y} had invalid number: {exc.args[0]}'
                ) from None

    def _parse_strata_points(self, block: Keyvalues) -> None:
        """Parse Strata Source's additional vertices block."""
        points: List[Optional[Vec]] = [None] * block.int('numpts')
        for child in block:
            if child.name == 'point':
                ind_str, pos_str = child.value.split(' ', 1)
                point = Vec.from_str(pos_str)
                try:
                    ind = int(ind_str)
                    existing = points[ind]
                except (ValueError, IndexError) as exc:
                    raise ValueError(
                        f'Invalid points value "{child.value}" '
                        f'for array of length {len(points)} in face #{self.id}!'
                    ) from exc
                if existing is not None:
                    raise ValueError(f'Duplicate points index "{ind}" in face #{self.id}!')
                points[ind] = point
        if None in points:
            raise ValueError(f'Missing points in array for face #{self.id}: {points}')
        self.strata_points = points  # type: ignore[assignment]  # We checked for None.

    @classmethod
    def from_plane(
        cls,
        vmf: VMF,
        position: AnyVec,
        normal: AnyVec,
        material: str = 'tools/toolsnodraw',
    ) -> 'Side':
        """Generate a new brush face, aligned to the specified plane.

        The normal vector points out of the face. This calculates a valid texture alignment, but
        does not specify an exact result.
        """
        orient = Matrix.from_basis(x=FrozenVec(normal))
        point = Vec(position)
        u = -orient.left()
        v = -orient.up()
        return cls(
            vmf,
            [
                point - 16 * u,
                point,
                point + 16 * v,
            ],
            mat=material,
            uaxis=UVAxis(*u),
            vaxis=UVAxis(*v),
        )

    def copy(
        self,
        des_id: int = -1,
        vmf_file: Optional[VMF] = None,
        side_mapping: MutableMapping[int, int] = EmptyMapping,
    ) -> 'Side':
        """Duplicate this brush side.

        des_id is the id which is desired for the new side.
        map is the VMF to add the new side to (defaults to the same map).
        If passed, side_mapping will be updated with an old -> new ID pair.
        """
        if vmf_file is not None and des_id == -1:
            des_id = self.id

        new_side = Side(
            vmf_file or self.map,
            [p.copy() for p in self.planes],
            des_id,
            self.lightmap,
            self.smooth,
            self.mat,
            self.ham_rot,
            self.uaxis.copy(),
            self.vaxis.copy(),
            self.disp_power,
        )
        side_mapping[self.id] = new_side.id
        if self.is_disp:
            assert self.disp_pos is not None
            assert self._disp_verts is not None
            new_side.disp_flags = self.disp_flags
            new_side.disp_elevation = self.disp_elevation
            new_side.disp_pos = self.disp_pos.copy()
            new_side._disp_verts = [
                DispVertex(
                    vert.x,
                    vert.y,
                    vert.normal.copy(),
                    vert.distance,
                    vert.offset.copy(),
                    vert.offset_norm.copy(),
                    vert.alpha,
                    vert.triangle_a,
                    vert.triangle_b,
                ) for vert in self._disp_verts
            ]
        if self.strata_points is not None:
            new_side.strata_points = [point.copy() for point in self.strata_points]
        return new_side

    # noinspection PyProtectedMember
    def export(self, buffer: IO[str], ind: str = '', disp_multiblend: bool = True) -> None:
        """Generate the strings required to define this side in a VMF.

        - disp_multiblend controls whether displacements produce their multiblend
          data (added in CSGO), or if it is skipped.
        """
        buffer.write(f'{ind}side\n')
        buffer.write(f'{ind}{{\n')
        buffer.write(
            f'{ind}\t"id" "{self.id}"\n'
            f'{ind}\t"plane" "({self.planes[0]}) ({self.planes[1]}) ({self.planes[2]})"\n'
            f'{ind}\t"material" "{self.mat}"\n'
            f'{ind}\t"uaxis" "{self.uaxis}"\n'
            f'{ind}\t"vaxis" "{self.vaxis}"\n'
            f'{ind}\t"rotation" "{self.ham_rot:g}\"\n'
            f'{ind}\t"lightmapscale" "{self.lightmap}"\n'
            f'{ind}\t"smoothing_groups" "{self.smooth}"\n'
        )
        if self.strata_points is not None:
            buffer.write(f'{ind}\t"point_data"\n')
            buffer.write(f'{ind}\t{{\n')
            buffer.write(f'{ind}\t\t"numpts" "{len(self.strata_points)}"\n')
            for i, point in enumerate(self.strata_points):
                buffer.write(f'{ind}\t\t"point" "{i} {point}"\n')
            buffer.write(f'{ind}\t}}\n')
        if self.disp_power > 0:
            self._export_displacement(buffer, ind, disp_multiblend)

        buffer.write(f'{ind}}}\n')

    def _export_displacement(self, buffer: IO[str], ind: str, disp_multiblend: bool) -> None:
        """Export displacement data."""
        assert self._disp_verts is not None
        assert self.disp_allowed_vert is not None
        buffer.write(
            f'{ind}\tdispinfo\n'
            f'{ind}\t{{\n'
            f'{ind}\t\t"power" "{self.disp_power}"\n'
            f'{ind}\t\t"startposition" "[{self.disp_pos}]"\n'
            f'{ind}\t\t"flags" "{_DISP_COLL_TO_FLAG[self.disp_flags & DispFlag.COLL_ALL]}"\n'
            f'{ind}\t\t"elevation" "{self.disp_elevation}"\n'
            f'{ind}\t\t"subdiv" "{"1" if DispFlag.SUBDIV in self.disp_flags else "0"}"\n'
        )

        size = self.disp_size
        self._export_disp_rowset('normals', 'normal', buffer, ind, size)
        self._export_disp_rowset('distances', 'distance', buffer, ind, size)
        self._export_disp_rowset('offsets', 'offset', buffer, ind, size)
        self._export_disp_rowset('offset_normals', 'offset_norm', buffer, ind, size)
        self._export_disp_rowset('alphas', 'alpha', buffer, ind, size)

        buffer.write(f'{ind}\t\ttriangle_tags\n{ind}\t\t{{\n')
        for y in range(size):
            row = [
                f'{vert.triangle_a.value} {vert.triangle_b.value}'
                for vert in self._disp_verts[size * y:size * (y+1)]
            ]
            buffer.write(f'{ind}\t\t"row{y}" "{" ".join(row)}"\n')
        buffer.write(ind + '\t\t}\n')

        buffer.write(ind + '\t\tallowed_verts\n')
        buffer.write(ind + '\t\t{\n')
        assert len(self.disp_allowed_vert) == 10, self.disp_allowed_vert
        buffer.write(f'{ind}\t\t"10" "{" ".join(map(str, self.disp_allowed_vert))}"\n')
        buffer.write(f'{ind}\t\t}}\n{ind}\t}}\n')

        if disp_multiblend and any(vert.multi_blend for vert in self._disp_verts):
            self._export_disp_rowset('multiblend', 'multi_blend', buffer, ind, size)
            self._export_disp_rowset('alphablend', 'multi_alpha', buffer, ind, size)
            for i in range(4):
                buffer.write(f'{ind}\t\tmultiblend_color_{i}\n{ind}\t\t{{\n')
                for y in range(size):
                    row = [
                        str(vert.multi_colors[i]) if vert.multi_colors is not None else '1'
                        for vert in self._disp_verts[size * y:size * (y+1)]
                    ]
                    buffer.write(f'{ind}\t\t"row{y}" "{" ".join(row)}"\n')
                buffer.write(ind + '\t\t}\n')

    def _export_disp_rowset(self, name: str, membr: str, f: IO[str], ind: str, size: int) -> None:
        """Write out one of the displacement vertex arrays."""
        assert self._disp_verts is not None
        f.write(f'{ind}\t\t{name}\n{ind}\t\t{{\n')
        rows = [
            str(getattr(vert, membr))
            for vert in self._disp_verts
        ]
        for y in range(size):
            f.write(f'{ind}\t\t"row{y}" "{" ".join(rows[size * y:size * (y+1)])}"\n')
        f.write(f'{ind}\t\t}}\n')

    def __str__(self) -> str:
        """Dump a user-friendly representation of the side."""
        st = "\tmat = " + self.mat
        st += "\n\trotation = " + str(self.ham_rot) + '\n'
        pl_str = ['(' + p.join(' ') + ')' for p in self.planes]
        st += '\tplane: ' + ", ".join(pl_str) + '\n'
        return st

    def __del__(self) -> None:
        """Forget this side's ID when the object is destroyed."""
        self.map.face_id.discard(self.id)

    def __getitem__(self, pos: Tuple[int, int]) -> DispVertex:
        """Return the displacement vertex at this position."""
        if self.disp_pos == 0 or self._disp_verts is None:
            raise ValueError('This face is not a displacement!')
        size = self.disp_size
        x, y = pos
        if 0 <= x < size and 0 <= y < size:
            return self._disp_verts[size * y + x]
        else:
            raise IndexError(
                f'Index {x}, {y} is not valid for a power '
                f'{self.disp_power} displacement, '
                f'both must be within 0-{size}!'
            )

    def __len__(self) -> int:
        """If a displacement, the face has disp_size*disp_size vertexes."""
        return self.disp_size ** 2

    def get_bbox(self) -> Tuple[Vec, Vec]:
        """Generate the highest and lowest points these planes form."""
        bbox_max = self.planes[0].copy()
        bbox_min = self.planes[0].copy()
        for v in self.planes[1:]:
            bbox_max.max(v)
            bbox_min.min(v)
        return bbox_min, bbox_max

    def get_origin(self) -> Vec:
        """Calculates a vector representing the exact center of this plane."""
        size_min, size_max = self.get_bbox()
        origin = (size_min + size_max) / 2
        return origin

    def translate(self, diff: AnyVec) -> None:
        """Move this side by the specified vector.

        - A tuple can be passed in instead if desired.
        """
        for p in self.planes:
            p += diff

        u_axis = Vec(self.uaxis.x, self.uaxis.y, self.uaxis.z)
        v_axis = Vec(self.vaxis.x, self.vaxis.y, self.vaxis.z)

        # Fix offset - see 2013 SDK utils/vbsp/map.cpp:2237
        self.uaxis.offset -= Vec.dot(u_axis, diff) / self.uaxis.scale
        self.vaxis.offset -= Vec.dot(v_axis, diff) / self.vaxis.scale

    def localise(self, origin: AnyVec, angles: Union[AnyMatrix, AnyAngle, None] = None) -> None:
        """Shift the face by the given origin and angles.

        This preserves texture offsets.
        """
        orient = to_matrix(angles)  # Only do this once.
        for p in self.planes:
            p.localise(origin, orient)

        self.uaxis = self.uaxis.localise(origin, orient)
        self.vaxis = self.vaxis.localise(origin, orient)
        if self.is_disp:
            assert self._disp_verts is not None
            assert self.disp_pos is not None
            self.disp_pos.localise(origin, orient)
            for vert in self._disp_verts:
                vert.offset @= orient
                vert.normal @= orient
                vert.offset_norm @= orient

    def disp_get_tri_verts(self, x: int, y: int) -> Tuple[
        DispVertex, DispVertex, DispVertex,
        DispVertex, DispVertex, DispVertex,
    ]:
        """Return the locations of the triangle vertexes.

        This is a set of 6 verts, representing the two triangles in order.
        See 2013 SDK src/public/builddisp.cpp:896-965.
        """
        if not self.is_disp:
            raise ValueError(
                f'This side (id={self.id}, mat={self.mat}) '
                'is not a displacement'
            )
        assert self._disp_verts is not None
        size = self.disp_size
        if x >= size or y >= size:
            raise IndexError(f'Indexes must be from 0-{size-1}, not ({x}, {y})')
        ind = y * size + x
        vert_tl = self._disp_verts[ind]
        vert_tr = self._disp_verts[ind + 1]
        vert_bl = self._disp_verts[ind + size]
        vert_br = self._disp_verts[ind + size + 1]
        if (y * size + x) % 2 == 1:
            # top left to bottom right
            return (
                vert_tl, vert_bl, vert_tr,
                vert_tr, vert_bl, vert_br,
            )
        else:
            # bottom left to top right
            return (
                vert_tl, vert_bl, vert_br,
                vert_tl, vert_br, vert_tr,
            )

    @deprecated('This is useless and will be removed.', category=DeprecationWarning)
    def plane_desc(self) -> str:
        """Return a string which describes this face.

         This is for use in texture randomisation.
         """
        return self.planes[0].join(' ') + self.planes[1].join(' ') + self.planes[2].join(' ')

    def normal(self) -> Vec:
        """Compute the unit vector which extends perpendicular to the face.

        """
        # The three points are in clockwise order, so compute differences
        # in the clockwise direction, then cross to get the normal.
        point_1 = self.planes[1] - self.planes[0]
        point_2 = self.planes[2] - self.planes[1]

        return Vec.cross(point_1, point_2).norm()

    def _scale_setter(self, value: float) -> None:
        """Set both scale attributes easily."""
        self.uaxis.scale = value
        self.vaxis.scale = value

    def _offset_setter(self, value: float) -> None:
        """Set both offset attributes easily."""
        self.uaxis.offset = value
        self.vaxis.offset = value

    if TYPE_CHECKING:
        scale = _FloatSetter()
        offset = _FloatSetter()
    else:
        scale = property(fset=_scale_setter, doc='Set both scale attributes easily.')
        offset = property(fset=_offset_setter, doc='Set both offset attributes easily.')
    del _scale_setter, _offset_setter


# Instead of using setattr(), define a setter function for each attribute
def _disprow_set_norm(vert: DispVertex, value: Vec) -> None:
    vert.normal = value


def _disprow_set_off(vert: DispVertex, value: Vec) -> None:
    vert.offset = value


def _disprow_set_off_norm(vert: DispVertex, value: Vec) -> None:
    vert.offset_norm = value


def _make_disprow_set_multiblend(ind: int) -> Callable[[DispVertex, Vec], None]:
    """Make a function to set a specific multiblend color value."""
    def setter(vert: DispVertex, value: Vec) -> None:
        """multi_colors could be None, but the caller should have initialised it."""
        assert vert.multi_colors is not None
        vert.multi_colors[ind] = value
    return setter


_disprow_multiblend = [
    (f'multiblend_color_{i}', _make_disprow_set_multiblend(i))
    for i in range(4)
]


class _KeyDict(Dict[str, str]):
    """Temporary class to allow the `Entity.keys` dict to be accessed directly, as well as call keys()."""
    def __call__(self) -> KeysView[str]:
        return self.keys()


class Entity(MutableMapping[str, str]):
    """A representation of either a point or brush entity.

    Creation:

    * :py:meth:`Entity(args) <Entity>` for a brand-new Entity
    * :py:meth:`Entity.parse(keyvalues) <Entity.parse>` if reading from a VMF file
    * :py:meth:`ent.copy() <Entity.copy()>` to duplicate an existing entity.
    * :py:meth:`vmf.create_ent() <VMF.create_ent>` to create an entity, then add it to the VMF file.

    Supports ``ent[key]`` operations to read and write keyvalues.
    To read instance ``$replace`` values operate on ``entity.fixup[var]``.
    """
    _fixup: Optional['EntityFixup']
    outputs: List['Output']
    solids: List[Solid]
    id: int
    hidden: bool
    groups: Set[int]

    visgroup_ids: Set[int]
    vis_shown: bool
    vis_auto_shown: bool
    editor_color: Vec
    logical_pos: str
    comments: str

    def __init__(
        self,
        vmf_file: VMF,
        keys: Mapping[str, ValidKVs] = EmptyMapping,
        fixup: Iterable['FixupValue'] = (),
        ent_id: int = -1,
        outputs: Iterable['Output'] = (),
        solids: Iterable[Solid] = (),
        hidden: bool = False,
        groups: Iterable[int] = (),
        vis_ids: Iterable[int] = (),
        vis_shown: bool = True,
        vis_auto_shown: bool = True,
        logical_pos: Optional[str] = None,
        editor_color: Union[Vec, Tuple[int, int, int]] = (255, 255, 255),
        comments: str = '',
    ) -> None:
        """Construct an entity from scratch."""
        self.map = vmf_file
        self._keys = _KeyDict()
        self.outputs = list(outputs)
        self.solids = list(solids)
        self.id = vmf_file.ent_id.get_id(ent_id)
        self.hidden = hidden
        self.groups = set(groups)

        self.visgroup_ids = set(vis_ids)
        self.vis_shown = vis_shown
        self.vis_auto_shown = vis_auto_shown
        self.editor_color = Vec(editor_color)
        self.logical_pos = logical_pos or f'[0 {self.id}]'
        self.comments = comments

        for k, v in keys.items():
            self[k] = v

        fixup_list = list(fixup)
        self._fixup = EntityFixup(fixup_list) if fixup_list else None

    if TYPE_CHECKING:
        def keys(self) -> KeysView[str]:
            """To type checkers, this is a regular method with only the non-deprecated usage."""
            ...
    else:
        @property
        def keys(self) -> _KeyDict:
            """Access the internal keyvalues dict.

            This use is deprecated, the entity is a MutableMapping. It can also be called to get
            a keys view, as with other mappings.
            """
            warnings.warn('This is private, use the entity as a mapping.', DeprecationWarning, stacklevel=2)
            return self._keys

        @keys.setter
        def keys(self, value: Dict[str, ValidKVs]) -> None:
            """Deprecated method to replace all keys."""
            warnings.warn('This is private, call .clear_keys() and update().', DeprecationWarning, stacklevel=2)
            self.clear_keys()
            self.update(value)

    @property
    def fixup(self) -> 'EntityFixup':
        """Access ``$replace`` variables on instances."""
        # Store None for non-instance entities, create if used.
        if self._fixup is None:
            self._fixup = EntityFixup()
        return self._fixup

    # Override MutableMapping, we compare by identity.
    if TYPE_CHECKING:
        def __eq__(self, other: object) -> bool:
            return self is other

        def __ne__(self, other: object) -> bool:
            return self is not other

        def __hash__(self) -> int:
            return object.__hash__(self)
    else:  # Directly assign for efficiency
        __eq__ = object.__eq__
        __ne__ = object.__ne__
        __hash__ = object.__hash__

    def copy(
        self,
        des_id: int = -1,
        vmf_file: Optional[VMF] = None,
        side_mapping: MutableMapping[int, int] = EmptyMapping,
        keep_vis: bool = True,
    ) -> 'Entity':
        """Duplicate this entity entirely, including solids and outputs."""
        new_solids = [
            solid.copy(vmf_file=vmf_file, side_mapping=side_mapping)
            for solid in
            self.solids
        ]
        outs = [o.copy() for o in self.outputs]

        return Entity(
            vmf_file=vmf_file or self.map,
            keys=self._keys,  # __init__() copies for us.
            fixup=self._fixup.copy_values() if self._fixup is not None else (),
            ent_id=des_id,
            outputs=outs,
            solids=new_solids,
            hidden=self.hidden if keep_vis else False,
            groups=self.groups,  # __init__() copies for us.

            editor_color=self.editor_color,
            logical_pos=self.logical_pos,
            vis_shown=self.vis_shown if keep_vis else True,
            vis_auto_shown=self.vis_auto_shown if keep_vis else True,
            vis_ids=self.visgroup_ids if keep_vis else (),
            comments=self.comments,
        )

    @staticmethod
    def parse(
        vmf_file: VMF, tree_list: Keyvalues,
        hidden: bool = False,
        _worldspawn: bool = False,
    ) -> 'Entity':
        """Parse a property tree into an Entity object.

        _worldspawn indicates if this is the worldspawn entity, which additionally contains
        the entity group definitions.
        """
        ent_id = -1
        solids: List[Solid] = []
        keys: Dict[str, str] = {}
        outputs: List[Output] = []
        fixup: List[FixupValue] = []
        group_ids: List[int] = []
        visgroup_ids: List[int] = []
        vis_shown = vis_auto_shown = True
        logical_pos = None
        comment = ''
        editor_color = Vec()
        for item in tree_list:
            name = item.name
            assert name is not None, repr(item)
            if item.has_children():
                if name == "solid":
                    solids.append(Solid.parse(vmf_file, item))
                elif name == "connections":
                    for out in item:
                        outputs.append(Output.parse(out))
                elif name == "editor":
                    for editor_prop in item:
                        if editor_prop.name == "visgroupshown":
                            vis_shown = srctools.conv_bool(editor_prop.value, default=True)
                        elif editor_prop.name == "visgroupautoshown":
                            vis_auto_shown = srctools.conv_bool(editor_prop.value, default=True)
                        elif editor_prop.name == 'color':
                            editor_color = Vec.from_str(editor_prop.value, 255, 255, 255)
                        elif editor_prop.name == 'logicalpos':
                            logical_pos = editor_prop.value
                        elif editor_prop.name == 'comments':
                            comment = editor_prop.value
                        elif editor_prop.name == 'group':
                            group_ids.append(int(editor_prop.value))
                        elif editor_prop.name == 'visgroupid':
                            try:
                                visgroup_ids.append(int(editor_prop.value))
                            except (TypeError, ValueError):
                                raise ValueError(f'Invalid visgroup ID "{editor_prop.value}"!') from None
                elif name == "hidden":
                    for brush_prop in item:
                        if brush_prop.name == "solid":
                            solids.append(Solid.parse(vmf_file, brush_prop, hidden=True))
                        else:
                            raise ValueError(f'Unknown hidden keyvalue "{brush_prop.name}"!')
                elif name == "group":
                    if not _worldspawn:
                        raise ValueError('Group blocks are only permitted on worldspawn!')
                    grp = EntityGroup.parse(vmf_file, item)
                    vmf_file.groups[grp.id] = grp
                else:
                    raise ValueError(f'Unrecognised block keyvalue "{name}" in entity!')
            elif name == "id" and item.value.isnumeric():
                ent_id = int(item.value)
            elif name.startswith('replace'):
                ind_str = name[-2:]  # Index is the last 2 digits
                try:
                    index = int(ind_str)
                except ValueError:  # Not a replace value!
                    keys[item.real_name] = item.value
                else:
                    # Parse the $replace value
                    try:
                        vals = item.value.split(" ", 1)
                        var = vals[0].lstrip('$')
                        try:
                            value = vals[1]
                        except IndexError:
                            # Might happen if entirely blank.
                            value = ''
                        fixup.append(FixupValue(var, value, index))
                    except ValueError:
                        # Failed!
                        keys[item.real_name] = item.value
            else:
                keys[item.real_name] = item.value

        return Entity(
            vmf_file,
            keys,
            fixup,
            ent_id,
            outputs,
            solids,
            hidden,
            group_ids,
            visgroup_ids,
            vis_shown,
            vis_auto_shown,
            logical_pos,
            editor_color,
            comment,
        )

    def is_brush(self) -> bool:
        """Is this Entity a brush entity?"""
        return len(self.solids) > 0

    def export(
        self,
        buffer: IO[str],
        ind: str = '',
        disp_multiblend: bool = True,
        _is_worldspawn: bool = False,
    ) -> None:
        """Generate the strings needed to create this entity.

        - disp_multiblend controls whether displacements produce their multiblend
          data (added in ASW), or if it is skipped.
        - _is_worldspawn is used interally to generate the special worldspawn block.
        """

        if self.hidden:
            buffer.write(f'{ind}hidden\n{ind}{{\n')
            ind += '\t'

        buffer.write(f'{ind}{"world" if _is_worldspawn else "entity"}\n')
        buffer.write(ind + '{\n')
        buffer.write(f'{ind}\t"id" "{self.id}"\n')
        for key, value in sorted(self._keys.items(), key=operator.itemgetter(0)):
            buffer.write(f'{ind}\t"{key}" "{escape_text(value)}"\n')

        if self._fixup is not None:
            self._fixup.export(buffer, ind)

        if self.is_brush():
            for s in self.solids:
                s.export(
                    buffer,
                    ind=ind+'\t',
                    disp_multiblend=disp_multiblend,
                    include_groups=not _is_worldspawn,
                )
        if len(self.outputs) > 0:
            buffer.write(ind + '\tconnections\n')
            buffer.write(ind + '\t{\n')
            for o in self.outputs:
                o.export(buffer, ind=ind+'\t\t')
            buffer.write(ind + '\t}\n')

        # For worldspawn, this includes all group blocks.
        if _is_worldspawn:
            for group in self.map.groups.values():
                group.export(buffer, ind + '\t')

        buffer.write(ind + '\teditor\n')
        buffer.write(ind + '\t{\n')
        buffer.write(f'{ind}\t\t"color" "{self.editor_color}"\n')
        # The editor{} block, indicating if shown/hidden.
        # Worldspawn can't be hidden, so skip these.
        if not _is_worldspawn:
            for group_id in self.groups:
                buffer.write(f'{ind}\t\t"groupid" "{group_id}"\n')

            for vis_id in self.visgroup_ids:
                buffer.write(f'{ind}\t\t"visgroupid" "{vis_id}"\n')

            buffer.write(f'{ind}\t\t"visgroupshown" "{srctools.bool_as_int(self.vis_shown)}"\n')
            buffer.write(f'{ind}\t\t"visgroupautoshown" "{srctools.bool_as_int(self.vis_auto_shown)}"\n')
            buffer.write(f'{ind}\t\t"logicalpos" "{self.logical_pos}"\n')

        if self.comments:
            buffer.write(f'{ind}\t\t"comments" "{escape_text(self.comments)}"\n')
        buffer.write(ind + '\t}\n')

        buffer.write(ind + '}\n')
        if self.hidden:
            buffer.write(ind[:-1] + '}\n')

    def sides(self) -> Iterable['Side']:
        """Iterate through all our brush sides."""
        for solid in self.solids:
            yield from solid

    def add_out(self, *outputs: 'Output') -> None:
        """Add the outputs to our list."""
        self.outputs.extend(outputs)

    def output_targets(self) -> Set[str]:
        """Return a set of the targetnames this entity triggers."""
        return {
            out.target
            for out in
            self.outputs
        }

    def remove(self) -> None:
        """Remove this entity from the map."""
        self.map.remove_ent(self)

    def make_unique(self, unnamed_prefix: str = '') -> 'Entity':
        """Ensure this entity is uniquely named, by adding a numeric suffix.

        If the entity doesn't start with a name, it will use the parameter.
        """
        orig_name = self['targetname']
        if orig_name:
            # If this name is already unique, preserve it.
            if self.map.by_target[orig_name] == {self}:
                return self

            self['targetname'] = ''  # Remove ourselves from the .by_target[] set.
        else:
            orig_name = unnamed_prefix

        base_name = orig_name.rstrip('0123456789')

        if self.map.by_target[base_name]:
            # Check every index in order.
            i = 1
            while True:
                name = base_name + str(i)
                if not self.map.by_target[name]:
                    self['targetname'] = name
                    break
                i += 1
        else:
            # The base name is free!
            self['targetname'] = base_name

        return self

    def __repr__(self) -> str:
        """Produce a short string identifying this entity."""
        desc: List[str] = []
        if classname := self['classname']:
            desc.append(classname)
        desc.append('Entity')
        if name := self['targetname']:
            desc.append(f'"{name}"({classname})')
        else:
            desc.append(classname)
        if hammerid := self['hammerid']:
            desc.append(f'#{hammerid}')
        if origin := self['origin']:
            desc.append(f'@ ({origin})')
        return f'<{" ".join(desc)}>'

    def __str__(self) -> str:
        """Dump a user-friendly representation of the entity."""
        st = "<Entity>: \n{\n"
        for k, v in self._keys.items():
            if not isinstance(v, list):
                st += f"\t {k} = \"{v}\"\n"
        if self._fixup is not None:
            for k, v in self.fixup.items():
                st += f"\t ${k} = \"{v}\"\n"

        for out in self.outputs:
            st += f'\t{out!s}\n'
        st += "}\n"
        return st

    def __len__(self) -> int:
        """The length of an entity is the number of keyvalues."""
        return len(self._keys)

    def __iter__(self) -> Iterator[str]:
        """Iteration iterates over all keyvalues."""
        return iter(self._keys)

    @overload
    def __getitem__(self, key: str) -> str: ...
    @overload
    def __getitem__(self, key: Tuple[str, T]) -> Union[str, T]: ...

    def __getitem__(self, key: Union[str, Tuple[str, T]]) -> Union[str, T]:
        """Allow using [] syntax to search for keyvalues.

        - This will return '' if the value is not present.
        - It ignores case-matching, but will use the first given version
          of a key.
        - If used via Entity.get() the default argument is available.
        - A tuple can be passed for the default to be set, inside the
          [] syntax.
        """
        default: Union[str, T]
        if isinstance(key, tuple):
            key, default = key
        else:
            default = ''

        key = key.casefold()
        for k in self._keys:
            if k.casefold() == key:
                return self._keys[k]
        return default

    def __setitem__(
        self,
        key: str,
        val: ValidKVs,
    ) -> None:
        """Allow using [] syntax to save a keyvalue.

        - It is case-insensitive, so it will overwrite a key which only
          differs by case.
        """
        str_val = conv_kv(val)
        key_fold = key.casefold()
        for k in self._keys:
            if k.casefold() == key_fold:
                # Check case-insensitively for this key first
                orig_val = self._keys.get(k)
                self._keys[k] = str_val
                key = k
                break
        else:
            orig_val = self._keys.get(key)
            self._keys[key] = str_val

        # TODO: if 'mapversion' is passed and self is self.map.spawn, update version there.

        # Update the by_class/target dicts with our new value
        if key_fold == 'classname':
            _remove_copyset(self.map.by_class, orig_val or '', self)
            if self in self.map.entities:
                self.map.by_class[str_val.casefold()].add(self)
            elif self is self.map.spawn:
                if str_val.casefold() != 'worldspawn':
                    self['classname'] = 'worldspawn'  # Revert the change.
                    raise ValueError('The worldspawn entity must remain worldspawn!')
                self.map.by_class['worldspawn'].add(self)
        elif key_fold == 'targetname':
            _remove_copyset(self.map.by_target, orig_val, self)
            if self in self.map.entities:
                self.map.by_target[str_val].add(self)
        elif key_fold == 'nodeid':
            try:
                node_id = int(orig_val)  # type: ignore  # Using as a cast
            except (TypeError, ValueError):
                pass
            else:
                self.map.node_id.discard(node_id)
            try:
                node_id = int(val)  # type: ignore  # Using as a cast
            except (TypeError, ValueError):
                pass
            else:
                self._keys[key] = str(self.map.node_id.get_id(node_id))

    def __delitem__(self, key: str) -> None:
        key = key.casefold()
        if key == 'targetname':
            _remove_copyset(self.map.by_target, self._keys.get('targetname', None), self)
            self.map.by_target[None].add(self)

        if key == 'classname':
            raise KeyError('Classnames cannot be deleted!')

        for k in self._keys:
            if k.casefold() == key:
                # After popping we break out and won't iterate.
                val = self._keys.pop(k)  # noqa: B909
                if key == 'nodeid':
                    try:
                        node_id = int(val)
                    except (TypeError, ValueError):
                        pass
                    else:
                        self.map.node_id.discard(node_id)
                break

    @overload
    def get(self, key: str, /) -> str: ...
    @overload
    def get(self, key: str, /, default: Union[str, T]) -> Union[str, T]: ...

    def get(self, key: str, /, default: Union[str, T] = '') -> Union[str, T]:
        """Allow using [] syntax to search for keyvalues.

        - This will return '' if the value is not present.
        - It ignores case-matching, but will use the first given version
          of a key.
        - If used via Entity.get() the default argument is available.
        - A tuple can be passed for the default to be set, inside the
          [] syntax.
        """
        key = key.casefold()
        for k in self._keys:
            if k.casefold() == key:
                return self._keys[k]
        return default

    def clear(self) -> None:
        """Remove all keyvalues from an item."""
        # Delete these so the .by_class/name values are cleared.
        self['classname'] = 'info_null'
        del self['targetname']
        self._keys.clear()
        # Clear $fixup as well.
        self._fixup = None
    clear_keys = clear

    def __contains__(self, key: object) -> bool:
        """Determine if a value exists for the given key."""
        if isinstance(key, str):
            key = key.casefold()
            for k in self._keys:
                if k.casefold() == key:
                    return True
        return False

    get_key = __contains__

    def __del__(self) -> None:
        """Forget this entity's ID when the object is destroyed."""
        self.map.ent_id.discard(self.id)

    def get_bbox(self) -> Tuple[Vec, Vec]:
        """Get two vectors representing the space this entity takes up."""
        if self.is_brush():
            bbox_min, bbox_max = self.solids[0].get_bbox()
            for s in self.solids[1:]:
                side_min, side_max = s.get_bbox()
                bbox_max.max(side_max)
                bbox_min.min(side_min)
            return bbox_min, bbox_max
        else:
            origin = self.get_origin()
            # the bounding box is 0x0 large for a point ent basically
            return origin, origin.copy()

    def get_origin(self) -> Vec:
        """Return a vector representing the center of this entity's brushes."""
        if self.is_brush():
            bbox_min, bbox_max = self.get_bbox()
            return (bbox_min + bbox_max) / 2
        else:
            return Vec.from_str(self['origin'])


@attrs.define(weakref_slot=False)
class FixupValue:
    """One $fixup variable with its replacement."""
    var: str  # The original casing of the variable name.
    value: str
    id: int  # replaceXX number used.


class EntityFixup(MutableMapping[str, str]):
    """A specialised mapping which keeps track of the variable indexes.

    This treats variable names case-insensitively, and optionally allows
    writing variables with $ signs in front. The case of the first assigned
    name for each key is preserved.

    Additionally, lookups never fail - returning '' instead. Pass in a non-string
    default or use `in` to distinguish.
    """

    # Because of the int(), bool(), float() methods, we need to use builtins.*
    # for the type annotations.
    __slots__ = ['_fixup', '_matcher']

    def __init__(self, fixup: Iterable[FixupValue] = ()) -> None:
        self._fixup: Dict[str, FixupValue] = {}
        self._matcher: Optional[Pattern[str]] = None
        # In _fixup each variable is stored as a tuple of (var_name,
        # value, index) with keys equal to the casefolded var name.
        # var_name is kept to allow restoring the original case when exporting.

        # Do a check to ensure all fixup values have valid indexes:
        used_indexes: Set[int] = set()
        extra_vals: List[FixupValue] = []
        for fix in fixup:
            if fix.id not in used_indexes:
                used_indexes.add(fix.id)
                self._fixup[intern(fix.var.casefold())] = fix
            else:
                extra_vals.append(fix)
        for fix in extra_vals:
            # Add these values wherever they'll fit.
            self[fix.var] = fix.value

    @overload
    def get(self, var: str) -> str: ...
    # noinspection PyMethodOverriding
    @overload
    def get(self, var: str, default: Union[str, T]) -> Union[str, T]: ...

    def get(self, var: str, default: Union[str, T] = '') -> Union[str, T]:
        """Get the value of an instance $replace variable.

        If not found, the default will be returned (an empty string).
        """
        if var[0] == '$':
            var = var[1:]
        folded_var = var.casefold()
        if folded_var in self._fixup:
            return self._fixup[folded_var].value
        else:
            return default

    def copy_values(self) -> List[FixupValue]:
        """Generate a list that can be passed to the constructor."""
        return list(self._fixup.values())

    def __copy__(self) -> 'EntityFixup':
        fix = EntityFixup.__new__(EntityFixup)
        fix._matcher = self._matcher
        fix._fixup = self._fixup.copy()
        return fix

    def __deepcopy__(self, memodict: Optional[Dict[int, Any]] = None) -> 'EntityFixup':
        fix = EntityFixup.__new__(EntityFixup)
        fix._matcher = self._matcher
        fix._fixup = self._fixup.copy()
        return fix

    def __getstate__(self) -> List[FixupValue]:
        return list(self._fixup.values())

    def __setstate__(self, state: List[FixupValue]) -> None:
        self._matcher = None
        self._fixup = {
            intern(tup.var.casefold()): tup
            for tup in state
        }

    def clear(self) -> None:
        """Wipe all the $fixup values."""
        self._fixup.clear()
        self._matcher = None

    @overload
    def setdefault(self, var: str, /, default: str = ...) -> str: ...
    # noinspection PyMethodOverriding
    @overload
    def setdefault(self, var: str, /, default: ValidKV_T) -> Union[str, ValidKV_T]: ...

    def setdefault(self, var: str, /, default: Union[ValidKV_T, str] = '') -> Union[str, ValidKV_T]:
        """Return $key, but if not present set it to the default and return that."""
        if var[0] == '$':
            var = var[1:]
        folded_var = var.casefold()
        if folded_var in self._fixup:
            return self._fixup[folded_var].value
        else:
            self[folded_var] = default
            return default

    def __len__(self) -> int:
        """Return the number of defined keys."""
        return len(self._fixup)

    @overload
    def __getitem__(self, key: str) -> str: ...

    @overload
    def __getitem__(self, key: Tuple[str, T]) -> Union[str, T]: ...

    def __getitem__(self, key: Union[Tuple[str, T], str]) -> Union[str, T]:
        """Retrieve keys via fixup[key] or fixup[key, default].

        See EntityFixup.get().
        """
        if isinstance(key, tuple):
            return self.get(key[0], default=key[1])
        else:
            return self.get(key)

    def __contains__(self, var: object) -> builtins.bool:
        """Check if a variable is present in the fixup list."""
        if isinstance(var, str):
            if var and var[0] == '$':
                var = var[1:]
            return var.casefold() in self._fixup
        return False

    def __setitem__(self, var: str, val: ValidKVs) -> None:
        """Set the value of an instance $replace variable."""
        if var[0] == '$':
            var = var[1:]

        sval = conv_kv(val)

        folded_var = intern(var.casefold())
        try:
            self._fixup[folded_var].value = sval
            # self._matcher is still correct.
        except KeyError:
            # Insert a new value. Use the lowest unused index.
            indexes = {
                fixup.id
                for fixup in
                self._fixup.values()
            }
            ind = 1
            while ind in indexes:
                ind += 1
            self._fixup[folded_var] = FixupValue(intern(var), sval, ind)
            # We've changed the keys so this needs to be regenerated.
            self._matcher = None

    def __delitem__(self, var: str) -> None:
        """Delete a instance $replace variable."""
        if var[0] == '$':
            var = var[1:]
        var = intern(var.casefold())
        if var in self._fixup:
            del self._fixup[var]
            # We've changed the keys so this needs to be regenerated.
            self._matcher = None

    def __iter__(self) -> Iterator[str]:
        """Iterate over all set variable names."""
        for fixup in self._fixup.values():
            yield fixup.var

    def items(self) -> 'ItemsView[str, str]':
        """Provides a view over all variable-value pairs."""
        return _EntityFixupItems(self)

    def values(self) -> 'ValuesView[str]':
        """Provides a view over all variable values."""
        return _EntityFixupValues(self)

    def export(self, buffer: IO[str], ind: str) -> None:
        """Export all the fixup values into the VMF."""
        for fixup in sorted(self._fixup.values(), key=operator.attrgetter('id')):
            # When exporting, pad the index with zeros if necessary
            buffer.write(
                f'{ind}\t"replace{fixup.id:02}" "${fixup.var} {escape_text(fixup.value)}"\n'
            )

    def __str__(self) -> str:
        items = '\n'.join(
            f'\t${fix.var} = {fix.value!r}'
            for fix in sorted(self._fixup.values(), key=operator.attrgetter('id'))
        )
        return f'{self.__class__.__name__}{{\n{items}\n}}'

    def __repr__(self) -> str:
        items = ', '.join(
            repr(tup)
            for tup in
            sorted(self._fixup.values(), key=operator.attrgetter('id'))
        )
        return f'{self.__class__.__name__}([{items}])'

    def substitute(
        self,
        text: str,
        default: Optional[str] = None,
        *,
        allow_invert: bool = False,
    ) -> str:
        """Substitute the fixup variables into the provided string.

        Variables are found based on the defined values, so constructions such as
        val$varval are valid (with no delimiter indicating the end of variables).
        Longer matches are preferred. If the name after $ is not found at all,
        a KeyError is raised, or if default is provided it is substituted.

        Any key is valid if defined in the instance, but only a-z, 0-9 and _ is
        detected for the default functionality.

        If allow_invert is enabled, a variable can additionally be specified
        like !$var to cause it to be inverted when substituted.
        """
        if '$' not in text:  # Early out, cannot substitute.
            return text

        # Cache the pattern used, we can reuse it whenever called again without adding new variables.
        if self._matcher is None:
            # Sort longer values first, so they are checked before smaller
            # counterparts.
            sections: Iterable[str] = map(re.escape, sorted(self._fixup.keys(), key=len, reverse=True))
            # ! maybe, $, any known fixups, then a default any-identifier check.
            self._matcher = re.compile(
                rf'(!)?\$({"|".join(sections)}|[a-z_][a-z0-9_]*)',
                re.IGNORECASE,
            )

        fixup = self._fixup  # Avoid making self a cell var.

        def replacer(match: 'Match[str]') -> str:
            """Handles the replacement semantics."""
            has_inv, varname = match.groups()
            try:
                res = fixup[varname.casefold()].value
            except KeyError:
                if default is None:
                    raise KeyError(f'${varname} not found, known: {["$"+var.var for var in fixup.values()]}') from None
                res = default
            if has_inv is not None:
                if allow_invert:
                    try:
                        res = '0' if srctools.BOOL_LOOKUP[res.casefold()] else '1'
                    except KeyError:
                        # If not bool, keep existing value.
                        pass
                else:
                    # Re-add the !, as if we didn't match it.
                    res = '!' + res
            return res

        return self._matcher.sub(replacer, text)

    def int(self, key: str, def_: Union[builtins.int, T] = 0) -> Union[builtins.int, T]:
        """Return the value of an integer key.

        Equivalent to int(fixup[key]), but with a default value if missing or
        invalid.
        """
        try:
            return int(self.get(key))
        except (ValueError, TypeError):
            return def_

    def float(self, key: str, def_: Union[builtins.float, T] = 0.0) -> Union[builtins.float, T]:
        """Return the value of an integer key.

        Equivalent to float(fixup[key]), but with a default value if missing or
        invalid.
        """
        try:
            return float(self.get(key))
        except (ValueError, TypeError):
            return def_

    def bool(self, key: str, def_: Union[builtins.bool, T] = False) -> Union[builtins.bool, T]:
        """Return a fixup interpreted as a boolean.

        The value may be case-insensitively 'true', 'false', '1', '0', 'T',
        'F', 'y', 'n', 'yes', or 'no'.
        """
        try:
            return BOOL_LOOKUP[self.get(key).casefold()]
        except KeyError:
            return def_

    def vec(
        self,
        key: str,
        x: builtins.float = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
    ) -> Vec:
        """Return the given fixup, converted to a vector."""
        return Vec.from_str(self.get(key), x, y, z)


# noinspection PyProtectedMember
class _EntityFixupValues(ValuesView[str]):
    """Implements Entity.fixup.values()."""
    __slots__ = ()
    _mapping: EntityFixup  # Defined in constructor.

    def __iter__(self) -> Iterator[str]:
        """Yield each value one by one."""
        for fixup in self._mapping._fixup.values():
            yield fixup.value

    def __contains__(self, item: object) -> bool:
        """Check if any fixup has the given value."""
        val = conv_kv(item)  # type: ignore
        for fixup in self._mapping._fixup.values():
            if fixup.value == val:
                return True
        return False


# noinspection PyProtectedMember
class _EntityFixupItems(ItemsView[str, str]):
    """Implements Entity.fixup.items()."""
    __slots__ = ()
    _mapping: EntityFixup  # Defined in constructor.

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        """Yield each key, value pair."""
        for fixup in self._mapping._fixup.values():
            yield fixup.var, fixup.value

    def __contains__(self, item: object) -> bool:
        """Check if any fixup has the given value."""
        if isinstance(item, tuple) and len(item) == 2:
            var, value = item
        else:
            return False
        if isinstance(var, str):
            if var and var[0] == '$':
                var = var[1:]
            try:
                return self._mapping._fixup[var.casefold()].value == conv_kv(value)
            except KeyError:
                return False
        return False


@attrs.define
class EntityGroup:
    """Represents the 'group' blocks in worldspawn.

    This allows the grouping of brushes.
    """
    vmf: VMF
    id: int = attrs.field(default=-1)
    shown: bool = True
    auto_shown: bool = True
    color: Vec = attrs.field(factory=lambda: Vec(255, 255, 255))

    def __attrs_post_init__(self) -> None:
        self.id = self.vmf.group_id.get_id(self.id)

    @classmethod
    def parse(cls, vmf_file: VMF, props: Keyvalues) -> 'EntityGroup':
        """Parse an entity group from the VMF file."""
        editor_block = props.find_block('editor', or_blank=True)
        return cls(
            vmf_file,
            props.int('id', -1),
            editor_block.bool('visgroupshown', True),
            editor_block.bool('visgroupsautoshown', True),
            editor_block.vec('color', 255, 255, 255),
        )

    def copy(self, vmf: Optional[VMF] = None) -> 'EntityGroup':
        """Duplicate an entity group."""
        if vmf is None:
            vmf = self.vmf

        return EntityGroup(
            vmf,
            self.id,
            self.shown,
            self.auto_shown,
            self.color.copy(),
        )

    def export(self, buffer: IO[str], ind: str) -> None:
        """Write out a group into a VMF file."""
        buffer.write(ind + 'group\n')
        buffer.write(ind + '{\n')
        buffer.write(f'{ind}\t"id" "{self.id}"\n')
        buffer.write(ind + '\teditor\n')
        buffer.write(ind + '\t{\n')
        buffer.write(ind + f'\t\t"visgroupshown" "{int(self.shown)}"\n')
        buffer.write(ind + f'\t\t"visgroupautoshown" "{int(self.auto_shown)}"\n')
        buffer.write(ind + f'\t\t"color" "{self.color}"\n')
        buffer.write(ind + '\t}\n')
        buffer.write(ind + '}\n')


class Output:
    """An output from one entity pointing to another.

    When parsing, either the post-L4D ``0x1B`` character can be used, or the previous ``,``
    delimiters can be used. For commas, extraneous ones will be treated as part of the parameters.
    """
    __slots__ = [
        'output',
        'inst_out',
        'target',
        'input',
        'inst_in',
        'params',
        'delay',
        'times',
        'comma_sep',
    ]
    output: str
    """The output which triggers this."""
    target: str
    """The target entity."""
    input: str
    """The input to fire."""
    params: str
    """Parameters to give the input, or ``""`` for none."""
    delay: float
    """The number of seconds before the output should fire."""
    inst_out: Optional[str]
    """The local entity for an instance output (``instance:name;Output``)"""
    inst_in: Optional[str]
    """The local entity we are really triggering in instance inputs (``instance:name;Input``)"""
    comma_sep: bool
    """Use a comma as a separator, instead of the :py:const:`OUTPUT_SEP` character."""
    times: int
    """The number of times to fire before being deleted. ``-1`` means forever, Hammer only uses ``-1`` and ``1``."""

    #: The character used to separate output values, after L4D. Before then commas (``,``) were used.
    SEP: Final = OUTPUT_SEP

    def __init__(
        self,
        out: str,
        targ: Union[Entity, str],
        inp: str,
        param: ValidKVs = '',
        delay: float = 0.0,
        *,
        times: int = -1,
        only_once: bool = False,
        inst_out: Optional[str] = None,
        inst_in: Optional[str] = None,
        comma_sep: bool = False,
    ) -> None:
        self.output = out
        self.inst_out = inst_out
        if isinstance(targ, Entity):
            self.target = targ['targetname']
        else:
            self.target = targ
        self.input = inp
        self.inst_in = inst_in
        self.params = conv_kv(param)
        self.delay = delay
        self.times = 1 if only_once else times
        self.comma_sep = comma_sep

    @property
    def only_once(self) -> bool:
        """Instead of setting ``times``, this provides an interface like how Hammer does."""
        return self.times == 1

    @only_once.setter
    def only_once(self, is_once: bool) -> None:
        self.times = 1 if is_once else -1

    @classmethod
    def parse(cls, prop: Keyvalues) -> 'Output':
        """Convert the VMF Property into an Output object."""
        if OUTPUT_SEP in prop.value:
            sep = False
            vals = prop.value.split(OUTPUT_SEP)
        else:
            sep = True
            vals = prop.value.split(',')

        try:
            targ, inp, param, delay, times = vals
        except ValueError as e:
            if sep and len(vals) > 5:
                # Special case, more than 4 commas, recombine the commas in the params.
                # Valve's code doesn't do this, but a mod could potentially edit VBSP & the game to
                # then support commas in params in a backward-compatible way. Commas in any other
                # field wouldn't be particularly useful.
                targ, inp, *param_lst, delay, times = vals
                param = ','.join(param_lst)
            else:
                raise ValueError(f'Bad output value: "{prop.value}"') from e

        inst_out, out = Output.parse_name(prop.real_name)
        inst_inp, inp = Output.parse_name(inp)

        return cls(
            out,
            targ,
            inp,
            param=param,
            delay=float(delay),
            times=int(times),
            inst_out=inst_out,
            inst_in=inst_inp,
            comma_sep=sep,
        )

    @classmethod
    def combine(cls, first: 'Output', second: 'Output') -> 'Output':
        """Combine two outputs into a merged form.

        This is suitable for combining a Trigger and OnTriggered pair into one,
        or similar values.
        """
        return cls(
            first.output,
            second.target,
            second.input,
            second.params or first.params,
            first.delay + second.delay,
            times=first.times if second.times < 0
            else second.times if first.times < 0
            else min(first.times, second.times),
            inst_out=first.inst_out,
            inst_in=second.inst_in,
            comma_sep=first.comma_sep and second.comma_sep,
        )

    @staticmethod
    def parse_name(name: str) -> Tuple[Optional[str], str]:
        """Extract the instance name from values of the form:

        'instance:local_name;Command'
        This then returns a local_name, command tuple.
        If not of this form, the first value will be None.
        """
        if name.casefold().startswith('instance:'):
            try:
                inst_part, command = name.split(';', 1)
            except ValueError:
                # This is an invalid instance: command, which will crash VBSP.
                raise ValueError(f'"Instance:" in/output without command! ({name})') from None
            else:
                return inst_part[9:], command
        return None, name

    def exp_out(self) -> str:
        """Combine the instance name with the output if necessary."""
        if self.inst_out:
            return 'instance:' + self.inst_out + ';' + self.output
        else:
            return self.output

    def exp_in(self) -> str:
        """Combine the instance name with the input if necessary."""
        if self.inst_in:
            return 'instance:' + self.inst_in + ';' + self.input
        else:
            return self.input

    def __repr__(self) -> str:
        vals = (
            f'{self.__class__.__name__}({self.output!r}, {self.target!r}, '
            f'{self.input!r}, {self.params!r}, delay={self.delay!r}'
        )
        if self.inst_in is not None:
            vals += ', inst_in=' + repr(self.inst_in)
        if self.inst_out is not None:
            vals += ', inst_out=' + repr(self.inst_out)

        if self.times == 1:
            # Use only_once  to be more clear
            vals += ', only_once=True'
        elif self.times != -1:
            # Use 'raw' value if a specific count
            vals += ', times=' + repr(self.times)
        # Omit if infinite, most common

        if self.comma_sep:
            vals += ', comma_sep=True'
        return vals + ')'

    def __str__(self) -> str:
        """Generate a user-friendly representation of this output."""
        st = "<Output> "
        if self.inst_out:
            st += f'instance:{self.inst_out};'
        st += f'''{self.output} -> {self.target or '""'} -> '''
        if self.inst_in:
            st += f"instance:{self.inst_in};"
        st += self.input

        if self.params and not self.inst_in:
            st += f" ({self.params})"
        if self.delay != 0:
            st += f" after {self.delay} seconds"
        if self.times != -1:
            st += " (once only)" if self.times == 1 else f" ({self.times!s} times only)"
        return st

    def __getstate__(self) -> Tuple[object, ...]:
        """Produce the state for pickling.

        We know output/input names tend to be the same often,
        so interning here will simplify the pickle.
        """
        basic: Tuple[object, ...] = (
            intern(self.output),
            intern(self.target),
            intern(self.input),
            self.comma_sep,
        )
        # Instance, delays and times are more rare - if unset don't include.
        if self.inst_in or self.inst_out or self.params or self.delay or self.times != -1:
            return (
                *basic,
                intern(self.inst_out) if self.inst_out is not None else None,
                intern(self.inst_in) if self.inst_in is not None else None,
                intern(self.params),
                self.delay,
                self.times,
            )
        else:
            return basic

    def __setstate__(self, state: Tuple[Any, ...]) -> None:
        """Restore the pickled state."""
        (
            self.output,
            self.target,
            self.input,
            self.comma_sep,
            *advanced,
        ) = state
        if advanced:
            (
                self.inst_out,
                self.inst_in,
                self.params,
                self.delay,
                self.times,
            ) = advanced
        else:
            self.inst_out = None
            self.inst_in = None
            self.params = ''
            self.delay = 0.0
            self.times = -1

    def export(self, buffer: IO[str], ind: str = '') -> None:
        """Generate the text required to define this output in the VMF."""
        buffer.write(ind + self.as_keyvalue())

    def as_keyvalue(self) -> str:
        """Generate the text form of the output.

        This backslash-escapes characters where necessary.
        """
        sep = ',' if self.comma_sep else self.SEP
        # Don't bother escaping the delay/times values, since those can't be text.
        return (
            f'"{escape_text(self.exp_out())}" "{escape_text(self.target)}{sep}'
            f'{escape_text(self.exp_in())}{sep}{escape_text(self.params)}{sep}'
            f'{self.delay:g}{sep}{self.times}"\n'
        )

    def copy(self) -> 'Output':
        """Duplicate this Output object."""
        return Output(
            self.output,
            self.target,
            self.input,
            self.params,
            self.delay,
            times=self.times,
            inst_out=self.inst_out,
            inst_in=self.inst_in,
            comma_sep=self.comma_sep,
        )

    def gen_addoutput(self, delim: str = ',') -> str:
        """Return the parameter needed to create this output via AddOutput.

        This assumes the target instance uses Prefix fixup, if inst_in is set.
        """
        if self.inst_out:
            raise ValueError('Inst_out is not useable in AddOutput.')

        if self.inst_in:
            target = f'{self.target}-{self.inst_in}'
        else:
            target = self.target

        return (
            f'{self.output} {target}{delim}{self.input}{delim}'
            f'{self.params}{delim}{self.delay}{delim}{self.times}'
        )
