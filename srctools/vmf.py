""" VMF Library
Wraps property_parser tree in a set of classes which smartly handle
specifics of VMF files.
"""
import io
import re
import operator
import builtins
import warnings
from array import ArrayType as Array
from enum import Flag
from collections import defaultdict
from contextlib import suppress
from sys import intern

from typing import (
    Optional, Union, overload, TypeVar, Generic,
    Dict, List, Tuple, Set, Mapping, MutableMapping, IO,
    Iterable, Iterator, AbstractSet, Pattern, Match,
)

import attr

from srctools import BOOL_LOOKUP, EmptyMapping
from srctools.math import Vec, Angle, Matrix, to_matrix
from srctools.property_parser import Property
import srctools

# Used to set the defaults for versioninfo
CURRENT_HAMMER_VERSION = 400
CURRENT_HAMMER_BUILD = 5304


# The character used to separate output values.
OUTPUT_SEP = chr(27)


T = TypeVar('T')
# Types we allow for setting keyvalues. These we can stringify into something
# matching Valve's usual encoding.
# Other types are just str()ed, which might produce a bad result.
ValidKVs = Union[str, int, bool, float, Vec, Angle, Matrix]
ValidKV_T = TypeVar('ValidKV_T', str, int, bool, float, Vec, Angle, Matrix)


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
    """Convert a type into a string matching Valve's syntax."""
    if val is True:
        return '1'
    elif val is False:
        return '0'
    elif isinstance(val, Matrix):
        return str(val.to_angle())
    else:
        return str(val)


class IDMan(AbstractSet[int]):
    """Allocate and manage a set of unique IDs.

    This implements some of MutableSet, but the adding methods cannot
    be used since the ID may need to change to ensure uniqueness.
    """
    def __init__(self, existing: Iterable[int] = ()):
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
    u_repeat: float=1,
    v_repeat: float=1,
    swap: bool=False,
    render_order: int=0,
) -> 'Entity':
    """Generate an overlay on an axis-aligned surface.

    - origin is the center point of the overlay.
    - uax is the direction and distance for the texture's width ('right').
    - vax is the direction and distance for the texture's height ('up').
    - normal is the normal of the surfaces (axis-aligned).
    - material is the material used.
    - u_ and v_repeat define how many times to repeat the texture in that
      direction.
    - If swap is true, the texture will be rotated 90.
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


def localise_overlay(over: 'Entity', origin: Vec, angles: Union[Angle, Matrix]=None) -> None:
    """Rotate an overlay like what is done in instances."""
    if angles is not None:
        for key in ('basisNormal', 'basisU', 'basisV'):
            ang = Vec.from_str(over[key]) @ angles
            over[key] = ang.join(' ')
    else:
        angles = Matrix()

    for key in ('basisOrigin', 'origin'):
        ang = Vec.from_str(over[key]) @ angles
        ang += origin
        over[key] = ang.join(' ')


class CopySet(Generic[T], Set[T]):
    """Modified version of a Set which allows modification during iteration.

    """
    __slots__ = ()  # No extra vars

    def __iter__(self) -> Iterator[T]:
        cur_items: frozenset[T] = frozenset(self)

        yield from cur_items
        # after iterating through ourselves, iterate through any new ents.
        yield from self - cur_items


@attr.frozen
class PrismFace:
    """Return value for VMF.make_prism()."""
    solid: 'Solid'
    top: 'Side'
    bottom: 'Side'
    north: 'Side'
    south: 'Side'
    east: 'Side'
    west: 'Side'


class VMF:
    """Represents a VMF file, and holds counters for various IDs used.

    Has functions for searching for specific entities or brushes, and
    converts to/from a property_parser tree.

    The dictionaries by_target and by_class allow quickly getting a set
    of entities with the given class or targetname.
    """
    def __init__(
        self,
        map_info=EmptyMapping,
        spawn: 'Entity'=None,
        entities: List['Entity']=None,
        brushes: List['Solid']=None,
        cameras: List['Camera']=None,
        cordons: List['Cordon']=None,
        vis_tree: List['VisGroup']=None,
        preserve_ids: bool=False,
    ):
        """Create a VMF.

        If preserve_ids is False (default), various IDs will be changed to
        ensure they are unique when adding items to the VMF. If True they will
        stay the same. New items will aquire a unique ID.
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
        self.by_target: defaultdict[Optional[str], CopySet[Entity]] = defaultdict(CopySet)
        self.by_class: defaultdict[Optional[str], CopySet[Entity]] = defaultdict(CopySet)

        self.entities: list[Entity] = []
        self.add_ents(entities or [])  # We need to set the by_ dicts too.
        self.brushes: list[Solid] = brushes or []
        self.cameras: list[Camera] = cameras or []
        self.cordons: list[Cordon] = cordons or []
        self.vis_tree: list[VisGroup] = vis_tree or []

        # mapspawn entity, which is the entity world brushes are saved to.
        self.spawn: Entity = spawn or Entity(self)
        self.spawn.solids = self.brushes
        self.is_prefab = srctools.conv_bool(map_info.get('prefab'), False)
        self.cordon_enabled = srctools.conv_bool(map_info.get('cordons_on'), False)
        self.map_ver = srctools.conv_int(map_info.get('mapversion'))

        if 'mapversion' in self.spawn:
            # This is saved only in the main VMF object, delete the copy.
            del self.spawn['mapversion']
        # The worldspawn entity should always be worldspawn.
        self.spawn['classname'] = 'worldspawn'

        # These three are mostly useless for us, but we'll preserve them anyway
        self.format_ver = srctools.conv_int(
            map_info.get('formatversion'), 100)
        self.hammer_ver = srctools.conv_int(
            map_info.get('editorversion'), CURRENT_HAMMER_VERSION)
        self.hammer_build = srctools.conv_int(
            map_info.get('editorbuild'), CURRENT_HAMMER_BUILD)

        # Various Hammer settings
        self.show_grid = srctools.conv_bool(
            map_info.get('showgrid'), True)
        self.show_3d_grid = srctools.conv_bool(
            map_info.get('show3dgrid'), False)
        self.snap_grid = srctools.conv_bool(
            map_info.get('snaptogrid'), True)
        self.show_logic_grid = srctools.conv_bool(
            map_info.get('showlogicalgrid'), False)
        self.grid_spacing = srctools.conv_int(
            map_info.get('gridspacing'), 64)
        self.active_cam = srctools.conv_int(
            map_info.get('active_cam'), -1)
        self.quickhide_count = srctools.conv_int(
            map_info.get('quickhide'), -1)

    def add_brush(self, item: 'Solid'):
        """Add a world brush to this map."""
        self.brushes.append(item)

    def remove_brush(self, brush: 'Solid'):
        """Remove a world brush from this map."""
        try:
            self.brushes.remove(brush)
        except ValueError:
            pass  # Already removed.

    def add_ent(self, item: 'Entity'):
        """Add an entity to the map.

        The entity should have been created with this VMF as a parent.
        """
        self.entities.append(item)
        self.by_class[item['classname', ''].casefold() or None].add(item)
        self.by_target[item['targetname', ''].casefold() or None].add(item)
        if 'nodeid' in item:
            try:
                node_id = int(item['nodeid'])
            except (TypeError, ValueError):
                pass
            else:
                item['nodeid'] = str(self.node_id.get_id(node_id))

    def remove_ent(self, item: 'Entity'):
        """Remove an entity from the map.

        After this is called, the entity will no longer be exported.
        The object still exists, so it can be reused.
        """
        try:
            self.entities.remove(item)
        except ValueError:
            pass  # Already removed.

        self.by_class[item['classname', ''].casefold() or None].discard(item)
        self.by_target[item['targetname', ''].casefold() or None].discard(item)
        if 'nodeid' in item:
            try:
                node_id = int(item['nodeid'])
            except (TypeError, ValueError):
                pass
            else:
                self.node_id.discard(node_id)

        self.ent_id.discard(item.id)

    def add_brushes(self, brushes: Iterable['Solid']):
        """Add multiple brushes to the map."""
        self.brushes.extend(brushes)

    def add_ents(self, ents: Iterable['Entity']):
        """Add multiple entities to the map."""
        ents = list(ents)
        self.entities.extend(ents)
        for item in ents:
            self.by_class[item['classname', ''].casefold() or None].add(item)
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

    def create_visgroup(self, name: str, color: Union[Vec, Tuple[int, int, int]]=(255, 255, 255)) -> 'VisGroup':
        """Convenience method for creating visgroups."""
        vis = VisGroup(self, name, -1, Vec(color))
        self.vis_tree.append(vis)
        return vis

    @staticmethod
    def parse(tree: Union[Property, str], preserve_ids=False):
        """Convert a property_parser tree into VMF classes.
        """
        if not isinstance(tree, Property):
            # if not a tree, try to read the file
            with open(tree) as file:
                tree = Property.parse(file)

        map_info = {}
        ver_info = tree.find_block('versioninfo', or_blank=True)
        for key in ('editorversion',
                    'mapversion',
                    'editorbuild',
                    'prefab'):
            map_info[key] = ver_info[key, '']

        map_info['formatversion'] = ver_info['formatversion', '100']
        if map_info['formatversion'] != '100':
            # If the version is different, we're probably about to fail horribly
            raise Exception(
                'Unknown VMF format version " ' +
                map_info['formatversion'] + '"!'
            )

        view_opt = tree.find_block('viewsettings', or_blank=True)
        view_dict = {
            'bSnapToGrid': 'snaptogrid',
            'bShowGrid': 'showgrid',
            'bShow3DGrid': 'show3dgrid',
            'bShowLogicalGrid': 'showlogicalgrid',
            'nGridSpacing': 'gridspacing'
            }
        for key in view_dict:
            map_info[view_dict[key]] = view_opt[key, '']

        cordons = tree.find_block('cordons', or_blank=True)
        map_info['cordons_on'] = cordons['active', '0']

        cam_props = tree.find_block('cameras', or_blank=True)
        map_info['active_cam'] = cam_props['activecamera', '-1']
        map_info['quickhide'] = tree.find_block('quickhide', or_blank=True)['count', '']

        # We have to create an incomplete map before parsing any data.
        # This ensures the ID manager objects have been created, so we can
        # ensure unique IDs in brushes, entities and faces.
        map_obj = VMF(map_info=map_info, preserve_ids=preserve_ids)

        for vis in tree.find_all('visgroups', 'visgroup'):
            map_obj.vis_tree.append(VisGroup.parse(map_obj, vis))

        for c in cam_props:
            if c.name != 'activecamera':
                Camera.parse(map_obj, c)

        for ent in cordons.find_all('cordon'):
            Cordon.parse(map_obj, ent)

        for ent in tree.find_all('Entity'):
            map_obj.add_ent(
                Entity.parse(map_obj, ent, hidden=False)
            )

        # find hidden entities
        for hidden_ent in tree.find_all('hidden'):
            for ent in hidden_ent:
                map_obj.add_ent(
                    Entity.parse(map_obj, ent, hidden=True)
                )

        map_spawn = tree.find_block('world', or_blank=True)
        if map_spawn is None:
            # Generate a fake default to parse through
            map_spawn = Property("world", [])
        map_obj.spawn = Entity.parse(map_obj, map_spawn)

        if map_obj.spawn.solids is not None:
            map_obj.brushes = map_obj.spawn.solids

        return map_obj

    @overload
    def export(self, *, inc_version: bool=True, minimal: bool=False, disp_multiblend: bool = True) -> str: ...
    @overload
    def export(self, dest_file: IO[str], *, inc_version: bool=True, minimal: bool=False, disp_multiblend: bool = True) -> None: ...
    def export(self, dest_file: IO[str]=None, *, inc_version=True, minimal=False, disp_multiblend: bool = True):
        """Serialises the object's contents into a VMF file.

        - If no file is given the map will be returned as a string.
        - By default, this will increment the map's version - set
          inc_version to False to suppress this.
        - If minimal is True, several blocks will be skipped
          (Viewsettings, cameras, cordons and visgroups)
        - disp_multiblend controls whether displacements produce their multiblend
          data (added in ASW), or if it is stripped.
        - named_cordons controls if multiple cordons may be used (post L4D), or
          if only a single cordon is allowed.
        """
        if dest_file is None:
            dest_file = io.StringIO()
            # acts like a file object but is actually a string. We're
            # using this to prevent having Python duplicate the entire
            # string every time we append
            ret_string = True
        else:
            ret_string = False

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
                            srctools.bool_as_int(self.show_3d_grid) + '"\n}\n')

        # The worldspawn version should always match the global value.
        # Also force the classname, since this will crash if it's different.
        self.spawn['mapversion'] = str(self.map_ver)
        self.spawn['classname'] = 'worldspawn'
        self.spawn.export(dest_file, ent_name='world', disp_multiblend=disp_multiblend)
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

        if ret_string:
            assert isinstance(dest_file, io.StringIO)
            string = dest_file.getvalue()
            dest_file.close()
            return string

    def iter_wbrushes(self, world: bool=True, detail: bool=True) -> Iterator['Solid']:
        """Iterate through all world and detail solids in the map."""
        if world:
            yield from self.brushes
        if detail:
            for ent in self.by_class['func_detail']:
                yield from ent.solids

    def iter_wfaces(self, world: bool=True, detail: bool=True) -> Iterator['Side']:
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
        vals: Mapping[str, str]=EmptyMapping,
        tags: Mapping[str, str]=EmptyMapping,
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
        p1: Vec,
        p2: Vec,
        mat: str='tools/toolsnodraw',
    ) -> PrismFace:
        """Create an axis-aligned brush connecting the two points.

        A PrismFaces tuple will be returned which contains the six
        faces, as well as the solid.
        All faces will be textured with 'mat'.
        """
        b_min = Vec(p1)
        b_max = Vec(p1)
        b_min.min(p2)
        b_max.max(p2)

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
        p1: Vec,
        p2: Vec,
        thick: float=16,
        mat: str='tools/toolsnodraw',
        inner_mat: str='',
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
    def parse(cls, vmf_file: VMF, tree: Property) -> 'Camera':
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

    def export(self, buffer: IO[str], ind: str='') -> None:
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
        is_active=True,
        name='Cordon',
    ):
        self.map = vmf_file
        self.name = name
        self.bounds_min = min_
        self.bounds_max = max_
        self.active = is_active
        vmf_file.cordons.append(self)

    @classmethod
    def parse(cls, vmf_file: VMF, tree: Property) -> 'Cordon':
        """Parse a cordon from the VMF file."""
        name = tree['name', 'cordon']
        is_active = tree.bool('active', False)
        bounds = tree.find_block('box', or_blank=True)
        min_ = bounds.vec('mins', 0, 0, 0)
        max_ = bounds.vec('maxs', 128, 128, 128)
        return Cordon(vmf_file, min_, max_, is_active, name)

    def export(self, buffer: IO[str], ind: str='') -> None:
        """Write the cordon into the VMF."""
        buffer.write(ind + 'cordon\n')
        buffer.write(ind + '{\n')
        buffer.write(ind + '\t"name" "' + self.name + '"\n')
        buffer.write(ind + '\t"active" "' +
                     srctools.bool_as_int(self.active) +
                     '"\n')
        buffer.write(ind + '\tbox\n')
        buffer.write(ind + '\t{\n')
        buffer.write(ind + '\t\t"mins" "(' +
                     self.bounds_min.join(' ') +
                     ')"\n')
        buffer.write(ind + '\t\t"maxs" "(' +
                     self.bounds_max.join(' ') +
                     ')"\n')
        buffer.write(ind + '\t}\n')
        buffer.write(ind + '}\n')

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


@attr.s(auto_attribs=True, hash=False, eq=False, order=False, getstate_setstate=True)
class VisGroup:
    """Defines one visgroup."""
    vmf: VMF
    name: str
    id: int = attr.ib(default=-1)
    color: Vec = attr.ib(factory=lambda: Vec(255, 255, 255))
    child_groups: List['VisGroup'] = attr.ib(converter=list, factory=list)

    def __attrs_post_init__(self) -> None:
        self.id = self.vmf.vis_id.get_id(self.id)

    @classmethod
    def parse(cls, vmf: VMF, props: Property) -> 'VisGroup':
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

    def export(self, buffer: IO[str], ind: str='') -> None:
        """Write out the VMF text for a visgroup"""
        buffer.write(
            f'{ind}visgroup\n'
            f'{ind}{{\n'
            f'{ind}\t"name" "{self.name}"\n'
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
        vmf: VMF=None,
        group_mapping: MutableMapping[int, int]=EmptyMapping,
        des_id: int=-1,
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


@attr.s(auto_attribs=True, hash=False, eq=False, order=False, getstate_setstate=True)
class Solid:
    """A single brush, serving as both world brushes and brush entities."""
    map: VMF
    id: int = attr.ib(default=-1)
    sides: List['Side'] = attr.ib(factory=list)
    visgroup_ids: Set[int] = attr.ib(default=(), converter=set)
    hidden: bool = False
    group_id: Optional[int] = None
    vis_shown: bool = True
    vis_auto_shown: bool = True
    cordon_solid: Optional[int] = None
    editor_color: Vec = attr.ib(factory=lambda: Vec(255, 255, 255))

    def __attrs_post_init__(self) -> None:
        self.id = self.map.solid_id.get_id(self.id)

    def copy(
        self,
        des_id: int=-1,
        vmf_file: VMF=None,
        side_mapping: MutableMapping[int, int]=EmptyMapping,
        keep_vis: bool=True,
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
            self.visgroup_ids if keep_vis else (),  # type: ignore
            self.hidden if keep_vis else False,
            self.group_id,
            self.vis_shown if keep_vis else True,
            self.vis_auto_shown if keep_vis else True,
            self.cordon_solid,
            self.editor_color,
        )

    @classmethod
    def parse(cls, vmf_file: VMF, tree: Property, hidden: bool=False) -> 'Solid':
        """Parse a Property tree into a Solid object."""
        solid_id = tree.int('id', -1)
        sides = []
        for side in tree.find_all("side"):
            sides.append(Side.parse(vmf_file, side))

        visgroups = set()
        group_id = None
        vis_shown = vis_auto_shown = True
        cordon_solid = None
        editor_color = Vec(255, 255, 255)

        for v in tree.find_children("editor"):
            if v.name == "visgroupshown":
                vis_shown = srctools.conv_bool(v.value, default=True)
            elif v.name == "visgroupautoshown":
                vis_auto_shown = srctools.conv_bool(v.value, default=True)
            elif v.name == "cordonsolid":
                cordon_solid = srctools.conv_int(v.value, default=None)
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
            cordon_solid,
            editor_color,
        )

    def export(
        self,
        buffer: IO[str],
        ind: str='',
        disp_multiblend: bool = True,
    ) -> None:
        """Generate the strings needed to define this brush.

        - disp_multiblend controls whether displacements produce their multiblend
          data (added in ASW), or if it is stripped.
        """
        if self.hidden:
            buffer.write(ind + 'hidden\n' + ind + '{\n')
            ind += '\t'
        buffer.write(ind + 'solid\n')
        buffer.write(ind + '{\n')
        buffer.write(ind + '\t"id" "' + str(self.id) + '"\n')
        for s in self.sides:
            s.export(buffer, ind + '\t', disp_multiblend)

        buffer.write(ind + '\teditor\n')
        buffer.write(ind + '\t{\n')
        buffer.write(f'{ind}\t\t"color" "{self.editor_color}"\n')
        if self.group_id is not None:
            buffer.write(f'{ind}\t\t"groupid" "{self.group_id}"\n')

        for group in self.visgroup_ids:
            buffer.write(f'{ind}\t\t"visgroupid" "{group}"\n')

        buffer.write('{}\t\t"visgroupshown" "{}"\n'.format(
            ind,
            srctools.bool_as_int(self.vis_shown),
        ))
        buffer.write('{}\t\t"visgroupautoshown" "{}"\n'.format(
            ind,
            srctools.bool_as_int(self.vis_auto_shown),
        ))
        if self.cordon_solid is not None:
            buffer.write(f'{ind}\t\t"cordonsolid" "{self.cordon_solid}"\n')

        buffer.write(ind + '\t}\n')

        buffer.write(ind + '}\n')
        if self.hidden:
            buffer.write(ind[:-1] + '}\n')

    def __str__(self) -> str:
        """Return a user-friendly description of our data."""
        st = "<solid:" + str(self.id) + ">\n{\n"
        for s in self.sides:
            st += str(s) + "\n"
        st += "}"
        return st

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

    def get_origin(self, bbox_min: Vec=None, bbox_max: Vec=None) -> Vec:
        """Calculates a vector representing the exact center of this brush."""
        if bbox_min is None or bbox_max is None:
            bbox_min, bbox_max = self.get_bbox()
        return (bbox_min + bbox_max) / 2

    def translate(self, diff: Vec):
        """Move this solid by the specified vector."""
        for s in self.sides:
            s.translate(diff)

    def localise(self, origin: Vec, angles: Union[Angle, Matrix]=None):
        """Shift this brush by the given origin/angles."""
        angles = to_matrix(angles)  # Only do this once.
        for s in self.sides:
            s.localise(origin, angles)


@attr.define(frozen=True, hash=True, order=True)
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


def _list4_validator(_: object, attrib: attr.Attribute, value: object) -> None:
    """Validate the value is a list of 4 elements."""
    if not isinstance(value, list):
        raise TypeError(attrib.name + ' should be a list!')
    if len(value) != 4:
        raise ValueError(attrib.name + ' must have 4 values!')


@attr.define(frozen=False)
class DispVertex:
    """A vertex in dislacements."""
    x: int  # Position of the vertex in the displacement.
    y: int

    normal: Vec = attr.ib(factory=Vec, validator=attr.validators.instance_of(Vec))
    distance: float = 0
    offset: Vec = attr.ib(factory=Vec, validator=attr.validators.instance_of(Vec))
    offset_norm: Vec = attr.ib(factory=Vec, validator=attr.validators.instance_of(Vec))
    alpha: float = 0.0  # 0-255
    # The pair of triangle tags for the quad in the +ve direction
    # from us. This means the last row/column's triangles are ignored.
    triangle_a: TriangleTag = TriangleTag.FLAT
    triangle_b: TriangleTag = TriangleTag.FLAT

    # These are for multiblend displacements, added in ASW+.
    multi_blend: Vec4 = Vec4()
    multi_alpha: Vec4 = Vec4()
    multi_colors: Optional[List[Vec]] = attr.ib(default=None, validator=attr.validators.optional(
        attr.validators.deep_iterable(attr.validators.instance_of(Vec), _list4_validator)
    ))


class UVAxis:
    """Values saved into Side.uaxis and Side.vaxis.

    These define the alignment of textures on a face.
    """
    __slots__ = [
        'x', 'y', 'z',
        'scale',
        'offset',
    ]

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        offset: float=0.0,
        scale: float=0.25,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.offset = offset
        self.scale = scale

    @staticmethod
    def parse(value: str) -> 'UVAxis':
        """Parse a UV axis from a string."""
        vals = value.split()
        return UVAxis(
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

    def __copy__(self) -> 'UVAxis':
        return UVAxis(
            x=self.x,
            y=self.y,
            z=self.z,
            offset=self.offset,
            scale=self.scale,
        )

    def __deepcopy__(self, memodict: object=None) -> 'UVAxis':
        return UVAxis(
            x=self.x,
            y=self.y,
            z=self.z,
            offset=self.offset,
            scale=self.scale,
        )

    def __getstate__(self) -> tuple:
        return (self.x, self.y, self.z, self.offset, self.scale)

    def __setstate__(self, state: tuple) -> None:
        (self.x, self.y, self.z, self.offset, self.scale) = state

    def vec(self) -> Vec:
        """Return the axis as a vector."""
        return Vec(self.x, self.y, self.z)

    def rotate(self, angles: Angle) -> 'UVAxis':
        """Rotate the axis by a vector.

        This doesn't handle offsets correctly.
        """
        vec = self.vec() @ angles
        return UVAxis(
            vec.x,
            vec.y,
            vec.z,
            self.offset,
            self.scale,
        )

    def localise(self, origin: Vec, angles: Union[Angle, Matrix]) -> 'UVAxis':
        """Rotate and translate the texture coordinates."""
        vec = self.vec() @ angles

        # Fix offset - see source-sdk: utils/vbsp/map.cpp line 2237
        offset = self.offset - origin.dot(vec) / self.scale

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
            f'[{self.x:g} {self.y:g} {self.z:g} '
            f'{self.offset:g}] {self.scale:g}'
        )

    def __repr__(self) -> str:
        rep = f'{self.__class__.__name__}({self.x:g}, {self.y:g}, {self.z:g}'
        if self.offset != 0:
            rep += f', offset={self.offset:g}'
        if self.scale != 0.25:
            rep += f', scale={self.scale:g}'
        return rep + ')'


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

    disp_power: int
    disp_pos: Optional[Vec]
    disp_elevation: float
    disp_flags: DispFlag
    disp_allowed_vert: Optional[Array]
    _disp_verts: Optional[List[DispVertex]]

    def __init__(
        self,
        vmf_file: VMF,
        planes: List[Vec],
        des_id: int=-1,
        lightmap: int=16,
        smoothing: int=0,
        mat: str='tools/toolsnodraw',
        rotation: float=0,
        uaxis: Optional[UVAxis]=None,
        vaxis: Optional[UVAxis]=None,
        disp_power: int=0,
    ):
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
    def parse(cls, vmf_file: VMF, tree: Property) -> 'Side':
        """Parse the property tree into a Side object."""
        # planes = "(x1 y1 z1) (x2 y2 z2) (x3 y3 z3)"
        verts = tree["plane", "(0 0 0) (0 0 0) (0 0 0)"][1:-1].split(") (")
        if len(verts) != 3:
            raise ValueError('Wrong number of solid planes in "' +
                             tree['plane', ''] +
                             '"')
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
            return side

        # Deal with displacements.
        disp_power = disp_tree.int('power', 4)
        side.disp_power = disp_power
        side.disp_pos = disp_tree.vec('startposition')
        side.disp_elevation = disp_tree.float('elevation')
        disp_flag_ind = disp_tree.int('flags')
        if 0 <= disp_flag_ind <= 16:
            side.disp_flags = _DISP_FLAG_TO_COLL[disp_flag_ind]
        else:
            raise ValueError(f'Invalid displacement flags {disp_flag_ind} in side {side.id}!')
        if disp_tree.bool('subdiv'):
            side.disp_flags |= DispFlag.SUBDIV

        # This always has a key of '10', with 10 '-1's...
        vert_key = disp_tree.find_key('allowed_verts')
        allowed_vert = Array('i', map(int, vert_key['10'].split()))
        if len(allowed_vert) != 10:
            raise ValueError(
                f'Displacement allowed_verts in side {side.id} '
                f'must be 10 long!'
            )
        side.disp_allowed_vert = allowed_vert

        size = side.disp_size
        side._disp_verts = [
            DispVertex(x, y)
            for y in range(size)
            for x in range(size)
        ]
        # Parse all the rows..
        side._parse_disp_vecrow(disp_tree, 'normals', 'normal')
        side._parse_disp_vecrow(disp_tree, 'offsets', 'offset')
        side._parse_disp_vecrow(disp_tree, 'offset_normals', 'offset_norm')

        for y, row in side._iter_disp_row(disp_tree, 'alphas', size):
            try:
                for x, alpha in enumerate(row):
                    side._disp_verts[y * size + x].alpha = float(alpha)
            except ValueError as exc:
                raise ValueError(
                    f'Displacement array for alpha in side {side.id}, '
                    f'row {y} had invalid number: {exc.args[0]}'
                ) from None

        for y, row in side._iter_disp_row(disp_tree, 'distances', size):
            try:
                for x, alpha in enumerate(row):
                    side._disp_verts[y * size + x].distance = float(alpha)
            except ValueError as exc:
                raise ValueError(
                    f'Displacement array for distances in side {side.id}, '
                    f'row {y} had invalid number: {exc.args[0]}'
                ) from None

        # Not the same, 1 less row and column since it's per-quad.
        tri_tags_count = 2 ** disp_power
        for y, row in side._iter_disp_row(disp_tree, 'triangle_tags', 2 * tri_tags_count):
            try:
                for x in range(tri_tags_count):
                    vert = side._disp_verts[y * size + x]
                    vert.triangle_a = TriangleTag(int(row[2 * x]))
                    vert.triangle_b = TriangleTag(int(row[2 * x + 1]))
            except ValueError as exc:
                raise ValueError(
                    f'Displacement array for triangle tags in side {side.id}, '
                    f'row {y} had invalid number: {exc.args[0]}'
                ) from None

        if 'multiblend' not in disp_tree:
            return side
        # Else: Parse multiblend too.
        # First initialise this list.
        for vert in side._disp_verts:
            vert.multi_colors = [Vec(1, 1, 1), Vec(1, 1, 1), Vec(1, 1, 1), Vec(1, 1, 1)]
        for i in range(4):
            side._parse_disp_vecrow(disp_tree, 'multiblend_color_' + str(i), i)

        for y, split in side._iter_disp_row(disp_tree, 'multiblend', 4 * size):
            try:
                for x in range(size):
                    side._disp_verts[y * size + x].multi_blend = Vec4(
                        float(split[4*x]),
                        float(split[4*x + 1]),
                        float(split[4*x + 2]),
                        float(split[4*x + 3]),
                    )
            except ValueError as exc:
                raise ValueError(
                    f'Displacement array for multiblend in side {side.id}, '
                    f'row {y} had invalid number: {exc.args[0]}'
                ) from None

        for y, split in side._iter_disp_row(disp_tree, 'alphablend', 4 * size):
            try:
                for x in range(size):
                    side._disp_verts[y * size + x].multi_alpha = Vec4(
                        float(split[4*x]),
                        float(split[4*x + 1]),
                        float(split[4*x + 2]),
                        float(split[4*x + 3]),
                    )
            except ValueError as exc:
                raise ValueError(
                    f'Displacement array for multiblend in side {side.id}, '
                    f'row {y} had invalid number: {exc.args[0]}'
                ) from None
        return side

    def _iter_disp_row(self, tree: Property, name: str, size: int) -> Iterator[Tuple[int, List[str]]]:
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

    def _parse_disp_vecrow(self, tree: Property, name: str, member: Union[str, int]) -> None:
        """Parse one of the very similar per-vert sections.

        If member is a string, it is an attribute to set on the DispVertex.
        Otherwise if it's an int, it's the channel for multi_colors (which must
        have been initialised beforehand).
        """
        assert self._disp_verts is not None
        size = self.disp_size
        for y, split in self._iter_disp_row(tree, name, 3 * size):
            try:
                for x in range(size):
                    res = Vec(float(split[3 * x]), float(split[3 * x + 1]), float(split[3 * x + 2]))
                    if isinstance(member, str):
                        setattr(self._disp_verts[y * size + x], member, res)
                    else:
                        # multi_colors could be None, but we make sure to fix
                        # that before calling this.
                        self._disp_verts[y * size + x].multi_colors[member] = res  # type: ignore
            except ValueError as exc:
                raise ValueError(
                    f'Displacement array for {name} in side {self.id}, '
                    f'row {y} had invalid number: {exc.args[0]}'
                ) from None

    def copy(
        self,
        des_id: int=-1,
        vmf_file: VMF=None,
        side_mapping: MutableMapping[int, int]=EmptyMapping,
    ) -> 'Side':
        """Duplicate this brush side.

        des_id is the id which is desired for the new side.
        map is the VMF to add the new side to (defaults to the same map).
        If passed, side_mapping will be updated with a old -> new ID pair.
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
        return new_side

    # noinspection PyProtectedMember
    def export(self, buffer: IO[str], ind: str='', disp_multiblend: bool = True) -> None:
        """Generate the strings required to define this side in a VMF.

        - disp_multiblend controls whether displacements produce their multiblend
          data (added in CSGO), or if it is skipped.
        """
        buffer.write(ind + 'side\n')
        buffer.write(ind + '{\n')
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
        if self.disp_power > 0:
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

        buffer.write(ind + '}\n')

    def _export_disp_rowset(self, name: str, membr: str, f: IO[str], ind: str, size: int) -> None:
        """Write out one of the displacement vertex arrays."""
        assert self._disp_verts is not None
        f.write(f'{ind}\t\t{name}\n{ind}\t\t{{\n')
        for y in range(size):
            row = [
                str(getattr(vert, membr))
                for vert in self._disp_verts[size * y:size * (y+1)]
            ]
            f.write(f'{ind}\t\t"row{y}" "{" ".join(row)}"\n')
        f.write(ind + '\t\t}\n')

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

    def translate(self, diff: Vec) -> None:
        """Move this side by the specified vector.

        - A tuple can be passed in instead if desired.
        """
        for p in self.planes:
            p += diff

        u_axis = Vec(self.uaxis.x, self.uaxis.y, self.uaxis.z)
        v_axis = Vec(self.vaxis.x, self.vaxis.y, self.vaxis.z)

        # Fix offset - see 2013 SDK utils/vbsp/map.cpp:2237
        self.uaxis.offset -= diff.dot(u_axis) / self.uaxis.scale
        self.vaxis.offset -= diff.dot(v_axis) / self.vaxis.scale

    def localise(self, origin: Vec, angles: Union[Matrix, Angle]=None) -> None:
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

    def plane_desc(self) -> str:
        """Return a string which describes this face.

         This is for use in texture randomisation.
         """
        warnings.warn('This is useless and will be removed.', DeprecationWarning)
        return (
            self.planes[0].join(' ') +
            self.planes[1].join(' ') +
            self.planes[2].join(' ')
            )

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

    scale = property(fset=_scale_setter, doc='Set both scale attributes easily.')
    offset = property(fset=_offset_setter, doc='Set both offset attributes easily.')
    del _scale_setter, _offset_setter


class Entity:
    """A representation of either a point or brush entity.

    Creation:
    Entity(args) for a brand-new Entity
    Entity.parse(property) if reading from a VMF file
    ent.copy() to duplicate an existing entity

    Supports [] operations to read and write keyvalues.
    To read instance $replace values operate on entity.fixup[]
    """
    def __init__(
        self,
        vmf_file: VMF,
        keys: Mapping[str, ValidKVs]=EmptyMapping,
        fixup: Iterable['FixupValue']=(),
        ent_id: int=-1,
        outputs: List['Output']=None,
        solids: List[Solid]=None,
        hidden: bool=False,
        groups: Iterable['EntityGroup']=(),
        vis_ids=(),
        vis_shown: bool=True,
        vis_auto_shown: bool=True,
        logical_pos: str=None,
        editor_color: Union[Vec, Tuple[int, int, int]]=(255, 255, 255),
        comments: str='',
    ):
        self.map = vmf_file
        self.keys: Dict[str, str] = {
            k: conv_kv(v)
            for k, v in
            keys.items()
        }
        self.fixup = EntityFixup(fixup)
        self.outputs: list[Output] = outputs or []
        self.solids: list[Solid] = solids or []
        self.id = vmf_file.ent_id.get_id(ent_id)
        self.hidden = hidden
        self.groups = list(groups)

        self.visgroup_ids = set(vis_ids)
        self.vis_shown = vis_shown
        self.vis_auto_shown = vis_auto_shown
        self.editor_color = Vec(editor_color)
        self.logical_pos = logical_pos or f'[0 {self.id}]'
        self.comments = comments

    def copy(
        self,
        des_id: int=-1,
        vmf_file: VMF=None,
        side_mapping: MutableMapping[int, int]=EmptyMapping,
        keep_vis=True,
    ) -> 'Entity':
        """Duplicate this entity entirely, including solids and outputs."""
        new_keys: dict[str, str] = {}
        new_fixup = self.fixup.copy_values()
        for key, value in self.keys.items():
            new_keys[key] = value

        new_solids = [
            solid.copy(vmf_file=vmf_file, side_mapping=side_mapping)
            for solid in
            self.solids
        ]
        outs = [o.copy() for o in self.outputs]

        new_groups = [group.copy() for group in self.groups]

        return Entity(
            vmf_file=vmf_file or self.map,
            keys=new_keys,
            fixup=new_fixup,
            ent_id=des_id,
            outputs=outs,
            solids=new_solids,
            hidden=self.hidden if keep_vis else False,
            groups=new_groups,

            editor_color=self.editor_color,
            logical_pos=self.logical_pos,
            vis_shown=self.vis_shown if keep_vis else True,
            vis_auto_shown=self.vis_auto_shown if keep_vis else True,
            vis_ids=self.visgroup_ids if keep_vis else (),
            comments=self.comments,
        )

    @staticmethod
    def parse(vmf_file, tree_list: Property, hidden=False):
        """Parse a property tree into an Entity object."""
        ent_id = -1
        solids = []
        keys = {}
        outputs = []
        fixup = []
        groups = []
        visgroups = []
        vis_shown = vis_auto_shown = True
        logical_pos = None
        comment = ''
        editor_color = Vec()
        for item in tree_list:
            name = item.name
            if name == "id" and item.value.isnumeric():
                ent_id = int(item.value)
            elif name.startswith('replace'):
                ind_str = name[-2:]  # Index is the last 2 digits
                try:
                    index = int(ind_str)
                except ValueError:  # Not a replace value!
                    keys[name] = item.value
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
                        fixup.append(FixupValue(var, value, int(index)))
                    except ValueError:
                        # Failed!
                        keys[name] = item.value
            elif name == "solid" and item.has_children():
                solids.append(Solid.parse(vmf_file, item))
            elif name == "connections" and item.has_children():
                for out in item:
                    outputs.append(Output.parse(out))
            elif name == "hidden" and item.has_children():
                solids.extend(
                    Solid.parse(vmf_file, br, hidden=True)
                    for br in
                    item
                )
            elif name == "group" and item.has_children():
                groups.append(EntityGroup.parse(vmf_file, item))
            elif name == "editor" and item.has_children():
                for v in item:
                    if v.name == "visgroupshown":
                        vis_shown = srctools.conv_bool(v.value, default=True)
                    elif v.name == "visgroupautoshown":
                        vis_auto_shown = srctools.conv_bool(v.value, default=True)
                    elif v.name == 'color':
                        editor_color = Vec.from_str(v.value, 255, 255, 255)
                    elif v.name == 'logicalpos':
                        logical_pos = v.value
                    elif v.name == 'comments':
                        comment = v.value
                    elif v.name == 'group':
                        groups.append(int(v.value))
                    elif v.name == 'visgroupid':
                        val = srctools.conv_int(v.value, default=-1)
                        if val:
                            visgroups.append(val)
            else:
                keys[item.name] = item.value

        return Entity(
            vmf_file,
            keys,
            fixup,
            ent_id,
            outputs,
            solids,
            hidden,
            groups,
            visgroups,
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
        ent_name: str='entity',
        ind: str='',
        disp_multiblend: bool = True,
    ) -> None:
        """Generate the strings needed to create this entity.

        - ent_name is the key used for the item's block, which is used to allow
          generating the MapSpawn data block from the entity object.
        - disp_multiblend controls whether displacements produce their multiblend
          data (added in ASW), or if it is skipped
        """

        if self.hidden:
            buffer.write(f'{ind}hidden\n{ind}{{\n')
            ind += '\t'

        buffer.write(f'{ind}{ent_name}\n')
        buffer.write(ind + '{\n')
        buffer.write(f'{ind}\t"id" "{str(self.id)}"\n')
        for key, value in sorted(self.keys.items(), key=operator.itemgetter(0)):
            buffer.write(f'{ind}\t"{key}" "{value!s}"\n')

        self.fixup.export(buffer, ind)

        if self.is_brush():
            for s in self.solids:
                s.export(buffer, ind=ind+'\t', disp_multiblend=disp_multiblend)
        if len(self.outputs) > 0:
            buffer.write(ind + '\tconnections\n')
            buffer.write(ind + '\t{\n')
            for o in self.outputs:
                o.export(buffer, ind=ind+'\t\t')
            buffer.write(ind + '\t}\n')

        buffer.write(ind + '\teditor\n')
        buffer.write(ind + '\t{\n')
        buffer.write(f'{ind}\t\t"color" "{self.editor_color}"\n')

        for group in self.groups:
            buffer.write(f'{ind}\t\t"groupid" "{group}"\n')

        for group in self.visgroup_ids:
            buffer.write(f'{ind}\t\t"visgroupid" "{group}"\n')

        buffer.write('{}\t\t"visgroupshown" "{}"\n'.format(
            ind,
            srctools.bool_as_int(self.vis_shown),
        ))
        buffer.write('{}\t\t"visgroupautoshown" "{}"\n'.format(
            ind,
            srctools.bool_as_int(self.vis_auto_shown),
        ))
        buffer.write(f'{ind}\t\t"logicalpos" "{self.logical_pos}"\n')
        if self.comments:
            buffer.write(f'{ind}\t\t"comments" "{self.comments}"\n')
        buffer.write(ind + '\t}\n')

        buffer.write(ind + '}\n')
        if self.hidden:
            buffer.write(ind[:-1] + '}\n')

    def sides(self) -> Iterable['Side']:
        """Iterate through all our brush sides."""
        if self.is_brush():
            for solid in self.solids:
                for face in solid:
                    yield face

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

    def make_unique(self, unnamed_prefix='') -> 'Entity':
        """Ensure this entity is uniquely named, by adding a numeric suffix.

        If the entity doesn't start with a name, it will use the parameter.
        """
        orig_name = self['targetname']
        if orig_name:
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

    def __str__(self) -> str:
        """Dump a user-friendly representation of the entity."""
        st = "<Entity>: \n{\n"
        for k, v in self.keys.items():
            if not isinstance(v, list):
                st += f"\t {k} = \"{v}\"\n"
        for k, v in self.fixup.items():
            st += f"\t ${k} = \"{v}\"\n"

        for out in self.outputs:
            st += f'\t{out!s}\n'
        st += "}\n"
        return st

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
        for k in self.keys:
            if k.casefold() == key:
                return self.keys[k]
        else:
            return default

    def __setitem__(
        self,
        key: str,
        val: ValidKVs,
    ) -> None:
        """Allow using [] syntax to save a keyvalue.

        - It is case-insensitive, so it will overwrite a key which only
          differs by case.
        - Booleans are treated specially, all other types are stringified.
        """
        str_val = conv_kv(val)
        key_fold = key.casefold()
        for k in self.keys:
            if k.casefold() == key_fold:
                # Check case-insensitively for this key first
                orig_val = self.keys.get(k)
                self.keys[k] = str_val
                key = k
                break
        else:
            orig_val = self.keys.get(key)
            self.keys[key] = str_val

        # Update the by_class/target dicts with our new value
        if key_fold == 'classname':
            with suppress(KeyError):
                self.map.by_class[orig_val].remove(self)
            self.map.by_class[str_val].add(self)
        elif key_fold == 'targetname':
            with suppress(KeyError):
                self.map.by_target[orig_val].remove(self)
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
                self.keys[key] = str(self.map.node_id.get_id(node_id))

    def __delitem__(self, key: str) -> None:
        key = key.casefold()
        if key == 'targetname':
            with suppress(KeyError):
                self.map.by_target[
                    self.keys.get('targetname', None)
                ].remove(self)
            self.map.by_target[None].add(self)

        if key == 'classname':
            with suppress(KeyError):
                self.map.by_class[
                    self.keys.get('classname', None)
                ].remove(self)
            self.map.by_class[None].add(self)

        for k in self.keys:
            if k.casefold() == key:
                val = self.keys.pop(k)
                if key == 'nodeid':
                    try:
                        node_id = int(val)
                    except (TypeError, ValueError):
                        pass
                    else:
                        self.map.node_id.discard(node_id)
                break

    def get(self, key: str, default: Union[str, T]='') -> Union[str, T]:
        """Allow using [] syntax to search for keyvalues.

        - This will return '' if the value is not present.
        - It ignores case-matching, but will use the first given version
          of a key.
        - If used via Entity.get() the default argument is available.
        - A tuple can be passed for the default to be set, inside the
          [] syntax.
        """
        key = key.casefold()
        for k in self.keys:
            if k.casefold() == key:
                return self.keys[k]
        else:
            return default

    def clear_keys(self) -> None:
        """Remove all keyvalues from an item."""
        # Delete these so the .by_class/name values are cleared.
        del self['targetname']
        del self['classname']
        self.keys.clear()
        # Clear $fixup as well.
        self.fixup.clear()

    def __contains__(self, key: str) -> bool:
        """Determine if a value exists for the given key."""
        key = key.casefold()
        for k in self.keys:
            if k.casefold() == key:
                return True
        else:
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


@attr.define(weakref_slot=False)
class FixupValue:
    """One $fixup variable with its replacement."""
    var: str  # The original casing of the variable name.
    value: str
    id: int  # replaceXX number used.


class EntityFixup(MutableMapping[str, str]):
    """A specialised mapping which keeps track of the variable indexes.

    This treats variable names case-insensitively, and optionally allows
    writing variables with $ signs in front. The first assigned name for each
    key is preserved however.

    Additionally, lookups never fail - returning '' instead. Pass in a non-string
    default or use `in` to distinguish,.
    """

    # Because of the int(), bool(), float() methods, we need to use builtins.*
    # for the type annotations.
    __slots__ = ['_fixup', '_matcher']

    def __init__(self, fixup: Iterable[FixupValue]=()):
        self._fixup: dict[str, FixupValue] = {}
        self._matcher: Optional[Pattern[str]] = None
        # In _fixup each variable is stored as a tuple of (var_name,
        # value, index) with keys equal to the casefolded var name.
        # var_name is kept to allow restoring the original case when exporting.

        # Do a check to ensure all fixup values have valid indexes:
        used_indexes: set[int] = set()
        extra_vals: list[FixupValue] = []
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

    def get(self, var: str, default: Union[str, T]='') -> Union[str, T]:
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

    def __deepcopy__(self, memodict=None) -> 'EntityFixup':
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

    def setdefault(self, var: str, default: Union[ValidKV_T, str]='') -> Union[str, ValidKV_T]:
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

    def keys(self) -> Iterator[str]:
        """Iterate over all set variable names."""
        for value in self._fixup.values():
            yield value.var

    def __iter__(self) -> Iterator[str]:
        """Iterate over all set variable names."""
        return self.keys()

    def items(self) -> Iterator[Tuple[str, str]]:
        """Iterate over all variable-value pairs."""
        for value in self._fixup.values():
            yield value.var, value.value

    def values(self) -> Iterator[str]:
        """Iterate over all variable values."""
        for value in self._fixup.values():
            yield value.value

    def export(self, buffer: IO[str], ind: str) -> None:
        """Export all the replace values into the VMF."""
        for fixup in sorted(self._fixup.values(), key=operator.attrgetter('id')):
            # When exporting, pad the index with zeros if needed
            buffer.write(
                f'{ind}\t"replace{fixup.id:02}" "${fixup.var} {fixup.value}"\n'
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

    def substitute(self, text: str, default: str=None, *, allow_invert: bool=False) -> str:
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
        if '$' not in text:
            return text

        # Cache the pattern used, we can reuse it whenever called again without
        # adding new variables.
        if self._matcher is None:
            # Sort longer values first, so they are checked before smaller
            # counterparts.
            sections = [
                re.escape(key)
                for key in
                sorted(self._fixup.keys(), key=len, reverse=True)
            ]
            # ! maybe, $, any known fixups, then a default any-identifier check.
            self._matcher = re.compile(
                rf'(!)?\$({"|".join(sections)}|[a-z_][a-z0-9_]*)',
                re.IGNORECASE,
            )

        def replacer(match: 'Match[str]') -> str:
            """Handles the replacement semantics."""
            has_inv, varname = match.groups()
            try:
                res = self._fixup[varname.casefold()].value
            except KeyError:
                if default is None:
                    raise KeyError('$' + varname) from None
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

    def int(self, key: str, def_: Union[builtins.int, T]=0) -> Union[builtins.int, T]:
        """Return the value of an integer key.

        Equivalent to int(fixup[key]), but with a default value if missing or
        invalid.
        """
        try:
            return int(self.get(key))
        except (ValueError, TypeError):
            return def_

    def float(self, key: str, def_: Union[builtins.float, T]=0.0) -> Union[builtins.float, T]:
        """Return the value of an integer key.

        Equivalent to float(fixup[key]), but with a default value if missing or
        invalid.
        """
        try:
            return float(self.get(key))
        except (ValueError, TypeError):
            return def_

    def bool(self, key: str, def_: Union[builtins.bool, T]=False) -> Union[builtins.bool, T]:
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
        x: builtins.float=0.0,
        y: builtins.float=0.0,
        z: builtins.float=0.0,
    ) -> Vec:
        """Return the given fixup, converted to a vector."""
        return Vec.from_str(self.get(key), x, y, z)


class EntityGroup:
    """Represents the 'group' blocks in entities.

    This allows the grouping of brushes.
    """
    def __init__(
        self,
        vmf_file: VMF,
        grp_id: int,
        vis_shown: bool=False,
        vis_auto_shown: bool=False,
    ) -> None:
        self.map = vmf_file
        self.id = vmf_file.group_id.get_id(grp_id)
        self.shown = vis_shown
        self.auto_shown = vis_auto_shown

    @classmethod
    def parse(cls, vmf_file: VMF, props: Property) -> 'EntityGroup':
        """Parse an entity group from the VMF file."""
        editor_block = props.find_block('editor', or_blank=True)
        return cls(
            vmf_file,
            props.int('id', -1),
            vis_shown=editor_block.bool('visgroupshown', True),
            vis_auto_shown=editor_block.bool('visgroupsautoshown', True),
        )

    def copy(self, vmf_file: VMF=None) -> 'EntityGroup':
        """Duplicate an entity group."""
        if vmf_file is None:
            vmf_file = self.map
        return EntityGroup(
            vmf_file,
            self.id,
            self.shown,
            self.auto_shown,
        )

    def export(self, buffer: IO[str], ind: str) -> None:
        """Write out a group into a VMF file."""
        buffer.write(ind + 'group\n')
        buffer.write(ind + '\t{\n')
        buffer.write(f'{ind}\t"id" "{str(self.id)}"\n')
        buffer.write(ind + '\teditor\n')
        buffer.write(ind + '\t\t{\n')
        buffer.write(ind + '\t\t"visgroupshown" "{}"'.format(
            srctools.bool_as_int(self.shown)
        ))
        buffer.write(ind + '\t\t"visgroupautoshown" "{}"'.format(
            srctools.bool_as_int(self.auto_shown)
        ))
        buffer.write(ind + '\t\t}\n')
        buffer.write(ind + '\t}')


class Output:
    """An output from one entity pointing to another.

    Attributes:
        output: The output which triggers this.
        target: The target entity.
        input: The input to fire.
        params: Parameters to give the input, or '' for none.
        delay: The number of seconds before the output should fire.

    Keyword only parameters:
        inst_out: The local entity for an instance output (instance:name;Output)
        inst_in: The local entity we are really triggering in instance inputs
            (instance:name;Input)
        comma_sep: Use a comma as a separator, instead of the OUTPUT_SEP
            character.
        times: The number of times to fire before being deleted.
            -1 means forever, Hammer only uses (-1, 1).
        only_once: Boolean alternative to 'times', setting -1/1 based on
            True/False.

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

    # Make this available here also.
    SEP = OUTPUT_SEP

    def __init__(
        self,
        out: str,
        targ: Union[Entity, str],
        inp: str,
        param: ValidKVs='',
        delay: float=0.0,
        *,
        times: int=-1,
        only_once: bool=False,
        inst_out: str=None,
        inst_in: str=None,
        comma_sep: bool=False,
    ):
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
        """Check if the output is active only once."""
        return self.times == 1

    @only_once.setter
    def only_once(self, is_once: bool) -> None:
        self.times = 1 if is_once else -1

    @classmethod
    def parse(cls, prop: Property) -> 'Output':
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
            times=min(first.times, second.times),
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
            except ValueError as e:
                # Incorrectly-formatted instance: names will crash VBSP,
                # so abort now.
                raise Exception(
                    f'"Instance:" in/output without command! ({name})'
                ).with_traceback(e.__traceback__)
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
            st += self.inst_out + ":"
        st += self.output + " -> " + self.target
        if self.inst_in:
            st += "-" + self.inst_in
        st += " -> " + self.input

        if self.params and not self.inst_in:
            st += " (" + self.params + ")"
        if self.delay != 0:
            st += " after " + str(self.delay) + " seconds"
        if self.times != -1:
            st += " (once" if self.times == 1 else f" ({self.times!s} times"
            st += " only)"
        return st

    def __getstate__(self) -> tuple:
        """Produce the state for pickling.

        We know output/input names tend to be the same often,
        so interning here will simplify the pickle.
        """
        basic = (
            intern(self.output),
            intern(self.target),
            intern(self.input),
            self.comma_sep,
        )
        # Instance, delays and times are more rare - if unset don't include.
        if self.inst_in or self.inst_out or self.params or self.delay or self.times != -1:
            return basic + (
                intern(self.inst_out) if self.inst_out is not None else None,
                intern(self.inst_in) if self.inst_in is not None else None,
                intern(self.params),
                self.delay,
                self.times,
            )
        else:
            return basic

    def __setstate__(self, state: tuple) -> None:
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

    def export(self, buffer: IO[str], ind: str='') -> None:
        """Generate the text required to define this output in the VMF."""
        buffer.write(ind + self.as_keyvalue())

    def as_keyvalue(self) -> str:
        """Generate the text form of the output."""
        sep = ',' if self.comma_sep else self.SEP
        return (
            f'"{self.exp_out()}" "{self.target}{sep}{self.exp_in()}'
            f'{sep}{self.params}{sep}{self.delay:g}{sep}{self.times}"\n'
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

    def gen_addoutput(self, delim: str=',') -> str:
        """Return the parameter needed to create this output via AddOutput.

        This assumes the target instance uses Prefix fixup, if inst_in is set.
        """
        if self.inst_out:
            raise ValueError('Inst_out is not useable in AddOutput.')

        if self.inst_in:
            target = self.target + '-' + self.inst_in
        else:
            target = self.target

        return (
            f'{self.output} {target}{delim}{self.input}{delim}'
            f'{self.params}{delim}{self.delay}{delim}{self.times}'
        )
