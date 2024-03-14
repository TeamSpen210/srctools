"""Implements support for collapsing instances."""
from typing import Container, Dict, Iterable, List, MutableMapping, Optional, Set, Tuple, Union
from typing_extensions import deprecated, overload
from enum import Enum
from pathlib import Path
import warnings

import attrs

from srctools.fgd import FGD, EntityDef, EntityTypes, ValueTypes
from srctools.filesys import FileSystem, FileSystemChain, RawFileSystem
from srctools.keyvalues import Keyvalues
from srctools.math import Angle, Matrix, Vec, format_float
from srctools.vmf import VMF, Entity, EntityFixup, FixupValue, Output, VisGroup
import srctools
import srctools.logger


__all__ = [
    'FixupStyle',
    'Instance', 'Manifest', 'Param', 'InstanceFile',
    'get_inst_locs', 'collapse_all', 'collapse_one',
]
LOGGER = srctools.logger.get_logger(__name__)
# Hidden variable to track the number of recursions.
RECUR_COUNT_ATTR = '_inst_recur_count'
# Prevent displaying errors for missing keyvalues multiple times.
_UNKNOWN_KV: Set[Tuple[str, str]] = set()


class FixupStyle(Enum):
    """The kind of fixup style to use."""
    PREFIX = 0  #: Entities will be named :samp:`{inst_name}-{ent_name}`.
    SUFFIX = 1  #: Entities will be named :samp:`{ent_name}-{inst_name}`.
    NONE = 2  #: Entities will remain named :samp:`{ent_name}`.


class Instance:
    """Represents an instance with all the values required to collapse it."""
    name: str
    filename: str
    pos: Vec
    orient: Matrix
    fixup_type: FixupStyle
    fixup: EntityFixup
    recur_count: int
    outputs: List[Output]

    ent_ids: Dict[int, int]
    face_ids: Dict[int, int]
    brush_ids: Dict[int, int]
    node_ids: Dict[int, int]
    visgroup_ids: Dict[int, int]

    def __init__(
        self,
        name: str,
        filename: str,
        pos: Vec, orient: Matrix,
        fixup_type: FixupStyle = FixupStyle.PREFIX,
        outputs: Iterable[Output] = (),
        fixup: Iterable[FixupValue] = (),
    ) -> None:
        self.name = name
        self.filename = filename
        self.pos = pos
        self.orient = orient
        self.fixup_type = fixup_type
        self.fixup = EntityFixup(fixup)
        self.outputs = list(outputs)
        # After collapsing, this is the original -> new ID mapping.
        self.ent_ids = {}
        self.face_ids = {}
        self.brush_ids = {}
        self.node_ids = {}
        self.visgroup_ids = {}
        # Keep track of recursive instances to handle loops.
        self.recur_count = 0

    @classmethod
    def from_entity(cls, ent: Entity) -> 'Instance':
        """Parse a func_instance entity."""
        name = ent['targetname']
        filename = ent['file']
        try:
            fixup_style = FixupStyle(int(ent['fixup_style', '0']))
        except ValueError:
            LOGGER.warning(
                'Invalid fixup style "{}" on func_instance "{}" at {} ({})',
                ent['fixup_style'], name, ent['origin'], filename,
            )
            fixup_style = FixupStyle.PREFIX
        inst = cls(
            name,
            filename,
            Vec.from_str(ent['origin']),
            Matrix.from_angstr(ent['angles']),
            fixup_style,
            ent.outputs,
            ent.fixup.copy_values(),
        )
        inst.recur_count = getattr(ent, RECUR_COUNT_ATTR, 0)
        return inst

    def fixup_name(self, name: str) -> str:
        """Apply the name fixup rules to this name."""
        if not name or name.startswith(('@', '!')):
            return name
        if self.fixup_type is FixupStyle.NONE:
            return name
        elif self.fixup_type is FixupStyle.PREFIX:
            return f'{self.name}-{name}'
        elif self.fixup_type is FixupStyle.SUFFIX:
            return f'{name}-{self.name}'
        else:
            raise AssertionError(f'Unknown fixup type {self.fixup_type}')

    def fixup_key(
        self,
        vmf: VMF,
        classnames: Container[str],
        type: ValueTypes,
        value: str,
    ) -> str:
        """Transform this keyvalue to the new instance's location and name.

        :param vmf: The rest of the map.
        :param classnames: This is a set of known entity classnames, used to avoid renaming those.
        :param type: The kind of FGD value.
        :param value: The value of the key.
        """
        # All three of these types are absolute positions.
        if type is ValueTypes.VEC or type is ValueTypes.VEC_ORIGIN or type is ValueTypes.VEC_LINE:
            return str(Vec.from_str(value) @ self.orient + self.pos)
        elif type is ValueTypes.ANGLES:
            return str(Angle.from_str(value) @ self.orient)
        elif type.is_ent_name:  # Target destination etc.
            return self.fixup_name(value)
        elif type is ValueTypes.TARG_DEST_CLASS:
            # Accepts both entity names and classnames - if this is a classname, pass it through
            # unaltered.
            if value.casefold() not in classnames:
                return self.fixup_name(value)
        elif type is ValueTypes.EXT_VEC_DIRECTION:
            return str(Vec.from_str(value) @ self.orient)
        elif type is ValueTypes.SIDE_LIST:
            # Remap old sides to new. If not found skip.
            sides = []
            for side in value.split():
                try:
                    new_side = self.face_ids[int(side)]
                except (KeyError, ValueError, TypeError):
                    pass
                else:
                    sides.append(str(new_side))
            sides.sort()
            return ' '.join(sides)
        elif type is ValueTypes.TARG_NODE_SOURCE or type is ValueTypes.TARG_NODE_DEST:
            # For each old ID always create a new ID.
            try:
                old_id = int(value)
            except (ValueError, TypeError):
                return value  # Skip.
            try:
                value = str(self.node_ids[old_id])
            except KeyError:
                self.node_ids[old_id] = new_id = vmf.node_id.get_id(old_id)
                value = str(new_id)
        elif type is ValueTypes.VEC_AXIS:
            # Two positions seperated by commas.
            first_str, second_str = value.split(',')
            first = Vec.from_str(first_str) @ self.orient + self.pos
            second = Vec.from_str(second_str) @ self.orient + self.pos
            value = f'{first}, {second}'
        elif type is ValueTypes.ANGLE_NEG_PITCH or type is ValueTypes.EXT_ANGLE_PITCH:
            raise ValueError(f'"{type.name}" keyvalue type must be fixed up on the instance as a whole!')
        elif type is ValueTypes.CHOICES:
            raise ValueError(
                'Choices keyvalue type is not meaningful, it should be swapped with another type!'
            )
        # All others = no change required.
        return value


class Manifest(Instance):
    """Additional options set in VMM manifests."""
    def __init__(
        self,
        name: str,
        filename: str,
        id: int,
        is_toplevel: bool = False,
    ) -> None:
        super().__init__(
            name, filename,
            Vec(), Matrix(),  # Collapsed directly at the existing position.
            FixupStyle.NONE,  # Names are unaltered.
        )
        self.id = id
        self.is_toplevel = is_toplevel

    @classmethod
    def parse(cls, tree: Keyvalues) -> List['Manifest']:
        """Parse a VMM file."""
        return [
            cls(
                prop['Name'], prop['File'],
                prop.int('InternalID'), prop.bool('TopLevel')
            )
            for prop in tree.find_all('Maps', 'VMF')
        ]


@attrs.define
class Param:
    """Configuration for a specific fixup variable."""
    name: str
    type: ValueTypes = ValueTypes.STRING
    default: str = ''


class InstanceFile:
    """Represents an instance VMF which has been parsed."""
    vmf: VMF
    params: Dict[str, Param]
    #: Inputs into the instance. The key is the parts of the instance:name;input string.
    proxy_inputs: Dict[Tuple[str, str], Output]
    #: Outputs out of the instance. The key is the parts of the instance:name;output string.
    #: The value is the ID of the entity to add the output to.
    proxy_outputs: Dict[Tuple[str, str], Tuple[int, Output]]

    #: If instructed to add in a proxy later, this is the local pos to place it.
    proxy_pos: Vec

    def __init__(self, vmf: VMF) -> None:
        self.vmf = vmf
        self.params = {}
        self.proxy_inputs = {}
        self.proxy_outputs = {}
        self.proxy_pos = Vec()

        self.parse()

    def parse(self) -> None:
        """Parse func_instance_params and io_proxies in the map."""
        for params_ent in self.vmf.by_class['func_instance_parms']:
            params_ent.remove()
            for key, value in params_ent.items():
                if not key.startswith('param'):
                    continue
                # Don't bother parsing the index, it doesn't matter.

                # It's allowed to omit values here. The default needs to allow
                # spaces as well.
                parts = value.split(' ', 3)
                name = parts[0]
                var_type = ValueTypes.STRING
                default = ''

                if len(parts) >= 2:
                    try:
                        var_type = ValueTypes(parts[1])
                    except ValueError:
                        pass
                if len(parts) == 3:
                    default = parts[2]
                self.params[name.casefold()] = Param(name, var_type, default)

        proxy_names: Set[str] = set()
        for proxy in self.vmf.by_class['func_instance_io_proxy']:
            proxy.remove()
            self.proxy_pos = Vec.from_str(proxy['origin'])
            proxy_names.add(proxy['targetname'])
            # First, inputs.
            for out in proxy.outputs:
                if out.output.casefold() == 'onproxyrelay':
                    self.proxy_inputs[out.target.casefold(), out.input.casefold()] = out
                    out.output = ''
        # Now, outputs.
        for ent in self.vmf.entities:
            for out in ent.outputs[:]:
                if out.input.casefold() == 'proxyrelay' and out.target.casefold() in proxy_names:
                    ent.outputs.remove(out)
                    self.proxy_outputs[ent['targetname'].casefold(), out.output.casefold()] = (ent.id, out)
                    out.input = out.target = ''


def get_inst_locs(map_filename: Path) -> FileSystemChain:
    """Given a map filename, find sdk_content and produce the lookup locations.

    The chained filesystem will first look relative to the map, then in
    ``sdk_content/maps/`` if that's a parent directory.
    """
    fsys_rel = RawFileSystem(map_filename.parent)
    fsys = FileSystemChain(fsys_rel)
    for parent in map_filename.parents:
        # parent.parent of a root returns self.
        if parent.stem == 'maps' and parent.parent.stem == 'sdk_content':
            fsys.add_sys(RawFileSystem(parent))
            break
    return fsys


def reset_keyvalue_warnings() -> None:
    """If an unknown keyvalue is encountered during collapsing, a log message is produced only once.

    This resets the internal tracker so the messages will be repeated.
    """
    _UNKNOWN_KV.clear()


@overload
def collapse_one(
    vmf: VMF,
    inst: Instance,
    file: InstanceFile,
    *,
    visgroup: Union[bool, VisGroup] = False,
    engine_cache: MutableMapping[str, EntityDef] = srctools.EmptyMapping,
) -> None: ...
@overload
@deprecated('fgd parameter is deprecated, use engine_cache instead', category=None)
def collapse_one(
    vmf: VMF,
    inst: Instance,
    file: InstanceFile,
    *,
    visgroup: Union[bool, VisGroup] = False,
    fgd: FGD,  # <<<
) -> None: ...


def collapse_one(
    vmf: VMF,
    inst: Instance,
    file: InstanceFile,
    *,
    visgroup: Union[bool, VisGroup] = False,
    engine_cache: MutableMapping[str, EntityDef] = srctools.EmptyMapping,
    fgd: Optional[FGD] = None,
) -> None:
    """Collapse a single instance into the map.

    The visgroup parameter controls how visgroups are handled:

    * If :external:py:data:`False`, visgroups are stripped.
    * If :external:py:data:`True`, the original visgroups will be kept.
    * If set to a specific visgroup, all ents and brushes will be added to it, with any existing
      visgroups in the instance added as a child.

    :param engine_cache: This is a dict used to cache parsed FGD data for remapping.
        If multiple instances are being collapsed, create an empty dict and pass it in to every call
        to allow reusing data.
    :param fgd: :deprecated: old method of passing in a FGD database.
    """
    origin = inst.pos
    orient = inst.orient
    id_to_ent: Dict[int, Entity] = {}

    if fgd is not None:
        warnings.warn(
            'fgd parameter is deprecated, use engine_cache instead',
            DeprecationWarning, stacklevel=2,
        )

    if visgroup is not False:
        for old_group in file.vmf.vis_tree:
            new_group = old_group.copy(vmf, inst.visgroup_ids)
            if visgroup is True:
                vmf.vis_tree.append(new_group)
            else:
                visgroup.child_groups.append(new_group)
    if isinstance(visgroup, VisGroup):
        ungrouped_group = {visgroup.id}
    else:
        ungrouped_group = set()

    for old_brush in file.vmf.brushes:
        if old_brush.hidden or not old_brush.vis_shown:
            continue
        new_brush = old_brush.copy(vmf_file=vmf, side_mapping=inst.face_ids, keep_vis=visgroup is not False)
        vmf.add_brush(new_brush)
        inst.brush_ids[old_brush.id] = new_brush.id
        new_brush.localise(origin, orient)
        # Convert across the IDs.
        if visgroup is not False:
            new_brush.visgroup_ids = {
                inst.visgroup_ids[old]
                for old in new_brush.visgroup_ids
            } or ungrouped_group.copy()

    # Before adding the ents, apply instance inputs.
    folded_inst_name = inst.name.casefold()
    for ent in vmf.entities:
        for out in ent.outputs:
            if out.target.casefold() != folded_inst_name or out.inst_in is None:
                continue
            try:
                proxy_out = file.proxy_inputs[out.inst_in, out.input]
            except KeyError:
                # Not an error, could be another instance with our name.
                continue
            # Output.combine(), but in-place.
            out.target = proxy_out.target
            out.input = proxy_out.input
            out.inst_in = None
            if proxy_out.params:
                out.params = proxy_out.params
            out.times = min(out.times, proxy_out.times)
            out.delay += proxy_out.delay
            if not proxy_out.comma_sep:
                out.comma_sep = False

    # Only modify keyvalues after all ents have been copied over, so brush
    # IDs are all present.
    new_ents: List[Entity] = []

    for old_ent in file.vmf.entities:
        if visgroup is False and (old_ent.hidden or not old_ent.vis_shown):
            continue
        new_ent = old_ent.copy(
            vmf_file=vmf,
            side_mapping=inst.face_ids,
            keep_vis=visgroup is not False
        )
        vmf.add_ent(new_ent)
        new_ents.append(new_ent)
        inst.ent_ids[old_ent.id] = new_ent.id
        id_to_ent[old_ent.id] = new_ent

        if visgroup is not False:
            new_ent.visgroup_ids = {
                inst.visgroup_ids[old]
                for old in old_ent.visgroup_ids
            } or ungrouped_group.copy()

        for old_brush, new_brush in zip(old_ent.solids, new_ent.solids):
            inst.brush_ids[old_brush.id] = new_brush.id
            new_brush.localise(origin, orient)

            # Convert across the IDs.
            if visgroup is not False:
                new_brush.visgroup_ids = {
                    inst.visgroup_ids[old]
                    for old in old_brush.visgroup_ids
                } or ungrouped_group.copy()

    for new_ent in new_ents:
        # Find the FGD to use.
        classname = new_ent['classname']
        try:
            ent_type = engine_cache[classname]
        except KeyError:
            try:
                ent_type = EntityDef.engine_def(classname)
            except KeyError:
                # Contains all base-entity keyvalues, as a fallback.
                try:
                    ent_type = EntityDef.engine_def('_CBaseEntity_')
                except KeyError:
                    LOGGER.warning('No CBaseEntity definition!')
                    ent_type = EntityDef(EntityTypes.BASE)
            engine_cache[classname] = ent_type

        # Set a hidden attribute to keep track of recursive instancing.
        if classname.casefold() == 'func_instance':
            setattr(new_ent, RECUR_COUNT_ATTR, inst.recur_count + 1)

        # Now keyvalues.
        # First extract a rotated angles value, handling the special "pitch" and "yaw" keys.
        angles = Angle.from_str(new_ent['angles'])
        if 'pitch' in new_ent:
            angles.pitch = srctools.conv_float(new_ent['pitch'])
            try:
                kv = ent_type.kv['pitch']
            except KeyError:
                pass
            else:
                if kv.type is ValueTypes.ANGLE_NEG_PITCH:
                    angles.pitch = -angles.pitch
        if 'yaw' in new_ent:
            angles.yaw = srctools.conv_float(new_ent['yaw'])
        angles @= orient

        for key, value in new_ent.items():
            folded = key.casefold()
            value = inst.fixup.substitute(value, '')
            # Hardcode these critical keyvalues to always be these types.
            if folded == 'origin':
                new_ent['origin'] = str(Vec.from_str(value) @ orient + origin)
                continue
            elif folded == 'angles':
                new_ent['angles'] = str(angles)
                continue
            elif folded == 'yaw':
                new_ent['yaw'] = format_float(angles.yaw)
                continue
            elif folded in {'classname', 'hammerid', 'spawnflags'}:
                continue

            try:
                kv = ent_type.kv[folded]
            except KeyError:
                if folded.startswith('$') and classname == 'func_instance':
                    # Dummy fixup names Hammer provides for convenience, ignore.
                    continue
                if (classname, key) not in _UNKNOWN_KV:
                    LOGGER.warning('Unknown keyvalue {}.{}', classname, key)
                    _UNKNOWN_KV.add((classname, key))
                continue
            # This has specific interactions with angles, it needs to be the pitch KV.
            if kv.type is ValueTypes.ANGLE_NEG_PITCH:
                if folded == 'pitch':
                    new_ent['pitch'] = format_float(-angles.pitch)
                elif (classname, key) not in _UNKNOWN_KV:
                    LOGGER.warning('angle_negative_pitch should only be applied to pitch, not {}.{}', classname, key)
                    _UNKNOWN_KV.add((classname, key))
                continue
            # Same, but not inverted.
            elif kv.type is ValueTypes.EXT_ANGLE_PITCH:
                if folded == 'pitch':
                    new_ent['pitch'] = format_float(angles.pitch)
                elif (classname, key) not in _UNKNOWN_KV:
                    LOGGER.warning('angle_pitch should only be applied to pitch, not {}.{}', classname, key)
                    _UNKNOWN_KV.add((classname, key))
                continue
            # Instance fixup commands shouldn't exist here.
            elif kv.type is ValueTypes.INST_VAR_REP:
                if (classname, key) not in _UNKNOWN_KV:
                    LOGGER.warning('instance_variable should only be applied to replaceXX, not {}.{}', classname, key)
                    _UNKNOWN_KV.add((classname, key))
                continue

            new_ent[key] = inst.fixup_key(vmf, EntityDef.engine_classes(), kv.type, value)

        # Remap fixups on instance entities too.
        for key, value in new_ent.fixup.items():
            # Match Valve's bad logic here. TODO: Load the InstanceFile and remap accordingly.
            if value and value[0] not in '@!-.0123456789':
                new_ent.fixup[key] = inst.fixup_name(value)

        # Outputs
        for out in new_ent.outputs:
            out.target = inst.fixup_name(inst.fixup.substitute(out.target, ''))

    for out in inst.outputs:
        # Non-instance output, ignore - on regular ents it'd never fire.
        if out.inst_out is None:
            continue
        try:
            ent_id, prox_out = file.proxy_outputs[out.inst_out.casefold(), out.output.casefold()]
        except KeyError:
            LOGGER.info('No output {},{} in {}', out.inst_out, out.output, inst.filename)
            continue
        id_to_ent[ent_id].add_out(Output.combine(prox_out, out))


def collapse_all(
    vmf: VMF,
    fsys: FileSystem,
    recur_limit: int = 100,
) -> None:
    """Searches for ``func_instance`` in the map, then collapses them.

    :param vmf: The map to analyse.
    :param fsys: The filesystem is used to find the relevant instances.
    :param recur_limit: The recursion limit indicates how many instances can be contained in
        another - if it's exceeded they're left in the map.
    """
    auto_inst_count = 0
    fgd_cache: MutableMapping[str, EntityDef] = {}
    file_cache: Dict[str, InstanceFile] = {}

    for _ in range(recur_limit):
        instances = list(vmf.by_class['func_instance'])
        if not instances:
            return  # No more instances, success!
        for inst_ent in instances:
            inst = Instance.from_entity(inst_ent)
            inst_ent.remove()
            LOGGER.debug('Collapse {} @ {}', inst.filename, inst.pos)
            if not inst.name:
                auto_inst_count += 1
                inst.name = f'InstanceAuto{auto_inst_count}'
            try:
                file = file_cache[inst.filename]
            except KeyError:
                props = fsys.read_kv1(inst.filename)
                # except FileNotFoundError - fail.
                file = file_cache[inst.filename] = InstanceFile(VMF.parse(props, preserve_ids=True))
            collapse_one(vmf, inst, file, engine_cache=fgd_cache)

    # Exhausted the range, we must have too much recursion in the instances.
    raise RecursionError('Loop in instances!')
