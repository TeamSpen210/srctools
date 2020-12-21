"""Implements support for collapsing instances."""
from enum import Enum
from typing import Dict, Tuple, List, Set, Iterable, AbstractSet, Container

from srctools import Matrix, Vec, Angle, conv_float, conv_int
from srctools.vmf import Entity, EntityFixup, FixupTuple, VMF, Output
from srctools.fgd import ValueTypes, FGD, EntityDef, EntityTypes
import srctools.logger


LOGGER = srctools.logger.get_logger(__name__)


class FixupStyle(Enum):
    """The kind of fixup style to use."""
    NONE = 'none'
    PREFIX = 'prefix'
    SUFFIX = 'suffix'


class Instance:
    """Represents an instance with all the values required to collapse it."""
    def __init__(
        self,
        name: str,
        filename: str,
        pos: Vec, orient: Matrix,
        fixup_type: FixupStyle,
        fixup: Iterable[FixupTuple]=(),
    ) -> None:
        self.name = name
        self.filename = filename
        self.pos = pos
        self.orient = orient
        self.fixup_type = fixup_type
        self.fixup = EntityFixup(fixup)
        # After collapsing, this is the original -> new ID mapping.
        self.ent_ids: Dict[int, int] = {}
        self.face_ids: Dict[int, int] = {}
        self.brush_ids: Dict[int, int] = {}
        self.node_ids: Dict[int, int] = {}

    @classmethod
    def from_entity(cls, ent: Entity) -> 'Instance':
        """Parse a func_instance entity."""
        name = ent['targetname']
        filename = ent['file']
        try:
            fixup_style = FixupStyle(int(ent['fixup_style', '0']))
        except ValueError:
            LOGGER.warning(
                'Invalid fixup style "{}" on func_instance "{}" at {}',
                ent['fixup_style'], name, ent['origin'],
            )
            fixup_style = FixupStyle.PREFIX
        return cls(
            name,
            filename,
            Vec.from_str(ent['origin']),
            Matrix.from_angle(Angle.from_str(ent['angles'])),
            fixup_style,
            ent.fixup.copy_values(),
        )

    def fixup_name(self, name: str) -> str:
        """Apply the name fixup rules to this name."""
        if name.startswith(('@', '!')):
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

        - classnames is a set of known entity classnames, used to avoid renaming
        those.
        """
        # All three are absolute positions.
        if type is ValueTypes.VEC or type is ValueTypes.VEC_ORIGIN or type is ValueTypes.VEC_LINE:
            return str(Vec.from_str(value) @ self.orient + self.pos)
        elif type is ValueTypes.ANGLES:
            return str(Angle.from_str(value) @ self.orient)
        elif type.is_ent_name:  # Target destination etc.
            return self.fixup_name(value)
        elif type is ValueTypes.TARG_DEST_CLASS:
            # Target destination, but also classnames which we don't want to change.
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
        elif type is ValueTypes.VEC_AXIS:
            value = str(Vec.from_str(value) @ self.orient)
        elif type is ValueTypes.TARG_NODE_SOURCE or ValueTypes.TARG_NODE_DEST:
            # For each old ID always create a new ID.
            try:
                old_id = int(value)
            except (ValueError, TypeError):
                return value  # Skip.
            try:
                value = str(self.node_ids[old_id])
            except KeyError:
                self.node_ids[old_id] = new_id = vmf.node_id.get_id()
                value = str(new_id)
        elif type is ValueTypes.VEC_AXIS:
            # Two positions seperated by commas.
            first, second = value.split(',')
            first = Vec.from_str(first) @ self.orient + self.pos
            second = Vec.from_str(second) @ self.orient + self.pos
            value = f'{first}, {second}'
        # All others = no change required.
        return value


class Param:
    """Configuration for a specific fixup variable."""
    def __init__(
        self,
        name: str,
        type: ValueTypes=ValueTypes.STRING,
        default: str='',
    ) -> None:
        self.name = name
        self.type = type
        self.default = default


class InstanceFile:
    """Represents an instance VMF which has been parsed."""
    def __init__(self, vmf: VMF) -> None:
        self.vmf = vmf
        self.params: Dict[str, Param] = {}
        # Inputs into the instance. The key is the parts of the instance:name;input string.
        self.proxy_inputs: Dict[Tuple[str, str], Output] = {}
        # Outputs out of the instance. The key is the parts of the instance:name;output string.
        # The value is the ID of the entity to add the output to.
        self.proxy_outputs: Dict[Tuple[str, str], Tuple[int, Output]] = {}

        # If instructed to add in a proxy later, this is the local pos to place
        # it.
        self.proxy_pos = Vec()

        self.parse()

    def parse(self) -> None:
        """Parse func_instance_params and io_proxies in the map."""
        for params_ent in self.vmf.by_class['func_instance_params']:
            params_ent.remove()
            for key, value in params_ent.keys.items():
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


def collapse_one(
    vmf: VMF,
    inst: Instance,
    file: InstanceFile,
    fgd: FGD=None,
) -> None:
    """Collapse a single instance into the map.

    The FGD is the data used to localise keyvalues. If none an internal database
    will be used.
    """
    origin = inst.pos
    orient = inst.orient

    if fgd is None:
        fgd = FGD.engine_dbase()
    # Contains all base-entity keyvalues, as a fallback.
    try:
        base_entity = fgd['_CBaseEntity_']
    except KeyError:
        LOGGER.warning('No CBaseEntity definition!')
        base_entity = EntityDef(EntityTypes.BASE)

    for old_brush in file.vmf.brushes:
        if old_brush.hidden or not old_brush.vis_shown:
            continue
        new_brush = old_brush.copy(vmf_file=vmf, side_mapping=inst.face_ids, keep_vis=False)
        inst.brush_ids[old_brush.id] = new_brush.id
        new_brush.localise(origin, orient)

    for old_ent in file.vmf.entities:
        if old_ent.hidden or not old_ent.vis_shown:
            continue
        new_ent = old_ent.copy(vmf_file=vmf, side_mapping=inst.face_ids, keep_vis=False)
        for old_brush, new_brush in zip(old_ent.solids, new_ent.solids):
            inst.brush_ids[old_brush.id] = new_brush.id
            new_brush.localise(origin, orient)

        # Find the FGD to use.
        try:
            ent_type = fgd[new_ent['classname']]
        except KeyError:
            ent_type = base_entity

        # Now keyvalues.
        # First extract a rotated angles value, handling the special "pitch" and "yaw" keys.
        angles = Angle.from_str(new_ent['angles'])
        if 'pitch' in new_ent:
            angles.pitch = conv_float(new_ent['pitch'])
        if 'yaw' in new_ent:
            angles.yaw = conv_float(new_ent['yaw'])
        angles @= orient

        for key, value in new_ent.keys.items():
            folded = key.casefold()
            # Hardcode these critical keyvalues to always be these types.
            if folded == 'origin':
                new_ent['origin'] = str(Vec.from_str(value) @ orient + origin)
            elif folded == 'angles':
                new_ent['angles'] = str(angles)
            elif folded == 'pitch':
                new_ent['pitch'] = str(angles.pitch)
            elif folded == 'yaw':
                new_ent['yaw'] = str(angles.yaw)
            elif folded in ('classname', 'hammerid', 'spawnflags', 'nodeid'):
                continue
            try:
                kv = ent_type.kv[folded]
            except KeyError:
                LOGGER.warning('Unknown keyvalue {}.{}', new_ent['classname'], key)
                continue
            # This has specific interactions with angles, it needs to be the pitch KV.
            if kv.type is ValueTypes.ANGLE_NEG_PITCH:
                LOGGER.warning('angle_negative_pitch should only be applied to pitch, not {}.{}', new_ent['classname'], key)
                continue
            elif kv.type is ValueTypes.INST_VAR_REP:
                LOGGER.warning('instance_variable should only be applied to replaceXX, not {}.{}', new_ent['classname'], key)
                continue

            new_ent.keys[key] = inst.fixup_key(vmf, fgd, kv.type, value)

        # Remap fixups on instance entities too.
        for key, value in new_ent.fixup.items():
            # Match Valve's bad logic here. TODO: Load the InstanceFile and remap accordingly.
            if value and value[0] not in '@!-.0123456789':
                new_ent.fixup[key] = inst.fixup_name(value)
