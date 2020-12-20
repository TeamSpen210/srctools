"""Implements support for collapsing instances."""
from enum import Enum
from typing import Dict, Tuple, List, Set, Iterable

from srctools import Matrix, Vec, Angle
from srctools.vmf import Entity, EntityFixup, FixupTuple, VMF, Output
from srctools.fgd import ValueTypes
import srctools.logger


LOGGER = srctools.logger.get_logger(__name__)


class FixupStyle(Enum):
    """The kind of fixup style to use."""
    NONE = 'none'
    PREFIX = 'prefix'
    SUFFIX = 'suffix'


class Instance:
    """Represents an instance with all values parsed."""
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
