"""Parse particle system files."""
from typing import IO, Dict, Iterable, List, Union, overload
import copy

import attrs

from srctools.dmx import Attribute, Element, ValueType


__all__ = [
    'FORMAT_NAME', 'FORMAT_VERSION',
    'Operator', 'Child', 'Particle',
]
# The name of the file format used in DMX files.
FORMAT_NAME: str = 'pcf'
# PCF files can have version 1 or 2.
FORMAT_VERSION: int = 2


class Operator:
    """A generic option in particles."""
    def __init__(self, name: str, function: str, options: Dict[str, Attribute]) -> None:
        self.function = function
        self.name = name
        self.options = options

    def __repr__(self) -> str:
        return f'<Op {self.name}.{self.function}({", ".join(map(repr, self.options.values()))})>'


@attrs.define(eq=False)
class Child:
    """Options for a child particle reference."""
    particle: str


@attrs.define(eq=False)
class Particle:
    """A particle system."""
    name: str
    options: Dict[str, Attribute] = attrs.Factory(dict)
    renderers: List[Operator] = attrs.field(factory=list)
    operators: List[Operator] = attrs.field(factory=list)
    initializers: List[Operator] = attrs.field(factory=list)
    emitters: List[Operator] = attrs.field(factory=list)
    forces: List[Operator] = attrs.field(factory=list)
    constraints: List[Operator] = attrs.field(factory=list)
    children: List[Child] = attrs.field(factory=list)

    @classmethod
    @overload
    def parse(
        cls,
        file: Element,
        version: int = FORMAT_VERSION,
    ) -> Dict[str, 'Particle']: ...

    @classmethod
    @overload
    def parse(
        cls,
        file: IO[bytes],
    ) -> Dict[str, 'Particle']: ...

    @classmethod
    def parse(
        cls,
        file: Union[IO[bytes], Element],
        version: int = FORMAT_VERSION,
    ) -> Dict[str, 'Particle']:
        """Parse a PCF file.

        The file can either be a binary file, or an already parsed DMX.
        If already parsed, the format version needs to be passed on.
        """
        systems: Dict[str, Particle] = {}
        if isinstance(file, Element):
            root = file
        else:
            with file:
                root, fmt_name, version = Element.parse(file)
            if fmt_name != FORMAT_NAME:
                raise ValueError(f'DMX file is a "{fmt_name}" format, not "{FORMAT_NAME}"!')

        if version not in (1, 2):
            raise ValueError(f'Unknown particle version {version}!')

        part_list = root['particleSystemDefinitions']
        if part_list.type != ValueType.ELEMENT or not part_list.is_array:
            raise ValueError('"particleSystemDefinitions" must be an element array!')

        def generic_attr(el: Element, name: str) -> List['Operator']:
            try:
                value = el.pop(name)
            except KeyError:
                return []
            if value.type is not ValueType.ELEMENT or not value.is_array:
                raise ValueError('{} must be an element array!')
            return [
                Operator(ele.name, ele.pop('functionName').val_str, copy.deepcopy(dict(ele)))
                for ele in value.iter_elem()
            ]

        for elem in part_list.iter_elem():
            renderers = generic_attr(elem, 'renderers')
            operators = generic_attr(elem, 'operators')
            initializers = generic_attr(elem, 'initializers')
            emitters = generic_attr(elem, 'emitters')
            forces = generic_attr(elem, 'forces')
            constraints = generic_attr(elem, 'constraints')
            try:
                child_attr = elem.pop('children')
                if child_attr.type is not ValueType.ELEMENT or not child_attr.is_array:
                    raise ValueError('Children must be an element array!')
            except KeyError:
                children = []
            else:
                children = [
                    Child(subelem.name) for subelem
                    in child_attr.iter_elem()
                ]
            # Everything else.
            options = {
                value.name.casefold(): copy.deepcopy(value)
                for value in elem.values()
            }

            systems[elem.name.casefold()] = Particle(
                elem.name,
                options,
                renderers,
                operators,
                initializers,
                emitters,
                forces,
                constraints,
                children,
            )
        return systems

    @classmethod
    def export(cls, particles: Iterable['Particle']) -> Element:
        """Reconstruct a DMX file with the specified particles."""
        root = Element('', 'DmElement')
        root['particleSystemDefinitions'] = part_list = Attribute.array('', ValueType.ELEMENT)

        name_to_elem: Dict[str, Element] = {}

        for part in particles:
            part_elem = Element(part.name, 'DmeParticleSystemDefinition')
            part_list.append(part_elem)
            part_elem.name = part.name
            name_to_elem[part.name.casefold()] = part_elem
            for identifier in [
                'renderers', 'operators', 'initializers',
                'emitters', 'forces', 'constraints',
            ]:
                op_list: List[Operator] = getattr(part, identifier)
                part_elem[identifier] = op_attrlist = Attribute.array(identifier, ValueType.ELEMENT)
                for operator in op_list:
                    op_elem = Element(operator.name, 'DmeParticleOperator')
                    op_attrlist.append(op_elem)
                    op_elem['functionName'] = operator.function
                    for op_attr in operator.options.values():
                        op_elem[op_attr.name.casefold()] = copy.deepcopy(op_attr)

            # Initialise early to cause it to be placed above regular options.
            part_elem['children'] = Attribute.array('children', ValueType.ELEMENT)

            for option in part.options.values():
                part_elem[option.name.casefold()] = copy.deepcopy(option)

        # Now append the children.
        for part in particles:
            child_attr: Attribute[Element] = name_to_elem[part.name.casefold()]['children']
            assert child_attr.type is ValueType.ELEMENT and child_attr.is_array, child_attr
            for child in part.children:
                child_attr.append(name_to_elem[child.particle.casefold()])

        return root
