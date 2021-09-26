"""Parse particle system files."""
from typing import Union, IO, Dict, overload, List, Iterable

import attr

from srctools.dmx import Element, ValueType, Value as DMXValue, UUID, Vec3, Attribute


# The name of the file format used in DMX files.
FORMAT_NAME: str = 'pcf'
# PCF files can have version 1 or 2.
FORMAT_VERSION: int = 2


class Operator:
    """A generic option in particles."""
    def __init__(self, name: str, function: str, options: Dict[str, DMXValue]) -> None:
        self.function = function
        self.name = name
        self.options = options

    def __repr__(self) -> str:
        return f'<Op {self.name}.{self.function}({self.options})>'


@attr.define(eq=False)
class Child:
    """Options for a child particle reference."""
    particle: str


@attr.define(eq=False)
class Particle:
    """A particle system."""
    name: str
    options: Dict[str, DMXValue] = attr.Factory({}.copy)
    renderers: List[Operator] = attr.ib(converter=list, factory=list)
    operators: List[Operator] = attr.ib(converter=list, factory=list)
    initializers: List[Operator] = attr.ib(converter=list, factory=list)
    emitters: List[Operator] = attr.ib(converter=list, factory=list)
    forces: List[Operator] = attr.ib(converter=list, factory=list)
    constraints: List[Operator] = attr.ib(converter=list, factory=list)
    children: List[Operator] = attr.ib(converter=list, factory=list)

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

        def generic_attr(el: Element, name: str) -> Iterable['Operator']:
            try:
                attr = el.pop(name)
            except KeyError:
                return ()
            return [
                Operator(elem.name, elem.pop('functionName').val_str, dict(elem))
                for elem in attr
            ]

        elem: Element
        for elem in part_list:
            renderers = generic_attr(elem, 'renderers')
            operators = generic_attr(elem, 'operators')
            initializers = generic_attr(elem, 'initializers')
            emitters = generic_attr(elem, 'emitters')
            forces = generic_attr(elem, 'forces')
            constraints = generic_attr(elem, 'constraints')
            try:
                child_attr = elem.pop('children')
                if child_attr.type != ValueType.ELEMENT:
                    raise ValueError('Children must be an element array!')
            except KeyError:
                children = []
            else:
                children = [
                    Child(subelem.name) for subelem
                    in child_attr
                ]

            systems[elem.name.casefold()] = Particle(
                elem.name,
                renderers,
                operators,
                initializers,
                emitters,
                forces,
                constraints,
                children,
            )
        return systems
