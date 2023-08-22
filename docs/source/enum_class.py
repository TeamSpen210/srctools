"""Implements a documenter for Enum classes."""
from collections import defaultdict

from typing import Any, Dict, Iterable, List, Tuple, Type
from dataclasses import dataclass
import enum
import functools

from sphinx.ext.autodoc import ClassDocumenter, AttributeDocumenter, ObjectMember, bool_option


@dataclass
class EnumInfo:
    """Computed member info."""
    cls: Type[enum.Enum]
    aliases: Dict[str, List[str]]
    canonical: List[enum.Enum]
    should_hex: bool = False


@functools.lru_cache(maxsize=None)
def enum_aliases(enum_obj: Type[enum.Enum]) -> EnumInfo:
    """Compute the aliases for this enum."""
    aliases: Dict[str, List[str]] = defaultdict(list)
    canonical: List[enum.Enum] = []
    for name, member in enum_obj.__members__.items():
        if name != member.name:
            aliases[member.name].append(name)
        else:
            canonical.append(member)
    return EnumInfo(enum_obj, dict(aliases), canonical)


class EnumDocumenter(ClassDocumenter):
    """Handle enum documentation specially."""
    objtype = 'srcenum'
    directivetype = ClassDocumenter.objtype
    priority = 10 + ClassDocumenter.priority
    option_spec = {
        **ClassDocumenter.option_spec,
        'hex': bool_option,
    }
    del option_spec['show-inheritance']

    def __init__(self, *args) -> None:
        """Force-enable show-inheritance, we want to show it's an enum."""
        super().__init__(*args)
        self.options = self.options.copy()
        self.options['show-inheritance'] = True

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any) -> bool:
        """We can only document Enums."""
        return isinstance(member, type) and issubclass(member, enum.Enum)

    def filter_members(self, members: Iterable[ObjectMember], want_all: bool) -> List[Tuple[str, Any, bool]]:
        """Specially handle enum members."""
        results: List[Tuple[str, object, bool]] = []

        info = enum_aliases(self.object)
        info.should_hex = self.options.hex

        for member in info.canonical:  # Keep in order.
            if member.name.isdigit() and isinstance(member.value, int):
                # add_unknown() pseudo-flags, skip.
                continue
            results.append((member.name, member, True))

        # Have super() handle any other members (properties, methods).
        results.extend(super().filter_members([
            obj for obj in members
            if obj.__name__ not in info.cls.__members__
        ], True))
        return results


class NoReprString(str):
    """A string which doesn't quote in the repr."""
    def __repr__(self) -> str:
        return self


class EnumMemberDocumenter(AttributeDocumenter):
    """Special documenter for enum members."""
    objtype = '__srcenummember'
    directivetype = AttributeDocumenter.objtype  # Unchanged.
    priority = 10000

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any) -> bool:
        """We only document members from EnumDocumenter."""
        return isinstance(parent, EnumDocumenter)

    def add_directive_header(self, sig: str) -> None:
        """Alter behaviour of the header."""
        # If we're
        info = enum_aliases(self.parent)
        if info.should_hex:
            self.object = NoReprString(f'0x{int(self.object):x}')
        super().add_directive_header(sig)

    def add_content(self, *args, **kwargs) -> None:
        sourcename = self.get_sourcename()
        info = enum_aliases(self.parent)
        try:
            aliases = info.aliases[self.objpath[-1]]
        except KeyError:
            aliases = []
        if aliases:
            self.add_line(
                ('*Aliases:* ' if len(aliases) > 1 else '*Alias:* ')
                + ', '.join([f'``{name}``' for name in aliases]),
                sourcename,
            )
            self.add_line('', sourcename)
        super().add_content(*args, **kwargs)
