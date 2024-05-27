"""Analyse structs."""
from typing import Iterator, Literal

import attrs


INDENT = ''


class Member:
    """A member in a struct."""
    def __len__(self) -> int:
        """Fixed in bytes."""
        raise NotImplementedError

    def display(self, indent: str, name: str) -> Iterator[str]:
        raise NotImplementedError


@attrs.frozen
class Fixed(Member):
    """A fixed size member."""
    size: int
    kind: Literal['s', 'u', 'f', 'pad', 'unk', 'str']
    def __len__(self) -> int:
        return self.size

    def display(self, indent: str, name: str) -> Iterator[str]:
        if self.kind == 'pad':
            yield from ['0'] * self.size
        elif self.kind == 'unk':
            yield from ['?'] * self.size
        else:
            if self.kind == 'str':
                yield f'{indent}str[{self.size}] {name}'
            else:
                yield f'{indent}{self.kind}{self.size*8} {name}'
            for _ in range(1, self.size):
                yield ' ' * len(indent) + '.'


@attrs.frozen(init=False)
class Struct(Member):
    """A structure."""
    contents: dict[str, Member]
    def __init__(self, **members: Member) -> None:
        self.__attrs_init__(members)

    def __len__(self) -> int:
        """A struct is the size of the contents."""
        return sum(map(len, self.contents.values()))

    def display(self, indent: str, name: str) -> Iterator[str]:
        for attr_name, attr in self.contents.items():
            if attr_name == 'base':
                yield from attr.display(indent, name)
            elif attr_name.startswith('_'):
                yield from [indent + '-'] * len(attr)   # Padding etc.
            else:
                yield from attr.display(indent + INDENT, f'{name}.{attr_name}' if name else attr_name)


type_uchar = Fixed(1, 'u')
type_char = Fixed(1, 's')
type_ushort = Fixed(2, 'u')
type_short = Fixed(2, 's')
type_uint = Fixed(4, 'u')
type_int = Fixed(4, 's')
type_ulong = Fixed(8, 'u')
type_long = Fixed(8, 's')

type_u8 = Fixed(1, 'u')
type_s8 = Fixed(1, 's')
type_u16 = Fixed(2, 'u')
type_s16 = Fixed(2, 's')
type_u32 = Fixed(4, 'u')
type_s32 = Fixed(4, 's')
type_u64 = Fixed(8, 'u')
type_s64 = Fixed(8, 's')

type_float = Fixed(4, 'f')
type_double = Fixed(8, 'f')

type_vector = Struct(x=type_float, y=type_float, z=type_float)
type_angles = Struct(pitch=type_float, yaw=type_float, roll=type_float)
type_color32 = Struct(r=type_u8, g=type_u8, b=type_u8, a=type_u8)


def display_all(ns: dict[str, object]) -> None:
    structs = []
    columns = []
    for name, value in ns.items():
        if name.startswith('prop_') and isinstance(value, Member):
            data = [f' {name[5:]} ({len(value)})']
            data.extend(value.display('', ''))
            print(data[0])
            columns.append(max(map(len, data)))
            data.insert(1, '=' * columns[-1])
            structs.append(data)

    print()

    iters = [(iter(lst), size) for lst, size in zip(structs, columns)]
    remaining = True
    while remaining:
        remaining = False
        print('| ', end='')
        for it, size in iters:
            try:
                cell = next(it)
                remaining = True
            except StopIteration:
                cell = '#' * size
            if cell in ('?', '0'):
                print('', cell * (size - 2), '', end=' | ')
            else:
                print(f'{cell: <{size}}', end=' | ')
        print()
