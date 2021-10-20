"""There's a lot of complexity with the value properties,
so it's better done in type stub form.
"""
from enum import Enum
from collections.abc import Iterable, Iterator, Mapping
from typing import Union, Optional, NamedTuple, TypeVar, Generic, NewType, Literal, Callable, IO, overload
from typing_extensions import TypeAlias
from uuid import UUID as UUID, uuid4 as get_uuid  # Re-export
from srctools import Matrix, Angle, Property
import builtins

from srctools import Vec_tuple as Vec3  # Re-export.
from srctools.tokenizer import Tokenizer


class ValueType(Enum):
    ELEMENT = 'element'  # Another attribute
    INTEGER = INT = 'int'
    FLOAT = 'float'
    BOOL = 'bool'
    STRING = STR = 'string'
    BINARY = BIN = VOID = 'void'  # IE "void *", binary blob.
    TIME = 'time'  # Seconds
    COLOR = COLOUR = 'color'
    VEC2 = 'vector2'
    VEC3 = 'vector3'
    VEC4 = 'vector4'
    ANGLE = 'qangle'
    QUATERNION = 'quaternion'
    MATRIX = 'vmatrix'


VAL_TYPE_TO_IND: dict[ValueType, int]
IND_TO_VALTYPE: dict[int, ValueType]
ARRAY_OFFSET: int


class Vec2(NamedTuple):
    x: float
    y: float


class Vec4(NamedTuple):
    x: float
    y: float
    z: float
    w: float


class Quaternion(NamedTuple):
    x: float
    y: float
    z: float
    w: float


class Color(NamedTuple):
    r: int
    g: int
    b: int
    a: int


class AngleTup(NamedTuple):
    pitch: float
    yaw: float
    roll: float

Time = NewType('Time', float)
Value = Union[
    int, float, bool, str, bytes,
    Color, Time,
    Vec2, Vec3,
    Vec4,
    AngleTup,
    Quaternion,
    Matrix,
    Element, None
]

ValueT = TypeVar(
    'ValueT',
    int, float, bool, str, bytes,
    Color, Time,
    Vec2, Vec3, Vec4,
    AngleTup,
    Quaternion,
    Matrix,
    Optional[Element],
)

TYPE_CONVERT: dict[tuple[ValueType, ValueType], Callable[[Value], Value]]
CONVERSIONS: dict[ValueType, Callable[[object], Value]]
SIZES: dict[ValueType, int]
TYPE_TO_VALTYPE: dict[type, ValueType]

def parse_vector(text: str, count: int) -> list[float]: ...

@overload
def deduce_type(value: list[ValueT]) -> tuple[ValueType, list[ValueT]]: ...
@overload
def deduce_type(value: ValueT) -> tuple[ValueType, ValueT]: ...

def deduce_type_array(value: list[ValueT]) -> tuple[ValueType, list[ValueT]]: ...
def deduce_type_single(value: ValueT) -> tuple[ValueType, ValueT]: ...


class _ValProps:
    """Properties which read/write as the various kinds of value types."""
    @property
    def val_int(self) -> int: ...
    @val_int.setter
    def val_int(self, value: int | float): ...

    val_float: float

    @property
    def val_time(self) -> Time: ...
    @val_time.setter
    def val_time(self, value: float | Time): ...

    @property
    def val_bool(self) -> bool: ...
    @val_bool.setter
    def val_bool(self, value: object): ...

    @property
    def val_vec2(self) -> Vec2: ...
    @val_vec2.setter
    def val_vec2(self, value: Vec2 | tuple[float, float] | Iterable[float]) -> None: ...

    @property
    def val_vec3(self) -> Vec3: ...
    @val_vec3.setter
    def val_vec3(self, value: Vec3 | tuple[float, float, float] | Iterable[float]) -> None: ...

    @property
    def val_vec4(self) -> Vec4: ...
    @val_vec4.setter
    def val_vec4(self, value: Vec4 | tuple[float, float, float, float] | Iterable[float]) -> None: ...

    @property
    def val_color(self) -> Color: ...
    @val_color.setter
    def val_color(self, value: Color | tuple[int, int, int, int] | Iterable[int]) -> None: ...

    @property
    def val_colour(self) -> Color: ...
    @val_colour.setter
    def val_colour(self, value: Color | tuple[int, int, int, int] | Iterable[int]) -> None: ...

    @property
    def val_str(self) -> str: ...
    @val_str.setter
    def val_str(self, value: object) -> None: ...

    @property
    def val_string(self) -> str: ...
    @val_string.setter
    def val_string(self, value: object) -> None: ...

    val_angle: AngleTup
    val_ang: AngleTup

    val_quat: Quaternion
    val_quaternion: Quaternion

    val_mat: Matrix
    val_matrix: Matrix

    val_bytes: bytes
    val_bin: bytes
    val_binary: bytes

    val_compound: Element | None
    val_elem: Element | None
    val: Element | None

class AttrMember(_ValProps):
    def __init__(self, owner: Attribute, index: int | str) -> None: ...


class Attribute(Generic[ValueT], _ValProps):
    name: str
    _typ: ValueType
    _value: Value | list

    # These are all properties, but no need to annotate like that.
    val_int: int
    val_float: float
    val_bool: bool

    val_vec2: Vec2
    val_vec3: Vec3
    val_vec4: Vec4

    val_color: Color
    val_colour: Color

    val_str: str
    val_string: str

    val_angle: AngleTup
    val_ang: AngleTup

    val_quat: Quaternion
    val_quaternion: Quaternion

    val_mat: Matrix
    val_matrix: Matrix

    val_elem: Element
    val_compound: Element

    # Readonly
    @property
    def type(self) -> ValueType: ...

    @property
    def is_array(self) -> bool: ...

    @overload
    def __init__(self: Attribute[Element | None], name: str, val_type: Literal[ValueType.ELEMENT], value: Element | None | list[Element | None]) -> None: ...
    @overload
    def __init__(self: Attribute[int], name: str, val_type: Literal[ValueType.INTEGER], value: int | list[int]) -> None: ...
    @overload
    def __init__(self: Attribute[float], name: str, val_type: Literal[ValueType.FLOAT], value: float | list[float]) -> None: ...
    @overload
    def __init__(self: Attribute[bool], name: str, val_type: Literal[ValueType.BOOL], value: bool | list[bool]) -> None: ...
    @overload
    def __init__(self: Attribute[str], name: str, val_type: Literal[ValueType.STR], value: str | list[str]) -> None: ...
    @overload
    def __init__(self: Attribute[bytes], name: str, val_type: Literal[ValueType.BIN], value: str | list[str]) -> None: ...
    @overload
    def __init__(self: Attribute[Time], name: str, val_type: Literal[ValueType.TIME], value: Time | list[Time]) -> None: ...
    @overload
    def __init__(self: Attribute[Color], name: str, val_type: Literal[ValueType.COLOR], value: Color | list[Color]) -> None: ...
    @overload
    def __init__(self: Attribute[Vec2], name: str, val_type: Literal[ValueType.VEC2], value: Vec2 | list[Vec2]) -> None: ...
    @overload
    def __init__(self: Attribute[Vec3], name: str, val_type: Literal[ValueType.VEC3], value: Vec3 | list[Vec3]) -> None: ...
    @overload
    def __init__(self: Attribute[Vec4], name: str, val_type: Literal[ValueType.VEC4], value: Vec4 | list[Vec4]) -> None: ...
    @overload
    def __init__(self: Attribute[AngleTup], name: str, val_type: Literal[ValueType.ANGLE], value: AngleTup | list[AngleTup]) -> None: ...
    @overload
    def __init__(self: Attribute[Quaternion], name: str, val_type: Literal[ValueType.QUATERNION], value: Quaternion | list[Quaternion]) -> None: ...
    @overload
    def __init__(self: Attribute[Matrix], name: str, val_type: Literal[ValueType.MATRIX], value: Matrix | list[Matrix]) -> None: ...

    @overload
    def __init__(self, name: str, type: ValueType, value: ValueT | list[ValueT]) -> None: ...

    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.ELEMENT]) -> Attribute[Element | None]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.INTEGER]) -> Attribute[int]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.FLOAT]) -> Attribute[float]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.BOOL]) -> Attribute[bool]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.STR]) -> Attribute[str]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.BIN]) -> Attribute[bytes]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.TIME]) -> Attribute[Time]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.COLOR]) -> Attribute[Color]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.VEC2]) -> Attribute[Vec2]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.VEC3]) -> Attribute[Vec3]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.VEC4]) -> Attribute[Vec4]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.ANGLE]) -> Attribute[AngleTup]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.QUATERNION]) -> Attribute[Quaternion]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.MATRIX]) -> Attribute[Matrix]: ...

    @classmethod
    @overload
    def array(cls, name: str, val_type: ValueType) -> Attribute: ...

    @classmethod
    def int(cls, name: str, value: builtins.int) -> Attribute[builtins.int]: ...
    @classmethod
    def float(cls, name: str, value: builtins.float) -> Attribute[builtins.float]: ...
    @classmethod
    def time(cls, name: str, value: Time | builtins.float) -> Attribute[Time]: ...
    @classmethod
    def bool(cls, name: str, value: builtins.bool) -> Attribute[builtins.bool]: ...
    @classmethod
    def string(cls, name: str, value: builtins.str) -> Attribute[builtins.str]: ...
    @classmethod
    def binary(cls, name: str, value: builtins.bytes) -> Attribute[builtins.bytes]: ...

    @overload
    @classmethod
    def vec2(cls, name: str, it: Iterable[builtins.float], /) -> Attribute[Vec2]: ...
    @overload
    @classmethod
    def vec2(cls, name: str, /, x: builtins.float = 0.0, y: builtins.float = 0.0) -> Attribute[Vec2]: ...

    @overload
    @classmethod
    def vec3(cls, name: str, it: Iterable[builtins.float], /) -> Attribute[Vec3]: ...
    @overload
    @classmethod
    def vec3(
        cls, name: str, /,
        x: builtins.float = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
    ) -> Attribute[Vec3]: ...

    @overload
    @classmethod
    def vec4(cls, name: str, __it: Iterable[builtins.float], /) -> Attribute[Vec4]: ...
    @overload
    @classmethod
    def vec4(
        cls, name: str, /,
        x: builtins.float | Iterable[builtins.float] = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
        w: builtins.float = 0.0,
    ) -> Attribute[Vec4]: ...

    @overload
    @classmethod
    def color(cls, name: str, __it: Iterable[builtins.int], /) -> Attribute[Color]: ...
    @overload
    @classmethod
    def color(
        cls, name: str, /,
        r: builtins.int = 0,
        g: builtins.int = 0,
        b: builtins.int = 0,
        a: builtins.int = 255,
    ) -> Attribute[Color]: ...

    @overload
    @classmethod
    def angle(cls, name: str, __it: Iterable[builtins.float], /) -> Attribute[AngleTup]: ...
    @overload
    @classmethod
    def angle(
        cls, name: str, /,
        pitch: builtins.float = 0.0,
        yaw: builtins.float = 0.0,
        roll: builtins.float = 0.0,
    ) -> Attribute[AngleTup]: ...

    @overload
    @classmethod
    def quaternion(cls, name: str, __it: Iterable[builtins.float], /) -> Attribute[Quaternion]: ...
    @overload
    @classmethod
    def quaternion(
        cls, name: str, /,
        x: builtins.float = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
        w: builtins.float = 0.0,
    ) -> Attribute[Quaternion]: ...

    def _read_val(self, newtype: ValueType) -> Value: ...
    def _write_val(self, newtype: ValueType, value: Value) -> None: ...

    def __repr__(self) -> str: ...

    def __getitem__(self, item: builtins.int) -> AttrMember: ...
    def __setitem__(self, item: builtins.int | slice, value: ValueT) -> None: ...
    def __delitem__(self, item: builtins.int | slice) -> None: ...
    def __len__(self) -> builtins.int: ...
    def __iter__(self) -> Iterator[ValueT]: ...
    def append(self, value: ValueT) -> None: ...


class Element(Mapping[str, Attribute]):
    name: str
    type: str
    uuid: UUID
    _members: dict[str, Attribute]

    def __init__(self, name: str, type: str, uuid: UUID=None) -> None: ...

    @classmethod
    def parse(cls, file: IO[bytes], unicode: bool = False) -> tuple[Element, str, int]: ...
    @classmethod
    def parse_bin(cls, file: IO[bytes], version: int, unicode: bool = False) -> Element: ...
    @classmethod
    def parse_kv2(cls, file: IO[str], version: int, unicode: bool = False) -> Element: ...

    @classmethod
    def _parse_kv2_element(
        cls, tok: Tokenizer,
        id_to_elem: dict[UUID, Element],
        fixups: list[tuple[Attribute, int | None, UUID, int]],
        name: str,
        typ_name: str,
    ) -> Element: ...

    def export_binary(
        self, file: IO[bytes],
        version: int = 5,
        fmt_name: str = 'dmx', fmt_ver: int = 1,
        unicode: Literal['ascii', 'format', 'silent'] = ...,
    ) -> None: ...
    def export_kv2(
        self, file: IO[bytes],
        fmt_name: str = 'dmx', fmt_ver: int = 1,
        *,
        flat: bool = False,
        unicode: Literal['ascii', 'format', 'silent'] = ...,
        cull_uuid: bool = False,
    ) -> None: ...

    def _export_kv2(
        self,
        file: IO[bytes],
        indent: bytes,
        roots: set[UUID],
        encoding: str,
        cull_uuid: bool,
    ) -> None: ...

    @classmethod
    def from_kv1(cls, props: Property, fmt_ext: bool=True) -> Element: ...

    def __repr__(self) -> str: ...

    def __getitem__(self, name: str) -> Attribute: ...
    def __setitem__(self, name: str, value: Attribute | Value | list[Value]) -> None: ...
    def __delitem__(self, name: str) -> None: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...

    def clear(self) -> None: ...
    @overload
    def pop(self, name: str) -> Attribute: ...
    @overload
    def pop(self, name: str, default: Attribute | Value | list[Value] = ...) -> Attribute: ...
    def popitem(self) -> tuple[str, Attribute]: ...
    def setdefault(self, name: str, default: Attribute | Value | list[Value]) -> Attribute: ...

    @overload
    def update(self, mapping: Mapping[str, Attribute], /, **kwargs: Attribute | Value | list[Value]) -> None: ...
    @overload
    def update(self, iterable: Iterable[tuple[str, ValueT] | Attribute], /, **kwargs: Attribute | Value | list[Value]) -> None: ...
    @overload
    def update(self, /, **kwargs: Attribute | Value | list[Value]) -> None: ...
