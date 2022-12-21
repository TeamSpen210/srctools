"""This module implements three classes for representing vectors, Euler angles and rotation matrices, 
following Valve conventions. `Vec` represents an XYZ position and `Angle` represents a pitch-yaw-roll 
angle (in degrees). `Matrix` represents a rotation also, but in a format that can be manipulated 
properly. `Angle` will automatically compute a matrix when required, but it is more efficient to 
perform repeated operations on a matrix.

These classes will be replaced by Cython-optimized versions where possible, which are interchangable
if pickled.

Rotations are performed via the matrix-multiplication operator `@`, where the left is rotated by
the right. Vectors can be rotated by matrices and angles and matrices can be rotated by angles,
but not vice-versa.

 - Vec @ Angle -> Vec
 - Vec @ Matrix -> Vec
 - 3-tuple @ Angle -> Vec
 - Angle @ Angle -> Angle
 - Angle @ Matrix -> Angle
 - Matrix @ Matrix -> Matrix
"""
from typing import (
    TYPE_CHECKING, Any, Callable, ClassVar, Dict, Iterable, Iterator, List, NamedTuple,
    Optional, Tuple, Type, TypeVar, Union, cast,
)
from typing_extensions import Final, Literal, Protocol, TypeGuard, final, overload
import contextlib
import math
import warnings


__all__ = [
    'parse_vec_str', 'to_matrix', 'lerp', 'quickhull',
    'Vec', 'FrozenVec', 'Vec_tuple', 'AnyVec',
    'Angle', 'FrozenAngle', 'AnyAngle',
    'Matrix', 'FrozenMatrix', 'AnyMatrix',
]

# Type aliases
Tuple3 = Tuple[float, float, float]
AnyVec = Union['VecBase', 'Vec_tuple', Tuple3]
VecUnion = Union['Vec', 'FrozenVec']
AnyAngle = Union['Angle', 'FrozenAngle']
AnyMatrix = Union['Matrix', 'FrozenMatrix']
VecT = TypeVar('VecT', bound='VecBase')
AngleT = TypeVar('AngleT', bound='AngleBase')
MatrixT = TypeVar('MatrixT', bound='MatrixBase')
T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')


def lerp(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Linearly interpolate from in to out.

    :raises ZeroDivisionError: If both ``in`` values are the same.
    """
    return out_min + ((x - in_min) * (out_max - out_min)) / (in_max - in_min)


def parse_vec_str(
    val: Union[str, 'VecBase', 'AngleBase'],
    x: Union[T1, float] = 0.0, y: Union[T2, float] = 0.0, z: Union[T3, float] = 0.0,
) -> Tuple[Union[T1, float], Union[T2, float], Union[T3, float]]:
    """Convert a string in the form ``(4 6 -4)`` into a set of floats.

    If the string is unparsable or an invalid type, this uses the defaults ``x``, ``y``, ``z``.
    The string can be surrounded by any of the ``()``, ``{}``, ``[]``, ``<>`` bracket types, which
    are simply ignored.

    If the 'string' is already a :py:class:`VecBase` or :py:class:`AngleBase`, this will be passed through.
    If you do want a specific class, use :py:meth:`Vec.from_str`, :py:meth:`Angle.from_str` or :py:meth:`Matrix.from_angstr`.
    """
    if isinstance(val, str):
        pass  # Fast path to skip the below code.
    elif isinstance(val, VecBase):
        return val.x, val.y, val.z
    elif isinstance(val, AngleBase):
        return val.pitch, val.yaw, val.roll
    else:
        # Not a string.
        return x, y, z

    val = val.strip()
    if val and val[0] in '({[<':
        val = val[1:]
    if val and val[-1] in ')}]>':
        val = val[:-1]

    try:
        str_x, str_y, str_z = val.split()
    except ValueError:
        return x, y, z

    try:
        return (
            float(str_x),
            float(str_y),
            float(str_z),
        )
    except ValueError:
        return x, y, z


def to_matrix(value: Union['AnyAngle', 'AnyMatrix', 'AnyVec', None]) -> 'Matrix | FrozenMatrix':
    """Convert various values to a rotation matrix.

    :py:class:`Vec` will be treated as angles, and :external:py:data:`None` as the identity.
    """
    if value is None:
        return Py_Matrix()
    elif isinstance(value, Matrix) or isinstance(value, FrozenMatrix):
        return value
    elif isinstance(value, AngleBase):
        return Matrix.from_angle(value)
    else:
        [p, y, r] = value
        return Matrix.from_angle(p, y, r)


def format_float(x: float, places: int=6) -> str:
    """Convert the specified float to a string, stripping off a .0 if it ends with that."""
    result = f'{x:.{places}f}'
    if '.' in result:
        result = result.rstrip('0')
    return result.rstrip('.')


def _check_tuple3(obj: object) -> TypeGuard[Tuple3]:
    """Check this object is a 3-tuple of floats (or int)."""
    if isinstance(obj, tuple) and len(obj) == 3:
        x, y, z = obj
        return (
            (isinstance(x, float) or isinstance(x, int)) and
            (isinstance(y, float) or isinstance(y, int)) and
            (isinstance(z, float) or isinstance(z, int))
        )
    return False


class Vec_tuple(NamedTuple):
    """An immutable tuple, useful for dictionary keys."""
    x: float
    y: float
    z: float

if not TYPE_CHECKING:
    _old_vec_tup = Vec_tuple.__new__
    def _vec_tup_new(*args, **kwargs):
        warnings.warn(
            'Vec_tuple is deprecated, use FrozenVec instead.',
            DeprecationWarning, stacklevel=2,
        )
        return _old_vec_tup(*args, **kwargs)
    # noinspection PyDeprecation
    Vec_tuple.__new__ = _vec_tup_new


if TYPE_CHECKING:
    class _InvAxis(Protocol):
        """Dummy class to type-check Vec.INV_AXIS"""
        @overload
        def __getitem__(self, item: Literal['x']) -> Tuple[Literal['y'], Literal['z']]: ...
        @overload
        def __getitem__(self, item: Literal['y']) -> Tuple[Literal['x'], Literal['z']]: ...
        @overload
        def __getitem__(self, item: Literal['z']) -> Tuple[Literal['x'], Literal['y']]: ...

        @overload
        def __getitem__(self, item: Tuple[Literal['y'], Literal['z']]) -> Literal['x']: ...
        @overload
        def __getitem__(self, item: Tuple[Literal['z'], Literal['y']]) -> Literal['x']: ...

        @overload
        def __getitem__(self, item: Tuple[Literal['x'], Literal['z']]) -> Literal['y']: ...
        @overload
        def __getitem__(self, item: Tuple[Literal['z'], Literal['x']]) -> Literal['y']: ...

        @overload
        def __getitem__(self, item: Tuple[Literal['x'], Literal['y']]) -> Literal['z']: ...
        @overload
        def __getitem__(self, item: Tuple[Literal['y'], Literal['x']]) -> Literal['z']: ...

        @overload
        def __getitem__(self, item: str) -> Tuple[str, str]: ...
        @overload
        def __getitem__(self, item: Tuple[str, str]) -> str: ...

        def __getitem__(self, item: Union[Tuple[str, str], str]) -> Union[Tuple[str, str], str]: ...
else:
    globals()['_InvAxis'] = None

# Use template code to reduce duplication in the various magic number methods.

_VEC_ADDSUB_TEMP = '''
def __{func}__(self, other: Union['Vec', tuple, float]):
    """``{op}`` operation.

    This additionally works on scalars (adds to all axes).
    """
    if isinstance(other, VecBase):
        return type(self)(
            self.x {op} other.x,
            self.y {op} other.y,
            self.z {op} other.z,
        )
    try:
        if _check_tuple3(other):
            x = self.x {op} other[0]
            y = self.y {op} other[1]
            z = self.z {op} other[2]
        else:
            x = self.x {op} other
            y = self.y {op} other
            z = self.z {op} other
    except TypeError:
        return NotImplemented
    else:
        return type(self)(x, y, z)

def __r{func}__(self, other: Union['VecBase', tuple, float]):
    """``{op}`` operation with reversed operands.

    This additionally works on scalars (adds to all axes).
    """
    if isinstance(other, VecBase):
        return type(self)(
            other.x {op} self.x,
            other.y {op} self.y,
            other.z {op} self.z,
        )
    try:
        if _check_tuple3(other):
            x = other[0] {op} self.x
            y = other[1] {op} self.y
            z = other[2] {op} self.z
        else:
            x = other {op} self.x
            y = other {op} self.y
            z = other {op} self.z
    except TypeError:
        return NotImplemented
    else:
        return type(self)(x, y, z)
'''


_VEC_ADDSUB_INPLACE_TEMP = '''
def __i{func}__(self, other: Union[VecBase, tuple, float]):
    """``{op}=`` operation.

    Like the normal one except without duplication.
    """
    if isinstance(other, VecBase):
        self.x {op}= other.x
        self.y {op}= other.y
        self.z {op}= other.z
    elif _check_tuple3(other):
        self.x {op}= other[0]
        self.y {op}= other[1]
        self.z {op}= other[2]
    elif isinstance(other, (int, float)):
        self.x {op}= other
        self.y {op}= other
        self.z {op}= other
    else:
        return NotImplemented
    return self
'''

# Multiplication and division doesn't work with two vectors - use dot/cross
# instead.

_VEC_MULDIV_TEMP = '''
def __{func}__(self, other: float):
    """``Vector {op} scalar`` operation."""
    if isinstance(other, VecBase):
        raise TypeError("Cannot {pretty} 2 Vectors.")
    else:
        try:
            return type(self)(
                self.x {op} other,
                self.y {op} other,
                self.z {op} other,
            )
        except TypeError:
            return NotImplemented

def __r{func}__(self, other: float):
    """``scalar {op} Vector`` operation."""
    if isinstance(other, VecBase):
        raise TypeError("Cannot {pretty} 2 Vectors.")
    else:
        try:
            return type(self)(
                other {op} self.x,
                other {op} self.y,
                other {op} self.z,
            )
        except TypeError:
            return NotImplemented
'''


_VEC_MULDIV_INPLACE_TEMP  = '''
def __i{func}__(self, other: float):
    """``{op}=`` operation.

    Like the normal one except without duplication.
    """
    if isinstance(other, VecBase):
        raise TypeError("Cannot {pretty} 2 Vectors.")
    else:
        self.x {op}= other
        self.y {op}= other
        self.z {op}= other
        return self
'''


class VecBase:
    """Internal Base class for 3D vectors, implementing common code."""
    __match_args__: Final = ('_x', '_y', '_z')
    __slots__ = ('_x', '_y', '_z')

    #: This is a dictionary containing complementary axes.
    #: ``INV_AXIS["x", "y"]`` gives ``"z"``, and ``INV_AXIS["y"]`` returns ``("x", "z")``.
    INV_AXIS = cast(_InvAxis, {
        'x': ('y', 'z'),
        'y': ('x', 'z'),
        'z': ('x', 'y'),

        ('y', 'z'): 'x',
        ('x', 'z'): 'y',
        ('x', 'y'): 'z',

        ('z', 'y'): 'x',
        ('z', 'x'): 'y',
        ('y', 'x'): 'z',
    })

    # Vectors pointing in all cardinal directions.
    # Variable annotation here, it has to be assigned after FrozenVec
    # is actually defined.
    N: ClassVar['FrozenVec']
    S: ClassVar['FrozenVec']
    E: ClassVar['FrozenVec']
    W: ClassVar['FrozenVec']
    T: ClassVar['FrozenVec']
    B: ClassVar['FrozenVec']
    north: ClassVar['FrozenVec']
    south: ClassVar['FrozenVec']
    east: ClassVar['FrozenVec']
    west: ClassVar['FrozenVec']
    top: ClassVar['FrozenVec']
    bottom: ClassVar['FrozenVec']
    y_pos: ClassVar['FrozenVec']
    y_neg: ClassVar['FrozenVec']
    x_pos: ClassVar['FrozenVec']
    x_neg: ClassVar['FrozenVec']
    z_pos: ClassVar['FrozenVec']
    z_neg: ClassVar['FrozenVec']

    _x: float
    _y: float
    _z: float

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """VecBase cannot be instantiated."""
        if type(self) is VecBase:
            raise TypeError('VecBase cannot be instantiated.')

    @property
    def x(self) -> float:
        """Return the X axis."""
        return self._x

    @property
    def y(self) -> float:
        """Return the Y axis."""
        return self._y

    @property
    def z(self) -> float:
        """Return the Z axis."""
        return self._z

    def copy(self: VecT) -> VecT:
        """Implemented by subclasses."""
        raise NotImplementedError

    @classmethod
    def from_str(cls: Type[VecT], val: Union[str, 'VecBase'], x: float=0.0, y: float=0.0, z: float=0.0) -> VecT:
        """Convert a string in the form ``(4 6 -4)`` into a Vector.

        If the string is unparsable, this uses the defaults ``(x,y,z)``.
        The string can be surrounded by any of the ``()``, ``{}``, ``[]``, ``<>`` bracket types,
        which are simply ignored.

        If the value is already a vector, a copy will be returned.
        To only do parsing, use :py:func:`parse_vec_str()`.
        """

        x, y, z = Py_parse_vec_str(val, x, y, z)
        return cls(x, y, z)

    @classmethod
    @overload
    def with_axes(cls: Type[VecT], axis1: str, val1: Union[float, 'VecBase']) -> VecT: ...

    @classmethod
    @overload
    def with_axes(
        cls: Type[VecT],
        axis1: str, val1: Union[float, 'VecBase'],
        axis2: str, val2: Union[float, 'VecBase'],
    ) -> VecT: ...

    @classmethod
    @overload
    def with_axes(
        cls: Type[VecT],
        axis1: str, val1: Union[float, 'VecBase'],
        axis2: str, val2: Union[float, 'VecBase'],
        axis3: str, val3: Union[float, 'VecBase'],
    ) -> VecT: ...

    @classmethod
    def with_axes(cls: Type[VecT], *args: Union[str, float, 'VecBase'], **kwargs: Union[str, float, 'VecBase']) -> VecT:
        """Create a Vector, given a number of axes and corresponding values.

        This is a convenience for doing the following::

            vec = Vec()
            vec[axis1] = val1
            vec[axis2] = val2
            vec[axis3] = val3

        The magnitudes can also be Vectors, in which case the matching
        axis will be used from the vector.
        """
        raise NotImplementedError

    @classmethod
    @overload
    def bbox(cls: Type[VecT],  __point: Iterable['VecBase']) -> Tuple[VecT, VecT]: ...
    @classmethod
    @overload
    def bbox(cls: Type[VecT], *points: 'VecBase') -> Tuple[VecT, VecT]: ...

    @classmethod
    def bbox(cls: Type[VecT], *points: Union[Iterable['VecBase'], 'VecBase']) -> Tuple[VecT, VecT]:
        """Compute the bounding box for a set of points.

        Pass either several Vecs, or an iterable of Vecs.
        Returns a ``(min, max)`` :external:py:class:`tuple`.
        """
        # Allow passing a single iterable, but also handle a single Vec.
        # The error messages match those produced by min()/max().
        first: VecBase
        point_coll: Iterable[VecBase]
        if len(points) == 1 and not isinstance(points[0], VecBase):
            try:
                [[first, *point_coll]] = points  # type: ignore # len() can't narrow
            except ValueError:
                raise ValueError('Vec.bbox() arg is an empty sequence') from None
        else:
            try:
                first, *point_coll = points  # type: ignore # len() can't narrow
            except ValueError:
                raise TypeError(
                    'Vec.bbox() expected at '
                    'least 1 argument, got 0.'
                ) from None

        bbox_min = Py_Vec(first)
        bbox_max = bbox_min.copy()
        for point in point_coll:
            bbox_min.min(point)
            bbox_max.max(point)
        if cls is Py_FrozenVec:
            return cls(bbox_min), cls(bbox_max)
        else:
            # We know cls is Py_Vec, and these are too.
            return bbox_min, bbox_max  # type: ignore

    @classmethod
    def iter_grid(
        cls: Type[VecT],
        min_pos: 'VecBase',
        max_pos: 'VecBase',
        stride: int=1,
    ) -> Iterator[VecT]:
        """Loop over points in a bounding box. All coordinates should be integers.

        Both borders will be included.
        """
        min_x = round(min_pos.x)
        min_y = round(min_pos.y)
        min_z = round(min_pos.z)

        max_x = round(max_pos.x)
        max_y = round(max_pos.y)
        max_z = round(max_pos.z)

        for x in range(min_x, max_x + 1, stride):
            for y in range(min_y, max_y + 1, stride):
                for z in range(min_z, max_z + 1, stride):
                    yield cls(x, y, z)

    def iter_line(self: VecT, end: 'VecBase', stride: int=1) -> Iterator[VecT]:
        """Yield points in a line (including both endpoints).

        :param stride: This specifies the distance between each point.
        :param end: The other end of the line.

        If the distance is less than the stride, only end-points will be yielded.
        If they are the same, that point will be yielded.
        """
        cls = type(self)
        offset = end - self
        length = offset.mag()
        if length < stride:
            # Not enough room, yield both
            yield self.copy()
            if self != end:
                yield cls(end)
            return

        direction = offset.norm()
        for pos in range(0, int(length), int(stride)):
            yield self + direction * pos
        yield cls(end)  # Directly yield - ensures no rounding errors.

    def axis(self) -> Literal['x', 'y', 'z']:
        """For an axis-aligned vector, return the axis it is on.

        :raises ValueError: If the vector is not on-axis.
        """
        x = abs(self.x) > 1e-6
        y = abs(self.y) > 1e-6
        z = abs(self.z) > 1e-6
        if x and not y and not z:
            return 'x'
        if not x and y and not z:
            return 'y'
        if not x and not y and z:
            return 'z'
        raise ValueError(
            f'({self.x:g}, {self.y:g}, {self.z:g}) is '
            f'not an on-axis vector!'
        )

    def to_angle(self, roll: float=0) -> 'Angle':
        """Convert a normal to a Source Engine angle.

        The angle will point its ``+x`` axis in the direction of this vector.
        The inverse of this is ``Vec(x=1) @ Angle(pitch, yaw, roll)``.

        :param roll: The roll is not affected by the direction of the vector, so it can be provided separately.
        """
        # Pitch is applied first, so we need to reconstruct the x-value
        horiz_dist = math.hypot(self.x, self.y)
        return Py_Angle(
            math.degrees(math.atan2(-self.z, horiz_dist)),
            math.degrees(math.atan2(self.y, self.x)) % 360,
            roll,
        )

    def __abs__(self: VecT) -> VecT:
        """Performing :external:py:func:`abs()` on a Vec takes the absolute value of all axes."""
        return type(self)(
            abs(self.x),
            abs(self.y),
            abs(self.z),
        )

    # The numeric magic methods are defined via exec(), so we need stubs
    # to annotate them in a way a type-checker can understand.
    # These are immediately overwritten.
    if TYPE_CHECKING:
        def __add__(self: VecT, other: Union['VecBase', Tuple3, int, float]) -> VecT: ...
        def __radd__(self: VecT, other: Union['VecBase', Tuple3, int, float]) -> VecT: ...

        def __sub__(self: VecT, other: Union['VecBase', Tuple3, int, float]) -> VecT: ...
        def __rsub__(self: VecT, other: Union['VecBase', Tuple3, int, float]) -> VecT: ...

        def __mul__(self: VecT, other: float) -> VecT: ...
        def __rmul__(self: VecT, other: float) -> VecT: ...

        def __truediv__(self: VecT, other: float) -> VecT: ...
        def __rtruediv__(self: VecT, other: float) -> VecT: ...

        def __floordiv__(self: VecT, other: float) -> VecT: ...
        def __rfloordiv__(self: VecT, other: float) -> VecT: ...

        def __mod__(self: VecT, other: float) -> VecT: ...
        def __rmod__(self: VecT, other: float) -> VecT: ...

    _funcname = _op = _pretty = ''
    # Use exec() to generate all the number magic methods. This reduces code
    # duplication since they're all very similar.

    for _funcname, _op in (('add', '+'), ('sub', '-')):
        exec(
            _VEC_ADDSUB_TEMP.format(func=_funcname, op=_op),
            globals(),
            locals(),
        )

    for _funcname, _op, _pretty in (
            ('mul', '*', 'multiply'),
            ('truediv', '/', 'divide'),
            ('floordiv', '//', 'floor-divide'),
            ('mod', '%', 'modulus'),
    ):
        exec(
            _VEC_MULDIV_TEMP.format(func=_funcname, op=_op, pretty=_pretty),
            globals(),
            locals(),
        )

    del _funcname, _op, _pretty

    # Divmod is entirely unique.
    def __divmod__(self: VecT, other: float) -> Tuple[VecT, VecT]:
        """Divide the vector by a scalar, returning the result and remainder."""
        if isinstance(other, VecBase):
            raise TypeError("Cannot divide 2 Vectors.")
        try:
            x1, x2 = divmod(self.x, other)
            y1, y2 = divmod(self.y, other)
            z1, z2 = divmod(self.z, other)
        except TypeError:
            return NotImplemented
        else:
            return type(self)(x1, y1, z1), type(self)(x2, y2, z2)

    def __rdivmod__(self: VecT, other: float) -> Tuple[VecT, VecT]:
        """Divide a scalar by a vector, returning the result and remainder."""
        if isinstance(other, VecBase):
            raise TypeError("Cannot divide 2 Vectors.")
        try:
            x1, x2 = divmod(other, self.x)
            y1, y2 = divmod(other, self.y)
            z1, z2 = divmod(other, self.z)
        except (TypeError, ValueError):
            return NotImplemented
        else:
            return type(self)(x1, y1, z1), type(self)(x2, y2, z2)

    def __matmul__(self: VecT, other: Union['AngleBase', 'MatrixBase']) -> VecT:
        """Rotate this vector by an angle or matrix."""
        if isinstance(other, MatrixBase):
            mat = other
        elif isinstance(other, AngleBase):
            mat = Py_Matrix.from_angle(other)
        else:
            return NotImplemented
        res = type(self)(self._x, self._y, self._z)
        # noinspection PyProtectedMember
        mat._vec_rot(res)
        return res

    def __bool__(self) -> bool:
        """Vectors are True if any axis is non-zero."""
        return self._x != 0 or self._y != 0 or self._z != 0

    def __eq__(self, other: object) -> bool:
        """Equality test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        A tolerance of ``1e-6`` is accounted for automatically.
        """
        if isinstance(other, VecBase):
            return (
                abs(other.x - self.x) < 1e-6 and
                abs(other.y - self.y) < 1e-6 and
                abs(other.z - self.z) < 1e-6
            )
        elif _check_tuple3(other):
            return (
                abs(self.x - other[0]) < 1e-6 and
                abs(self.y - other[1]) < 1e-6 and
                abs(self.z - other[2]) < 1e-6
            )
        else:
            return NotImplemented

    def __ne__(self, other: object) -> bool:
        """Inequality test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        A tolerance of 1e-6 is accounted for automatically.
        """
        if isinstance(other, VecBase):
            return (
                abs(other.x - self.x) >= 1e-6 or
                abs(other.y - self.y) >= 1e-6 or
                abs(other.z - self.z) >= 1e-6
            )
        elif _check_tuple3(other):
            return (
                abs(self.x - other[0]) >= 1e-6 or
                abs(self.y - other[1]) >= 1e-6 or
                abs(self.z - other[2]) >= 1e-6
            )
        else:
            return NotImplemented

    def __lt__(self, other: AnyVec) -> bool:
        """``A<B`` test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        A tolerance of 1e-6 is accounted for automatically.
        """
        if isinstance(other, VecBase):
            return (
                (other.x - self.x) > 1e-6 and
                (other.y - self.y) > 1e-6 and
                (other.z - self.z) > 1e-6
            )
        elif _check_tuple3(other):
            return (
                (other[0] - self.x) > 1e-6 and
                (other[1] - self.y) > 1e-6 and
                (other[2] - self.z) > 1e-6
            )
        else:
            return NotImplemented

    def __le__(self, other: AnyVec) -> bool:
        """``A<=B`` test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        A tolerance of 1e-6 is accounted for automatically.
        """
        if isinstance(other, VecBase):
            return (
                (self.x - other.x) <= 1e-6 and
                (self.y - other.y) <= 1e-6 and
                (self.z - other.z) <= 1e-6
            )
        elif _check_tuple3(other):
            return (
                (self.x - other[0]) <= 1e-6 and
                (self.y - other[1]) <= 1e-6 and
                (self.z - other[2]) <= 1e-6
            )
        else:
            return NotImplemented

    def __gt__(self, other: AnyVec) -> bool:
        """``A>B`` test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        A tolerance of 1e-6 is accounted for automatically.
        """
        if isinstance(other, VecBase):
            return (
                (self.x - other.x) > 1e-6 and
                (self.y - other.y) > 1e-6 and
                (self.z - other.z) > 1e-6
            )
        elif _check_tuple3(other):
            return (
                (self.x - other[0]) > 1e-6 and
                (self.y - other[1]) > 1e-6 and
                (self.z - other[2]) > 1e-6
            )
        else:
            return NotImplemented

    def __ge__(self, other: AnyVec) -> bool:
        """``A>=B`` test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        A tolerance of 1e-6 is accounted for automatically.
        """
        if isinstance(other, VecBase):
            return (
                (other.x - self.x) <= 1e-6 and
                (other.y - self.y) <= 1e-6 and
                (other.z - self.z) <= 1e-6
            )
        elif _check_tuple3(other):
            return (
                (other[0] - self.x) <= 1e-6 and
                (other[1] - self.y) <= 1e-6 and
                (other[2] - self.z) <= 1e-6
            )
        else:
            return NotImplemented

    @classmethod
    def lerp(cls: Type[VecT], x: float, in_min: float, in_max: float, out_min: 'VecBase', out_max: 'VecBase') -> VecT:
        """Linerarly interpolate between two vectors.

        :raises ZeroDivisionError: If ``in_min`` and ``in_max`` are the same.
        """
        x_off = x - in_min
        diff = in_max - in_min
        return cls(
            out_min.x + (x_off * (out_max.x - out_min.x)) / diff,
            out_min.y + (x_off * (out_max.y - out_min.y)) / diff,
            out_min.z + (x_off * (out_max.z - out_min.z)) / diff,
        )

    def __round__(self: VecT, ndigits: int=0) -> VecT:
        """Performing :external:py:func:`round()` on a vector rounds each axis."""
        return type(self)(
            round(self.x, ndigits),
            round(self.y, ndigits),
            round(self.z, ndigits),
        )

    def mag(self) -> float:
        """Compute the distance from the vector and the origin."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def join(self, delim: str=', ') -> str:
        """Return a string with all numbers joined by the passed delimiter.

        This strips off the ``.0`` if no decimal portion exists.
        """
        return f'{format_float(self.x)}{delim}{format_float(self.y)}{delim}{format_float(self.z)}'

    def __str__(self) -> str:
        """Return the values, separated by spaces.

        This is the main format in Valve's file formats.
        This strips off the .0 if no decimal portion exists.
        """
        return f'{format_float(self.x)} {format_float(self.y)} {format_float(self.z)}'

    def __format__(self, format_spec: str) -> str:
        """Control how the text is formatted.

        This returns each axis formatted with the provided specification, joined by spaces.
        """
        if not format_spec:
            return str(self)

        x = format(self.x, format_spec)
        if '.' in x:
            x = x.rstrip('0')

        y = format(self.y, format_spec)
        if '.' in y:
            y = y.rstrip('0')

        z = format(self.z, format_spec)
        if '.' in z:
            z = z.rstrip('0')
        return f'{x.rstrip(".")} {y.rstrip(".")} {z.rstrip(".")}'

    def __iter__(self) -> Iterator[float]:
        """Iterating through the vector yields each axis in order."""
        yield self.x
        yield self.y
        yield self.z

    def __reversed__(self) -> Iterator[float]:
        """Allow iterating through the dimensions, in reverse."""
        yield self.z
        yield self.y
        yield self.x

    def __getitem__(self, ind: Union[str, int]) -> float:
        """Allow reading values by index instead of name if desired.

        This accepts either ``0``, ``1``, ``2`` or ``x``, ``y``, ``z`` to read values.
        Useful in conjunction with a loop to apply commands to all values.
        """
        if ind == 0 or ind == "x":
            return self.x
        elif ind == 1 or ind == "y":
            return self.y
        elif ind == 2 or ind == "z":
            return self.z
        raise KeyError(f'Invalid axis: {ind!r}')

    def in_bbox(self, a: AnyVec, b: AnyVec) -> bool:
        """Check if this point is inside the specified bounding box."""
        return (
            min(a[0], b[0]) - 1e-6 <= self.x <= max(a[0], b[0]) + 1e-6 and
            min(a[1], b[1]) - 1e-6 <= self.y <= max(a[1], b[1]) + 1e-6 and
            min(a[2], b[2]) - 1e-6 <= self.z <= max(a[2], b[2]) + 1e-6
        )

    @staticmethod
    def bbox_intersect(min1: 'VecBase', max1: 'VecBase', min2: 'VecBase', max2: 'VecBase') -> bool:
        """Check if the ``(min1, max1)`` bounding box intersects the ``(min2, max2)`` bounding box."""
        if (min2.x - max1.x) > 1e-6 or (min1.x - max2.x) > 1e-6:
            return False
        if (min2.y - max1.y) > 1e-6 or (min1.y - max2.y) > 1e-6:
            return False
        if (min2.z - max1.z) > 1e-6 or (min1.z - max2.z) > 1e-6:
            return False
        return True

    def other_axes(self, axis: str) -> Tuple[float, float]:
        """Get the values for the other two axes."""
        if axis == 'x':
            return self.y, self.z
        if axis == 'y':
            return self.x, self.z
        if axis == 'z':
            return self.x, self.y
        raise KeyError('Bad axis "{}"'.format(axis))

    def as_tuple(self) -> Vec_tuple:
        """Return the Vector as a tuple."""
        return Vec_tuple(round(self.x, 6), round(self.y, 6), round(self.z, 6))

    def len_sq(self) -> float:
        """Return the magnitude squared, which is slightly faster."""
        return self.x**2 + self.y**2 + self.z**2

    def __len__(self) -> int:
        """The :external:py:func:`len()` of a vector is always ``3``."""
        return 3

    def __contains__(self, val: float) -> bool:
        """Check to see if an axis is set to the given value."""
        return abs(val - self.x) < 1e-6 or abs(val - self.y) < 1e-6 or abs(val - self.z) < 1e-6

    def __neg__(self: VecT) -> VecT:
        """The inverted form of a Vector has inverted axes."""
        return type(self)(-self.x, -self.y, -self.z)

    def __pos__(self: VecT) -> VecT:
        """``+`` on a Vector simply copies it."""
        return type(self)(self.x, self.y, self.z)

    def norm(self: VecT) -> VecT:
        """Normalise the Vector.

         This is done by transforming it to have a magnitude of 1 but the same
         direction.
         The vector is left unchanged if it is equal to ``(0, 0, 0)``, instead of raising.
         """
        if self.x == 0 and self.y == 0 and self.z == 0:
            # Don't do anything for this - otherwise we'd get division
            # by zero errors - we want this to be a valid normal!
            return self.copy()
        else:
            # Adding 0 clears -0 values - we don't want those.
            mag = self.mag()
            return type(self)(
                self.x / mag + 0,
                self.y / mag + 0,
                self.z / mag + 0,
            )

    def dot(self, other: AnyVec) -> float:
        """Return the dot product of both Vectors.

        Tip: using this in the form ``Vec.dot(a, b)`` may be more readable.
        """
        return (
            self.x * other[0] +
            self.y * other[1] +
            self.z * other[2]
        )

    def cross(self: VecT, other: AnyVec) -> VecT:
        """Return the cross product of both Vectors.

        If this is called as a method (``a.cross(b)``), the result will have the
        same type as ``a``. Otherwise, if called as ``Vec.cross(a, b)`` or ``FrozenVec.cross(a, b)``, the
        type of the class takes priority.
        """
        return type(self)(
            self.y * other[2] - self.z * other[1],
            self.z * other[0] - self.x * other[2],
            self.x * other[1] - self.y * other[0],
        )

    def norm_mask(self: VecT, normal: 'Vec | FrozenVec') -> VecT:
        """Subtract the components of this vector not in the direction of the normal.

        If the normal is axis-aligned, this will zero out the other axes.
        If not axis-aligned, it will do the equivalent.
        """
        norm = normal.norm()
        return type(self)(norm * self.dot(norm))

    len = mag
    mag_sq = len_sq


@final
class FrozenVec(VecBase):
    """Immutable vector class. This cannot be changed once created, but is hashable."""
    __slots__ = ()

    def __new__(
        cls,
        x: Union[int, float, 'VecBase', Iterable[float]]=0.0,
        y: float=0.0,
        z: float=0.0,
    ) -> 'FrozenVec':
        """Create a ``FrozenVec``.

        All values are converted to :external:py:class:`float`\\ s automatically.
        If no value is given, that axis will be set to ``0``.
        An iterable can be passed in (as the ``x`` argument), which will be
        used for ``x``, ``y``, and ``z``.
        """
        # Already a FrozenVec.
        if isinstance(x, cls):
            return x
        res = object.__new__(cls)
        if isinstance(x, (int, float)):
            res._x = float(x)
            res._y = float(y)
            res._z = float(z)
        elif isinstance(x, VecBase):
            # Mutable, we know it's safe to copy.
            res._x = x._x
            res._y = x._y
            res._z = x._z
        else:
            it = iter(x)
            res._x = float(next(it, 0.0))
            res._y = float(next(it, y))
            res._z = float(next(it, z))
        return res

    @classmethod
    @overload
    def with_axes(cls, axis1: str, val1: Union[float, VecBase]) -> 'FrozenVec': ...

    @classmethod
    @overload
    def with_axes(
        cls,
        axis1: str, val1: Union[float, VecBase],
        axis2: str, val2: Union[float, VecBase],
    ) -> 'FrozenVec': ...

    @classmethod
    @overload
    def with_axes(
        cls,
        axis1: str, val1: Union[float, VecBase],
        axis2: str, val2: Union[float, VecBase],
        axis3: str, val3: Union[float, VecBase],
    ) -> 'FrozenVec': ...

    @classmethod
    def with_axes(
        cls,
        axis1: str,
        val1: Union[float, VecBase],
        axis2: Optional[str] = None,
        val2: Union[float, VecBase]=0.0,
        axis3: Optional[str] = None,
        val3: Union[float, VecBase]=0.0,
    ) -> 'FrozenVec':
        """Create a Vector, given a number of axes and corresponding values.

        This is a convenience for doing the following::

            vec = Vec()
            vec[axis1] = val1
            vec[axis2] = val2
            vec[axis3] = val3

        The magnitudes can also be Vectors, in which case the matching
        axis will be used from the vector.
        """
        vals = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        if isinstance(val1, VecBase):
            vals[axis1] = val1[axis1]
        else:
            vals[axis1] = val1

        if axis2 is not None:
            if isinstance(val2, VecBase):
                vals[axis2] = val2[axis2]
            else:
                vals[axis2] = val2

            if axis3 is not None:
                if isinstance(val3, VecBase):
                    vals[axis3] = val3[axis3]
                else:
                    vals[axis3] = val3

        return cls(**vals)

    def copy(self) -> 'FrozenVec':
        """FrozenVec is immutable."""
        return self

    def __copy__(self) -> 'FrozenVec':
        """FrozenVec is immutable."""
        return self

    def __deepcopy__(self, memodict: Optional[Dict[Any, Any]]=None) -> 'FrozenVec':
        """FrozenVec is immutable."""
        return self

    def __repr__(self) -> str:
        """Code required to reproduce this vector."""
        return f"FrozenVec({format_float(self.x)}, {format_float(self.y)}, {format_float(self.z)})"

    def __reduce__(self) -> Tuple[Callable[[float, float, float], 'FrozenVec'], Tuple3]:
        """Pickling support.

        This redirects to a global function, so C/Python versions
        interoperate.
        """
        return _mk_fvec, (self._x, self._y, self._z)

    def __hash__(self) -> int:
        """Hashing a frozen vec is the same as hashing the tuple form."""
        return hash((round(self._x, 6), round(self._y, 6), round(self._z, 6)))

    def cross(self: VecUnion, other: AnyVec) -> 'FrozenVec':
        """Return the cross product of both Vectors.

        If this is called as a method (``a.cross(b)``), the result will have the
        same type as ``a``. Otherwise, if called as ``Vec.cross(a, b)`` or ``FrozenVec.cross(a, b)``, the
        type of the class takes priority.
        """
        return Py_FrozenVec(
            self.y * other[2] - self.z * other[1],
            self.z * other[0] - self.x * other[2],
            self.x * other[1] - self.y * other[0],
        )

    def thaw(self) -> 'Vec':
        """Return a mutable copy of this vector."""
        vec = Py_Vec.__new__(Py_Vec)
        vec._x = self._x
        vec._y = self._y
        vec._z = self._z
        return vec
    

VecBase.W = VecBase.west   = VecBase.x_neg = FrozenVec(x=-1)
VecBase.E = VecBase.east   = VecBase.x_pos = FrozenVec(x=+1)
VecBase.S = VecBase.south  = VecBase.y_neg = FrozenVec(y=-1)
VecBase.N = VecBase.north  = VecBase.y_pos = FrozenVec(y=+1)
VecBase.B = VecBase.bottom = VecBase.z_neg = FrozenVec(z=-1)
VecBase.T = VecBase.top    = VecBase.z_pos = FrozenVec(z=+1)


@final
class Vec(VecBase):
    """A 3D Vector. This has most standard Vector functions.

    >>> Vec(1, 2, z=3)  # Positional or vec, defaults to 0.
    Vec(1, 2, 3)
    >> Vec(range(3))  # Any 1,2 or 3 long iterable
    Vec(0, 1, 2)
    >>> Vec(1, 2, 3) * 2
    Vec(2, 4, 6)
    >>> Vec.from_str('<4 2 -45>')  # Parse strings.
    Vec(4, 2, -45)

    Operators and comparisons will treat 3-tuples interchangably with vectors, which is more
    convenient when specifying constant values.
    >>> Vec(3, 8, 7) - (0, 3, 4)
    Vec(3, 5, 3)

    Addition/subtraction can be performed between either vectors or scalar values (applying equally
    to all axes). Multiplication/division must be performed between a vector and scalar to scale -
    use `Vec.dot()` or `Vec.cross()` for those operations.

    Values can be modified by either setting/getting `x`, `y` and `z` attributes.
    In addition, the following indexes are allowed (case-insensitive):
    * `0`  `1`  `2`
    * `"x"`, `"y"`, `"z"`
    """
    __slots__ = ()

    # noinspection PyMissingConstructor
    def __init__(
        self,
        x: Union[int, float, 'VecBase', Iterable[float]]=0.0,
        y: float=0.0,
        z: float=0.0,
    ) -> None:
        """Create a Vector.

        All values are converted to :external:py:class:`float`\\ s automatically.
        If no value is given, that axis will be set to ``0``.
        An iterable can be passed in (as the ``x`` argument), which will be
        used for ``x``, ``y``, and ``z``.
        """
        if isinstance(x, (int, float)):
            self._x = float(x)
            self._y = float(y)
            self._z = float(z)
        elif isinstance(x, Py_Vec):
            self._x = x.x
            self._y = x.y
            self._z = x.z
        else:
            it = iter(x)
            self._x = float(next(it, 0.0))
            self._y = float(next(it, y))
            self._z = float(next(it, z))

    @property
    def x(self) -> float:
        """The X axis of the vector."""
        return self._x

    @x.setter
    def x(self, value: float) -> None:
        self._x = float(value)

    @property
    def y(self) -> float:
        """The Y axis of the vector."""
        return self._y

    @y.setter
    def y(self, value: float) -> None:
        self._y = float(value)

    @property
    def z(self) -> float:
        """The Z axis of the vector."""
        return self._z

    @z.setter
    def z(self, value: float) -> None:
        self._z = float(value)

    @classmethod
    @overload
    def with_axes(cls, axis1: str, val1: Union[float, VecBase]) -> 'Vec': ...

    @classmethod
    @overload
    def with_axes(
        cls,
        axis1: str, val1: Union[float, VecBase],
        axis2: str, val2: Union[float, VecBase],
    ) -> 'Vec': ...

    @classmethod
    @overload
    def with_axes(
        cls,
        axis1: str, val1: Union[float, VecBase],
        axis2: str, val2: Union[float, VecBase],
        axis3: str, val3: Union[float, VecBase],
    ) -> 'Vec': ...

    @classmethod
    def with_axes(
        cls,
        axis1: str,
        val1: Union[float, VecBase],
        axis2: Optional[str] = None,
        val2: Union[float, VecBase]=0.0,
        axis3: Optional[str] = None,
        val3: Union[float, VecBase]=0.0,
    ) -> 'Vec':
        """Create a Vector, given a number of axes and corresponding values.

        This is a convenience for doing the following::

            vec = Vec()
            vec[axis1] = val1
            vec[axis2] = val2
            vec[axis3] = val3

        The magnitudes can also be Vectors, in which case the matching
        axis will be used from the vector.
        """
        vec = cls()
        vec[axis1] = val1[axis1] if isinstance(val1, VecBase) else val1
        if axis2 is not None:
            vec[axis2] = val2[axis2] if isinstance(val2, VecBase) else val2
            if axis3 is not None:
                vec[axis3] = val3[axis3] if isinstance(val3, VecBase) else val3
        return vec

    def __repr__(self) -> str:
        """Code required to reproduce this vector."""
        return f"Vec({format_float(self.x)}, {format_float(self.y)}, {format_float(self.z)})"

    def freeze(self) -> FrozenVec:
        """Return an immutable version of this vector."""
        return Py_FrozenVec(self.x, self.y, self.z)

    def copy(self) -> 'Vec':
        """Create a duplicate of this vector."""
        return Py_Vec(self.x, self.y, self.z)

    __copy__ = copy  # copy module support.

    def cross(self: VecUnion, other: AnyVec) -> 'Vec':
        """Return the cross product of both Vectors.

        If this is called as a method (``a.cross(b)``), the result will have the
        same type as ``a``. Otherwise, if called as ``Vec.cross(a, b)`` or ``FrozenVec.cross(a, b)``, the
        type of the class takes priority.
        """
        return Py_Vec(
            self.y * other[2] - self.z * other[1],
            self.z * other[0] - self.x * other[2],
            self.x * other[1] - self.y * other[0],
        )

    if TYPE_CHECKING:
        def __iadd__(self, other: Union['VecBase', Tuple3, int, float]) -> 'Vec': ...
        def __isub__(self, other: Union['VecBase', Tuple3, int, float]) -> 'Vec': ...
        def __imul__(self, other: float) -> 'Vec': ...
        def __itruediv__(self, other: float) -> 'Vec': ...
        def __ifloordiv__(self, other: float) -> 'Vec': ...
        def __imod__(self, other: float) -> 'Vec': ...

    _funcname = _op = _pretty = ''
    # Use exec() to generate all the number magic methods. This reduces code
    # duplication since they're all very similar.
    for _funcname, _op in (('add', '+'), ('sub', '-')):
        exec(
            _VEC_ADDSUB_INPLACE_TEMP.format(func=_funcname, op=_op),
            globals(),
            locals(),
        )

    for _funcname, _op, _pretty in (
            ('mul', '*', 'multiply'),
            ('truediv', '/', 'divide'),
            ('floordiv', '//', 'floor-divide'),
            ('mod', '%', 'modulus'),
    ):
        exec(
            _VEC_MULDIV_INPLACE_TEMP.format(func=_funcname, op=_op, pretty=_pretty),
            globals(),
            locals(),
        )

    del _funcname, _op, _pretty

    def __imatmul__(self, other: Union['AngleBase', 'MatrixBase']) -> 'Vec':
        """We need to define this, so it's in-place."""
        if isinstance(other, MatrixBase):
            mat = other
        elif isinstance(other, AngleBase):
            mat = Py_Matrix.from_angle(other)
        else:
            return NotImplemented
        # noinspection PyProtectedMember
        mat._vec_rot(self)
        return self

    def __reduce__(self) -> Tuple[Callable[[float, float, float], 'Vec'], Tuple3]:
        """Pickling support.

        This redirects to a global function, so C/Python versions
        interoperate.
        """
        return _mk_vec, (self.x, self.y, self.z)

    def __setitem__(self, ind: Union[str, int], val: float) -> None:
        """Allow editing values by index instead of name if desired.

        This accepts either ``0``, ``1``, ``2`` or ``x``, ``y``, ``z`` to edit values.
        Useful in conjunction with a loop to apply commands to all values.
        """
        if ind == 0 or ind == "x":
            self.x = float(val)
        elif ind == 1 or ind == "y":
            self.y = float(val)
        elif ind == 2 or ind == "z":
            self.z = float(val)
        else:
            raise KeyError(f'Invalid axis: {ind!r}')

    # Deprecated, so no need to duplicate for FrozenVec.
    def rotate(
        self,
        pitch: float=0.0,
        yaw: float=0.0,
        roll: float=0.0,
        round_vals: bool=True,
    ) -> 'Vec':
        """Old method to rotate a vector by a Source rotational angle.

        :deprecated: do ``Vec(...) @ Angle(...)`` instead.

        If round is True, all values will be rounded to 6 decimals
        (since these calculations always have small inprecision.)
        """
        warnings.warn("Use vec @ Angle() instead.", DeprecationWarning, stacklevel=2)
        mat = Py_Matrix.from_angle(Py_Angle(pitch, yaw, roll))
        # noinspection PyProtectedMember
        mat._vec_rot(self)
        if round_vals:
            self.x = round(self.x, 6)
            self.y = round(self.y, 6)
            self.z = round(self.z, 6)
        return self

    def rotate_by_str(
        self, ang: str,
        pitch: float=0.0, yaw: float=0.0, roll: float=0.0,
        round_vals: bool=True,
    ) -> 'Vec':
        """Rotate a vector, using a string instead of a vector.

        :deprecated: use `Vec(...) @ Angle.from_str(...)` instead.
        """
        warnings.warn("Use vec @ Angle.from_str() instead.", DeprecationWarning, stacklevel=2)
        mat = Py_Matrix.from_angle(Py_Angle.from_str(ang, pitch, yaw, roll))
        # noinspection PyProtectedMember
        mat._vec_rot(self)
        if round_vals:
            self.x = round(self.x, 6)
            self.y = round(self.y, 6)
            self.z = round(self.z, 6)
        return self

    def to_angle_roll(self, z_norm: VecUnion, stride: int=0) -> 'Angle':
        """Produce a Source Engine angle with roll.

        :deprecated: Use :py:func:`Matrix.from_basis()` and then :py:func:`Matrix.to_angle()`. ``from_basis()`` can take any two direction pairs.
        :param z_norm: This must be at right angles to this vector. The resulting angle's ``+z``
            axis will point in this direction.
        :param stride: is no longer used, it defined the roll angles to try.
        """
        warnings.warn('Use Matrix.from_basis().to_angle()', DeprecationWarning)
        return Py_Matrix.from_basis(x=self, z=z_norm).to_angle()

    def rotation_around(self, rot: float=90) -> 'Angle':
        """For an axis-aligned normal, return the angles which rotate around it.

        :deprecated: Use :py:func:`Matrix.axis_angle()` and then :py:func:`Matrix.to_angle()`. :py:func:`~Matrix.axis_angle()` works for any arbitary axis.
        """
        warnings.warn('Use Matrix.axis_angle().to_angle()', DeprecationWarning)
        if self.x and not self.y and not self.z:
            return Py_Angle(roll=math.copysign(rot, self.x))
        elif self.y and not self.x and not self.z:
            return Py_Angle(pitch=math.copysign(rot, self.y))
        elif self.z and not self.x and not self.y:
            return Py_Angle(yaw=math.copysign(rot, self.z))
        else:
            raise ValueError('Zero vector!')

    def max(self, other: AnyVec) -> None:
        """Set this vector's values to the maximum of the two vectors."""
        if self.x < other[0]:
            self.x = other[0]
        if self.y < other[1]:
            self.y = other[1]
        if self.z < other[2]:
            self.z = other[2]

    def min(self, other: AnyVec) -> None:
        """Set this vector's values to be the minimum of the two vectors."""
        if self.x > other[0]:
            self.x = other[0]
        if self.y > other[1]:
            self.y = other[1]
        if self.z > other[2]:
            self.z = other[2]

    def localise(
        self,
        origin: Union['Vec', Tuple3],
        angles: Union[AnyAngle, AnyMatrix, None] = None,
    ) -> None:
        """Shift this point to be local to the given position and angles.

        This effectively translates local-space offsets to a global location,
        given the parent's origin and angles.
        This is an in-place version of ``self @ angles + origin``.
        """
        mat = to_matrix(angles)
        # noinspection PyProtectedMember
        mat._vec_rot(self)
        self.__iadd__(origin)

    @contextlib.contextmanager
    def transform(self) -> Iterator['Matrix']:
        """Perform rotations on this Vector efficiently.

        Used as a context manager, which returns a matrix.
        When the body is exited safely, the matrix is applied to
        the angle.
        """
        mat = Py_Matrix()
        yield mat
        # noinspection PyProtectedMember
        mat._vec_rot(self)

# Maps (1, 1) -> ._bb attributes, for getting/setting by XY coordinate.
_IND_TO_SLOT: Dict[Tuple[int, int], Literal[
    '_aa', '_ab', '_ac',
    '_ba', '_bb', '_bc',
    '_ca', '_cb', '_cc',
]] = {
    (0, 0): '_aa', (0, 1): '_ab', (0, 2): '_ac',
    (1, 0): '_ba', (1, 1): '_bb', (1, 2): '_bc',
    (2, 0): '_ca', (2, 1): '_cb', (2, 2): '_cc',
}


class MatrixBase:
    """Common code for both matrix versions."""
    __slots__ = [
        '_aa', '_ab', '_ac',
        '_ba', '_bb', '_bc',
        '_ca', '_cb', '_cc'
    ]
    _aa: float
    _ab: float
    _ac: float
    _ba: float
    _bb: float
    _bc: float
    _ca: float
    _cb: float
    _cc: float

    def __init__(self, matrix: 'MatrixBase | None' = None) -> None:
        """MatrixBase cannot be instantiated."""
        if type(self) is MatrixBase:
            raise TypeError('MatrixBase cannot be instantiated.')

    @classmethod
    def _from_raw(
        cls: Type[MatrixT],
        aa: float, ab: float, ac: float,
        ba: float, bb: float, bc: float,
        ca: float, cb: float, cc: float,
    ) -> MatrixT:
        """Construct from individual data values."""
        self = cls.__new__(cls)
        self._aa, self._ab, self._ac = aa, ab, ac
        self._ba, self._bb, self._bc = ba, bb, bc
        self._ca, self._cb, self._cc = ca, cb, cc
        return self

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MatrixBase):
            return (
                self._aa == other._aa and self._ab == other._ab and self._ac == other._ac and
                self._ba == other._ba and self._bb == other._bb and self._bc == other._bc and
                self._ca == other._ca and self._cb == other._cb and self._cc == other._cc
            )
        return NotImplemented

    def __repr__(self) -> str:
        return (
            f'<{self.__class__.__name__} '
            f'{self._aa:.3} {self._ab:.3} {self._ac:.3}, '
            f'{self._ba:.3} {self._bb:.3} {self._bc:.3}, '
            f'{self._ca:.3} {self._cb:.3} {self._cc:.3}'
            '>'
        )

    def copy(self: MatrixT) -> MatrixT:
        """Duplicate this matrix."""
        raise NotImplementedError

    @classmethod
    def from_pitch(cls: Type[MatrixT], pitch: float) -> MatrixT:
        """Return the matrix representing a pitch rotation (Y axis)."""
        rad_pitch = math.radians(pitch)
        cos = math.cos(rad_pitch)
        sin = math.sin(rad_pitch)

        rot = cls.__new__(cls)

        rot._aa, rot._ab, rot._ac = cos, 0.0, -sin
        rot._ba, rot._bb, rot._bc = 0.0, 1.0, 0.0
        rot._ca, rot._cb, rot._cc = sin, 0.0, cos

        return rot

    @classmethod
    def from_yaw(cls: Type[MatrixT], yaw: float) -> MatrixT:
        """Return the matrix representing a yaw rotation (Z axis)."""
        rad_yaw = math.radians(yaw)
        sin = math.sin(rad_yaw)
        cos = math.cos(rad_yaw)

        rot = cls.__new__(cls)

        rot._aa, rot._ab, rot._ac = cos, sin, 0.0
        rot._ba, rot._bb, rot._bc = -sin, cos, 0.0
        rot._ca, rot._cb, rot._cc = 0.0, 0.0, 1.0

        return rot

    @classmethod
    def from_roll(cls: Type[MatrixT], roll: float) -> MatrixT:
        """Return the matrix representing a roll rotation (X axis)."""
        rad_roll = math.radians(roll)
        cos_r = math.cos(rad_roll)
        sin_r = math.sin(rad_roll)

        rot = cls.__new__(cls)

        rot._aa, rot._ab, rot._ac = 1.0, 0.0, 0.0
        rot._ba, rot._bb, rot._bc = 0.0, cos_r, sin_r
        rot._ca, rot._cb, rot._cc = 0.0, -sin_r, cos_r

        return rot

    @classmethod
    @overload
    def from_angle(cls: Type[MatrixT], __angle: 'AngleBase') -> MatrixT: ...
    @classmethod
    @overload
    def from_angle(cls: Type[MatrixT], pitch: float, yaw: float, roll: float) -> MatrixT: ...
    @classmethod
    def from_angle(
        cls: Type[MatrixT],
        pitch: Union['AngleBase', float],
        yaw: Optional[float]=None,
        roll: Optional[float]=None,
    ) -> MatrixT:
        """Return the rotation representing an Euler angle.

        Either an Angle can be passed, or the raw pitch/yaw/roll angles.
        """
        if isinstance(pitch, AngleBase):
            if yaw is not None or roll is not None:
                raise TypeError('Matrix.from_angles() accepts a single Angle or 3 floats!')
            rad_pitch = math.radians(pitch.pitch)
            rad_yaw = math.radians(pitch.yaw)
            rad_roll = math.radians(pitch.roll)
        elif yaw is None or roll is None:
            raise TypeError('Matrix.from_angles() accepts a single Angle or 3 floats!')
        else:
            rad_pitch = math.radians(pitch)
            rad_yaw = math.radians(yaw)
            rad_roll = math.radians(roll)

        cos_p = math.cos(rad_pitch)
        sin_p = math.sin(rad_pitch)
        sin_y = math.sin(rad_yaw)
        cos_y = math.cos(rad_yaw)
        cos_r = math.cos(rad_roll)
        sin_r = math.sin(rad_roll)

        rot = cls.__new__(cls)

        rot._aa = cos_p * cos_y
        rot._ab = cos_p * sin_y
        rot._ac = -sin_p

        cos_r_cos_y = cos_r * cos_y
        cos_r_sin_y = cos_r * sin_y
        sin_r_cos_y = sin_r * cos_y
        sin_r_sin_y = sin_r * sin_y

        rot._ba = sin_p * sin_r_cos_y - cos_r_sin_y
        rot._bb = sin_p * sin_r_sin_y + cos_r_cos_y
        rot._bc = sin_r * cos_p

        rot._ca = (sin_p * cos_r_cos_y + sin_r_sin_y)
        rot._cb = (sin_p * cos_r_sin_y - sin_r_cos_y)
        rot._cc = cos_r * cos_p
        return rot

    @classmethod
    def from_angstr(
        cls: Type[MatrixT],
        val: Union[str, 'Angle'],
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
    ) -> MatrixT:
        """Parse a string of the form "pitch yaw roll", then convert to a Matrix.

        This is equivalent to combining :py:func:`Matrix.from_angle()` and
        :py:func:`Angle.from_str`, except more efficient.
        """
        pitch, yaw, roll = Py_parse_vec_str(val, pitch, yaw, roll)
        return cls.from_angle(pitch, yaw, roll)

    @classmethod
    def axis_angle(cls: Type[MatrixT], axis: Union[Vec, Tuple3], angle: float) -> MatrixT:
        """Compute the rotation matrix forming a rotation around an axis by a specific angle."""
        x, y, z = Vec(axis).norm()
        # Invert, so it matches the orientation of Angles().
        angle_rad = -math.radians(angle)
        cos = math.cos(angle_rad)
        icos = 1 - cos
        sin = math.sin(angle_rad)

        mat = cls.__new__(cls)

        mat._aa = x*x * icos + cos
        mat._ab = x*y * icos - z*sin
        mat._ac = x*z * icos + y*sin

        mat._ba = y*x * icos + z*sin
        mat._bb = y*y * icos + cos
        mat._bc = y*z * icos - x*sin

        mat._ca = z*x * icos - y*sin
        mat._cb = z*y * icos + x*sin
        mat._cc = z*z * icos + cos

        return mat

    def forward(self, mag: float = 1.0) -> 'Vec':
        """Return a vector with the given magnitude pointing along the X axis."""
        return Py_Vec(mag * self._aa, mag * self._ab, mag * self._ac)

    def left(self, mag: float = 1.0) -> 'Vec':
        """Return a vector with the given magnitude pointing along the Y axis."""
        return Py_Vec(mag * self._ba, mag * self._bb, mag * self._bc)

    def up(self, mag: float = 1.0) -> 'Vec':
        """Return a vector with the given magnitude pointing along the Z axis."""
        return Py_Vec(mag * self._ca, mag * self._cb, mag * self._cc)

    def __getitem__(self, item: Tuple[int, int]) -> float:
        """Retrieve an individual matrix value by x, y position (0-2)."""
        return cast(float, getattr(self, _IND_TO_SLOT[item]))

    __iter__: None = None
    """Iteration doesn't make much sense."""

    def to_angle(self) -> 'Angle':
        """Return an Euler angle replicating this rotation."""
        return self._to_angle(Py_Angle.__new__(Py_Angle))

    def _to_angle(self, ang: AngleT) -> AngleT:
        """Set the specified angle to this rotation.

        Internal, modifies the specified angle even if frozen -ensure it's new!
        """
        # https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/mathlib/mathlib_base.cpp#L208
        for_x = self._aa
        for_y = self._ab
        for_z = self._ac
        left_x = self._ba
        left_y = self._bb
        left_z = self._bc
        # up_x = self.ca
        # up_y = self.cb
        up_z = self._cc

        horiz_dist = math.sqrt(for_x**2 + for_y**2)
        if horiz_dist > 0.001:
            ang._yaw = math.degrees(math.atan2(for_y, for_x)) % 360.0
            ang._pitch = math.degrees(math.atan2(-for_z, horiz_dist)) % 360.0
            ang._roll = math.degrees(math.atan2(left_z, up_z)) % 360.0
        else:
            # Vertical, gimbal lock (yaw=roll)...
            ang._yaw = math.degrees(math.atan2(-left_x, left_y)) % 360.0
            ang._pitch = math.degrees(math.atan2(-for_z, horiz_dist)) % 360.0
            ang._roll = 0.0  # Can't produce.
        return ang

    def transpose(self: MatrixT) -> MatrixT:
        """Return the transpose of this matrix."""
        cls: Type[MatrixT] = type(self)
        rot = cls.__new__(cls)

        rot._aa, rot._ab, rot._ac = self._aa, self._ba, self._ca
        rot._ba, rot._bb, rot._bc = self._ab, self._bb, self._cb
        rot._ca, rot._cb, rot._cc = self._ac, self._bc, self._cc

        return rot

    @classmethod
    @overload
    def from_basis(cls: Type[MatrixT], *, x: VecUnion, y: VecUnion, z: VecUnion) -> MatrixT: ...
    @classmethod
    @overload
    def from_basis(cls: Type[MatrixT], *, x: VecUnion, y: VecUnion) -> MatrixT: ...
    @classmethod
    @overload
    def from_basis(cls: Type[MatrixT], *, y: VecUnion, z: VecUnion) -> MatrixT: ...
    @classmethod
    @overload
    def from_basis(cls: Type[MatrixT], *, x: VecUnion, z: VecUnion) -> MatrixT: ...
    @classmethod
    def from_basis(
        cls: Type[MatrixT], *,
        x: Optional[VecUnion] = None,
        y: Optional[VecUnion] = None,
        z: Optional[VecUnion] = None,
    ) -> MatrixT:
        """Construct a matrix from at least two basis vectors.

        The third is computed, if not provided.
        """
        if x is None and y is not None and z is not None:
            x = Vec.cross(y, z)
        elif y is None and x is not None and z is not None:
            y = Vec.cross(z, x)
        elif z is None and x is not None and y is not None:
            z = Vec.cross(x, y)
        if x is None or y is None or z is None:
            raise TypeError('At least two vectors must be provided!')
        mat = cls.__new__(cls)
        mat._aa, mat._ab, mat._ac = x.norm()
        mat._ba, mat._bb, mat._bc = y.norm()
        mat._ca, mat._cb, mat._cc = z.norm()
        return mat

    def _mat_mul(self, other: 'MatrixBase') -> None:
        """Rotate myself by the other matrix."""
        # We don't use each row after assigning to the set, so we can re-assign.
        # 3-tuple unpacking is optimised.
        self._aa, self._ab, self._ac = (
            self._aa * other._aa + self._ab * other._ba + self._ac * other._ca,
            self._aa * other._ab + self._ab * other._bb + self._ac * other._cb,
            self._aa * other._ac + self._ab * other._bc + self._ac * other._cc,
        )

        self._ba, self._bb, self._bc = (
            self._ba * other._aa + self._bb * other._ba + self._bc * other._ca,
            self._ba * other._ab + self._bb * other._bb + self._bc * other._cb,
            self._ba * other._ac + self._bb * other._bc + self._bc * other._cc,
        )

        self._ca, self._cb, self._cc = (
            self._ca * other._aa + self._cb * other._ba + self._cc * other._ca,
            self._ca * other._ab + self._cb * other._bb + self._cc * other._cb,
            self._ca * other._ac + self._cb * other._bc + self._cc * other._cc,
        )

    def _vec_rot(self, vec: VecBase) -> None:
        """Rotate a vector by our value, inplace (even if frozen)."""
        x = vec.x
        y = vec.y
        z = vec.z
        vec._x = (x * self._aa) + (y * self._ba) + (z * self._ca)
        vec._y = (x * self._ab) + (y * self._bb) + (z * self._cb)
        vec._z = (x * self._ac) + (y * self._bc) + (z * self._cc)

    def __matmul__(self: MatrixT, other: 'MatrixBase | AngleBase') -> MatrixT:
        if isinstance(other, MatrixBase):
            mat = self.copy()
            mat._mat_mul(other)
            return mat
        elif isinstance(other, AngleBase):
            mat = self.copy()
            mat._mat_mul(Py_Matrix.from_angle(other))
            return mat
        else:
            return NotImplemented

    @overload
    def __rmatmul__(self, other: FrozenVec) -> FrozenVec: ...
    @overload
    def __rmatmul__(self, other: 'Vec | Tuple3') -> 'Vec': ...
    @overload
    def __rmatmul__(self, other: MatrixT) -> MatrixT: ...
    @overload
    def __rmatmul__(self, other: AngleT) -> AngleT: ...

    def __rmatmul__(self, other: 'VecBase | Tuple3 | MatrixBase | AngleBase') -> 'Vec | FrozenVec | MatrixBase | AngleBase':
        mat: MatrixBase
        result: VecUnion
        if isinstance(other, Py_Vec) or isinstance(other, tuple):
            result = Py_Vec(other)
            self._vec_rot(result)
            return result
        elif isinstance(other, Py_FrozenVec):
            # We need to actually copy this!
            # noinspection PyProtectedMember
            result = Py_FrozenVec(other._x, other._y, other._z)
            self._vec_rot(result)
            return result
        elif isinstance(other, AngleBase):
            mat = Py_Matrix.from_angle(other)
            mat._mat_mul(self)
            cls = type(other)
            return mat._to_angle(cls.__new__(cls))
        elif isinstance(other, MatrixBase):
            mat = other.copy()
            mat._mat_mul(self)
            return mat
        else:
            return NotImplemented


@final
class FrozenMatrix(MatrixBase):
    """Represents an immutable rotation via a transformation matrix.

    When performing multiple rotations, it is more efficient to create one of these instead of using
    an ``Angle`` directly. To construct a rotation, use one of the several classmethods available
    depending on what rotation is desired.
    """
    def __new__(cls, matrix: 'MatrixBase | None' = None) -> 'FrozenMatrix':
        """Create a new matrix.

        If an existing matrix is supplied, it will be copied. Otherwise, an identity matrix is
        produced.
        """
        if isinstance(matrix, FrozenMatrix):
            return matrix
        self = super().__new__(cls)
        if matrix is not None:
            self._aa, self._ab, self._ac = matrix._aa, matrix._ab, matrix._ac
            self._ba, self._bb, self._bc = matrix._ba, matrix._bb, matrix._bc
            self._ca, self._cb, self._cc = matrix._ca, matrix._cb, matrix._cc
        else:
            self._aa, self._ab, self._ac = 1.0, 0.0, 0.0
            self._ba, self._bb, self._bc = 0.0, 1.0, 0.0
            self._ca, self._cb, self._cc = 0.0, 0.0, 1.0
        return self

    def thaw(self) -> 'Matrix':
        """Return a mutable copy of this matrix."""
        rot = Py_Matrix.__new__(Py_Matrix)

        rot._aa, rot._ab, rot._ac = self._aa, self._ab, self._ac
        rot._ba, rot._bb, rot._bc = self._ba, self._bb, self._bc
        rot._ca, rot._cb, rot._cc = self._ca, self._cb, self._cc

        return rot

    def copy(self) -> 'FrozenMatrix':
        """Frozen matrices are immutable."""
        return self

    __copy__ = copy

    def __deepcopy__(self, memodict: object=...) -> 'FrozenMatrix':
        """Frozen matrices are immutable."""
        return self

    def __reduce__(self) -> Tuple[
        Callable[[float, float, float, float, float, float, float, float, float], 'FrozenMatrix'],
        Tuple[float, float, float, float, float, float, float, float, float]
    ]:
        """Pickling support.

        This redirects to a global function, so C/Python versions
        interoperate.
        """
        return (_mk_fmat, (
            self._aa, self._ab, self._ac,
            self._ba, self._bb, self._bc,
            self._ca, self._cb, self._cc
        ))


@final
class Matrix(MatrixBase):
    """Represents a rotation via a transformation matrix.

    When performing multiple rotations, it is more efficient to create one of these instead of using
    an ``Angle`` directly. To construct a rotation, use one of the several classmethods available
    depending on what rotation is desired.
    """
    # noinspection PyMissingConstructor   # Does nothing.
    def __init__(self, matrix: 'MatrixBase | None' = None) -> None:
        """Create a new matrix.

        If an existing matrix is supplied, it will be copied. Otherwise, an identity matrix is
        produced.
        """
        if matrix is not None:
            self._aa, self._ab, self._ac = matrix._aa, matrix._ab, matrix._ac
            self._ba, self._bb, self._bc = matrix._ba, matrix._bb, matrix._bc
            self._ca, self._cb, self._cc = matrix._ca, matrix._cb, matrix._cc
        else:
            self._aa, self._ab, self._ac = 1.0, 0.0, 0.0
            self._ba, self._bb, self._bc = 0.0, 1.0, 0.0
            self._ca, self._cb, self._cc = 0.0, 0.0, 1.0

    def freeze(self) -> FrozenMatrix:
        """Return a frozen copy of this matrix."""
        rot = Py_FrozenMatrix.__new__(Py_FrozenMatrix)

        rot._aa, rot._ab, rot._ac = self._aa, self._ab, self._ac
        rot._ba, rot._bb, rot._bc = self._ba, self._bb, self._bc
        rot._ca, rot._cb, rot._cc = self._ca, self._cb, self._cc

        return rot

    def copy(self) -> 'Matrix':
        """Duplicate this matrix."""
        rot = Py_Matrix.__new__(Py_Matrix)

        rot._aa, rot._ab, rot._ac = self._aa, self._ab, self._ac
        rot._ba, rot._bb, rot._bc = self._ba, self._bb, self._bc
        rot._ca, rot._cb, rot._cc = self._ca, self._cb, self._cc

        return rot

    __copy__ = copy

    def __deepcopy__(self, memodict: object=...) -> 'Matrix':
        """Duplicate this matrix."""
        rot = Py_Matrix.__new__(Py_Matrix)

        rot._aa, rot._ab, rot._ac = self._aa, self._ab, self._ac
        rot._ba, rot._bb, rot._bc = self._ba, self._bb, self._bc
        rot._ca, rot._cb, rot._cc = self._ca, self._cb, self._cc

        return rot

    def __reduce__(self) -> Tuple[
        Callable[[float, float, float, float, float, float, float, float, float], 'Matrix'],
        Tuple[float, float, float, float, float, float, float, float, float]
    ]:
        """Pickling support.

        This redirects to a global function, so C/Python versions
        interoperate.
        """
        return (_mk_mat, (
            self._aa, self._ab, self._ac,
            self._ba, self._bb, self._bc,
            self._ca, self._cb, self._cc
        ))

    def __setitem__(self, item: Tuple[int, int], value: float) -> None:
        """Set an individual matrix value by x, y position (0-2)."""
        setattr(self, _IND_TO_SLOT[item], value)

    def __imatmul__(self, other: 'MatrixBase | AngleBase') -> 'Matrix':
        if isinstance(other, MatrixBase):
            self._mat_mul(other)
            return self
        elif isinstance(other, AngleBase):
            self._mat_mul(Py_Matrix.from_angle(other))
            return self
        else:
            return NotImplemented


# noinspection PyArgumentList
class AngleBase:
    """Internal base class for Euler angles, implements common code."""
    # When normalising, we have to double-modulus because -1e-14 % 360.0 = 360.0.

    # Use the private attrs for matching also, we only hook assignment in the mutable one.
    __match_args__ = ('_pitch', '_yaw', '_roll')
    __slots__ = ('_pitch', '_yaw', '_roll')

    _pitch: float
    _yaw: float
    _roll: float

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """AngleBase cannot be instantiated."""
        if type(self) is AngleBase:
            raise TypeError('AngleBase cannot be instantiated.')

    @property
    def pitch(self) -> float:
        """The Y-axis rotation, performed second."""
        return self._pitch

    @property
    def yaw(self) -> float:
        """The Z-axis rotation, performed last."""
        return self._yaw

    @property
    def roll(self) -> float:
        """The X-axis rotation, performed first."""
        return self._roll

    @classmethod
    def from_str(
        cls: Type[AngleT], val: Union[str, 'AngleBase'],
        pitch: float=0.0, yaw: float=0.0, roll: float=0.0,
    ) -> AngleT:
        """Convert a string in the form ``(4 6 -4)`` into an Angle.

        If the string is unparsable, the provided default values are used instead.
        The string can be surrounded by any of the ``()``, ``{}``, ``[]``, ``<>`` bracket types,
        which are simply ignored.

        If the value is already an Angle, a copy will be returned.
        To only do parsing, use :py:func:`parse_vec_str()`.
        """

        pitch, yaw, roll = Py_parse_vec_str(val, pitch, yaw, roll)
        return cls(pitch, yaw, roll)

    @classmethod
    @overload
    def from_basis(cls: Type[AngleT], *, x: VecUnion, y: VecUnion, z: VecUnion) -> AngleT: ...

    @classmethod
    @overload
    def from_basis(cls: Type[AngleT], *, x: VecUnion, y: VecUnion) -> AngleT: ...

    @classmethod
    @overload
    def from_basis(cls: Type[AngleT], *, y: VecUnion, z: VecUnion) -> AngleT: ...

    @classmethod
    @overload
    def from_basis(cls: Type[AngleT], *, x: VecUnion, z: VecUnion) -> AngleT: ...

    @classmethod
    def from_basis(cls: Type[AngleT], **kwargs: VecUnion) -> AngleT:
        """Return the rotation which results in the specified local axes.

        At least two must be specified, with the third computed if necessary.
        """
        # We just delegate to Matrix's arg validation.
        # noinspection PyProtectedMember
        return Py_Matrix.from_basis(**kwargs)._to_angle(cls.__new__(cls))

    @classmethod
    @overload
    def with_axes(cls: Type[AngleT], axis1: str, val1: Union[float, 'AngleBase']) -> AngleT: ...

    @classmethod
    @overload
    def with_axes(
        cls: Type[AngleT],
        axis1: str, val1: Union[float, 'AngleBase'],
        axis2: str, val2: Union[float, 'AngleBase'],
    ) -> AngleT:
        ...

    @classmethod
    @overload
    def with_axes(
        cls: Type[AngleT],
        axis1: str, val1: Union[float, 'AngleBase'],
        axis2: str, val2: Union[float, 'AngleBase'],
        axis3: str, val3: Union[float, 'AngleBase'],
    ) -> AngleT:
        ...

    @classmethod
    def with_axes(
        cls: Type[AngleT],
        *args: Union[str, float, 'AngleBase'],
        **kwargs: Union[str, float, 'AngleBase'],
    ) -> AngleT:
        """Create an Angle, given a number of axes and corresponding values.

        This is a convenience for doing the following:
            ang = Angle()
            ang[axis1] = val1
            ang[axis2] = val2
            ang[axis3] = val3
        The magnitudes can also be Angles, in which case the matching
        axis will be used from the angle.
        """
        raise NotImplementedError

    def join(self, delim: str=', ') -> str:
        """Return a string with all numbers joined by the passed delimiter.

        This strips off the .0 if no decimal portion exists.
        """
        return f'{format_float(self._pitch)}{delim}{format_float(self._yaw)}{delim}{format_float(self._roll)}'

    def __str__(self) -> str:
        """Return the values, separated by spaces.

        This is the main format in Valve's file formats, though identical to
        vectors.
        This strips off the .0 if no decimal portion exists.
        """
        return f"{format_float(self._pitch)} {format_float(self._yaw)} {format_float(self._roll)}"

    def __format__(self, format_spec: str) -> str:
        """Control how the text is formatted."""
        if not format_spec:
            return str(self)

        pitch = format(self._pitch, format_spec)
        if '.' in pitch:
            pitch = pitch.rstrip('0')

        yaw = format(self._yaw, format_spec)
        if '.' in yaw:
            yaw = yaw.rstrip('0')

        roll = format(self._roll, format_spec)
        if '.' in roll:
            roll = roll.rstrip('0')
        return f'{pitch.rstrip(".")} {yaw.rstrip(".")} {roll.rstrip(".")}'

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return the Angle as a tuple."""
        return Vec_tuple(self._pitch, self._yaw, self._roll)

    def __len__(self) -> int:
        """The length of an Angle is always 3."""
        return 3

    def __iter__(self) -> Iterator[float]:
        """Iterating over the angles returns each value in turn."""
        yield self._pitch
        yield self._yaw
        yield self._roll

    def __reversed__(self) -> Iterator[float]:
        """Iterating over the angles returns each value in turn."""
        yield self._roll
        yield self._yaw
        yield self._pitch

    def __getitem__(self, ind: Union[str, int]) -> float:
        """Allow reading values by index instead of name if desired.

        This accepts the following indexes to read values:
        - ``0``, ``1``, ``2``
        - ``"pitch"``, ``"yaw"``, ``"roll"``
        - ``"pit"``, ``"yaw"``, ``"rol"``
        - ``"p"``, ``"y"``, ``"r"``
        Useful in conjunction with a loop to apply commands to all values.
        """
        if ind in (0, 'p', 'pit', 'pitch'):
            return self._pitch
        elif ind in (1, 'y', 'yaw'):
            return self._yaw
        elif ind in (2, 'r', 'rol', 'roll'):
            return self._roll
        raise KeyError('Invalid axis: {!r}'.format(ind))

    def __eq__(self, other: object) -> bool:
        """== test.

        Two Angles are equal if all three axes are the same.
        An Angle can be compared with a 3-tuple as if it was an Angle also.
        A tolerance of 1e-6 is accounted for automatically.
        """
        if isinstance(other, AngleBase):
            return (
                abs(other._pitch - self._pitch) <= 1e-6 and
                abs(other._yaw - self._yaw) <= 1e-6 and
                abs(other._roll - self._roll) <= 1e-6
            )
        elif _check_tuple3(other):
            pit = other[0] % 360.0 % 360.0
            yaw = other[1] % 360.0 % 360.0
            rol = other[2] % 360.0 % 360.0
            return (
                abs(self._pitch - pit) <= 1e-6 and
                abs(self._yaw - yaw) <= 1e-6 and
                abs(self._roll - rol) <= 1e-6
            )
        else:
            return NotImplemented

    def __ne__(self, other: object) -> bool:
        """!= test.

        Two Angles are equal if all three axes are the same.
        An Angle can be compared with a 3-tuple as if it was an Angle also.
        A tolerance of 1e-6 is accounted for automatically.
        """
        if isinstance(other, AngleBase):
            return (
                abs(other._pitch - self._pitch) > 1e-6 or
                abs(other._yaw - self._yaw) > 1e-6 or
                abs(other._roll - self._roll) > 1e-6
            )
        elif _check_tuple3(other):
            pit = other[0] % 360.0 % 360.0
            yaw = other[1] % 360.0 % 360.0
            rol = other[2] % 360.0 % 360.0
            return (
                abs(self._pitch - pit) > 1e-6 or
                abs(self._yaw   - yaw) > 1e-6 or
                abs(self._roll  - rol) > 1e-6
            )
        else:
            return NotImplemented

    # No ordering, there isn't any sensible relationship.

    def __mul__(self: AngleT, other: Union[int, float]) -> AngleT:
        """Angle * float multiplies each value."""
        if isinstance(other, (int, float)):
            return type(self)(
                self._pitch * other,
                self._yaw * other,
                self._roll * other,
            )
        return NotImplemented

    def __rmul__(self: AngleT, other: Union[int, float]) -> AngleT:
        """Angle * float multiplies each value."""
        if isinstance(other, (int, float)):
            return type(self)(
                other * self._pitch,
                other * self._yaw,
                other * self._roll,
            )
        return NotImplemented

    def _rotate_angle(self, target: 'AngleBase', cls: Type[AngleT]) -> AngleT:
        """Rotate the target by this angle.

        Inefficient if we have more than one rotation to do.
        """
        mat = Py_Matrix.from_angle(target)
        mat @= self
        # noinspection PyProtectedMember
        return mat._to_angle(cls.__new__(cls))

    # noinspection PyProtectedMember
    def __matmul__(self: AngleT, other: 'AngleBase | MatrixBase') -> AngleT:
        """Angle @ Angle or Angle @ Matrix rotates the first by the second."""
        if isinstance(other, AngleBase):
            return other._rotate_angle(self, type(self))
        elif isinstance(other, MatrixBase):
            mat = Py_Matrix.from_angle(self)
            mat._mat_mul(other)
            cls = type(self)
            return mat._to_angle(cls.__new__(cls))
        else:
            return NotImplemented

    @overload
    def __rmatmul__(self: AngleT, other: 'AngleBase') -> AngleT: ...
    @overload
    def __rmatmul__(self, other: Tuple3) -> 'Vec': ...
    @overload
    def __rmatmul__(self, other: VecT) -> VecT: ...

    def __rmatmul__(self: AngleT, other: 'AngleBase | Tuple3 | VecT') -> 'Vec | FrozenVec | VecT | AngleT':
        """Vec @ Angle rotates the first by the second."""
        if isinstance(other, (Vec, FrozenVec)):
            return other @ Py_Matrix.from_angle(self)
        elif isinstance(other, tuple):
            return Vec(other) @ Py_Matrix.from_angle(self)
        elif isinstance(other, AngleBase):
            # Should always be done by __matmul__!
            return self._rotate_angle(other, type(self))
        return NotImplemented


@final
class FrozenAngle(AngleBase):
    """Represents an immutable pitch-yaw-roll Euler angle."""
    __slots__ = ()

    def __new__(
        cls,
        pitch: Union[int, float, Iterable[Union[int, float]]]=0.0,
        yaw: Union[int, float]=0.0,
        roll: Union[int, float]=0.0,
    ) -> 'FrozenAngle':
        """Create a FrozenAngle.

        All values are converted to Floats automatically.
        If no value is given, that axis will be set to 0.
        An iterable can be passed in (as the pitch argument), which will be
        used for pitch, yaw, and roll. This includes Vectors and other Angles.
        """
        # Already a FrozenVec.
        if isinstance(pitch, cls):
            return pitch
        res = object.__new__(cls)
        if isinstance(pitch, (int, float)):
            res._pitch = float(pitch) % 360.0 % 360.0
            res._yaw = float(yaw) % 360.0 % 360.0
            res._roll = float(roll) % 360.0 % 360.0
        elif isinstance(pitch, AngleBase):
            # Bypass modulo, iteration and float conversion.
            res._pitch = pitch._pitch
            res._yaw = pitch._yaw
            res._roll = pitch._roll
        else:
            it = iter(pitch)
            res._pitch = float(next(it, 0.0)) % 360.0 % 360.0
            res._yaw = float(next(it, yaw)) % 360.0 % 360.0
            res._roll = float(next(it, roll)) % 360.0 % 360.0
        return res

    @classmethod
    @overload
    def with_axes(cls, axis1: str, val1: Union[float, AngleBase]) -> 'FrozenAngle': ...
    @classmethod
    @overload
    def with_axes(
        cls,
        axis1: str, val1: Union[float, AngleBase],
        axis2: str, val2: Union[float, AngleBase],
    ) -> 'FrozenAngle': ...
    @classmethod
    @overload
    def with_axes(
        cls,
        axis1: str, val1: Union[float, AngleBase],
        axis2: str, val2: Union[float, AngleBase],
        axis3: str, val3: Union[float, AngleBase],
    ) -> 'FrozenAngle': ...
    @classmethod
    def with_axes(
        cls,
        axis1: str,
        val1: Union[float, AngleBase],
        axis2: Optional[str] = None,
        val2: Union[float, AngleBase] = 0.0,
        axis3: Optional[str] = None,
        val3: Union[float, AngleBase] = 0.0,
    ) -> 'FrozenAngle':
        """Create an Angle, given a number of axes and corresponding values.

        This is a convenience for doing the following::

            ang = Angle()
            ang[axis1] = val1
            ang[axis2] = val2
            ang[axis3] = val3

        The magnitudes can also be Angles, in which case the matching
        axis will be used from the angle.
        """
        res = {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        res[axis1] = val1[axis1] if isinstance(val1, AngleBase) else val1
        if axis2 is not None:
            res[axis2] = val2[axis2] if isinstance(val2, AngleBase) else val2
            if axis3 is not None:
                res[axis3] = val3[axis3] if isinstance(val3, AngleBase) else val3
        return Py_FrozenAngle(**res)

    def __reduce__(self) -> Tuple[Callable[[float, float, float], 'FrozenAngle'], Tuple3]:
        """Pickling support.

        This redirects to a global function, so C/Python versions
        interoperate.
        """
        return _mk_fang, (self._pitch, self._yaw, self._roll)
    
    def thaw(self) -> 'Angle':
        """Return a mutable copy of this angle."""
        ang = Py_Angle.__new__(Py_Angle)
        ang._pitch = self._pitch
        ang._yaw = self._yaw
        ang._roll = self._roll
        return ang

    def copy(self) -> 'FrozenAngle':
        """FrozenAngle is immutable."""
        return self

    def __copy__(self) -> 'FrozenAngle':
        """FrozenAngle is immutable."""
        return self

    def __deepcopy__(self, memodict: Any=None) -> 'FrozenAngle':
        """FrozenAngle is immutable."""
        return self

    def __repr__(self) -> str:
        return f"FrozenAngle({format_float(self._pitch)}, {format_float(self._yaw)}, {format_float(self._roll)})"


@final
class Angle(AngleBase):
    """Represents a pitch-yaw-roll Euler angle.

    >>> Angle(45, 0, z=-90)  # Positional or vec, defaults to 0.
    Angle(45, 0, 270)
    >> Vec(range(0, 270, 90))  # Any 1,2 or 3 long iterable
    Vec(0, 90, 180)
    >>> Vec(1, 2, 3) @ Angle(0, 0, 45)
    Vec(1, -0.707107, 3.53553)
    >>> Angle.from_str('(45 90 0)')  # Parse strings.
    Angle(45, 90, 0)

    Addition and subtraction can be performed 
    between angles, while division/multiplication must be between an angle and scalar (to scale).

    Like vectors, each axis can be accessed by getting/setting ``pitch``/``yaw`` and ``roll`` attributes.
    In addition, the following indexes are allowed (case-insensitive):

    * ``0``, ``1``, ``2``
    * ``"p"``, ``"y"``, ``r``
    * ``"pitch"``, ``"yaw"``, ``"roll"``
    * ``"pit"``, ``"yaw"``, ``"rol"``

    All values are remapped to between ``0-360`` when set.
    """
    __slots__ = ()

    # noinspection PyMissingConstructor
    def __init__(
        self,
        pitch: Union[int, float, Iterable[Union[int, float]]]=0.0,
        yaw: Union[int, float]=0.0,
        roll: Union[int, float]=0.0,
    ) -> None:
        """Create an Angle.

        All values are converted to :external:py:class`float`\\ s automatically.
        If no value is given, that axis will be set to ``0``.
        An iterable can be passed in (as the ``pitch`` argument), which will be
        used for ``pitch``, ``yaw``, and ``roll``. This includes Vectors and other Angles.
        """
        if isinstance(pitch, (int, float)):
            self._pitch = float(pitch) % 360 % 360
            self._yaw = float(yaw) % 360 % 360
            self._roll = float(roll) % 360 % 360
        elif isinstance(pitch, AngleBase):
            # Bypass modulo, iteration and float conversion.
            self._pitch = pitch._pitch
            self._yaw = pitch._yaw
            self._roll = pitch._roll
        else:
            it = iter(pitch)
            self._pitch = float(next(it, 0.0)) % 360 % 360
            self._yaw = float(next(it, yaw)) % 360 % 360
            self._roll = float(next(it, roll)) % 360 % 360

    def copy(self) -> 'Angle':
        """Create a duplicate of this angle."""
        return Py_Angle(self._pitch, self._yaw, self._roll)

    __copy__ = copy

    def __reduce__(self) -> Tuple[Callable[[float, float, float], 'Angle'], Tuple3]:
        """Pickling support.

        This redirects to a global function, so C/Python versions
        interoperate.
        """
        return _mk_ang, (self._pitch, self._yaw, self._roll)
    
    def freeze(self) -> FrozenAngle:
        """Return an immutable copy of this angle."""
        ang = Py_FrozenAngle.__new__(Py_FrozenAngle)
        ang._pitch = self._pitch
        ang._yaw = self._yaw
        ang._roll = self._roll
        return ang

    @property
    def pitch(self) -> float:
        """The Y-axis rotation, performed second."""
        return self._pitch

    @pitch.setter
    def pitch(self, pitch: float) -> None:
        self._pitch = float(pitch) % 360 % 360

    @property
    def yaw(self) -> float:
        """The Z-axis rotation, performed last."""
        return self._yaw

    @yaw.setter
    def yaw(self, yaw: float) -> None:
        self._yaw = float(yaw) % 360 % 360

    @property
    def roll(self) -> float:
        """The X-axis rotation, performed first."""
        return self._roll

    @roll.setter
    def roll(self, roll: float) -> None:
        self._roll = float(roll) % 360 % 360

    def __repr__(self) -> str:
        return f'Angle({format_float(self._pitch)}, {format_float(self._yaw)}, {format_float(self._roll)})'

    @classmethod
    @overload
    def with_axes(cls, axis1: str, val1: Union[float, AngleBase]) -> 'Angle':
        ...

    @classmethod
    @overload
    def with_axes(
        cls,
        axis1: str, val1: Union[float, AngleBase],
        axis2: str, val2: Union[float, AngleBase],
    ) -> 'Angle':
        ...

    @classmethod
    @overload
    def with_axes(
        cls,
        axis1: str, val1: Union[float, AngleBase],
        axis2: str, val2: Union[float, AngleBase],
        axis3: str, val3: Union[float, AngleBase],
    ) -> 'Angle':
        ...

    @classmethod
    def with_axes(
        cls,
        axis1: str,
        val1: Union[float, AngleBase],
        axis2: Optional[str] = None,
        val2: Union[float, AngleBase] = 0.0,
        axis3: Optional[str] = None,
        val3: Union[float, AngleBase] = 0.0,
    ) -> 'Angle':
        """Create an Angle, given a number of axes and corresponding values.

        This is a convenience for doing the following::

            ang = Angle()
            ang[axis1] = val1
            ang[axis2] = val2
            ang[axis3] = val3

        The magnitudes can also be Angles, in which case the matching
        axis will be used from the angle.
        """
        ang = cls()
        ang[axis1] = val1[axis1] if isinstance(val1, AngleBase) else val1
        if axis2 is not None:
            ang[axis2] = val2[axis2] if isinstance(val2, AngleBase) else val2
            if axis3 is not None:
                ang[axis3] = val3[axis3] if isinstance(val3, AngleBase) else val3
        return ang

    def __setitem__(self, ind: Union[str, int], val: float) -> None:
        """Allow editing values by index instead of name if desired.

        This accepts the following indexes to edit values:
        - ``0``, ``1``, ``2``
        - ``"pitch"``, ``"yaw"``, ``"roll"``
        - ``"pit"``, ``"yaw"``, ``"rol"``
        - ``"p"``, ``"y"``, ``"r"``
        Useful in conjunction with a loop to apply commands to all values.
        """
        if ind in (0, 'p', 'pit', 'pitch'):
            self._pitch = float(val) % 360.0 % 360.0
        elif ind in (1, 'y', 'yaw'):
            self._yaw = float(val) % 360.0 % 360.0
        elif ind in (2, 'r', 'rol', 'roll'):
            self._roll = float(val) % 360.0 % 360.0
        else:
            raise KeyError('Invalid axis: {!r}'.format(ind))

    def __imul__(self, other: Union[int, float]) -> 'Angle':
        """Angle *= float multiplies each value."""
        if isinstance(other, (int, float)):
            self._pitch = self._pitch * other % 360.0 % 360.0
            self._yaw = self._yaw * other % 360.0 % 360.0
            self._roll = self._roll * other % 360.0 % 360.0
            return self
        return NotImplemented

    # noinspection PyProtectedMember
    def __imatmul__(self, other: Union[AngleBase, MatrixBase]) -> 'Angle':
        """Angle @ Angle or Angle @ Matrix rotates the first by the second."""
        if isinstance(other, AngleBase):
            mat = Py_Matrix.from_angle(self)
            mat @= other
            return mat._to_angle(self)  # Inplace
        elif isinstance(other, Py_Matrix):
            mat = Py_Matrix.from_angle(self)
            mat._mat_mul(other)
            return mat._to_angle(self)
        else:
            return NotImplemented

    @contextlib.contextmanager
    def transform(self) -> Iterator[Matrix]:
        """Perform transformations on this angle.

        Used as a context manager, which returns a matrix.
        When the body is exited safely, the matrix is applied to
        the angle.
        """
        mat = Py_Matrix.from_angle(self)
        yield mat
        new_ang = mat.to_angle()
        self._pitch = new_ang._pitch
        self._yaw = new_ang._yaw
        self._roll = new_ang._roll


def quickhull(vertexes: Iterable[Vec]) -> List[Tuple[Vec, Vec, Vec]]:
    """Use the quickhull algorithm to construct a convex hull around the provided points.

    This is only available when the C extension is compiled.
    """
    raise NotImplementedError('Requires C extension!')


def _mk_vec(x: float, y: float, z: float) -> Vec:
    """Unpickle a Vec object, maintaining compatibility with C versions.

    Shortened name shrinks the data size.
    """
    # Skip __init__'s checks and coercion/iteration.
    v = Vec.__new__(Vec)
    v.x = x
    v.y = y
    v.z = z
    return v


def _mk_fvec(x: float, y: float, z: float) -> FrozenVec:
    """Unpickle a FrozenVec object, maintaining compatibility with C versions.

    Shortened name shrinks the data size.
    """
    # We can't skip, C-frozen is truly immutable.
    return FrozenVec(x, y, z)


def _mk_fang(pitch: float, yaw: float, roll: float) -> FrozenAngle:
    """Unpickle a FrozenAngle object, maintaining compatibility with C versions.

    Shortened name shrinks the data size.
    """
    # We can't skip, C-frozen is truly immutable.
    return FrozenAngle(pitch, yaw, roll)


def _mk_ang(pitch: float, yaw: float, roll: float) -> Angle:
    """Unpickle an Angle object, maintaining compatibility with C versions.

    Shortened name shrinks the data size.
    """
    # Skip __init__'s checks and coercion/iteration.
    ang = Angle.__new__(Angle)
    ang.pitch = pitch
    ang.yaw = yaw
    ang.roll = roll
    return ang


def _mk_mat(
    aa: float, ab: float, ac: float,
    ba: float, bb: float, bc: float,
    ca: float, cb: float, cc: float,
) -> Matrix:
    """Unpickle a Matrix object, maintaining compatibility with C versions.

    Shortened name shrinks the data size.
    """
    # Skip __init__'s checks and coercion/iteration.
    mat = Matrix.__new__(Matrix)
    mat[0, 0] = aa
    mat[0, 1] = ab
    mat[0, 2] = ac

    mat[1, 0] = ba
    mat[1, 1] = bb
    mat[1, 2] = bc

    mat[2, 0] = ca
    mat[2, 1] = cb
    mat[2, 2] = cc
    return mat


def _mk_fmat(
    aa: float, ab: float, ac: float,
    ba: float, bb: float, bc: float,
    ca: float, cb: float, cc: float,
) -> FrozenMatrix:
    """Unpickle a FrozenMatrix object, maintaining compatibility with C versions.

    Shortened name shrinks the data size.
    """
    # Need a backdoor to construct from raw values.
    # noinspection PyProtectedMember
    return FrozenMatrix._from_raw(aa, ab, ac, ba, bb, bc, ca, cb, cc)


# Older name, keep alias for pickle compatibility
_mk = _mk_vec

# A little dance to preserve both the Cython and Python versions,
# and choose an appropriate unprefixed version. Static analysis then
# also assumes all three are the Python version.

Cy_Vec = Py_Vec = Vec
Cy_FrozenVec = Py_FrozenVec = FrozenVec
Cy_parse_vec_str = Py_parse_vec_str = parse_vec_str
Cy_to_matrix = Py_to_matrix = to_matrix
Cy_lerp = Py_lerp = lerp
Cy_Angle = Py_Angle = Angle
Cy_FrozenAngle = Py_FrozenAngle = FrozenAngle
Cy_Matrix = Py_Matrix = Matrix
Cy_FrozenMatrix = Py_FrozenMatrix = FrozenMatrix

# Do it this way, so static analysis ignores this.
if not TYPE_CHECKING:
    _glob = globals()
    try:
        from . import _math  # noqa
    except ImportError:
        pass
    else:
        for _name in [
            'Vec', 'FrozenVec',
            'Angle', 'FrozenAngle',
            'Matrix', 'FrozenMatrix',
            'parse_vec_str', 'to_matrix', 'lerp',
        ]:
            _glob[_name] = _glob['Cy_' + _name] = getattr(_math, _name)
        del _glob, _name, _math
