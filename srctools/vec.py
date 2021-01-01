"""A 3D vector class which matches Valve conventions.

    >>> Vec(1, 2, 3)
    Vec(1, 2, 3)
    >>> Vec(1, 2, 3) * 2
    Vec(2, 4, 6)
    >>> Vec.from_str('<4 2 -45>')
    Vec(4, 2, -45)

Vectors support arithmetic with scalars, applying the operation to the three
components.
Call Vec.as_tuple() to get a tuple-version of the vector, useful as a
dictionary key. Vec will treat 3-tuples as equivalent to itself, converting it
when used in math operations and comparing values.

Index via .x, .y, .z attributes, or 'x', 'y', 'z', 0, 1, 2 index access.

Rotations are represented by Euler angles, but modifications need to be
performed using matrices.

Rotations are implemented as a matrix-multiplication, where the left is rotated
by the right. Vectors can be rotated by matrices and angles and matrices
can be rotated by angles, but not vice-versa.

Scales magnitude:
 - Vec * Scalar
 - Scalar * Vec
 - Angle * Scalar
 - Scalar * Angle

Rotates LHS by RHS:
 - Vec @ Angle
 - Vec @ Matrix
 - Angle @ Angle
 - Angle @ Matrix
 - Matrix @ Matrix
"""
import math
import contextlib
import warnings

from typing import (
    Union, Tuple, overload, Type,
    Dict, NamedTuple,
    Iterator, Iterable, ContextManager,
)


__all__ = [
    'parse_vec_str', 'to_matrix',
    'Vec', 'Vec_tuple',
    'Angle', 'Matrix',
]

# Type aliases
Tuple3 = Tuple[float, float, float]
AnyVec = Union['Vec', 'Vec_tuple', Tuple3]


def parse_vec_str(val: Union[str, 'Vec', 'Angle'], x=0.0, y=0.0, z=0.0) -> Tuple3:
    """Convert a string in the form '(4 6 -4)' into a set of floats.

    If the string is unparsable, this uses the defaults (x,y,z).
    The string can start with any of the (), {}, [], <> bracket
    types.

     If the 'string' is actually a Vec, the values will be returned.
     """
    if isinstance(val, str):
        pass  # Fast path to skip the below code.
    elif isinstance(val, Py_Vec):
        return val.x, val.y, val.z
    elif isinstance(val, Py_Angle):
        return val.pitch, val.yaw, val.roll
    else:
        # Not a string.
        return x, y, z

    try:
        str_x, str_y, str_z = val.split()
    except ValueError:
        return x, y, z

    if str_x[0] in '({[<':
        str_x = str_x[1:]
    if str_z[-1] in ')}]>':
        str_z = str_z[:-1]
    try:
        return (
            float(str_x),
            float(str_y),
            float(str_z),
        )
    except ValueError:
        return x, y, z


def to_matrix(value: Union['Angle', 'Matrix', 'Vec', Tuple3, None]) -> 'Matrix':
    """Convert various values to a rotation matrix.

    Vectors will be treated as angles, and None as the identity.
    """
    if value is None:
        return Py_Matrix()
    elif isinstance(value, Matrix):
        return value
    elif isinstance(value, Angle):
        return Matrix.from_angle(value)
    else:
        [p, y, r] = value
        return Matrix.from_angle(Angle(p, y, r))


class Vec_tuple(NamedTuple):
    """An immutable tuple, useful for dictionary keys."""
    x: float
    y: float
    z: float


# Use template code to reduce duplication in the various magic number methods.

_VEC_ADDSUB_TEMP = '''
def __{func}__(self, other: Union['Vec', tuple, float]):
    """{op} operation.

    This additionally works on scalars (adds to all axes).
    """
    if isinstance(other, Py_Vec):
        return Py_Vec(
            self.x {op} other.x,
            self.y {op} other.y,
            self.z {op} other.z,
        )
    try:
        if isinstance(other, tuple):
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
        return Py_Vec(x, y, z)

def __r{func}__(self, other: Union['Vec', tuple, float]):
    """{op} operation with reversed operands.

    This additionally works on scalars (adds to all axes).
    """
    if isinstance(other, Py_Vec):
        return Py_Vec(
            other.x {op} self.x,
            other.y {op} self.y,
            other.z {op} self.z,
        )
    try:
        if isinstance(other, tuple):
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
        return Py_Vec(x, y, z)

def __i{func}__(self, other: Union['Vec', tuple, float]):
    """{op}= operation.

    Like the normal one except without duplication.
    """
    if isinstance(other, Py_Vec):
        self.x {op}= other.x
        self.y {op}= other.y
        self.z {op}= other.z
    elif isinstance(other, tuple):
        self.x {op}= other[0]
        self.y {op}= other[1]
        self.z {op}= other[2]
    elif isinstance(other, (int, float)):
        orig = self.x, self.y, self.z
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
    """Vector {op} scalar operation."""
    if isinstance(other, Py_Vec):
        raise TypeError("Cannot {pretty} 2 Vectors.")
    else:
        try:
            return Py_Vec(
                self.x {op} other,
                self.y {op} other,
                self.z {op} other,
            )
        except TypeError:
            return NotImplemented

def __r{func}__(self, other: float):
    """scalar {op} Vector operation."""
    if isinstance(other, Py_Vec):
        raise TypeError("Cannot {pretty} 2 Vectors.")
    else:
        try:
            return Py_Vec(
                other {op} self.x,
                other {op} self.y,
                other {op} self.z,
            )
        except TypeError:
            return NotImplemented


def __i{func}__(self, other: float):
    """{op}= operation.

    Like the normal one except without duplication.
    """
    if isinstance(other, Py_Vec):
        raise TypeError("Cannot {pretty} 2 Vectors.")
    else:
        self.x {op}= other
        self.y {op}= other
        self.z {op}= other
        return self
'''


class Vec:
    """A 3D Vector. This has most standard Vector functions.

    Many of the functions will accept a 3-tuple for comparison purposes.
    """
    __slots__ = ('x', 'y', 'z')
    # Make type checkers understand that you can't do str->str or tuple->tuple.
    INV_AXIS = {
        'x': ('y', 'z'),
        'y': ('x', 'z'),
        'z': ('x', 'y'),

        ('y', 'z'): 'x',
        ('x', 'z'): 'y',
        ('x', 'y'): 'z',

        ('z', 'y'): 'x',
        ('z', 'x'): 'y',
        ('y', 'x'): 'z',
    }  # type: Union[Dict[str, Tuple[str, str]], Dict[Tuple[str, str], str]]

    # Vectors pointing in all cardinal directions
    N = north = y_pos = Vec_tuple(0, 1, 0)
    S = south = y_neg = Vec_tuple(0, -1, 0)
    E = east = x_pos = Vec_tuple(1, 0, 0)
    W = west = x_neg = Vec_tuple(-1, 0, 0)
    T = top = z_pos = Vec_tuple(0, 0, 1)
    B = bottom = z_neg = Vec_tuple(0, 0, -1)

    def __init__(
        self,
        x: Union[int, float, 'Vec', Iterable[float]]=0.0,
        y: float=0.0,
        z: float=0.0,
    ) -> None:
        """Create a Vector.

        All values are converted to Floats automatically.
        If no value is given, that axis will be set to 0.
        An iterable can be passed in (as the x argument), which will be
        used for x, y, and z.
        """
        if isinstance(x, (int, float)):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
        elif isinstance(x, Py_Vec):
            self.x = x.x
            self.y = x.y
            self.z = x.z
        else:
            it = iter(x)
            self.x = float(next(it, 0.0))
            self.y = float(next(it, y))
            self.z = float(next(it, z))

    def copy(self) -> 'Vec':
        """Create a duplicate of this vector."""
        return Py_Vec(self.x, self.y, self.z)

    __copy__ = copy  # copy module support.

    def __reduce__(self) -> tuple:
        """Pickling support.

        This redirects to a global function, so C/Python versions
        interoperate.
        """
        return _mk, (self.x, self.y, self.z)

    @classmethod
    def from_str(cls, val: Union[str, 'Vec'], x: float=0.0, y: float=0.0, z: float=0.0) -> 'Vec':
        """Convert a string in the form '(4 6 -4)' into a Vector.

         If the string is unparsable, this uses the defaults (x,y,z).
         The string can start with any of the (), {}, [], <> bracket
         types, or none.

         If the value is already a vector, a copy will be returned.
         """

        x, y, z = Py_parse_vec_str(val, x, y, z)
        return cls(x, y, z)

    @classmethod
    @overload
    def with_axes(cls, axis1: str, val1: Union[float, 'Vec']) -> 'Vec': ...

    @classmethod
    @overload
    def with_axes(
        cls,
        axis1: str, val1: Union[float, 'Vec'],
        axis2: str, val2: Union[float, 'Vec'],
    ) -> 'Vec': ...

    @classmethod
    @overload
    def with_axes(
        cls,
        axis1: str, val1: Union[float, 'Vec'],
        axis2: str, val2: Union[float, 'Vec'],
        axis3: str, val3: Union[float, 'Vec'],
    ) -> 'Vec': ...

    @classmethod
    def with_axes(
        cls,
        axis1: str,
        val1: Union[float, 'Vec'],
        axis2: str=None,
        val2: Union[float, 'Vec']=0.0,
        axis3: str=None,
        val3: Union[float, 'Vec']=0.0,
    ) -> 'Vec':
        """Create a Vector, given a number of axes and corresponding values.

        This is a convenience for doing the following:
            vec = Vec()
            vec[axis1] = val1
            vec[axis2] = val2
            vec[axis3] = val3
        The magnitudes can also be Vectors, in which case the matching
        axis will be used from the vector.
        """
        vec = cls()
        vec[axis1] = val1[axis1] if isinstance(val1, Py_Vec) else val1
        if axis2 is not None:
            vec[axis2] = val2[axis2] if isinstance(val2, Py_Vec) else val2
            if axis3 is not None:
                vec[axis3] = val3[axis3] if isinstance(val3, Py_Vec) else val3
        return vec

    def rotate(
        self,
        pitch: float=0.0,
        yaw: float=0.0,
        roll: float=0.0,
        round_vals: bool=True,
    ) -> 'Vec':
        """Rotate a vector by a Source rotational angle.
        Returns the vector, so you can use it in the form
        val = Vec(0,1,0).rotate(p, y, r)

        If round is True, all values will be rounded to 6 decimals
        (since these calculations always have small inprecision.)
        """
        warnings.warn("Use vec @ Angle() instead.", DeprecationWarning)
        self @= Py_Angle(pitch, yaw, roll)
        if round_vals:
            self.x = round(self.x, 6)
            self.y = round(self.y, 6)
            self.z = round(self.z, 6)
        return self

    def rotate_by_str(self, ang: str, pitch=0.0, yaw=0.0, roll=0.0, round_vals=True) -> 'Vec':
        """Rotate a vector, using a string instead of a vector.

        If the string cannot be parsed, use the passed in values instead.
        """
        warnings.warn("Use vec @ Angle.from_str() instead.", DeprecationWarning)
        self @= Py_Angle.from_str(ang, pitch, yaw, roll)
        if round_vals:
            self.x = round(self.x, 6)
            self.y = round(self.y, 6)
            self.z = round(self.z, 6)
        return self

    @staticmethod
    @overload
    def bbox(_point: Iterable['Vec']) -> Tuple['Vec', 'Vec']: ...
    @staticmethod
    @overload
    def bbox(*points: 'Vec') -> Tuple['Vec', 'Vec']: ...

    @staticmethod
    def bbox(*points: Union[Iterable['Vec'], 'Vec']) -> Tuple['Vec', 'Vec']:
        """Compute the bounding box for a set of points.

        Pass either several Vecs, or an iterable of Vecs.
        Returns a (min, max) tuple.
        """
        # Allow passing a single iterable, but also handle a single Vec.
        # The error messages match those produced by min()/max().
        if len(points) == 1 and not isinstance(points[0], Py_Vec):
            try:
                [[first, *point_coll]] = points
            except ValueError:
                raise ValueError('Vec.bbox() arg is an empty sequence') from None
        else:
            try:
                first, *point_coll = points
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
        return bbox_min, bbox_max

    @classmethod
    def iter_grid(
        cls,
        min_pos: 'Vec',
        max_pos: 'Vec',
        stride: int=1,
    ) -> Iterator['Vec']:
        """Loop over points in a bounding box. All coordinates should be integers.

        Both borders will be included.
        """
        min_x = int(min_pos.x)
        min_y = int(min_pos.y)
        min_z = int(min_pos.z)

        max_x = int(max_pos.x)
        max_y = int(max_pos.y)
        max_z = int(max_pos.z)

        for x in range(min_x, max_x + 1, stride):
            for y in range(min_y, max_y + 1, stride):
                for z in range(min_z, max_z + 1, stride):
                    yield cls(x, y, z)

    def iter_line(self, end: 'Vec', stride: int=1) -> Iterator['Vec']:
        """Yield points between this point and 'end' (including both endpoints).

        Stride specifies the distance between each point.
        If the distance is less than the stride, only end-points will be yielded.
        If they are the same, that point will be yielded.
        """
        offset = end - self
        length = offset.mag()
        if length < stride:
            # Not enough room, yield both
            yield self.copy()
            if self != end:
                yield end.copy()
            return

        direction = offset.norm()
        for pos in range(0, int(length), int(stride)):
            yield self + direction * pos
        yield end.copy()  # Directly yield - ensures no rounding errors.

    def axis(self) -> str:
        """For a normal vector, return the axis it is on."""
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

        A +x axis vector will result in a 0, 0, 0 angle. The roll is not
        affected by the direction of the normal.

        The inverse of this is `Vec(x=1) @ Angle(pitch, yaw, roll)`.
        """
        # Pitch is applied first, so we need to reconstruct the x-value
        horiz_dist = math.hypot(self.x, self.y)
        return Py_Angle(
            math.degrees(math.atan2(-self.z, horiz_dist)),
            math.degrees(math.atan2(self.y, self.x)) % 360,
            roll,
        )

    def to_angle_roll(self, z_norm: 'Vec', stride: int=...) -> 'Angle':
        """Produce a Source Engine angle with roll.

        The z_normal should point in +z, and must be at right angles to this
        vector.
        This is deprecated, use Matrix.from_basis().to_angle().
        Stride is no longer used.
        """
        warnings.warn('Use Matrix.from_basis().to_angle()', DeprecationWarning)
        return Py_Matrix.from_basis(x=self, z=z_norm).to_angle()

    def rotation_around(self, rot: float=90) -> 'Vec':
        """For an axis-aligned normal, return the angles which rotate around it."""
        warnings.warn('Use Matrix.axis_angle().to_angle()', DeprecationWarning)
        if self.x and not self.y and not self.z:
            return Py_Vec(z=math.copysign(rot, self.x))
        elif self.y and not self.x and not self.z:
            return Py_Vec(x=math.copysign(rot, self.y))
        elif self.z and not self.x and not self.y:
            return Py_Vec(y=math.copysign(rot, self.z))
        else:
            raise ValueError('Zero vector!')

    def __abs__(self) -> 'Vec':
        """Performing abs() on a Vec takes the absolute value of all axes."""
        return Py_Vec(
            abs(self.x),
            abs(self.y),
            abs(self.z),
        )

    # The numeric magic methods are defined via exec(), so we need stubs
    # to annotate them in a way a type-checker can understand.
    # These are immediately overwritten.

    def __add__(self, other: Union['Vec', Tuple3, int, float]) -> 'Vec': pass
    def __radd__(self, other: Union['Vec', Tuple3, int, float]) -> 'Vec': pass
    def __iadd__(self, other: Union['Vec', Tuple3, int, float]) -> 'Vec': pass

    def __sub__(self, other: Union['Vec', Tuple3, int, float]) -> 'Vec': pass
    def __rsub__(self, other: Union['Vec', Tuple3, int, float]) -> 'Vec': pass
    def __isub__(self, other: Union['Vec', Tuple3, int, float]) -> 'Vec': pass

    def __mul__(self, other: float) -> 'Vec': pass
    def __rmul__(self, other: float) -> 'Vec': pass
    def __imul__(self, other: float) -> 'Vec': pass

    def __truediv__(self, other: float) -> 'Vec': pass
    def __rtruediv__(self, other: float) -> 'Vec': pass
    def __itruediv__(self, other: float) -> 'Vec': pass

    def __floordiv__(self, other: float) -> 'Vec': pass
    def __rfloordiv__(self, other: float) -> 'Vec': pass
    def __ifloordiv__(self, other: float) -> 'Vec': pass

    def __mod__(self, other: float) -> 'Vec': pass
    def __rmod__(self, other: float) -> 'Vec': pass
    def __imod__(self, other: float) -> 'Vec': pass

    funcname = op = pretty = None

    # Use exec() to generate all the number magic methods. This reduces code
    # duplication since they're all very similar.

    for funcname, op in (('add', '+'), ('sub', '-')):
        exec(
            _VEC_ADDSUB_TEMP.format(func=funcname, op=op),
            globals(),
            locals(),
        )

    for funcname, op, pretty in (
            ('mul', '*', 'multiply'),
            ('truediv', '/', 'divide'),
            ('floordiv', '//', 'floor-divide'),
            ('mod', '%', 'modulus'),
    ):
        exec(
            _VEC_MULDIV_TEMP.format(func=funcname, op=op, pretty=pretty),
            globals(),
            locals(),
        )

    del funcname, op, pretty

    # Divmod is entirely unique.
    def __divmod__(self, other: float) -> Tuple['Vec', 'Vec']:
        """Divide the vector by a scalar, returning the result and remainder."""
        if isinstance(other, Py_Vec):
            raise TypeError("Cannot divide 2 Vectors.")
        else:
            try:
                x1, x2 = divmod(self.x, other)
                y1, y2 = divmod(self.y, other)
                z1, z2 = divmod(self.z, other)
            except TypeError:
                return NotImplemented
            else:
                return Py_Vec(x1, y1, z1), Py_Vec(x2, y2, z2)

    def __rdivmod__(self, other: float) -> Tuple['Vec', 'Vec']:
        """Divide a scalar by a vector, returning the result and remainder."""
        try:
            x1, x2 = divmod(other, self.x)
            y1, y2 = divmod(other, self.y)
            z1, z2 = divmod(other, self.z)
        except (TypeError, ValueError):
            return NotImplemented
        else:
            return Py_Vec(x1, y1, z1), Py_Vec(x2, y2, z2)

    def __imatmul__(self, other: Union['Angle', 'Matrix']) -> 'Vec':
        """We need to define this, so it's in-place."""
        if isinstance(other, Py_Matrix):
            mat = other
        elif isinstance(other, Py_Angle):
            mat = Py_Matrix.from_angle(other)
        else:
            return NotImplemented
        mat._vec_rot(self)
        return self

    def __bool__(self) -> bool:
        """Vectors are True if any axis is non-zero."""
        return self.x != 0 or self.y != 0 or self.z != 0

    def __eq__(self, other: object) -> bool:
        """== test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        A tolerance of 1e-6 is accounted for automatically.
        """
        if isinstance(other, Py_Vec):
            return (
                abs(other.x - self.x) < 1e-6 and
                abs(other.y - self.y) < 1e-6 and
                abs(other.z - self.z) < 1e-6
            )
        elif isinstance(other, tuple):
            return (
                abs(self.x - other[0]) < 1e-6 and
                abs(self.y - other[1]) < 1e-6 and
                abs(self.z - other[2]) < 1e-6
            )
        else:
            return NotImplemented

    def __ne__(self, other: object) -> bool:
        """!= test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        A tolerance of 1e-6 is accounted for automatically.
        """
        if isinstance(other, Py_Vec):
            return (
                abs(other.x - self.x) >= 1e-6 or
                abs(other.y - self.y) >= 1e-6 or
                abs(other.z - self.z) >= 1e-6
            )
        elif isinstance(other, tuple):
            return (
                abs(self.x - other[0]) >= 1e-6 or
                abs(self.y - other[1]) >= 1e-6 or
                abs(self.z - other[2]) >= 1e-6
            )
        else:
            return NotImplemented

    def __lt__(self, other: AnyVec) -> bool:
        """A<B test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        A tolerance of 1e-6 is accounted for automatically.
        """
        if isinstance(other, Py_Vec):
            return (
                (other.x - self.x) > 1e-6 and
                (other.y - self.y) > 1e-6 and
                (other.z - self.z) > 1e-6
            )
        elif isinstance(other, tuple):
            return (
                (other[0] - self.x) > 1e-6 and
                (other[1] - self.y) > 1e-6 and
                (other[2] - self.z) > 1e-6
            )
        else:
            return NotImplemented

    def __le__(self,other: AnyVec) -> bool:
        """A<=B test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        A tolerance of 1e-6 is accounted for automatically.
        """
        if isinstance(other, Py_Vec):
            return (
                (self.x - other.x) <= 1e-6 and
                (self.y - other.y) <= 1e-6 and
                (self.z - other.z) <= 1e-6
            )
        elif isinstance(other, tuple):
            return (
                (self.x - other[0]) <= 1e-6 and
                (self.y - other[1]) <= 1e-6 and
                (self.z - other[2]) <= 1e-6
            )
        else:
            return NotImplemented

    def __gt__(self,other: AnyVec) -> bool:
        """A>B test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        A tolerance of 1e-6 is accounted for automatically.
        """
        if isinstance(other, Py_Vec):
            return (
                (self.x - other.x) > 1e-6 and
                (self.y - other.y) > 1e-6 and
                (self.z - other.z) > 1e-6
            )
        elif isinstance(other, tuple):
            return (
                (self.x > other[0]) > 1e-6 and
                (self.y > other[1]) > 1e-6 and
                (self.z > other[2]) > 1e-6
            )
        else:
            return NotImplemented

    def __ge__(self, other: AnyVec) -> bool:
        """A>=B test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        A tolerance of 1e-6 is accounted for automatically.
        """
        if isinstance(other, Py_Vec):
            return (
                (other.x - self.x) <= 1e-6 and
                (other.y - self.y) <= 1e-6 and
                (other.z - self.z) <= 1e-6
            )
        elif isinstance(other, tuple):
            return (
                (other[0] - self.x) <= 1e-6 and
                (other[1] - self.y) <= 1e-6 and
                (other[2] - self.z) <= 1e-6
            )
        else:
            return NotImplemented

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

    def __round__(self, n: int=0) -> 'Vec':
        """Performing round() on a Py_Vec rounds each axis."""
        return Py_Vec(
            round(self.x, n),
            round(self.y, n),
            round(self.z, n),
        )

    def mag(self) -> float:
        """Compute the distance from the vector and the origin."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def join(self, delim: str=', ') -> str:
        """Return a string with all numbers joined by the passed delimiter.

        This strips off the .0 if no decimal portion exists.
        """
        # :g strips the .0 off of floats if it's an integer.
        return f'{self.x:g}{delim}{self.y:g}{delim}{self.z:g}'

    def __str__(self) -> str:
        """Return the values, separated by spaces.

        This is the main format in Valve's file formats.
        This strips off the .0 if no decimal portion exists.
        """
        return f"{self.x:g} {self.y:g} {self.z:g}"

    def __repr__(self) -> str:
        """Code required to reproduce this vector."""
        return f"Vec({self.x:g}, {self.y:g}, {self.z:g})"

    def __iter__(self) -> Iterator[float]:
        """Allow iterating through the dimensions."""
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, ind: Union[str, int]) -> float:
        """Allow reading values by index instead of name if desired.

        This accepts either 0,1,2 or 'x','y','z' to read values.
        Useful in conjunction with a loop to apply commands to all values.
        """
        if ind == 0 or ind == "x":
            return self.x
        elif ind == 1 or ind == "y":
            return self.y
        elif ind == 2 or ind == "z":
            return self.z
        raise KeyError(f'Invalid axis: {ind!r}')

    def __setitem__(self, ind: Union[str, int], val: float) -> None:
        """Allow editing values by index instead of name if desired.

        This accepts either 0,1,2 or 'x','y','z' to edit values.
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

    def other_axes(self, axis: str) -> Tuple[float, float]:
        """Get the values for the other two axes."""
        if axis == 'x':
            return self.y, self.z
        if axis == 'y':
            return self.x, self.z
        if axis == 'z':
            return self.x, self.y
        raise KeyError('Bad axis "{}"'.format(axis))

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return the Vector as a tuple."""
        return Vec_tuple(round(self.x, 6), round(self.y, 6), round(self.z, 6))

    def len_sq(self) -> float:
        """Return the magnitude squared, which is slightly faster."""
        return self.x**2 + self.y**2 + self.z**2

    def __len__(self) -> int:
        """The len() of a vector is the number of non-zero axes."""
        return (
            (abs(self.x) > 1e-6) +
            (abs(self.y) > 1e-6) +
            (abs(self.z) > 1e-6)
        )

    def __contains__(self, val: float) -> bool:
        """Check to see if an axis is set to the given value.
        """
        return abs(val - self.x) < 1e-6 or abs(val - self.y) < 1e-6 or abs(val - self.z) < 1e-6

    def __neg__(self) -> 'Vec':
        """The inverted form of a Vector has inverted axes."""
        return Py_Vec(-self.x, -self.y, -self.z)

    def __pos__(self) -> 'Vec':
        """+ on a Vector simply copies it."""
        return Py_Vec(self.x, self.y, self.z)

    def norm(self) -> 'Vec':
        """Normalise the Vector.

         This is done by transforming it to have a magnitude of 1 but the same
         direction.
         The vector is left unchanged if it is equal to (0,0,0)
         """
        if self.x == 0 and self.y == 0 and self.z == 0:
            # Don't do anything for this - otherwise we'd get division
            # by zero errors - we want this to be a valid normal!
            return self.copy()
        else:
            # Adding 0 clears -0 values - we don't want those.
            val = self / self.mag()
            val += 0
            return val

    def dot(self, other: AnyVec) -> float:
        """Return the dot product of both Vectors."""
        return (
            self.x * other[0] +
            self.y * other[1] +
            self.z * other[2]
        )

    def cross(self, other: AnyVec) -> 'Vec':
        """Return the cross product of both Vectors."""
        return Py_Vec(
            self.y * other[2] - self.z * other[1],
            self.z * other[0] - self.x * other[2],
            self.x * other[1] - self.y * other[0],
        )

    def localise(
        self,
        origin: Union['Vec', Tuple3],
        angles: Union['Angle', 'Matrix']=None,
    ) -> None:
        """Shift this point to be local to the given position and angles.

        This effectively translates local-space offsets to a global location,
        given the parent's origin and angles.
        """
        mat = to_matrix(angles)
        mat._vec_rot(self)
        self += origin

    def norm_mask(self, normal: 'Vec') -> 'Vec':
        """Subtract the components of this vector not in the direction of the normal.

        If the normal is axis-aligned, this will zero out the other axes.
        If not axis-aligned, it will do the equivalent.
        """
        norm = normal.norm()
        return norm * self.dot(norm)

    len = mag
    mag_sq = len_sq

    @contextlib.contextmanager
    def transform(self) -> ContextManager['Matrix']:
        """Perform rotations on this Vector efficiently.

        Used as a context manager, which returns a matrix.
        When the body is exited safely, the matrix is applied to
        the angle.
        """
        mat = Py_Matrix()
        yield mat
        mat._vec_rot(self)

_IND_TO_SLOT = {
    (x, y): f'_{chr(ord("a")+x)}{chr(ord("a")+y)}'
    for x in (0, 1, 2)
    for y in (0, 1, 2)
}


class Matrix:
    """Represents a matrix via a transformation matrix."""
    __slots__ = [
        '_aa', '_ab', '_ac',
        '_ba', '_bb', '_bc',
        '_ca', '_cb', '_cc'
    ]

    def __init__(self) -> None:
        """Create a matrix set to the identity transform."""
        self._aa, self._ab, self._ac = 1.0, 0.0, 0.0
        self._ba, self._bb, self._bc = 0.0, 1.0, 0.0
        self._ca, self._cb, self._cc = 0.0, 0.0, 1.0

    def __eq__(self, other: object) -> object:
        if isinstance(other, Py_Matrix):
            return (
                self._aa == other._aa and self._ab == other._ab and self._ac == other._ac and
                self._ba == other._ba and self._bb == other._bb and self._bc == other._bc and
                self._ca == other._ca and self._cb == other._cb and self._cc == other._cc
            )
        return NotImplemented
        
    def __repr__(self) -> str:
        return (
            '<Matrix '
            f'{self._aa:.3} {self._ab:.3} {self._ac:.3}, '
            f'{self._ba:.3} {self._bb:.3} {self._bc:.3}, '
            f'{self._ca:.3} {self._cb:.3} {self._cc:.3}'
            '>'
        )

    def copy(self) -> 'Matrix':
        """Duplicate this matrix."""
        rot = Py_Matrix.__new__(Py_Matrix)

        rot._aa, rot._ab, rot._ac = self._aa, self._ab, self._ac
        rot._ba, rot._bb, rot._bc = self._ba, self._bb, self._bc
        rot._ca, rot._cb, rot._cc = self._ca, self._cb, self._cc

        return rot

    @classmethod
    def from_pitch(cls: Type['Matrix'], pitch: float) -> 'Matrix':
        """Return the matrix representing a pitch rotation.

        This is a rotation around the Y axis.
        """
        rad_pitch = math.radians(pitch)
        cos = math.cos(rad_pitch)
        sin = math.sin(rad_pitch)

        rot: Matrix = cls.__new__(cls)

        rot._aa, rot._ab, rot._ac = cos, 0.0, -sin
        rot._ba, rot._bb, rot._bc = 0.0, 1.0, 0.0
        rot._ca, rot._cb, rot._cc = sin, 0.0, cos

        return rot

    @classmethod
    def from_yaw(cls: Type['Matrix'], yaw: float) -> 'Matrix':
        """Return the matrix representing a yaw rotation.

        """
        rad_yaw = math.radians(yaw)
        sin = math.sin(rad_yaw)
        cos = math.cos(rad_yaw)

        rot: Matrix = cls.__new__(cls)

        rot._aa, rot._ab, rot._ac = cos, sin, 0.0
        rot._ba, rot._bb, rot._bc = -sin, cos, 0.0
        rot._ca, rot._cb, rot._cc = 0.0, 0.0, 1.0

        return rot
        
    @classmethod
    def from_roll(cls: Type['Matrix'], roll: float) -> 'Matrix':
        """Return the matrix representing a roll rotation.

        This is a rotation around the X axis.
        """
        rad_roll = math.radians(roll)
        cos_r = math.cos(rad_roll)
        sin_r = math.sin(rad_roll)

        rot: Matrix = cls.__new__(cls)

        rot._aa, rot._ab, rot._ac = 1.0, 0.0, 0.0
        rot._ba, rot._bb, rot._bc = 0.0, cos_r, sin_r
        rot._ca, rot._cb, rot._cc = 0.0, -sin_r, cos_r

        return rot
        
    @classmethod
    def from_angle(cls, angle: 'Angle') -> 'Matrix':
        """Return the rotation representing an Euler angle."""
        rad_pitch = math.radians(angle.pitch)
        cos_p = math.cos(rad_pitch)
        sin_p = math.sin(rad_pitch)
        rad_yaw = math.radians(angle.yaw)
        sin_y = math.sin(rad_yaw)
        cos_y = math.cos(rad_yaw)
        rad_roll = math.radians(angle.roll)
        cos_r = math.cos(rad_roll)
        sin_r = math.sin(rad_roll)

        rot: Matrix = cls.__new__(cls)

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
    def axis_angle(cls, axis: Union[Vec, Tuple3], angle: float) -> 'Matrix':
        """Compute the rotation matrix forming a rotation around an axis by a specific angle."""
        x, y, z = Vec(axis).norm()
        # Invert, so it matches the orientation of Angles().
        angle_rad = -math.radians(angle)
        cos = math.cos(angle_rad)
        icos = 1 - cos
        sin = math.sin(angle_rad)

        mat: Matrix = cls.__new__(cls)

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

    def forward(self) -> 'Vec':
        """Return a normalised vector pointing in the +X direction."""
        return Py_Vec(self._aa, self._ab, self._ac)

    def left(self) -> 'Vec':
        """Return a normalised vector pointing in the +Y direction."""
        return Py_Vec(self._ba, self._bb, self._bc)

    def up(self) -> 'Vec':
        """Return a normalised vector pointing in the +Z direction."""
        return Py_Vec(self._ca, self._cb, self._cc)

    def __getitem__(self, item: Tuple[int, int]) -> float:
        """Retrieve an individual matrix value by x, y position (0-2)."""
        return getattr(self, _IND_TO_SLOT[item])

    def __setitem__(self, item: Tuple[int, int], value: float) -> None:
        """Set an individual matrix value by x, y position (0-2)."""
        setattr(self, _IND_TO_SLOT[item], value)

    def to_angle(self) -> 'Angle':
        """Return an Euler angle replicating this rotation."""

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
            return Py_Angle(
                yaw=math.degrees(math.atan2(for_y, for_x)),
                pitch=math.degrees(math.atan2(-for_z, horiz_dist)),
                roll=math.degrees(math.atan2(left_z, up_z)),
            )
        else:
            # Vertical, gimbal lock (yaw=roll)...
            return Py_Angle(
                yaw=math.degrees(math.atan2(-left_x, left_y)),
                pitch=math.degrees(math.atan2(-for_z, horiz_dist)),
                roll=0,  # Can't produce.
            )

    def transpose(self) -> 'Matrix':
        """Return the transpose of this matrix."""
        rot = Py_Matrix.__new__(Py_Matrix)

        rot._aa, rot._ab, rot._ac = self._aa, self._ba, self._ca
        rot._ba, rot._bb, rot._bc = self._ab, self._bb, self._cb
        rot._ca, rot._cb, rot._cc = self._ac, self._bc, self._cc

        return rot

    @classmethod
    @overload
    def from_basis(cls, *, x: Vec, y: Vec, z: Vec) -> 'Matrix': ...
    @classmethod
    @overload
    def from_basis(cls, *, x: Vec, y: Vec) -> 'Matrix': ...
    @classmethod
    @overload
    def from_basis(cls, *, y: Vec, z: Vec) -> 'Matrix': ...
    @classmethod
    @overload
    def from_basis(cls, *, x: Vec, z: Vec) -> 'Matrix': ...
    @classmethod
    def from_basis(
        cls, *,
        x: Vec=None,
        y: Vec=None,
        z: Vec=None,
    ) -> 'Matrix':
        """Construct a matrix from at least two basis vectors.

        The third is computed, if not provided.
        """
        if x is None and y is not None and z is not None:
            x = Vec.cross(y, z)
        elif y is None and x is not None and z is not None:
            y = Vec.cross(z, x)
        elif z is None and x is not None and y is not None:
            z = Vec.cross(x, y)
        elif x is None and y is None and z is None:
            raise TypeError('At least two vectors must be provided!')
        mat: Matrix = cls.__new__(cls)
        mat._aa, mat._ab, mat._ac = x.norm()
        mat._ba, mat._bb, mat._bc = y.norm()
        mat._ca, mat._cb, mat._cc = z.norm()
        return mat

    @overload
    def __matmul__(self, other: 'Matrix') -> 'Matrix': ...
    @overload
    def __matmul__(self, other: 'Angle') -> 'Matrix': ...
    @overload
    def __matmul__(self, other: object) -> NotImplemented: ...

    def __matmul__(self, other: object) -> 'Matrix':
        if isinstance(other, Py_Matrix):
            mat = self.copy()
            mat._mat_mul(other)
            return mat
        elif isinstance(other, Py_Angle):
            mat = self.copy()
            mat._mat_mul(Py_Matrix.from_angle(other))
            return mat
        else:
            return NotImplemented

    @overload
    def __rmatmul__(self, other: Vec) -> Vec: ...
    @overload
    def __rmatmul__(self, other: 'Matrix') -> 'Matrix': ...
    @overload
    def __rmatmul__(self, other: 'Angle') -> 'Angle': ...
    @overload
    def __rmatmul__(self, other: object) -> NotImplemented: ...
    
    def __rmatmul__(self, other):
        if isinstance(other, Py_Vec):
            result = other.copy()
            self._vec_rot(result)
            return result
        elif isinstance(other, Py_Angle):
            mat = Py_Matrix.from_angle(other)
            mat._mat_mul(self)
            return mat.to_angle()
        elif isinstance(other, Py_Matrix):
            mat = other.copy()
            return mat._mat_mul(self)
        else:
            return NotImplemented

    @overload
    def __imatmul__(self, other: 'Matrix') -> 'Matrix': ...
    @overload
    def __imatmul__(self, other: 'Angle') -> 'Matrix': ...
    @overload
    def __imatmul__(self, other: object) -> NotImplemented: ...

    def __imatmul__(self, other):
        if isinstance(other, Py_Matrix):
            self._mat_mul(other)
            return self
        elif isinstance(other, Py_Angle):
            self._mat_mul(Py_Matrix.from_angle(other))
            return self
        else:
            return NotImplemented
            
    def _mat_mul(self, other: 'Matrix') -> None:
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
    
    def _vec_rot(self, vec: Vec) -> None:
        """Rotate a vector by our value."""
        x = vec.x
        y = vec.y
        z = vec.z
        vec.x = (x * self._aa) + (y * self._ba) + (z * self._ca)
        vec.y = (x * self._ab) + (y * self._bb) + (z * self._cb)
        vec.z = (x * self._ac) + (y * self._bc) + (z * self._cc)

    
class Angle:
    """Represents a pitch-yaw-roll Euler angle.

    All values are remapped to between 0-360 when set.
    Addition and subtraction modify values, matrix-multiplication with 
    Vec, Angle or Matrix rotates (RHS rotating LHS).
    """
    # We have to double-modulus because -1e-14 % 360.0 = 360.0.
    __slots__ = ['_pitch', '_yaw', '_roll']
    
    def __init__(
        self,
        pitch: Union[int, float, Iterable[Union[int, float]]]=0.0,
        yaw: Union[int, float]=0.0,
        roll: Union[int, float]=0.0,
    ) -> None:
        """Create an Angle.

        All values are converted to Floats automatically.
        If no value is given, that axis will be set to 0.
        An iterable can be passed in (as the pitch argument), which will be
        used for pitch, yaw, and roll. This includes Vectors and other Angles.
        """
        if isinstance(pitch, (int, float)):
            self._pitch = float(pitch) % 360 % 360
            self._yaw = float(yaw) % 360 % 360
            self._roll = float(roll) % 360 % 360
        else:
            it = iter(pitch)
            self._pitch = float(next(it, 0.0)) % 360 % 360
            self._yaw = float(next(it, yaw)) % 360 % 360
            self._roll = float(next(it, roll)) % 360 % 360

    def copy(self) -> 'Angle':
        """Create a duplicate of this vector."""
        return Py_Angle(self._pitch, self._yaw, self._roll)

    @classmethod
    def from_str(cls, val: Union[str, 'Angle'], pitch=0.0, yaw=0.0, roll=0.0):
        """Convert a string in the form '(4 6 -4)' into an Angle.

         If the string is unparsable, this uses the defaults.
         The string can start with any of the (), {}, [], <> bracket
         types, or none.

         If the value is already a Angle, a copy will be returned.
         """

        pitch, yaw, roll = Py_parse_vec_str(val, pitch, yaw, roll)
        return cls(pitch, yaw, roll)

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

    def __str__(self) -> str:
        """Return the values, separated by spaces.

        This is the main format in Valve's file formats, though identical to
        vectors.
        This strips off the .0 if no decimal portion exists.
        """
        return f"{self._pitch:g} {self._yaw:g} {self._roll:g}"

    def __repr__(self) -> str:
        return f'Angle({self._pitch:g}, {self._yaw:g}, {self._roll:g})'

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return the Angle as a tuple."""
        return Vec_tuple(self._pitch, self._yaw, self._roll)

    def __iter__(self) -> Iterator[float]:
        """Iterating over the angles returns each value in turn."""
        yield self._pitch
        yield self._yaw
        yield self._roll

    @classmethod
    @overload
    def with_axes(cls, axis1: str, val1: Union[float, 'Angle']) -> 'Angle':
        ...

    @classmethod
    @overload
    def with_axes(
        cls,
        axis1: str, val1: Union[float, 'Angle'],
        axis2: str, val2: Union[float, 'Angle'],
    ) -> 'Angle':
        ...

    @classmethod
    @overload
    def with_axes(
        cls,
        axis1: str, val1: Union[float, 'Angle'],
        axis2: str, val2: Union[float, 'Angle'],
        axis3: str, val3: Union[float, 'Angle'],
    ) -> 'Angle':
        ...

    @classmethod
    def with_axes(
        cls,
        axis1: str,
        val1: Union[float, 'Angle'],
        axis2: str = None,
        val2: Union[float, 'Angle'] = 0.0,
        axis3: str = None,
        val3: Union[float, 'Angle'] = 0.0,
    ) -> 'Angle':
        """Create an Angle, given a number of axes and corresponding values.

        This is a convenience for doing the following:
            ang = Angle()
            ang[axis1] = val1
            ang[axis2] = val2
            ang[axis3] = val3
        The magnitudes can also be Angles, in which case the matching
        axis will be used from the angle.
        """
        ang = cls()
        ang[axis1] = val1[axis1] if isinstance(val1, Py_Angle) else val1
        if axis2 is not None:
            ang[axis2] = val2[axis2] if isinstance(val2, Py_Angle) else val2
            if axis3 is not None:
                ang[axis3] = val3[axis3] if isinstance(val3, Py_Angle) else val3
        return ang

    @classmethod
    @overload
    def from_basis(cls, *, x: Vec, y: Vec, z: Vec) -> 'Angle': ...

    @classmethod
    @overload
    def from_basis(cls, *, x: Vec, y: Vec) -> 'Angle': ...

    @classmethod
    @overload
    def from_basis(cls, *, y: Vec, z: Vec) -> 'Angle': ...

    @classmethod
    @overload
    def from_basis(cls, *, x: Vec, z: Vec) -> 'Angle': ...

    @classmethod
    def from_basis(
        cls, *,
        x: Vec=None,
        y: Vec=None,
        z: Vec=None,
    ) -> 'Angle':
        """Return the rotation which results in the specified local axes.

        At least two must be specified, with the third computed if necessary.
        """
        return Matrix.from_basis(x=x, y=y, z=z).to_angle()

    def __getitem__(self, ind: Union[str, int]) -> float:
        """Allow reading values by index instead of name if desired.

        This accepts the following indexes to read values:
        - 0, 1, 2
        - pitch, yaw, roll
        - pit, yaw, rol
        - p, y, r
        Useful in conjunction with a loop to apply commands to all values.
        """
        if ind in (0, 'p', 'pit', 'pitch'):
            return self._pitch
        elif ind in (1, 'y', 'yaw'):
            return self._yaw
        elif ind in (2, 'r', 'rol', 'roll'):
            return self._roll
        raise KeyError('Invalid axis: {!r}'.format(ind))

    def __setitem__(self, ind: Union[str, int], val: float) -> None:
        """Allow editing values by index instead of name if desired.

        This accepts either 0,1,2 or 'x','y','z' to edit values.
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

    @overload
    def __mul__(self, other: Union[int, float]) -> 'Angle': ...
    @overload
    def __mul__(self, other: object) -> NotImplemented: ...

    def __mul__(self, other):
        """Angle * float multiplies each value."""
        if isinstance(other, (int, float)):
            return Py_Angle(
                self._pitch * other,
                self._yaw * other,
                self._roll * other,
            )
        return NotImplemented

    @overload
    def __rmul__(self, other: Union[int, float]) -> 'Angle': ...
    @overload
    def __rmul__(self, other: object) -> NotImplemented: ...

    def __rmul__(self, other):
        """Angle * float multiplies each value."""
        if isinstance(other, (int, float)):
            return Py_Angle(
                other * self._pitch,
                other * self._yaw,
                other * self._roll,
            )
        return NotImplemented

    @overload
    def __matmul__(self, other: 'Angle') -> 'Angle': ...
    @overload
    def __matmul__(self, other: object) -> NotImplemented: ...

    def __matmul__(self, other):
        """Angle @ Angle rotates the first by the second.
        """
        if isinstance(other, Py_Angle):
            return other._rotate_angle(self)
        else:
            return NotImplemented

    @overload
    def __rmatmul__(self, other: 'Angle') -> 'Angle': ...
    @overload
    def __rmatmul__(self, other: 'Vec') -> 'Vec': ...
    @overload
    def __rmatmul__(self, other: object) -> NotImplemented: ...

    def __rmatmul__(self, other):
        """Vec @ Angle rotates the first by the second."""
        if isinstance(other, Py_Vec):
            return other @ Py_Matrix.from_angle(self)
        elif isinstance(other, Py_Angle):
            # Should always be done by __mul__!
            return self._rotate_angle(other)
        return NotImplemented

    def _rotate_angle(self, target: 'Angle') -> 'Angle':
        """Rotate the target by this angle.
        
        Inefficient if we have more than one rotation to do.
        """
        mat = Py_Matrix.from_angle(target)
        mat @= self
        return mat.to_angle()

    @contextlib.contextmanager
    def transform(self) -> ContextManager[Matrix]:
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


def _mk(x: float, y: float, z: float) -> Vec:
    """Unpickle the Vec object, maintaining compatibility with C versions.

    Shortened name shrinks the data size.
    """
    # Skip __init__'s checks and coercion/iteration.
    v = Vec.__new__(Vec)
    v.x = x
    v.y = y
    v.z = z
    return v


# A little dance to import both the Cython and Python versions,
# and choose an appropriate unprefixed version.

Cy_Vec = Py_Vec = Vec
Cy_parse_vec_str = Py_parse_vec_str = parse_vec_str
Cy_to_matrix = Py_to_matrix = to_matrix
Cy_Angle = Py_Angle = Angle
Cy_Matrix = Py_Matrix = Matrix

# Do it this way, so static analysis ignores this.
_glob = globals()
try:
    from srctools import _vec
except ImportError:
    pass
else:
    for _name in ['Vec', 'Angle', 'Matrix', 'parse_vec_str', 'to_matrix']:
        _glob[_name] = _glob['Cy_' + _name] = getattr(_vec, _name)
    del _glob, _name, _vec
