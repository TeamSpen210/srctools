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

from typing import (
    Union, Tuple, overload,
    Dict, NamedTuple,
    SupportsFloat, Iterator, Iterable,
    ContextManager
)


__all__ = ['parse_vec_str', 'Vec', 'Vec_tuple', 'Angle', 'Rotation']

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
    elif isinstance(val, Vec):
        return val.x, val.y, val.z
    elif isinstance(val, Angle):
        return val.pitch, val.yaw, val.roll
    else:
        # Not a string.
        return x, y, z

    try:
        str_x, str_y, str_z = val.split(' ')
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


Vec_tuple = NamedTuple('Vec_tuple', [('x', float), ('y', float), ('z', float)])

# Use template code to reduce duplication in the various magic number methods.

_VEC_ADDSUB_TEMP = '''
def __{func}__(self, other: Union['Vec', tuple, float]):
    """{op} operation.

    This additionally works on scalars (adds to all axes).
    """
    if isinstance(other, Vec):
        return Vec(
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
        return Vec(x, y, z)

def __r{func}__(self, other: Union['Vec', tuple, float]):
    """{op} operation with reversed operands.

    This additionally works on scalars (adds to all axes).
    """
    if isinstance(other, Vec):
        return Vec(
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
        return Vec(x, y, z)

def __i{func}__(self, other: Union['Vec', tuple, float]):
    """{op}= operation.

    Like the normal one except without duplication.
    """
    if isinstance(other, Vec):
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
    if isinstance(other, Vec):
        raise TypeError("Cannot {pretty} 2 Vectors.")
    else:
        try:
            return Vec(
                self.x {op} other,
                self.y {op} other,
                self.z {op} other,
            )
        except TypeError:
            return NotImplemented

def __r{func}__(self, other: float):
    """scalar {op} Vector operation."""
    if isinstance(other, Vec):
        raise TypeError("Cannot {pretty} 2 Vectors.")
    else:
        try:
            return Vec(
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
    if isinstance(other, Vec):
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
        elif isinstance(x, Vec):
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
        return Vec(self.x, self.y, self.z)

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

        x, y, z = parse_vec_str(val, x, y, z)
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
        vec[axis1] = val1[axis1] if isinstance(val1, Vec) else val1
        if axis2 is not None:
            vec[axis2] = val2[axis2] if isinstance(val2, Vec) else val2
            if axis3 is not None:
                vec[axis3] = val3[axis3] if isinstance(val3, Vec) else val3
        return vec

    def _mat_mul(
        self,
        a: float, b: float, c: float,
        d: float, e: float, f: float,
        g: float, h: float, i: float,
    ) -> None:
        """Multiply this vector by a 3x3 rotation matrix.

        Used for Vec.rotate().
        The matrix should follow the following pattern:
        [ a b c ]
        [ d e f ]
        [ g h i ]
        """
        x, y, z = self.x, self.y, self.z

        self.x = (x * a) + (y * b) + (z * c)
        self.y = (x * d) + (y * e) + (z * f)
        self.z = (x * g) + (y * h) + (z * i)

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

        If round is True, all values will be rounded to 3 decimals
        (since these calculations always have small inprecision.)
        """
        # pitch is in the y axis
        # yaw is the z axis
        # roll is the x axis

        rad_pitch = math.radians(pitch)
        rad_yaw = math.radians(yaw)
        rad_roll = math.radians(roll)
        cos_p = math.cos(rad_pitch)
        cos_y = math.cos(rad_yaw)
        cos_r = math.cos(rad_roll)

        sin_p = math.sin(rad_pitch)
        sin_y = math.sin(rad_yaw)
        sin_r = math.sin(rad_roll)

        # Need to do transformations in roll, pitch, yaw order
        self._mat_mul(  # Roll = X
            1, 0, 0,
            0, cos_r, -sin_r,
            0, sin_r, cos_r,
        )

        self._mat_mul(  # Pitch = Y
            cos_p, 0, sin_p,
            0, 1, 0,
            -sin_p, 0, cos_p,
        )

        self._mat_mul(  # Yaw = Z
            cos_y, -sin_y, 0,
            sin_y, cos_y, 0,
            0, 0, 1,
        )

        if round_vals:
            self.x = round(self.x, 6)
            self.y = round(self.y, 6)
            self.z = round(self.z, 6)

        return self

    def rotate_by_str(self, ang: str, pitch=0.0, yaw=0.0, roll=0.0, round_vals=True) -> 'Vec':
        """Rotate a vector, using a string instead of a vector.

        If the string cannot be parsed, use the passed in values instead.
        """
        pitch, yaw, roll = parse_vec_str(ang, pitch, yaw, roll)
        return self.rotate(
            pitch,
            yaw,
            roll,
            round_vals,
        )

    @staticmethod
    @overload
    def bbox(points: Iterable['Vec']) -> Tuple['Vec', 'Vec']: ...
    @staticmethod
    @overload
    def bbox(*points: 'Vec') -> Tuple['Vec', 'Vec']: ...

    @staticmethod
    def bbox(*points):
        """Compute the bounding box for a set of points.

        Pass either several Vecs, or an iterable of Vecs.
        Returns a (min, max) tuple.
        """
        # Allow passing a single iterable, but also handle a single Vec.
        # The error messages match those produced by min()/max().
        if len(points) == 1 and not isinstance(points[0], Vec):
            try:
                [[first, *points]] = points
            except ValueError:
                raise ValueError('Vec.bbox() arg is an empty sequence') from None
        else:
            try:
                first, *points = points
            except ValueError:
                raise TypeError(
                    'Vec.bbox() expected at '
                    'least 1 argument, got 0.'
                ) from None

        bbox_min = Vec(first)
        bbox_max = bbox_min.copy()
        for point in points:
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
        x = self.x != 0
        y = self.y != 0
        z = self.z != 0
        if x and not y and not z:
            return 'x'
        if not x and y and not z:
            return 'y'
        if not x and not y and z:
            return 'z'
        raise ValueError(
            '({}, {}, {}) is '
            'not an on-axis vector!'.format(self.x, self.y, self.z)
        )

    def to_angle(self, roll: float=0) -> 'Vec':
        """Convert a normal to a Source Engine angle.

        A +x axis vector will result in a 0, 0, 0 angle. The roll is not
        affected by the direction of the normal.

        The inverse of this is `Vec(x=1).rotate(pitch, yaw, roll)`.
        """
        # Pitch is applied first, so we need to reconstruct the x-value
        horiz_dist = math.sqrt(self.x ** 2 + self.y ** 2)
        return Vec(
            math.degrees(math.atan2(-self.z, horiz_dist)),
            math.degrees(math.atan2(self.y, self.x)) % 360,
            roll,
        )

    def to_angle_roll(self, z_norm: 'Vec', stride: int=90) -> 'Vec':
        """Produce a Source Engine angle with roll.

        The z_normal should point in +z, and must be at right angles to this
        vector. Stride determines the angles chosen - the normal must point
        in one of these.
        """
        angle = self.to_angle()
        for roll in range(0, 360, stride):
            result = Vec(z=1).rotate(angle.x, angle.y, roll)
            if result == z_norm:
                angle.z = roll
                return angle
        else:
            raise ValueError(
                'Normal of {} does not have a valid angle'
                ' in ({}, {}, z) at increments of {}'.format(
                    z_norm, angle.x, angle.y, stride,
                )
            )

    def rotation_around(self, rot: float=90) -> 'Vec':
        """For an axis-aligned normal, return the angles which rotate around it."""
        if self.x:
            return Vec(z=self.x * rot)
        elif self.y:
            return Vec(x=self.y * rot)
        elif self.z:
            return Vec(y=self.z * rot)
        else:
            raise ValueError('Zero vector!')

    def __abs__(self) -> 'Vec':
        """Performing abs() on a Vec takes the absolute value of all axes."""
        return Vec(
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
        if isinstance(other, Vec):
            raise TypeError("Cannot divide 2 Vectors.")
        else:
            try:
                x1, x2 = divmod(self.x, other)
                y1, y2 = divmod(self.y, other)
                z1, z2 = divmod(self.z, other)
            except TypeError:
                return NotImplemented
            else:
                return Vec(x1, y1, z1), Vec(x2, y2, z2)

    def __rdivmod__(self, other: float) -> Tuple['Vec', 'Vec']:
        """Divide a scalar by a vector, returning the result and remainder."""
        try:
            x1, x2 = divmod(other, self.x)
            y1, y2 = divmod(other, self.y)
            z1, z2 = divmod(other, self.z)
        except (TypeError, ValueError):
            return NotImplemented
        else:
            return Vec(x1, y1, z1), Vec(x2, y2, z2)

    def __bool__(self) -> bool:
        """Vectors are True if any axis is non-zero."""
        return self.x != 0 or self.y != 0 or self.z != 0

    def __eq__(self, other: object) -> bool:
        """== test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        """
        if isinstance(other, Vec):
            return other.x == self.x and other.y == self.y and other.z == self.z
        elif isinstance(other, tuple):
            return (
                self.x == other[0] and
                self.y == other[1] and
                self.z == other[2]
            )
        else:
            return NotImplemented

    def __ne__(self, other: object) -> bool:
        """!= test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        """
        if isinstance(other, Vec):
            return other.x != self.x or other.y != self.y or other.z != self.z
        elif isinstance(other, tuple):
            return (
                self.x != other[0] or
                self.y != other[1] or
                self.z != other[2]
            )
        else:
            return NotImplemented

    def __lt__(
        self,
        other: Union['Vec', Tuple3, SupportsFloat],
    ) -> bool:
        """A<B test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        """
        if isinstance(other, Vec):
            return (
                self.x < other.x and
                self.y < other.y and
                self.z < other.z
            )
        elif isinstance(other, tuple):
            return (
                self.x < other[0] and
                self.y < other[1] and
                self.z < other[2]
            )
        else:
            return NotImplemented

    def __le__(
        self,
        other: Union['Vec', Tuple3, SupportsFloat],
    ) -> bool:
        """A<=B test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        """
        if isinstance(other, Vec):
            return (
                self.x <= other.x and
                self.y <= other.y and
                self.z <= other.z
            )
        elif isinstance(other, tuple):
            return (
                self.x <= other[0] and
                self.y <= other[1] and
                self.z <= other[2]
            )
        else:
            return NotImplemented

    def __gt__(
        self,
        other: Union['Vec', Tuple3, SupportsFloat],
    ) -> bool:
        """A>B test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        """
        if isinstance(other, Vec):
            return (
                self.x > other.x and
                self.y > other.y and
                self.z > other.z
            )
        elif isinstance(other, tuple):
            return (
                self.x > other[0] and
                self.y > other[1] and
                self.z > other[2]
            )
        else:
            return NotImplemented

    def __ge__(
        self,
        other: Union['Vec', Tuple3, SupportsFloat],
    ) -> bool:
        """A>=B test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        """
        if isinstance(other, Vec):
            return (
                self.x >= other.x and
                self.y >= other.y and
                self.z >= other.z
            )
        elif isinstance(other, tuple):
            return (
                self.x >= other[0] and
                self.y >= other[1] and
                self.z >= other[2]
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
        return Vec(
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
        return '{x:g}{delim}{y:g}{delim}{z:g}'.format(
            x=self.x,
            y=self.y,
            z=self.z,
            delim=delim,
        )

    def __str__(self):
        """Return the values, separated by spaces.

        This is the main format in Valve's file formats.
        This strips off the .0 if no decimal portion exists.
        """
        return "{:g} {:g} {:g}".format(self.x, self.y, self.z)

    def __repr__(self):
        """Code required to reproduce this vector."""
        return self.__class__.__name__ + "(" + self.join() + ")"

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
        raise KeyError('Invalid axis: {!r}'.format(ind))

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
            raise KeyError('Invalid axis: {!r}'.format(ind))

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
        return Vec_tuple(self.x, self.y, self.z)

    def len_sq(self) -> float:
        """Return the magnitude squared, which is slightly faster."""
        return self.x**2 + self.y**2 + self.z**2

    def __len__(self) -> int:
        """The len() of a vector is the number of non-zero axes."""
        return (
            (self.x != 0) +
            (self.y != 0) +
            (self.z != 0)
        )

    def __contains__(self, val: float) -> bool:
        """Check to see if an axis is set to the given value.
        """
        return val == self.x or val == self.y or val == self.z

    def __neg__(self) -> 'Vec':
        """The inverted form of a Vector has inverted axes."""
        return Vec(-self.x, -self.y, -self.z)

    def __pos__(self) -> 'Vec':
        """+ on a Vector simply copies it."""
        return Vec(self.x, self.y, self.z)

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
        return Vec(
            self.y * other[2] - self.z * other[1],
            self.z * other[0] - self.x * other[2],
            self.x * other[1] - self.y * other[0],
        )

    def localise(
        self,
        origin: Union['Vec', Tuple3],
        angles: Union['Vec', Tuple3]=None,
    ) -> None:
        """Shift this point to be local to the given position and angles.

        This effectively translates local-space offsets to a global location,
        given the parent's origin and angles.
        """
        if angles is not None:
            self.rotate(angles[0], angles[1], angles[2])
        self.__iadd__(origin)

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
    def transform(self) -> ContextManager['Rotation']:
        """Perform rotations on this Vector efficiently.

        Used as a context manager, which returns a matrix.
        When the body is exited safely, the matrix is applied to
        the angle.
        """
        mat = Rotation()
        yield mat
        mat._vec_rot(self)


class Rotation:
    """Represents a rotation via a transformation matrix.
    
    The internal values are private, to allow changing the implementation.
    """
    __slots__ = [
        'a', 'b', 'c', 
        'd', 'e', 'f', 
        'g', 'h', 'i'
    ]

    def __init__(self) -> None:
        """Create a rotation set to the identity transform."""
        self.a, self.b, self.c = 1.0, 0.0, 0.0
        self.d, self.e, self.f = 0.0, 1.0, 0.0
        self.g, self.h, self.i = 0.0, 0.0, 1.0

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Rotation):
            return (
                self.a == other.a and self.b == other.b and self.c == other.c and
                self.d == other.d and self.e == other.e and self.f == other.f and
                self.g == other.g and self.h == other.h and self.i == other.i
            )
        return False
        
    def __repr__(self) -> str:
        return (
            'Rotation(\n'
            '\t<{0.a}, {0.b}, {0.c}>\n'
            '\t<{0.d}, {0.e}, {0.f}>\n'
            '\t<{0.g}, {0.h}, {0.i}>\n'
            ')'
        ).format(self)

    def copy(self) -> 'Rotation':
        """Duplicate this matrix."""
        rot = Rotation.__new__(Rotation)

        rot.a, rot.b, rot.c = self.a, self.b, self.c
        rot.d, rot.e, rot.f = self.d, self.e, self.f
        rot.g, rot.h, rot.i = self.g, self.h, self.i

        return rot

    @classmethod
    def from_pitch(cls, pitch: float) -> 'Rotation':
        """Return the rotation representing a pitch rotation.

        This is a rotation around the Y axis.
        """
        rad_pitch = math.radians(pitch)
        cos = math.cos(rad_pitch)
        sin = math.sin(rad_pitch)

        rot = cls.__new__(cls)

        rot.a, rot.b, rot.c = cos, 0.0, sin
        rot.d, rot.e, rot.f = 0.0, 1.0, 0.0
        rot.g, rot.h, rot.i = -sin, 0.0, cos

        return rot

    @classmethod
    def from_yaw(cls, yaw: float) -> 'Rotation':
        """Return the rotation representing a yaw rotation.

        """
        rad_yaw = math.radians(yaw)
        sin = math.sin(rad_yaw)
        cos = math.cos(rad_yaw)

        rot = cls.__new__(cls)

        rot.a, rot.b, rot.c = cos, -sin, 0.0
        rot.d, rot.e, rot.f = sin, cos, 0.0
        rot.g, rot.h, rot.i = 0.0, 0.0, 1.0

        return rot
        
    @classmethod
    def from_roll(cls, roll: float) -> 'Rotation':
        """Return the rotation representing a roll rotation.

        This is a rotation around the X axis.
        """
        rad_roll = math.radians(roll)
        cos_r = math.cos(rad_roll)
        sin_r = math.sin(rad_roll)

        rot = cls.__new__(cls)

        rot.a, rot.b, rot.c = 1.0, 0.0, 0.0
        rot.d, rot.e, rot.f = 0.0, cos_r, -sin_r
        rot.g, rot.h, rot.i = 0.0, sin_r, cos_r

        return rot
        
    @classmethod
    def from_angle(cls, angle: 'Angle') -> 'Rotation':
        """Return the rotation representing an Euler angle."""
        rot = cls()
        rot @= cls.from_roll(angle.roll)
        rot @= cls.from_pitch(angle.pitch)
        rot @= cls.from_yaw(angle.yaw)
        return rot

    def to_angle(self) -> 'Angle':
        """Return an Euler angle replicating this rotation."""

        # https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/mathlib/mathlib_base.cpp#L208
        for_x = self.a
        for_y = self.b
        for_z = self.c
        left_x = self.d
        left_y = self.e
        left_z = self.f
        # up_x = self.g
        # up_y = self.h
        up_z = self.i
        
        horiz_dist = math.sqrt(for_x**2 + for_y**2)
        if horiz_dist > 0.001:
            return Angle(
                yaw=-math.degrees(math.atan2(for_y, for_x)),
                pitch=-math.degrees(math.atan2(-for_z, horiz_dist)),
                roll=-math.degrees(math.atan2(left_z, up_z)),
            )
        else:
            # Vertical, gimbal lock (yaw=roll)...
            return Angle(
                yaw=-math.degrees(math.atan2(-left_x, left_y)),
                pitch=-math.degrees(math.atan2(-for_z, horiz_dist)),
                roll=0,  # Can't produce.
            )

    @overload
    def __matmul__(self, other: 'Rotation') -> 'Rotation': ...
    @overload
    def __matmul__(self, other: 'Angle') -> 'Rotation': ...
    @overload
    def __matmul__(self, other: object) -> NotImplemented: ...

    def __matmul__(self, other: object) -> 'Rotation':
        if isinstance(other, Rotation):
            mat = self.copy()
            mat._mat_mul(other)
            return mat
        elif isinstance(other, Angle):
            mat = self.copy()
            mat._mat_mul(Rotation.from_angle(other))
            return mat
        else:
            return NotImplemented

    @overload
    def __rmatmul__(self, other: Vec) -> Vec: ...
    @overload
    def __rmatmul__(self, other: 'Rotation') -> 'Rotation': ...
    @overload
    def __rmatmul__(self, other: 'Angle') -> 'Angle': ...
    @overload
    def __rmatmul__(self, other: object) -> NotImplemented: ...
    
    def __rmatmul__(self, other):
        if isinstance(other, Vec):
            result = other.copy()
            self._vec_rot(result)
            return result
        elif isinstance(other, Angle):
            mat = Rotation.from_angle(other)
            mat._mat_mul(self)
            return mat.to_angle()
        elif isinstance(other, Rotation):
            mat = other.copy()
            return mat._mat_mul(self)
        else:
            return NotImplemented

    @overload
    def __imatmul__(self, other: 'Rotation') -> 'Rotation': ...
    @overload
    def __imatmul__(self, other: 'Angle') -> 'Rotation': ...
    @overload
    def __imatmul__(self, other: object) -> NotImplemented: ...

    def __imatmul__(self, other):
        if isinstance(other, Rotation):
            self._mat_mul(other)
            return self
        elif isinstance(other, Angle):
            self._mat_mul(Rotation.from_angle(other))
            return self
        else:
            return NotImplemented
            
    def _mat_mul(self, other: 'Rotation') -> None:
        """Rotate myself by the other matrix."""
        a, b, c = self.a, self.b, self.c
        d, e, f = self.d, self.e, self.f
        g, h, i = self.g, self.h, self.i

        self.a = a * other.a + b * other.d + c * other.g
        self.b = a * other.b + b * other.e + c * other.h
        self.c = a * other.c + b * other.f + c * other.i

        self.d = d * other.a + e * other.d + f * other.g
        self.e = d * other.b + e * other.e + f * other.h
        self.f = d * other.c + e * other.f + f * other.i

        self.g = g * other.a + h * other.d + i * other.g
        self.h = g * other.b + h * other.e + i * other.h
        self.i = g * other.c + h * other.f + i * other.i
    
    def _vec_rot(self, vec: Vec) -> None:
        """Rotate a vector by our value."""
        x = vec.x
        y = vec.y
        z = vec.z

        vec.x = (x * self.a) + (y * self.b) + (z * self.c)
        vec.y = (x * self.d) + (y * self.e) + (z * self.f)
        vec.z = (x * self.g) + (y * self.h) + (z * self.i)
        
    
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
    ):
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
        return Angle(self._pitch, self._yaw, self._roll)

    @classmethod
    def from_str(cls, val: Union[str, 'Angle'], pitch=0.0, yaw=0.0, roll=0.0):
        """Convert a string in the form '(4 6 -4)' into an Angle.

         If the string is unparsable, this uses the defaults.
         The string can start with any of the (), {}, [], <> bracket
         types, or none.

         If the value is already a Angle, a copy will be returned.
         """

        pitch, yaw, roll = parse_vec_str(val, pitch, yaw, roll)
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

    def __repr__(self) -> str:
        return 'Angle({0._pitch:g}, {0._yaw:g}, {0._roll:g})'.format(self)

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
        ang[axis1] = val1[axis1] if isinstance(val1, Angle) else val1
        if axis2 is not None:
            ang[axis2] = val2[axis2] if isinstance(val2, Angle) else val2
            if axis3 is not None:
                ang[axis3] = val3[axis3] if isinstance(val3, Angle) else val3
        return ang

    def __getitem__(self, ind: Union[str, int]) -> float:
        """Allow reading values by index instead of name if desired.

        This accepts the following indexes to read values:
        - 0, 1, 2
        - pitch, yaw, roll
        - pit, yaw, rol
        - p, y, r
        Useful in conjunction with a loop to apply commands to all values.
        """
        if ind in (0, 'p', 'pit','pitch'):
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
        if ind in (0, 'p', 'pit','pitch'):
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
            return Angle(
                self._pitch * other,
                self._yaw * other,
                self._roll * other,
            )
        return NotImplemented

    @overload
    def __rmul__(self, other: Union[int, float]) -> 'Angle': ...
    @overload
    def __rmul__(self, other: object) -> NotImplemented: ...

    def __rmul__(self, other: float):
        """Angle * float multiplies each value."""
        if isinstance(other, (int, float)):
            return Angle(
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
        if isinstance(other, Angle):
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
        if isinstance(other, Vec):
            return other @ Rotation.from_angle(self)
        elif isinstance(other, Angle):
            # Should always be done by __mul__!
            return self._rotate_angle(other)
        return NotImplemented

    def _rotate_angle(self, target: 'Angle') -> 'Angle':
        """Rotate the target by this angle.
        
        Inefficient if we have more than one rotation to do.
        """
        mat = Rotation()
        mat @= target
        mat @= self
        return mat.to_angle()

    @contextlib.contextmanager
    def transform(self) -> ContextManager[Rotation]:
        """Perform transformations on this angle.

        Used as a context manager, which returns a matrix.
        When the body is exited safely, the matrix is applied to
        the angle.
        """
        mat = Rotation()
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
