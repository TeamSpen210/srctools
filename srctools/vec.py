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

Index via .x, .y, .z attributes, or 'x', 'y', 'z', 0, 1, 3 index access.


"""
import collections
import math
from collections import abc

from typing import Union, Tuple, SupportsFloat, Iterator, Iterable


__all__ = ['parse_vec_str', 'Vec', 'Vec_tuple']


def parse_vec_str(val: Union[str, 'Vec'], x=0.0, y=0.0, z=0.0) -> Tuple[float, float, float]:
    """Convert a string in the form '(4 6 -4)' into a set of floats.

     If the string is unparsable, this uses the defaults (x,y,z).
     The string can start with any of the (), {}, [], <> bracket
     types.

     If the 'string' is actually a Vec, the values will be returned.
     """
    if isinstance(val, Vec):
        return val.x, val.y, val.z

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


Vec_tuple = collections.namedtuple('Vec_tuple', ['x', 'y', 'z'])

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
    else:
        orig = self.x, self.y, self.z
        try:
            self.x {op}= other
            self.y {op}= other
            self.z {op}= other
        except TypeError as e:
            self.x, self.y, self.z = orig
            raise TypeError(
                'Cannot add {{}} to Vector!'.format(type(other))
            ) from e
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

    INV_AXIS = {
        'x': 'yz',
        'y': 'xz',
        'z': 'xy',

        ('y', 'z'): 'x',
        ('x', 'z'): 'y',
        ('x', 'y'): 'z',

        ('z', 'y'): 'x',
        ('z', 'x'): 'y',
        ('y', 'x'): 'z',
    }
    # Vectors pointing in all cardinal directions
    N = north = y_pos = Vec_tuple(0, 1, 0)
    S = south = y_neg = Vec_tuple(0, -1, 0)
    E = east = x_pos = Vec_tuple(1, 0, 0)
    W = west = x_neg = Vec_tuple(-1, 0, 0)
    T = top = z_pos = Vec_tuple(0, 0, 1)
    B = bottom = z_neg = Vec_tuple(0, 0, -1)

    def __init__(
            self,
            x: Union[int, float, 'Vec', Iterable[Union[int, float]]]=0.0,
            y: Union[int, float]=0.0,
            z: Union[int, float]=0.0,
    ):
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
        else:
            it = iter(x)
            self.x = float(next(it, 0.0))
            self.y = float(next(it, y))
            self.z = float(next(it, z))

    def copy(self) -> 'Vec':
        """Create a duplicate of this vector."""
        return Vec(self.x, self.y, self.z)

    @classmethod
    def from_str(cls, val: Union[str, 'Vec'], x=0.0, y=0.0, z=0.0):
        """Convert a string in the form '(4 6 -4)' into a Vector.

         If the string is unparsable, this uses the defaults (x,y,z).
         The string can start with any of the (), {}, [], <> bracket
         types, or none.

         If the value is already a vector, a copy will be returned.
         """

        x, y, z = parse_vec_str(val, x, y, z)
        return cls(x, y, z)

    @classmethod
    def with_axes(
            cls,
            axis1: str,
            val1: Union[float, 'Vec'],
            axis2: str=None,
            val2: Union[float, 'Vec']=None,
            axis3: str=None,
            val3: Union[float, 'Vec']=None
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

    def mat_mul(self, matrix) -> None:
        """Multiply this vector by a 3x3 rotation matrix.

        Used for Vec.rotate().
        The matrix should be a 9-tuple, following the pattern:
        [ a b c ]
        [ d e f ]
        [ g h i ]
        """
        a, b, c, d, e, f, g, h, i = matrix
        x, y, z = self.x, self.y, self.z

        self.x = (x * a) + (y * b) + (z * c)
        self.y = (x * d) + (y * e) + (z * f)
        self.z = (x * g) + (y * h) + (z * i)

    def rotate(self, pitch=0.0, yaw=0.0, roll=0.0, round_vals=True) -> 'Vec':
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

        mat_roll = (  # X
            1, 0, 0,
            0, cos_r, -sin_r,
            0, sin_r, cos_r,
        )
        mat_yaw = (  # Z
            cos_y, -sin_y, 0,
            sin_y, cos_y, 0,
            0, 0, 1,
        )

        mat_pitch = (  # Y
            cos_p, 0, sin_p,
            0, 1, 0,
            -sin_p, 0, cos_p,
        )

        # Need to do transformations in roll, pitch, yaw order
        self.mat_mul(mat_roll)
        self.mat_mul(mat_pitch)
        self.mat_mul(mat_yaw)

        if round_vals:
            self.x = round(self.x, 3)
            self.y = round(self.y, 3)
            self.z = round(self.z, 3)

        return self

    def rotate_by_str(self, ang, pitch=0.0, yaw=0.0, roll=0.0, round_vals=True):
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
    def bbox(*points: 'Vec') -> Tuple['Vec', 'Vec']:
        """Compute the bounding box for a set of points.

        Pass either several Vecs, or an iterable of Vecs.
        Returns a (min, max) tuple.
        """
        if len(points) == 1:  # Allow passing a single iterable
            (first, *points), = points
        else:
            first, *points = points
        bbox_min = first.copy()
        bbox_max = first.copy()
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
        min_x, min_y, min_z = map(int, min_pos)
        max_x, max_y, max_z = map(int, max_pos)
        for x in range(min_x, max_x + 1, stride):
            for y in range(min_y, max_y + 1, stride):
                for z in range(min_z, max_z + 1, stride):
                    yield cls(x, y, z)

    def iter_line(self, end: 'Vec', stride: int=1):
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
        """For a normal vector, return the axis it is on.

        This will not function correctly if not a on-axis normal vector!
        """
        return (
            'x' if self.x != 0 else
            'y' if self.y != 0 else
            'z'
        )

    def to_angle(self, roll=0) -> 'Vec':
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

    def rotation_around(self, rot=90):
        """For an axis-aligned normal, return the angles which rotate around it."""
        if self.x:
            return Vec(z=self.x * rot)
        elif self.y:
            return Vec(x=self.y * rot)
        elif self.z:
            return Vec(y=self.z * rot)
        else:
            raise ValueError('Zero vector!')

    def __abs__(self):
        """Performing abs() on a Vec takes the absolute value of all axes."""
        return Vec(
            abs(self.x),
            abs(self.y),
            abs(self.z),
        )

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
        if isinstance(other, Vec):
            return NotImplemented
        else:
            try:
                x1, x2 = divmod(other, self.x)
                y1, y2 = divmod(other, self.y)
                z1, z2 = divmod(other, self.z)
            except TypeError:
                return NotImplemented
            else:
                return Vec(x1, y1, z1), Vec(x2, y2, z2)

    def __bool__(self) -> bool:
        """Vectors are True if any axis is non-zero."""
        return self.x != 0 or self.y != 0 or self.z != 0

    def __eq__(
            self,
            other: Union['Vec', tuple, SupportsFloat],
            ) -> bool:
        """== test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        Otherwise the other value will be compared with the magnitude.
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
            try:
                return self.mag() == float(other)
            except ValueError:
                return NotImplemented

    def __ne__(
            self,
            other: Union['Vec', tuple, SupportsFloat],
            ) -> bool:
        """!= test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        Otherwise the other value will be compared with the magnitude.
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
            try:
                return self.mag() != float(other)
            except ValueError:
                return NotImplemented

    def __lt__(
            self,
            other: Union['Vec', abc.Sequence, SupportsFloat],
            ) -> bool:
        """A<B test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        Otherwise the other value will be compared with the magnitude.
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
            try:
                return self.mag() < float(other)
            except ValueError:
                return NotImplemented

    def __le__(
            self,
            other: Union['Vec', tuple, SupportsFloat],
            ) -> bool:
        """A<=B test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        Otherwise the other value will be compared with the magnitude.
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
            try:
                return self.mag() <= float(other)
            except ValueError:
                return NotImplemented

    def __gt__(
            self,
            other: Union['Vec', tuple, SupportsFloat],
            ) -> bool:
        """A>B test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        Otherwise the other value will be compared with the magnitude.
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
            try:
                return self.mag() > float(other)
            except ValueError:
                return NotImplemented

    def __ge__(
            self,
            other: Union['Vec', tuple, SupportsFloat],
    ) -> bool:
        """A>=B test.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        Otherwise the other value will be compared with the magnitude.
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
            try:
                return self.mag() >= float(other)
            except ValueError:
                return NotImplemented

    def max(self, other: Union['Vec', Vec_tuple]):
        """Set this vector's values to the maximum of the two vectors."""
        if self.x < other.x:
            self.x = other.x
        if self.y < other.y:
            self.y = other.y
        if self.z < other.z:
            self.z = other.z

    def min(self, other: Union['Vec', Vec_tuple]):
        """Set this vector's values to be the minimum of the two vectors."""
        if self.x > other.x:
            self.x = other.x
        if self.y > other.y:
            self.y = other.y
        if self.z > other.z:
            self.z = other.z

    def __round__(self, n=0):
        return Vec(
            round(self.x, n),
            round(self.y, n),
            round(self.z, n),
        )

    def mag(self):
        """Compute the distance from the vector and the origin."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def join(self, delim=', '):
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

    def __setitem__(self, ind: Union[str, int], val: float):
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

    def as_tuple(self):
        """Return the Vector as a tuple."""
        return Vec_tuple(self.x, self.y, self.z)

    def len_sq(self):
        """Return the magnitude squared, which is slightly faster."""
        return self.x**2 + self.y**2 + self.z**2

    def __len__(self):
        """The len() of a vector is the number of non-zero axes."""
        return (
            (self.x != 0) +
            (self.y != 0) +
            (self.z != 0)
        )

    def __contains__(self, val):
        """Check to see if an axis is set to the given value.
        """
        return val == self.x or val == self.y or val == self.z

    def __neg__(self):
        """The inverted form of a Vector has inverted axes."""
        return Vec(-self.x, -self.y, -self.z)

    def __pos__(self):
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

    def dot(self, other):
        """Return the dot product of both Vectors."""
        return (
            self.x * other.x +
            self.y * other.y +
            self.z * other.z
        )

    def cross(self, other):
        """Return the cross product of both Vectors."""
        return Vec(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def localise(
            self,
            origin: Union['Vec', tuple],
            angles: Union['Vec', tuple]=None,
    ):
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
        normal = normal.norm()
        return normal * self.dot(normal)

    len = mag
    mag_sq = len_sq
