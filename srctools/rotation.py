"""Performs various rotations.

Rotations are represented by Euler angles, but modifications need to be
performed using rotation matrices.

Rotations are implemented as a multiplication, where the left is rotated by the right.
Vecs can be rotated by matrices and angles and matrices can be rotated by angles.
"""
import math

from typing import Union, Iterable
from srctools.vec import Vec, parse_vec_str

__all__ = ['RotationMatrix', 'Angle']


class RotationMatrix:
    """Represents a 3x3 rotation matrix.

    Valid operations:
    matrix * angle
    matrix * matrix
    vec * matrix
    """
    __slots__ = [
        'aa', 'ab', 'ac',
        'ba', 'bb', 'bc',
        'ca', 'cb', 'cc',
    ]

    def __init__(
        self,
        aa: float=1.0, ab: float=0.0, ac: float=0.0,
        ba: float=0.0, bb: float=1.0, bc: float=0.0,
        ca: float=0.0, cb: float=0.0, cc: float=1.0,
    ):
        self.aa = aa
        self.ab = ab
        self.ac = ac
        self.ba = ba
        self.bb = bb
        self.bc = bc
        self.ca = ca
        self.cb = cb
        self.cc = cc

    def copy(self):
        return RotationMatrix(
            self.aa, self.ab, self.ac,
            self.ba, self.bb, self.bc,
            self.ca, self.cb, self.cc,
        )

    def __repr__(self):
        return '<Matrix {} {} {}, {} {} {}, {} {} {}>'.format(
            self.aa, self.ab, self.ac,
            self.ba, self.bb, self.bc,
            self.ca, self.cb, self.cc,
        )

    def __getitem__(self, axes):
        x, y = axes
        return getattr(self, 'abc'[y] + 'abc'[x])

    def __setitem__(self, axes, value: float):
        x, y = axes
        setattr(self, 'abc'[y] + 'abc'[x], float(value))

    def rotate_vec(self, vec: Vec):
        """Rotate a Vec by ourselves."""
        x = vec.x
        y = vec.y
        z = vec.z
        vec.x = round((x * self.aa) + (y * self.ab) + (z * self.ac), 3)
        vec.y = round((x * self.ba) + (y * self.bb) + (z * self.bc), 3)
        vec.z = round((x * self.ca) + (y * self.cb) + (z * self.cc), 3)

    def rotate_by_mat(self, other: 'RotationMatrix'):
        """Multiply ourselves by another rotation matrix."""

        # We don't use A row after the first 3, so we can re-assign.
        # 3-tuple unpacking is optimised.
        self.aa, self.ab, self.ac = (
            self.aa * other.aa + self.ab * other.ba + self.ac * other.ca,
            self.aa * other.ab + self.ab * other.bb + self.ac * other.cb,
            self.aa * other.ac + self.ab * other.bc + self.ac * other.cc,
        )

        self.ba, self.bb, self.bc = (
            self.ba * other.aa + self.bb * other.ba + self.bc * other.ca,
            self.ba * other.ab + self.bb * other.bb + self.bc * other.cb,
            self.ba * other.ac + self.bb * other.bc + self.bc * other.cc,
        )

        self.ca, self.cb, self.cc = (
            self.ca * other.aa + self.cb * other.ba + self.cc * other.ca,
            self.ca * other.ab + self.cb * other.bb + self.cc * other.cb,
            self.ca * other.ac + self.cb * other.bc + self.cc * other.cc,
        )

    @classmethod
    def pitch(cls, pitch):
        """Return the rotation matrix rotating around pitch/y."""
        ang = math.radians(pitch)
        return cls(
            math.cos(ang), 0, math.sin(ang),
            0, 1, 0,
            -math.sin(ang), 0, math.cos(ang),
        )
    @classmethod
    def yaw(cls, yaw):
        """Return the rotation matrix rotating around yaw/z."""
        ang = math.radians(yaw)
        return cls(
            math.cos(ang), -math.sin(ang), 0,
            math.sin(ang), math.cos(ang), 0,
            0, 0, 1,
        )

    @classmethod
    def roll(cls, roll):
        """Return the rotation matrix rotating around roll/x."""
        ang = math.radians(roll)
        return cls(
            1, 0, 0,
            0, math.cos(ang), -math.sin(ang),
            0, math.sin(ang), math.cos(ang),
        )

    def rotate_by_angle(self, ang: 'Angle'):
        """Rotate ourselves by an angle."""
        # pitch is in the y axis
        # yaw is the z axis
        # roll is the x axis
        # Need to do transformations in roll, pitch, yaw order
        if ang.roll:
            self.rotate_by_mat(RotationMatrix.roll(ang.roll))  # X
        if ang.pitch:
            self.rotate_by_mat(RotationMatrix.pitch(ang.pitch))  # Z
        if ang.yaw:
            self.rotate_by_mat(RotationMatrix.yaw(ang.yaw))  # Y

    def to_angle(self) -> 'Angle':
        """Convert this to a pitch-yaw-roll angle."""
        ang = Angle()
        ang.yaw = math.degrees(math.atan2(self.ab, self.aa))

        # Rotate so yaw = 0, then pitch and roll are aligned.
        copy = self.copy()
        copy.rotate_by_mat(self.yaw(-ang.yaw))

        ang.pitch = math.degrees(math.atan2(self.ac, self.aa))
        ang.roll = math.degrees(math.atan2(self.cb, self.aa))
        return ang

    def __mul__(self, other):
        copy = self.copy()
        if isinstance(other, Angle):
            copy.rotate_by_angle(other)
            return copy
        elif isinstance(other, RotationMatrix):
            copy.rotate_by_mat(other)
            return copy
        else:
            return NotImplemented

    def __imul__(self, other):
        if isinstance(other, Angle):
            self.rotate_by_angle(other)
            return self
        elif isinstance(other, RotationMatrix):
            self.rotate_by_mat(other)
            return self
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Vec):
            copy = other.copy()
            self.rotate_vec(copy)
            return copy
        elif isinstance(other, Angle):
            raise TypeError("Can't rotate angle!")
        else:
            return NotImplemented


class Angle:
    """Represents a pitch-yaw-roll angle.

    All values are remapped to between 0-360 when set.
    Addition and subtraction modify values, multiplication with Vec or Angle
    rotate (RHS rotating LHS).
    Angle * Vec is not allowed for consistency.
    """

    def __init__(
        self,
        pitch: Union[int, float, Iterable[Union[int, float]]]=0.0,
        yaw: Union[int, float]=0.0,
        roll: Union[int, float]=0.0,
    ):
        """Create a Vector.

        All values are converted to Floats automatically.
        If no value is given, that axis will be set to 0.
        An iterable can be passed in (as the pitch argument), which will be
        used for pitch, yaw, and roll.
        """
        if isinstance(pitch, (int, float)):
            self._pitch = float(pitch) % 360
            self._yaw = float(yaw) % 360
            self._roll = float(roll) % 360
        else:
            it = iter(pitch)
            self._pitch = float(next(it, 0.0)) % 360
            self._yaw = float(next(it, yaw)) % 360
            self._roll = float(next(it, roll)) % 360

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
    def pitch(self):
        """The Y-axis rotation, performed second."""
        return self._pitch

    @pitch.setter
    def pitch(self, pitch):
        self._pitch = float(pitch) % 360

    @property
    def yaw(self):
        """The Z-axis rotation, performed last."""
        return self._yaw

    @yaw.setter
    def yaw(self, yaw):
        self._yaw = float(yaw) % 360

    @property
    def roll(self):
        """The X-axis rotation, performed first."""
        return self._roll

    @roll.setter
    def roll(self, roll):
        self._roll = float(roll) % 360

    def __repr__(self):
        return 'Angle({:g}, {:g}, {:g})'.format(self._pitch, self._yaw, self._roll)

    def __rmul__(self, other: Union[Vec, 'Angle']):
        """(Vec or Angle) * Angle rotates the first by the second."""
        if isinstance(other, Vec):
            other = other.copy()
            mat = RotationMatrix()
            mat.rotate_by_angle(self)
            mat.rotate_vec(other)
            return other
        elif isinstance(other, Angle):
            return self._rotate_angle(other)
        else:
            return Angle(
                self._pitch * other,
                self._roll * other,
                self._yaw * other,
            )

    __rmatmul__ = __rmul__

    def _rotate_angle(self, target: 'Angle'):
        """Rotate the target by this angle."""
        mat = RotationMatrix()
        mat.rotate_by_angle(target)
        mat.rotate_by_angle(self)
        return mat.to_angle()

    class _AngleTransform:
        """Implements Angle.transform"""

        def __init__(self, angle: 'Angle'):
            self._angle = angle
            self._mat = None  # type: RotationMatrix

        def __enter__(self):
            self._mat = RotationMatrix()
            self._mat.rotate_by_angle(self._angle)
            return self._mat

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None or self._mat is None:
                return
            new_ang = self._mat.to_angle()
            self._angle._pitch = new_ang._pitch
            self._angle._yaw = new_ang._yaw
            self._angle._roll = new_ang._roll

    transform = property(
        fget=_AngleTransform,
        doc="""Perform transformations on this angle.

        Used as a context manager, which returns a matrix.
        When the body is exited safely, the matrix is applied to
        the angle.
        """,
    )

