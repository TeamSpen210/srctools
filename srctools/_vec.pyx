# cython: language_level=3, embedsignature=True, auto_pickle=False
# """Optimised Vector object."""
from libc cimport math
from libc.string cimport memcpy
from cpython.object cimport PyObject, PyTypeObject, Py_LT, Py_LE, Py_EQ, Py_NE, Py_GT, Py_GE
from cpython.ref cimport Py_INCREF
cimport cython

# Lightweight struct just holding the three values.
# We can use this for temporaries. It's also used for angles.
cdef struct vec_t:
    double x
    double y
    double z

ctypedef double[3][3] mat_t

cdef inline Vec _vector(double x, double y, double z):
    """Make a Vector directly."""
    cdef Vec vec = Vec.__new__(Vec)
    vec.val.x = x
    vec.val.y = y
    vec.val.z = z
    return vec

cdef inline Angle _angle(double pitch, double yaw, double roll):
    """Make an Angle directly."""
    cdef Angle vec = Angle.__new__(Angle)
    vec.val.x = pitch
    vec.val.y = yaw
    vec.val.z = roll
    return vec


# Shared func that we use to do unpickling.
# It's defined in the Python module, so all versions
# produce the same pickle value.
cdef object unpickle_func

# Grab the Vec_Tuple class.
cdef object Vec_tuple
from srctools.vec import _mk as unpickle_func, Vec_tuple

# Sanity check.
if not issubclass(Vec_tuple, tuple):
    raise RuntimeError('Vec_tuple is not a tuple subclass!')


DEF PI = 3.141592653589793238462643383279502884197
# Multiply to convert.
DEF rad_2_deg = 180.0 / PI
DEF deg_2_rad = PI / 180.0

cdef extern from *:  # Allow ourselves to access one of the feature flag macros.
    cdef bint USE_TYPE_INTERNALS "CYTHON_USE_TYPE_SLOTS"

cdef object _make_tuple(x, y, z):
    # Fast-construct a Vec_tuple. We make a normal tuple (fast),
    # then assign the namedtuple type. The type is on the heap
    # so we need to incref it.
    cdef tuple tup = (x, y, z)
    if USE_TYPE_INTERNALS:
        Py_INCREF(Vec_tuple)
        (<PyObject *>tup).ob_type = <PyTypeObject*>Vec_tuple
        return tup
    else: # Not CPython, use more correct method.
        with cython.optimize.unpack_method_calls(False):
            return Vec_tuple._make(*tup)


cdef unsigned char _parse_vec_str(vec_t *vec, object value, double x, double y, double z) except False:
    cdef unicode str_x, str_y, str_z

    if isinstance(value, Vec):
        vec.x = (<Vec>value).val.x
        vec.y = (<Vec>value).val.y
        vec.z = (<Vec>value).val.z
        return True
    elif isinstance(value, Angle):
        vec.x = (<Angle>value).val.x
        vec.y = (<Angle>value).val.y
        vec.z = (<Angle>value).val.z
        return True

    try:
        str_x, str_y, str_z = (<unicode?>value).split(' ')
    except ValueError:
        vec.x = x
        vec.y = y
        vec.z = z
        return True

    if str_x[0] in '({[<':
        str_x = str_x[1:]
    if str_z[-1] in ')}]>':
        str_z = str_z[:-1]
    try:
        vec.x = float(str_x)
        vec.y = float(str_y)
        vec.z = float(str_z)
    except ValueError:
        vec.x = x
        vec.y = y
        vec.z = z
    return True


def parse_vec_str(val, double x=0.0, double y=0.0, double z=0.0):
    """Convert a string in the form '(4 6 -4)' into a set of floats.

     If the string is unparsable, this uses the defaults (x,y,z).
     The string can start with any of the (), {}, [], <> bracket
     types.

     If the 'string' is actually a Vec, the values will be returned.
     """
    cdef vec_t vec
    _parse_vec_str(&vec, val, x, y, z)
    return _make_tuple(vec.x, vec.y, vec.z)

cdef inline unsigned char _conv_vec(
    vec_t *result,
    object vec,
    bint scalar,
) except False:
    """Convert some object to a unified Vector struct. 
    
    If scalar is True, allow int/float to set all axes.
    """
    if isinstance(vec, Vec):
        result.x = (<Vec>vec).val.x
        result.y = (<Vec>vec).val.y
        result.z = (<Vec>vec).val.z
    elif isinstance(vec, float) or isinstance(vec, int):
        if scalar:
            result.x = result.y = result.z = vec
        else:
            # No need to do argument checks.
            raise TypeError('Cannot use scalars here.')
    elif isinstance(vec, tuple):
        result.x, result.y, result.z = <tuple>vec
    else:
        try:
            result.x = vec.x
            result.y = vec.y
            result.z = vec.z
        except AttributeError:
            raise TypeError(f'{type(vec)} is not a Vec-like object!')
    return True

cdef inline unsigned char _conv_angles(
    vec_t *result,
    object ang,
    bint scalar,
) except False:
    """Convert some object to a unified Angle struct. 
    
    If scalar is True, allow int/float to set all axes.
    """
    cdef double x, y, z
    if isinstance(ang, Angle):
        result.x = (<Angle>ang).val.x
        result.y = (<Angle>ang).val.y
        result.z = (<Angle>ang).val.z
    elif isinstance(ang, float) or isinstance(ang, int):
        if scalar:
            result.x = result.y = result.z = <double>ang % 360.0 % 360.0
        else:
            # No need to do argument checks.
            raise TypeError('Cannot use scalars here.')
    elif isinstance(ang, tuple):
        x, y, z = <tuple>ang
        result.x = x % 360.0 % 360.0
        result.y = y % 360.0 % 360.0
        result.z = z % 360.0 % 360.0
    else:
        try:
            result.x = <double>ang.x % 360.0 % 360.0
            result.y = <double>ang.y % 360.0 % 360.0
            result.z = <double>ang.z % 360.0 % 360.0
        except AttributeError:
            raise TypeError(f'{type(ang)} is not an Angle-like object!')
    return True


cdef inline double _vec_mag(vec_t *vec):
    return math.sqrt(vec.x**2 + vec.y**2 + vec.z**2)

cdef inline void _vec_normalise(vec_t *out, vec_t *inp):
    """Normalise the vector, writing to out. inp and out may be the same."""
    cdef double mag = _vec_mag(inp)

    if mag == 0:
        # Vec(0, 0, 0).norm = Vec(0, 0, 0), as a special case.
        out.x = out.y = out.z = 0
    else:
        # Disable ZeroDivisionError check, we just checked that.
        with cython.cdivision(True):
            out.x = inp.x / mag
            out.y = inp.y / mag
            out.z = inp.z / mag


cdef inline void _mat_mul(mat_t targ, mat_t rot):
    """Rotate target by the rotator matrix."""
    # We don't use each row after assigning to the set, so we can re-assign.
    targ[0][0], targ[0][1], targ[0][2] = (
        (targ[0][0]) * (rot[0][0]) + (targ[0][1]) * (rot[1][0]) + (targ[0][2]) * (rot[2][0]),
        targ[0][0] * rot[0][1] + targ[0][1] * rot[1][1] + targ[0][2] * rot[2][1],
        targ[0][0] * rot[0][2] + targ[0][1] * rot[1][2] + targ[0][2] * rot[2][2],
    )

    targ[1][0], targ[1][1], targ[1][2] = (
        targ[1][0] * rot[0][0] + targ[1][1] * rot[1][0] + targ[1][2] * rot[2][0],
        targ[1][0] * rot[0][1] + targ[1][1] * rot[1][1] + targ[1][2] * rot[2][1],
        targ[1][0] * rot[0][2] + targ[1][1] * rot[1][2] + targ[1][2] * rot[2][2],
    )

    targ[2][0], targ[2][1], targ[2][2] = (
        targ[2][0] * rot[0][0] + targ[2][1] * rot[1][0] + targ[2][2] * rot[2][0],
        targ[2][0] * rot[0][1] + targ[2][1] * rot[1][1] + targ[2][2] * rot[2][1],
        targ[2][0] * rot[0][2] + targ[2][1] * rot[1][2] + targ[2][2] * rot[2][2],
    )


cdef inline void _vec_rot(vec_t *vec, mat_t mat):
    """Rotate a vector by our value."""
    cdef double x = vec.x
    cdef double y = vec.y
    cdef double z = vec.z
    vec.x = (x * mat[0][0]) + (y * mat[1][0]) + (z * mat[2][0])
    vec.y = (x * mat[0][1]) + (y * mat[1][1]) + (z * mat[2][1])
    vec.z = (x * mat[0][2]) + (y * mat[1][2]) + (z * mat[2][2])


cdef inline void _vec_cross(vec_t *res, vec_t *a, vec_t *b):
    """Compute the cross product of A x B. """
    res.x = a.y * b.z - a.z * b.y
    res.y = a.z * b.x - a.x * b.z
    res.z = a.x * b.y - a.y * b.x


cdef void _mat_from_angle(mat_t res, vec_t *angle):
    cdef double cos_r_cos_y, cos_r_sin_y, sin_r_cos_y, sin_r_sin_y
    cdef double rad_pitch = deg_2_rad * angle.x
    cdef double cos_p = math.cos(rad_pitch)
    cdef double sin_p = math.sin(rad_pitch)
    cdef double rad_yaw = deg_2_rad * angle.y
    cdef double sin_y = math.sin(rad_yaw)
    cdef double cos_y = math.cos(rad_yaw)
    cdef double rad_roll = deg_2_rad * angle.z
    cdef double cos_r = math.cos(rad_roll)
    cdef double sin_r = math.sin(rad_roll)

    res[0][0] = cos_p * cos_y
    res[0][1] = cos_p * sin_y
    res[0][2] = -sin_p

    cos_r_cos_y = cos_r * cos_y
    cos_r_sin_y = cos_r * sin_y
    sin_r_cos_y = sin_r * cos_y
    sin_r_sin_y = sin_r * sin_y

    res[1][0] = sin_p * sin_r_cos_y - cos_r_sin_y
    res[1][1] = sin_p * sin_r_sin_y + cos_r_cos_y
    res[1][2] = sin_r * cos_p

    res[2][0] = sin_p * cos_r_cos_y + sin_r_sin_y
    res[2][1] = sin_p * cos_r_sin_y - sin_r_cos_y
    res[2][2] = cos_r * cos_p


cdef inline void _mat_to_angle(vec_t *ang, mat_t mat):
    # https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/mathlib/mathlib_base.cpp#L208
    cdef double horiz_dist = math.sqrt(mat[0][0]**2 + mat[0][1]**2)
    if horiz_dist > 0.001:
        ang.x = rad_2_deg * math.atan2(mat[0][1], mat[0][0])
        ang.y = rad_2_deg * math.atan2(-mat[0][2], horiz_dist)
        ang.z = rad_2_deg * math.atan2(mat[1][2], mat[2][2])
    else:
        # Vertical, gimbal lock (yaw=roll)...
        ang.x = rad_2_deg * math.atan2(-mat[1][0], mat[1][1])
        ang.y = rad_2_deg * math.atan2(-mat[0][2], horiz_dist)
        ang.z = 0.0  # Can't produce.

@cython.final
cdef class VecIter:
    """Implements iter(Vec)."""
    cdef Vec vec
    cdef unsigned char index

    def __cinit__(self, Vec vec not None):
        self.vec = vec
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 3:
            raise StopIteration
        self.index += 1
        if self.index == 1:
            return self.vec.val.x
        elif self.index == 2:
            return self.vec.val.y
        elif self.index == 3:
            # Drop our reference.
            ret = self.vec.val.z
            self.vec = None
            return ret
        
@cython.final
cdef class AngleIter:
    """Implements iter(Angle)."""
    cdef Angle ang
    cdef unsigned char index

    def __cinit__(self, Angle ang not None):
        self.ang = ang
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 3:
            raise StopIteration
        self.index += 1
        if self.index == 1:
            return self.ang.val.x
        elif self.index == 2:
            return self.ang.val.y
        elif self.index == 3:
            # Drop our reference.
            ret = self.ang.val.z
            self.ang = None
            return ret


@cython.freelist(16)
@cython.final
cdef class Vec:
    """A 3D Vector. This has most standard Vector functions.

    Many of the functions will accept a 3-tuple for comparison purposes.
    """
    # Various constants.
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
    }
    # Vectors pointing in all cardinal directions.
    # Tuple.__new__() can't be unpacked...
    N = north = y_pos = _make_tuple(0, 1, 0)
    S = south = y_neg = _make_tuple(0, -1, 0)
    E = east = x_pos = _make_tuple(1, 0, 0)
    W = west = x_neg = _make_tuple(-1, 0, 0)
    T = top = z_pos = _make_tuple(0, 0, 1)
    B = bottom = z_neg = _make_tuple(0, 0, -1)


    # This is a sub-struct, so we can pass pointers to it to other
    # functions.
    cdef vec_t val

    @property
    def x(self):
        """The X axis of the vector."""
        return self.val.x

    @x.setter
    def x(self, value):
        self.val.x = value

    @property
    def y(self):
        """The Y axis of the vector."""
        return self.val.y

    @y.setter
    def y(self, value):
        self.val.y = value

    @property
    def z(self):
        return self.val.z

    @z.setter
    def z(self, value):
        """The Z axis of the vector."""
        self.val.z = value

    def __init__ (
        self,
        x=0.0,
        y=0.0,
        z=0.0,
    ):
        """Create a Vector.

        All values are converted to Floats automatically.
        If no value is given, that axis will be set to 0.
        An iterable can be passed in (as the x argument), which will be
        used for x, y, and z.
        """
        cdef tuple tup
        if isinstance(x, float) or isinstance(x, int):
            self.val.x = x
            self.val.y = y
            self.val.z = z
        elif isinstance(x, Vec):
            self.val.x = (<Vec>x).val.x
            self.val.y = (<Vec>x).val.y
            self.val.z = (<Vec>x).val.z
        elif isinstance(x, tuple):
            tup = <tuple>x
            if len(tup) >= 1:
                self.val.x = tup[0]
            else:
                self.val.x = 0

            if len(tup) >= 2:
                self.val.y = tup[1]
            else:
                self.val.y = y

            if len(tup) >= 3:
                self.val.z = tup[2]
            else:
                self.val.z = z

        else:
            it = iter(x)
            try:
                self.val.x = next(it)
            except StopIteration:
                self.val.x = 0
                self.val.y = y
                self.val.z = z
                return

            try:
                self.val.y = next(it)
            except StopIteration:
                self.val.y = y
                self.val.z = z
                return

            try:
                self.val.z = next(it)
            except StopIteration:
                self.val.z = z

    def copy(self):
        """Create a duplicate of this vector."""
        return _vector(self.val.x, self.val.y, self.val.z)

    def __copy__(self):
        """Create a duplicate of this vector."""
        return _vector(self.val.x, self.val.y, self.val.z)

    def __reduce__(self):
        return unpickle_func, (self.val.x, self.val.y, self.val.z)

    @classmethod
    def from_str(cls, value, double x=0, double y=0, double z=0):
        """Convert a string in the form '(4 6 -4)' into a Vector.

        If the string is unparsable, this uses the defaults (x,y,z).
        The string can start with any of the (), {}, [], <> bracket
        types, or none.

        If the value is already a vector, a copy will be returned.
        """
        cdef Vec vec = Vec.__new__(Vec)
        _parse_vec_str(&vec.val, value, x, y, z)
        return vec

    @staticmethod
    @cython.boundscheck(False)
    def with_axes(*args) -> 'Vec':
        """Create a Vector, given a number of axes and corresponding values.

        This is a convenience for doing the following:
            vec = Vec()
            vec[axis1] = val1
            vec[axis2] = val2
            vec[axis3] = val3
        The magnitudes can also be Vectors, in which case the matching
        axis will be used from the vector.
        """
        cdef Py_ssize_t arg_count = len(args)
        if arg_count not in (2, 4, 6):
            raise TypeError(
                f'Vec.with_axis() takes 2, 4 or 6 positional arguments '
                f'but {arg_count} were given'
            )

        cdef Vec vec = Vec.__new__(Vec)
        cdef Py_UCS4 axis
        cdef unsigned char i
        for i in range(0, arg_count, 2):
            axis_val = args[i+1]
            axis_obj = args[i]
            if isinstance(axis_obj, str) and len(<str>axis_obj) == 1:
                axis = (<str>axis_obj)[0]
            else:
                raise KeyError(f'Invalid axis {axis_obj!r}' '!')
            if axis == 'x':
                if isinstance(axis_val, Vec):
                    vec.val.x = (<Vec>axis_val).val.x
                else:
                    vec.val.x = axis_val
            elif axis == 'y':
                if isinstance(axis_val, Vec):
                    vec.val.y = (<Vec>axis_val).val.y
                else:
                    vec.val.y = axis_val
            elif axis == 'z':
                if isinstance(axis_val, Vec):
                    vec.val.z = (<Vec>axis_val).val.z
                else:
                    vec.val.z = axis_val
            else:
                raise KeyError(f'Invalid axis {axis_obj!r}' '!')

        return vec

    def rotate(
        self,
        double pitch: float=0.0,
        double yaw: float=0.0,
        double roll: float=0.0,
        bint round_vals: bool=True,
    ) -> 'Vec':
        """Rotate a vector by a Source rotational angle.
        Returns the vector, so you can use it in the form
        val = Vec(0,1,0).rotate(p, y, r)

        If round is True, all values will be rounded to 3 decimals
        (since these calculations always have small inprecision.)
        """
        self._rotate(pitch, yaw, roll, round_vals)
        return self

    def rotate_by_str(
        self,
        ang,
        double pitch=0.0,
        double yaw=0.0,
        double roll=0.0,
        bint round_vals=True,
    ) -> 'Vec':
        """Rotate a vector, using a string instead of a vector.

        If the string cannot be parsed, use the passed in values instead.
        """
        cdef vec_t angle
        _parse_vec_str(&angle, ang, pitch, yaw, roll)
        self._rotate(
            angle.x,
            angle.y,
            angle.z,
            round_vals,
        )
        return self

    @staticmethod
    def bbox(*points: Vec) -> 'Tuple[Vec, Vec]':
        """Compute the bounding box for a set of points.

        Pass either several Vecs, or an iterable of Vecs.
        Returns a (min, max) tuple.
        """
        cdef Vec bbox_min = Vec.__new__(Vec)
        cdef Vec bbox_max = Vec.__new__(Vec)
        cdef Vec sing_vec
        cdef vec_t vec
        cdef Py_ssize_t i
        # Allow passing a single iterable, but also handle a single Vec.
        # The error messages match those produced by min()/max().

        if len(points) == 1:
            if isinstance(points[0], Vec):
                # Special case, don't iter over the vec, just copy.
                sing_vec = <Vec>points[0]
                bbox_min.val = sing_vec.val
                bbox_max.val = sing_vec.val
                return bbox_min, bbox_max
            points_iter = iter(points[0])
            try:
                first = next(points_iter)
            except StopIteration:
                raise ValueError('Empty iterator!') from None

            _conv_vec(&bbox_min.val, first, scalar=False)
            bbox_max.val = bbox_min.val

            try:
                while True:
                    point = next(points_iter)
                    _conv_vec(&vec, point, scalar=False)

                    if bbox_max.val.x < vec.x:
                        bbox_max.val.x = vec.x

                    if bbox_max.val.y < vec.y:
                        bbox_max.val.y = vec.y

                    if bbox_max.val.z < vec.z:
                        bbox_max.val.z = vec.z

                    if bbox_min.val.x > vec.x:
                        bbox_min.val.x = vec.x

                    if bbox_min.val.y > vec.y:
                        bbox_min.val.y = vec.y

                    if bbox_min.val.z > vec.z:
                        bbox_min.val.z = vec.z
            except StopIteration:
                pass
        elif len(points) == 0:
            raise TypeError(
                'Vec.bbox() expected at '
                'least 1 argument, got 0.'
            )
        else:
            # Tuple-specific.
            _conv_vec(&bbox_min.val, points[0], scalar=False)
            bbox_max.val = bbox_min.val

            for i in range(1, len(points)):
                point = points[i]
                _conv_vec(&vec, point, scalar=False)

                if bbox_max.val.x < vec.x:
                    bbox_max.val.x = vec.x

                if bbox_max.val.y < vec.y:
                    bbox_max.val.y = vec.y

                if bbox_max.val.z < vec.z:
                    bbox_max.val.z = vec.z

                if bbox_min.val.x > vec.x:
                    bbox_min.val.x = vec.x

                if bbox_min.val.y > vec.y:
                    bbox_min.val.y = vec.y

                if bbox_min.val.z > vec.z:
                    bbox_min.val.z = vec.z

        return bbox_min, bbox_max


    def axis(self) -> str:
        """For a normal vector, return the axis it is on."""
        if self.val.x != 0 and self.val.y == 0 and self.val.z == 0:
            return 'x'
        if self.val.x == 0 and self.val.y != 0 and self.val.z == 0:
            return 'y'
        if self.val.x == 0 and self.val.y == 0 and self.val.z != 0:
            return 'z'
        raise ValueError(
            f'({self.val.x:g}, {self.val.y:g}, {self.val.z:g}) is '
            'not an on-axis vector!'
        )

    @cython.boundscheck(False)
    def other_axes(self, object axis) -> 'Tuple[float, float]':
        """Get the values for the other two axes."""
        cdef char axis_chr
        if isinstance(axis, str) and len(<str>axis) == 1:
            axis_chr = (<str>axis)[0]
        else:
            raise KeyError(f'Invalid axis {axis!r}' '!')
        if axis_chr == b'x':
            return self.val.y, self.val.z
        elif axis_chr == b'y':
            return self.val.x, self.val.z
        elif axis_chr == b'z':
            return self.val.x, self.val.y
        else:
            raise KeyError(f'Invalid axis {axis!r}' '!')

    def as_tuple(self) -> 'Tuple[float, float, float]':
        """Return the Vector as a tuple."""
        # Use tuple.__new__(cls, iterable) instead of calling the
        # Python __new__.
        return _make_tuple(self.val.x, self.val.y, self.val.z)

    def to_angle(self, double roll: float=0):
        """Convert a normal to a Source Engine angle.

        A +x axis vector will result in a 0, 0, 0 angle. The roll is not
        affected by the direction of the normal.

        The inverse of this is `Vec(x=1).rotate(pitch, yaw, roll)`.
        """
        # Pitch is applied first, so we need to reconstruct the x-value.
        cdef double horiz_dist = math.sqrt(self.val.x ** 2 + self.val.y ** 2)

        return _angle(
            rad_2_deg * math.atan2(-self.val.z, horiz_dist),
            (math.atan2(self.val.y, self.val.x) * rad_2_deg) % 360.0,
            roll,
        )

    def rotation_around(self, double rot: float=90) -> 'Vec':
        """For an axis-aligned normal, return the angles which rotate around it."""
        cdef Vec vec = Vec.__new__(Vec)
        vec.val.x = vec.val.y = vec.val.z = 0.0

        if self.val.x != 0 and self.val.y == 0 and self.val.z == 0:
            vec.val.z = math.copysign(rot, self.val.x)
        elif self.val.x == 0 and self.val.y != 0 and self.val.z == 0:
            vec.val.x = math.copysign(rot, self.val.y)
        elif self.val.x == 0 and self.val.y == 0 and self.val.z != 0:
            vec.val.y = math.copysign(rot, self.val.z)
        else:
            raise ValueError(
                f'({self.val.x}, {self.val.y}, {self.val.z}) is '
                'not an on-axis vector!'
            )
        return vec

    def __abs__(self):
        """Performing abs() on a Vec takes the absolute value of all axes."""
        return _vector(
            abs(self.val.x),
            abs(self.val.y),
            abs(self.val.z),
        )

    def __neg__(self):
        """The inverted form of a Vector has inverted axes."""
        return _vector(
            -self.val.x,
            -self.val.y,
            -self.val.z,
        )

    def __pos__(self):
        """+ on a Vector simply copies it."""
        return _vector(
            self.val.x,
            self.val.y,
            self.val.z,
        )

    def __contains__(self, val) -> bool:
        """Check to see if an axis is set to the given value."""
        cdef double val_d
        try:
            val_d = val
        except (TypeError, ValueError): # Non-floats should return False!
            return False
        if val_d == self.val.x:
            return True
        if val_d == self.val.y:
            return True
        if val_d == self.val.z:
            return True
        return False

    # Non-in-place operators. Arg 1 may not be a Vec.

    def __add__(obj_a, obj_b):
        """+ operation.

        This additionally works on scalars (adds to all axes).
        """
        cdef vec_t vec_a, vec_b

        try:
            _conv_vec(&vec_a, obj_a, scalar=True)
            _conv_vec(&vec_b, obj_b, scalar=True)
        except (TypeError, ValueError):
            return NotImplemented

        cdef Vec result = Vec.__new__(Vec)
        result.val.x = vec_a.x + vec_b.x
        result.val.y = vec_a.y + vec_b.y
        result.val.z = vec_a.z + vec_b.z
        return result

    def __sub__(obj_a, obj_b):
        """- operation.

        This additionally works on scalars (adds to all axes).
        """
        cdef vec_t vec_a, vec_b

        try:
            _conv_vec(&vec_a, obj_a, scalar=True)
            _conv_vec(&vec_b, obj_b, scalar=True)
        except (TypeError, ValueError):
            return NotImplemented

        cdef Vec result = Vec.__new__(Vec)
        result.val.x = vec_a.x - vec_b.x
        result.val.y = vec_a.y - vec_b.y
        result.val.z = vec_a.z - vec_b.z
        return result

    def __mul__(obj_a, obj_b):
        """Vector * scalar operation."""
        cdef Vec vec = Vec.__new__(Vec)
        cdef double scalar
        # Vector * Vector is disallowed.
        if isinstance(obj_a, (int, float)):
            # scalar * vector
            scalar = obj_a
            _conv_vec(&vec.val, obj_b, scalar=False)
            vec.val.x = scalar * vec.val.x
            vec.val.y = scalar * vec.val.y
            vec.val.z = scalar * vec.val.z
        elif isinstance(obj_b, (int, float)):
            # vector * scalar.
            _conv_vec(&vec.val, obj_a, scalar=False)
            scalar = obj_b
            vec.val.x = vec.val.x * scalar
            vec.val.y = vec.val.y * scalar
            vec.val.z = vec.val.z * scalar

        elif isinstance(obj_a, Vec) and isinstance(obj_b, Vec):
            raise TypeError('Cannot multiply 2 Vectors.')
        else:
            # Both vector-like or vector * something else.
            return NotImplemented
        return vec

    def __truediv__(obj_a, obj_b):
        """Vector / scalar operation."""
        cdef Vec vec = Vec.__new__(Vec)
        cdef double scalar
        # Vector / Vector is disallowed.
        if isinstance(obj_a, (int, float)):
            # scalar / vector
            scalar = obj_a
            _conv_vec(&vec.val, obj_b, scalar=False)
            vec.val.x = scalar / vec.val.x
            vec.val.y = scalar / vec.val.y
            vec.val.z = scalar / vec.val.z
        elif isinstance(obj_b, (int, float)):
            # vector / scalar.
            _conv_vec(&vec.val, obj_a, scalar=False)
            scalar = obj_b
            vec.val.x = vec.val.x / scalar
            vec.val.y = vec.val.y / scalar
            vec.val.z = vec.val.z / scalar

        elif isinstance(obj_a, Vec) and isinstance(obj_b, Vec):
            raise TypeError('Cannot divide 2 Vectors.')
        else:
            # Both vector-like or vector * something else.
            return NotImplemented
        return vec

    def __floordiv__(obj_a, obj_b):
        """Vector // scalar operation."""
        cdef Vec vec = Vec.__new__(Vec)
        cdef double scalar
        # Vector // Vector is disallowed.
        if isinstance(obj_a, (int, float)):
            # scalar // vector
            scalar = obj_a
            _conv_vec(&vec.val, obj_b, scalar=False)
            vec.val.x = scalar // vec.val.x
            vec.val.y = scalar // vec.val.y
            vec.val.z = scalar // vec.val.z
        elif isinstance(obj_b, (int, float)):
            # vector // scalar.
            _conv_vec(&vec.val, obj_a, scalar=False)
            scalar = obj_b
            vec.val.x = vec.val.x // scalar
            vec.val.y = vec.val.y // scalar
            vec.val.z = vec.val.z // scalar

        elif isinstance(obj_a, Vec) and isinstance(obj_b, Vec):
            raise TypeError('Cannot floor-divide 2 Vectors.')
        else:
            # Both vector-like or vector * something else.
            return NotImplemented
        return vec

    def __mod__(obj_a, obj_b):
        """Vector % scalar operation."""
        cdef Vec vec = Vec.__new__(Vec)
        cdef double scalar
        # Vector % Vector is disallowed.
        if isinstance(obj_a, (int, float)):
            # scalar % vector
            scalar = obj_a
            _conv_vec(&vec.val, obj_b, scalar=False)
            vec.val.x = scalar % vec.val.x
            vec.val.y = scalar % vec.val.y
            vec.val.z = scalar % vec.val.z
        elif isinstance(obj_b, (int, float)):
            # vector % scalar.
            _conv_vec(&vec.val, obj_a, scalar=False)
            scalar = obj_b
            vec.val.x = vec.val.x % scalar
            vec.val.y = vec.val.y % scalar
            vec.val.z = vec.val.z % scalar

        elif isinstance(obj_a, Vec) and isinstance(obj_b, Vec):
            raise TypeError('Cannot modulus 2 Vectors.')
        else:
            # Both vector-like or vector * something else.
            return NotImplemented
        return vec

    def __matmul__(first, second):
        cdef mat_t temp
        cdef Vec res
        if isinstance(first, Vec):
            res = Vec.__new__(Vec)
            res.val = (<Vec>first).val
            if isinstance(second, Angle):
                _mat_from_angle(temp, &(<Angle>second).val)
                _vec_rot(&res.val, temp)
                return res
            elif isinstance(second, Matrix):
                _vec_rot(&res.val, (<Matrix>second).mat)
                return res
        return NotImplemented

    # In-place operators. Self is always a Vec.

    def __iadd__(self, other: 'Union[Vec, tuple, float]'):
        """+= operation.

        Like the normal one except without duplication.
        """
        cdef vec_t vec_other
        try:
            _conv_vec(&vec_other, other, scalar=True)
        except (TypeError, ValueError):
            return NotImplemented

        self.val.x += vec_other.x
        self.val.y += vec_other.y
        self.val.z += vec_other.z

        return self

    def __isub__(self, other: 'Union[Vec, tuple, float]'):
        """-= operation.

        Like the normal one except without duplication.
        """
        cdef vec_t vec_other
        try:
            _conv_vec(&vec_other, other, scalar=True)
        except (TypeError, ValueError):
            return NotImplemented

        self.val.x -= vec_other.x
        self.val.y -= vec_other.y
        self.val.z -= vec_other.z

        return self

    def __imul__(self, object other: float):
        """*= operation.

        Like the normal one except without duplication.
        """
        cdef double scalar
        if isinstance(other, (int, float)):
            scalar = other
            self.val.x *= scalar
            self.val.y *= scalar
            self.val.z *= scalar
            return self
        elif isinstance(other, Vec):
            raise TypeError("Cannot multiply 2 Vectors.")
        else:
            return NotImplemented

    def __itruediv__(self, other: float):
        """/= operation.

        Like the normal one except without duplication.
        """
        cdef double scalar
        if isinstance(other, (int, float)):
            scalar = other
            self.val.x /= scalar
            self.val.y /= scalar
            self.val.z /= scalar
            return self
        elif isinstance(other, Vec):
            raise TypeError("Cannot divide 2 Vectors.")
        else:
            return NotImplemented

    def __ifloordiv__(self, other: float):
        """//= operation.

        Like the normal one except without duplication.
        """
        cdef double scalar
        if isinstance(other, (int, float)):
            scalar = other
            self.val.x //= scalar
            self.val.y //= scalar
            self.val.z //= scalar
            return self
        elif isinstance(other, Vec):
            raise TypeError("Cannot floor-divide 2 Vectors.")
        else:
            return NotImplemented

    def __imod__(self, other: float):
        """%= operation.

        Like the normal one except without duplication.
        """
        cdef double scalar
        if isinstance(other, (int, float)):
            scalar = other
            self.val.x %= scalar
            self.val.y %= scalar
            self.val.z %= scalar
            return self
        elif isinstance(other, Vec):
            raise TypeError("Cannot modulus 2 Vectors.")
        else:
            return NotImplemented

    def __imatmul__(self, other):
        """@= operation: rotate the vector by a matrix/angle."""
        cdef mat_t temp
        if isinstance(other, Angle):
            _mat_from_angle(temp, &(<Angle>other).val)
            _vec_rot(&self.val, temp)
        elif isinstance(other, Matrix):
            _vec_rot(&self.val, (<Matrix>other).mat)
        else:
            return NotImplemented
        return self

    def __divmod__(obj_a, obj_b):
        """Divide the vector by a scalar, returning the result and remainder."""
        cdef Vec vec
        cdef Vec res_1 = Vec.__new__(Vec)
        cdef Vec res_2 = Vec.__new__(Vec)
        cdef double other_d

        if isinstance(obj_a, Vec) and isinstance(obj_b, Vec):
            raise TypeError("Cannot divide 2 Vectors.")
        elif isinstance(obj_a, Vec):
            # vec / val
            vec = <Vec>obj_a
            try:
                other_d = <double ?>obj_b
            except TypeError:
                return NotImplemented

            # We put % first, since Cython then produces a 'divmod' error.

            res_2.val.x = vec.val.x % other_d
            res_1.val.x = vec.val.x // other_d
            res_2.val.y = vec.val.y % other_d
            res_1.val.y = vec.val.y // other_d
            res_2.val.z = vec.val.z % other_d
            res_1.val.z = vec.val.z // other_d
        elif isinstance(obj_b, Vec):
            # val / vec
            vec = <Vec>obj_b
            try:
                other_d = <double ?>obj_a
            except TypeError:
                return NotImplemented

            res_2.val.x = other_d % vec.val.x
            res_1.val.x = other_d // vec.val.x
            res_2.val.y = other_d % vec.val.y
            res_1.val.y = other_d // vec.val.y
            res_2.val.z = other_d % vec.val.z
            res_1.val.z = other_d // vec.val.z
        else:
            raise TypeError("Called with non-vectors??")

        return res_1, res_2

    def __bool__(self) -> bool:
        """Vectors are True if any axis is non-zero."""
        if self.val.x != 0:
            return True
        if self.val.y != 0:
            return True
        if self.val.z != 0:
            return True
        return False

    def __len__(self):
        """The len() of a vector is the number of non-zero axes."""
        return (
            (self.val.x != 0) +
            (self.val.y != 0) +
            (self.val.z != 0)
        )

    # All the comparisons are similar, so we can use richcmp to
    # nicely combine the parsing code.
    def __richcmp__(self, other_obj, int op):
        """Rich Comparisons.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        """
        cdef vec_t other
        try:
            _conv_vec(&other, other_obj, False)
        except (TypeError, ValueError):
            return NotImplemented

        if op == Py_EQ:
            return (
                self.val.x == other.x and
                self.val.y == other.y and
                self.val.z == other.z
            )
        elif op == Py_NE:
            return (
                self.val.x != other.x or
                self.val.y != other.y or
                self.val.z != other.z
            )
        elif op == Py_LT:
            return (
                self.val.x < other.x and
                self.val.y < other.y and
                self.val.z < other.z
            )
        elif op == Py_GT:
            return (
                self.val.x > other.x and
                self.val.y > other.y and
                self.val.z > other.z
            )
        elif op == Py_LE:
            return (
                self.val.x <= other.x and
                self.val.y <= other.y and
                self.val.z <= other.z
            )
        elif op == Py_GE:
            return (
                self.val.x >= other.x and
                self.val.y >= other.y and
                self.val.z >= other.z
            )
        else:
            raise RuntimeError('Bad operation!')

    def max(self, other):
        """Set this vector's values to the maximum of the two vectors."""
        cdef vec_t vec
        _conv_vec(&vec, other, scalar=False)
        if self.val.x < vec.x:
            self.val.x = vec.x

        if self.val.y < vec.y:
            self.val.y = vec.y

        if self.val.z < vec.z:
            self.val.z = vec.z

    def min(self, other):
        """Set this vector's values to be the minimum of the two vectors."""
        cdef vec_t vec
        _conv_vec(&vec, other, scalar=False)
        if self.val.x > vec.x:
            self.val.x = vec.x

        if self.val.y > vec.y:
            self.val.y = vec.y

        if self.val.z > vec.z:
            self.val.z = vec.z

    def __round__(self, object n=0):
        """Performing round() on a Vec rounds each axis."""
        cdef Vec vec = Vec.__new__(Vec)

        vec.val.x = round(self.val.x, n)
        vec.val.y = round(self.val.y, n)
        vec.val.z = round(self.val.z, n)

        return vec

    cdef inline double _mag_sq(self):
        return self.val.x**2 + self.val.y**2 + self.val.z**2

    def mag_sq(self):
        """Compute the distance from the vector and the origin."""
        return self._mag_sq()

    def len_sq(self):
        """Compute the distance from the vector and the origin."""
        return self._mag_sq()

    def mag(self):
        """Compute the distance from the vector and the origin."""
        return _vec_mag(&self.val)

    def len(self):
        """Compute the distance from the vector and the origin."""
        return _vec_mag(&self.val)

    def norm(self):
        """Normalise the Vector.

         This is done by transforming it to have a magnitude of 1 but the same
         direction.
         The vector is left unchanged if it is equal to (0,0,0).
         """
        cdef Vec vec = Vec.__new__(Vec)
        _vec_normalise(&vec.val, &self.val)
        return vec

    def norm_mask(self, normal: 'Vec') -> 'Vec':
        """Subtract the components of this vector not in the direction of the normal.

        If the normal is axis-aligned, this will zero out the other axes.
        If not axis-aligned, it will do the equivalent.
        """
        cdef vec_t norm

        _conv_vec(&norm, normal, False)

        _vec_normalise(&norm, &norm)

        cdef double dot = (
            self.val.x * norm.x +
            self.val.y * norm.y +
            self.val.z * norm.z
        )

        return _vector(
            norm.x * dot,
            norm.y * dot,
            norm.z * dot,
        )


    def dot(self, other) -> float:
        """Return the dot product of both Vectors."""
        cdef vec_t oth

        _conv_vec(&oth, other, False)

        return (
            self.val.x * oth.x +
            self.val.y * oth.y +
            self.val.z * oth.z
        )

    def cross(self, other) -> 'Vec':
        """Return the cross product of both Vectors."""
        cdef vec_t oth
        cdef Vec res

        _conv_vec(&oth, other, False)
        res = Vec.__new__(Vec)
        _vec_cross(&res.val, &self.val, &oth)
        return res


    def join(self, delim: str=', ') -> str:
        """Return a string with all numbers joined by the passed delimiter.

        This strips off the .0 if no decimal portion exists.
        """
        # :g strips the .0 off of floats if it's an integer.
        return f'{self.val.x:g}{delim}{self.val.y:g}{delim}{self.val.z:g}'

    def __str__(self):
        """Return the values, separated by spaces.

        This is the main format in Valve's file formats.
        This strips off the .0 if no decimal portion exists.
        """
        return f"{self.val.x:g} {self.val.y:g} {self.val.z:g}"

    def __repr__(self):
        """Code required to reproduce this vector."""
        return f"Vec({self.val.x:g}, {self.val.y:g}, {self.val.z:g})"

    def __iter__(self):
        return VecIter.__new__(VecIter, self)

    def __getitem__(self, ind_obj: 'Union[str, int]') -> float:
        """Allow reading values by index instead of name if desired.

        This accepts either 0,1,2 or 'x','y','z' to read values.
        Useful in conjunction with a loop to apply commands to all values.
        """
        cdef Py_UCS4 ind
        if isinstance(ind_obj, int):
            try:
                ind = ind_obj
            except (TypeError, ValueError, OverflowError):
                pass
            else:
                if ind == 0:
                    return self.val.x
                elif ind == 1:
                    return self.val.y
                elif ind == 2:
                    return self.val.z
            raise KeyError(f'Invalid axis: {ind!r}' '!')
        else:
            if isinstance(ind_obj, str) and len(<str>ind_obj) == 1:
                ind = (<str>ind_obj)[0]
            else:
                raise KeyError(f'Invalid axis {ind_obj!r}' '!')

            if ind == "x":
                return self.val.x
            elif ind == "y":
                return self.val.y
            elif ind == "z":
                return self.val.z
            else:
                raise KeyError(f'Invalid axis {ind_obj!r}' '!')


    def __setitem__(self, ind_obj, double val: float) -> None:
        """Allow editing values by index instead of name if desired.

        This accepts either 0,1,2 or 'x','y','z' to edit values.
        Useful in conjunction with a loop to apply commands to all values.
        """
        cdef Py_UCS4 ind

        if isinstance(ind_obj, int):
            try:
                ind = ind_obj
            except (TypeError, ValueError, OverflowError):
                pass
            else:
                if ind == 0:
                    self.val.x = val
                    return
                elif ind == 1:
                    self.val.y = val
                    return
                elif ind == 2:
                    self.val.z = val
                    return
            raise KeyError(f'Invalid axis: {ind!r}')
        else:
            if isinstance(ind_obj, str) and len(<str>ind_obj) == 1:
                ind = (<str>ind_obj)[0]
            else:
                raise KeyError(f'Invalid axis {ind_obj!r}' '!')

            if ind == "x":
                self.val.x = val
            elif ind == "y":
                self.val.y = val
            elif ind == "z":
                self.val.z = val
            else:
                raise KeyError(f'Invalid axis {ind_obj!r}' '!')


    # @contextlib.contextmanager
    # def transform(self) -> ContextManager['Matrix']:
    #     """Perform rotations on this Vector efficiently.
    #
    #     Used as a context manager, which returns a matrix.
    #     When the body is exited safely, the matrix is applied to
    #     the angle.
    #     """
    #     mat = Matrix()
    #     yield mat
    #     mat._vec_rot(self)


@cython.freelist(16)
@cython.final
cdef class Matrix:
    """Represents a matrix via a transformation matrix."""
    cdef mat_t mat

    def __init__(self) -> None:
        """Create a matrix set to the identity transform."""
        self.mat[0] = [1.0, 0.0, 0.0]
        self.mat[1] = [0.0, 1.0, 0.0]
        self.mat[2] = [0.0, 0.0, 1.0]

    def __eq__(self, other: object) -> object:
        if isinstance(other, Matrix):
            return self.mat == (<Matrix>other).mat
        return NotImplemented

    def __repr__(self) -> str:
        return (
            '<Matrix '
            f'{self[0][0]:.3} {self[0][1]:.3} {self[0][2]:.3}, '
            f'{self[1][0]:.3} {self[1][1]:.3} {self[1][2]:.3}, '
            f'{self[2][0]:.3} {self[2][1]:.3} {self[2][2]:.3}'
            '>'
        )

    def copy(self) -> 'Matrix':
        """Duplicate this matrix."""
        cdef Matrix copy = Matrix.__new__(Matrix)
        memcpy(copy.mat, self.mat, sizeof(mat_t))
        return copy

    @classmethod
    def from_pitch(cls, double pitch):
        """Return the matrix representing a pitch rotation.

        This is a rotation around the Y axis.
        """
        cdef double rad_pitch = deg_2_rad * pitch
        cdef double cos = math.cos(rad_pitch)
        cdef double sin = math.sin(rad_pitch)

        cdef Matrix rot = cls.__new__(cls)

        rot.mat[0] = cos, 0.0, -sin
        rot.mat[1] = 0.0, 1.0, 0.0
        rot.mat[2] = sin, 0.0, cos

        return rot

    @classmethod
    def from_yaw(cls, double yaw):
        """Return the matrix representing a yaw rotation.

        """
        cdef double rad_yaw = deg_2_rad * yaw
        cdef double sin = math.sin(rad_yaw)
        cdef double cos = math.cos(rad_yaw)

        cdef Matrix rot = cls.__new__(cls)

        rot.mat[0] = cos, sin, 0.0
        rot.mat[1] = -sin, cos, 0.0
        rot.mat[2] = 0.0, 0.0, 1.0

        return rot

    @classmethod
    def from_roll(cls, double roll):
        """Return the matrix representing a roll rotation.

        This is a rotation around the X axis.
        """
        cdef double rad_roll = deg_2_rad * roll
        cdef double cos = math.cos(rad_roll)
        cdef double sin = math.sin(rad_roll)

        cdef Matrix rot = cls.__new__(cls)

        rot.mat[0] = [1.0, 0.0, 0.0]
        rot.mat[1] = [0.0, cos, sin]
        rot.mat[2] = [0.0, -sin, cos]

        return rot

    @classmethod
    def from_angle(cls, angle):
        """Return the rotation representing an Euler angle."""
        cdef Matrix rot = Matrix.__new__(cls)
        cdef vec_t ang
        _conv_angles(&ang, angle, scalar=False)
        _mat_from_angle(rot.mat, &ang)
        return rot

    def forward(self):
        """Return a normalised vector pointing in the +X direction."""
        return _vector(self.mat[0][0], self.mat[0][1], self.mat[0][2])

    def left(self):
        """Return a normalised vector pointing in the +Y direction."""
        return _vector(self.mat[1][0], self.mat[1][1], self.mat[1][2])

    def up(self):
        """Return a normalised vector pointing in the +Z direction."""
        return _vector(self.mat[2][0], self.mat[2][1], self.mat[2][2])

    def to_angle(self):
        """Return an Euler angle replicating this rotation."""
        cdef Angle ang = Angle.__new__(Angle)
        _mat_to_angle(&ang.val, self.mat)
        return ang

    def transpose(self) -> 'Matrix':
        """Return the transpose of this matrix."""
        cdef Matrix rot = Matrix.__new__(Matrix)

        rot.mat[0] = self.mat[0][0], self.mat[1][0], self.mat[2][0]
        rot.mat[1] = self.mat[0][1], self.mat[1][1], self.mat[2][1]
        rot.mat[2] = self.mat[0][2], self.mat[1][2], self.mat[2][2]

        return rot

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
        cdef Matrix mat = Matrix.__new__(cls)
        cdef vec_t res

        if x is None:
            if y is not None and z is not None:
                _vec_cross(&res, &y.val, &z.val)
            else:
                raise TypeError('At least two vectors must be provided!')
        else:
            res = x.val

        _vec_normalise(&res, &res)
        mat.mat[0] = res.x, res.y, res.z

        if y is None:
            if x is not None and z is not None:
                _vec_cross(&res, &z.val, &x.val)
            else:
                raise TypeError('At least two vectors must be provided!')
        else:
            res = y.val

        _vec_normalise(&res, &res)
        mat.mat[1] = res.x, res.y, res.z

        if z is None:
            if x is not None and y is not None:
                _vec_cross(&res, &x.val, &y.val)
            else:
                raise TypeError('At least two vectors must be provided!')
        else:
            res = z.val

        _vec_normalise(&res, &res)
        mat.mat[2] = res.x, res.y, res.z

        return mat

    def __matmul__(first, second):
        """Rotate two objects."""
        cdef mat_t temp, temp2
        cdef Vec vec
        cdef Matrix mat
        cdef Angle ang
        if isinstance(first, Matrix):
            mat = Matrix.__new__(Matrix)
            memcpy(mat.mat, (<Matrix>first).mat, sizeof(mat_t))
            if isinstance(second, Matrix):
                _mat_mul(mat.mat, (<Matrix>second).mat)
            elif isinstance(second, Angle):
                _mat_from_angle(temp, &(<Angle>second).val)
                _mat_mul(mat.mat, temp)
            else:
                return NotImplemented
            return mat
        elif isinstance(second, Matrix):
            if isinstance(first, Vec):
                vec = Vec.__new__(Vec)
                _vec_rot(&vec.val, (<Matrix>second).mat)
                return vec
            elif isinstance(first, Angle):
                ang = Angle.__new__(Angle)
                _mat_from_angle(temp, &(<Angle>first).val)
                _mat_mul(temp, (<Matrix>second).mat)
                _mat_to_angle(&ang.val, temp)
                return ang
            else:
                return NotImplemented
        else:
            raise SystemError('Neither are Matrices?')

    def __imatmul__(self, other):
        cdef mat_t temp
        if isinstance(other, Matrix):
            _mat_mul(self.mat, (<Matrix>other).mat)
            return self
        elif isinstance(other, Angle):
            _mat_from_angle(temp, &(<Angle>other).val)
            _mat_mul(self.mat, temp)
            return self
        else:
            return NotImplemented


# Lots of temporaries are expected.
@cython.freelist(16)
@cython.final
cdef class Angle:
    """Represents a pitch-yaw-roll Euler angle.

    All values are remapped to between 0-360 when set.
    Addition and subtraction modify values, matrix-multiplication with
    Vec, Angle or Matrix rotates (RHS rotating LHS).
    """
    # We have to double-modulus because -1e-14 % 360.0 = 360.0.
    cdef vec_t val

    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0) -> None:
        """Create an Angle.

        All values are converted to Floats automatically.
        If no value is given, that axis will be set to 0.
        An iterable can be passed in (as the pitch argument), which will be
        used for pitch, yaw, and roll. This includes Vectors and other Angles.
        """
        cdef tuple tup
        if isinstance(pitch, float) or isinstance(pitch, int):
            self.val.x = <double>pitch % 360 % 360.0
            self.val.y = <double>yaw % 360 % 360.0
            self.val.z = <double>roll % 360 % 360.0
        elif isinstance(pitch, Angle):
            self.val.x = (<Angle>pitch).val.x
            self.val.y = (<Angle>pitch).val.y
            self.val.z = (<Angle>pitch).val.z
        elif isinstance(pitch, tuple):
            tup = <tuple>pitch
            if len(tup) >= 1:
                self.val.x = <double?>(tup[0]) % 360 % 360.0
            else:
                self.val.x = 0.0

            if len(tup) >= 2:
                self.val.y = <double?>(tup[1]) % 360 % 360.0
            else:
                self.val.y = yaw

            if len(tup) >= 3:
                self.val.z = <double?>(tup[2]) % 360 % 360.0
            else:
                self.val.z = roll

        else:
            it = iter(pitch)
            try:
                self.val.x = next(it)
            except StopIteration:
                self.val.x = 0
                self.val.y = <double?>yaw % 360 % 360.0
                self.val.z = <double?>roll % 360 % 360.0
                return

            try:
                self.val.y = next(it)
            except StopIteration:
                self.val.y = <double?>yaw % 360 % 360.0
                self.val.z = <double?>roll % 360 % 360.0
                return

            try:
                self.val.z = next(it)
            except StopIteration:
                self.val.z = <double?>roll % 360 % 360.0

    def copy(self) -> 'Angle':
        """Create a duplicate of this vector."""
        return _angle(self.val.x, self.val.y, self.val.z)

    @classmethod
    def from_str(cls, val, double pitch=0.0, double yaw=0.0, double roll=0.0):
        """Convert a string in the form '(4 6 -4)' into an Angle.

        If the string is unparsable, this uses the defaults.
        The string can start with any of the (), {}, [], <> bracket
        types, or none.

        If the value is already a Angle, a copy will be returned.
        """
        cdef Angle ang = Angle.__new__(Angle)
        _parse_vec_str(&ang.val, val, pitch, yaw, roll)
        return ang

    @property
    def pitch(self) -> float:
        """The Y-axis rotation, performed second."""
        return self.val.x

    @pitch.setter
    def pitch(self, double pitch) -> None:
        self.val.y = pitch % 360 % 360.0

    @property
    def yaw(self) -> float:
        """The Z-axis rotation, performed last."""
        return self.val.y

    @yaw.setter
    def yaw(self, double yaw) -> None:
        self.val.y = yaw % 360.0 % 360.0

    @property
    def roll(self) -> float:
        """The X-axis rotation, performed first."""
        return self.val.z

    @roll.setter
    def roll(self, double roll) -> None:
        self.val.z = roll % 360.0 % 360.0

    def __str__(self) -> str:
        """Return the values, separated by spaces.

        This is the main format in Valve's file formats, though identical to
        vectors.
        This strips off the .0 if no decimal portion exists.
        """
        return f"{self.val.x:g} {self.val.y:g} {self.val.z:g}"

    def __repr__(self) -> str:
        return f'Angle({self.val.x:g}, {self.val.y:g}, {self.val.z:g})'

    def as_tuple(self):
        """Return the Angle as a tuple."""
        return Vec_tuple(self.val.x, self.val.y, self.val.z)

    def __iter__(self):
        """Iterating over the angles returns each value in turn."""
        return AngleIter.__new__(AngleIter)


    @staticmethod
    @cython.boundscheck(False)
    def with_axes(*args):
        """Create an Angle, given a number of axes and corresponding values.

        This is a convenience for doing the following:
            vec = Angle()
            vec[axis1] = val1
            vec[axis2] = val2
            vec[axis3] = val3
        The magnitudes can also be Vectors, in which case the matching
        axis will be used from the vector.
        """
        cdef Py_ssize_t arg_count = len(args)
        if arg_count not in (2, 4, 6):
            raise TypeError(
                f'Angle.with_axis() takes 2, 4 or 6 positional arguments '
                f'but {arg_count} were given'
            )

        cdef Angle ang = Angle.__new__(Angle)
        cdef str axis
        cdef unsigned char i
        for i in range(0, arg_count, 2):
            axis_val = args[i+1]
            axis = args[i]
            if axis in ('p', 'pit', 'pitch'):
                if isinstance(axis_val, Angle):
                    ang.val.x = (<Angle>axis_val).val.x
                else:
                    ang.val.x = (<double?>axis_val) % 360 % 360
            elif axis in ('y', 'yaw'):
                if isinstance(axis_val, Angle):
                    ang.val.y = (<Angle>axis_val).val.y
                else:
                    ang.val.y = (<double?>axis_val) % 360 % 360
            elif axis in ('r', 'rol', 'roll'):
                if isinstance(axis_val, Angle):
                    ang.val.z = (<Angle>axis_val).val.z
                else:
                    ang.val.z = (<double?>axis_val) % 360 % 360

        return ang

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

    def __getitem__(self, pos):
        """Allow reading values by index instead of name if desired.

        This accepts the following indexes to read values:
        - 0, 1, 2
        - pitch, yaw, roll
        - pit, yaw, rol
        - p, y, r
        Useful in conjunction with a loop to apply commands to all values.
        """
        cdef str key
        cdef int index

        if isinstance(pos, int):
            index = <int>pos
            if pos == 0:
                return self.val.x
            if pos == 1:
                return self.val.y
            if pos == 2:
                return self.val.z
        elif isinstance(pos, str):
            key = <str>pos
            if key in ('p', 'pit', 'pitch'):
                return self.val.x
            elif key in ('y', 'yaw'):
                return self.val.y
            elif key in ('r', 'rol', 'roll'):
                return self.val.z
        raise KeyError(f'Invalid axis: {pos!r}')

    def __setitem__(self, pos, double val) -> None:
        """Allow editing values by index instead of name if desired.

        This accepts either 0,1,2 or 'x','y','z' to edit values.
        Useful in conjunction with a loop to apply commands to all values.
        """
        cdef str key
        cdef int index
        val = val % 360.0 % 360.0

        if isinstance(pos, int):
            index = pos
            if pos == 0:
                self.val.x = val
            if pos == 1:
                self.val.y = val
            if pos == 2:
                self.val.z = val
        elif isinstance(pos, str):
            key = <str>pos
            if key in ('p', 'pit', 'pitch'):
                self.val.x = val
            elif key in ('y', 'yaw'):
                self.val.y = val
            elif key in ('r', 'rol', 'roll'):
                self.val.z = val
        raise KeyError(f'Invalid axis: {pos!r}')

    def __mul__(first, second):
        """Angle * float multiplies each value."""
        cdef double scalar
        cdef Angle angle
        if isinstance(first, Angle) and isinstance(second, (int, float)):
            scalar = second
            angle = first
        elif isinstance(first, (int, float)) and isinstance(second, Angle):
            scalar = first
            angle = second
        else:
            return NotImplemented
        return _angle(
            (angle.val.x * scalar) % 360.0 % 360.0,
            (angle.val.y * scalar) % 360.0 % 360.0,
            (angle.val.z * scalar) % 360.0 % 360.0,
        )

    def __matmul__(first, second):
        cdef mat_t temp1, temp2
        if isinstance(first, Angle):
            _mat_from_angle(temp1, &(<Angle>first).val)
            if isinstance(second, Angle):
                _mat_from_angle(temp2, &(<Angle>second).val)
                _mat_mul(temp1, temp2)
            elif isinstance(second, Matrix):
                _mat_mul(temp1, (<Matrix>second).mat)
            else:
                return NotImplemented
            res = Angle.__new__(Angle)
            _mat_to_angle(&(<Angle>res).val, temp1)
            return res
        elif isinstance(second, Angle):
            # These classes should do this themselves, but this is here for
            # completeness.
            _mat_from_angle(temp2, &(<Angle>second).val)
            if isinstance(first, Matrix):
                res = Matrix.__new__(Matrix)
                memcpy((<Matrix>res).mat, (<Matrix>first).mat, sizeof(mat_t))
                _mat_mul((<Matrix>res).mat, temp2)
                return res
            elif isinstance(first, Vec):
                res = Vec.__new__(Vec)
                memcpy(&(<Vec>res).val, &(<Vec>first).val, sizeof(vec_t))
                _vec_rot(&(<Vec>res).val, temp2)
                return res

        return NotImplemented

    # @contextlib.contextmanager
    # def transform(self) -> ContextManager[Matrix]:
    #     """Perform transformations on this angle.
    #
    #     Used as a context manager, which returns a matrix.
    #     When the body is exited safely, the matrix is applied to
    #     the angle.
    #     """
    #     mat = Matrix()
    #     yield mat
    #     new_ang = mat.to_angle()
    #     self._pitch = new_ang._pitch
    #     self._yaw = new_ang._yaw
    #     self._roll = new_ang._roll
