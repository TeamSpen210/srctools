# cython: language_level=3, auto_pickle=False, binding=True, c_api_binop_methods=True
# """Optimised Vector object."""
from libc cimport math
from libc.math cimport sin, cos, tan, llround, NAN
from libc.string cimport memcpy, memcmp, memset
from libc.stdint cimport uint_fast8_t
from libc.stdio cimport sscanf
from cpython.object cimport PyObject, PyTypeObject, Py_LT, Py_LE, Py_EQ, Py_NE, Py_GT, Py_GE
from cpython.ref cimport Py_INCREF
from cpython.exc cimport PyErr_WarnEx
from libcpp.vector cimport vector
from srctools cimport quickhull
cimport cython.operator

cdef extern from *:
    const char* PyUnicode_AsUTF8AndSize(str string, Py_ssize_t *size) except NULL

cdef inline Vec _vector(double x, double y, double z):
    """Make a Vector directly."""
    cdef Vec vec = Vec.__new__(Vec)
    vec.val.x = x
    vec.val.y = y
    vec.val.z = z
    return vec

cdef inline Angle _angle(double pitch, double yaw, double roll):
    """Make an Angle directly."""
    cdef Angle ang = Angle.__new__(Angle)
    ang.val.x = pitch
    ang.val.y = yaw
    ang.val.z = roll
    return ang

cdef object Tuple, Iterator, Union  # Keep private.
from typing import Tuple, Iterator, Union

# Shared functions that we use to do unpickling.
# It's defined in the Python module, so all versions
# produce the same pickle value.
cdef object unpickle_vec, unpickle_ang, unpickle_mat

# Grab the Vec_Tuple class for quick construction as well
cdef object Vec_tuple
from srctools.math import Vec_tuple, _mk_vec as unpickle_vec, _mk_ang as unpickle_ang, _mk_mat as unpickle_mat

# Sanity check.
if not issubclass(Vec_tuple, tuple):
    raise RuntimeError('Vec_tuple is not a tuple subclass!')

# If we don't directly construct this is the fallback.
cdef object tuple_new = tuple.__new__

# For convenience, an iterator which immediately fails.
cdef object EMPTY_ITER = iter(())


DEF PI = 3.141592653589793238462643383279502884197
# Multiply to convert.
DEF rad_2_deg = 180.0 / PI
DEF deg_2_rad = PI / 180.0
DEF ROUND_TO = 6
DEF TOL = 1e-6

cdef extern from *:  # Allow ourselves to access one of the feature flag macros.
    cdef bint USE_TYPE_INTERNALS "CYTHON_USE_TYPE_SLOTS"

cdef inline object _make_tuple(double x, double y, double z):
    # Fast-construct a Vec_tuple. We make a normal tuple (fast),
    # then assign the namedtuple type. The type is on the heap
    # so we need to incref it.
    cdef tuple tup = (round(x, ROUND_TO), round(y, ROUND_TO), round(z, ROUND_TO))
    if USE_TYPE_INTERNALS:
        Py_INCREF(Vec_tuple)
        (<PyObject *>tup).ob_type = <PyTypeObject*>Vec_tuple
        return tup
    else: # Not CPython, use more correct but slow method.
        with cython.optimize.unpack_method_calls(False):
            return tuple_new(Vec_tuple, tup)


cdef inline double norm_ang(double val):
    """Normalise an angle to 0-360."""
    # We have to double-modulus because -1e-14 % 360.0 = 360.0.
    val = val % 360.0 % 360.0
    return val


cdef int _parse_vec_str(vec_t *vec, object value, double x, double y, double z) except -1:
    cdef const char *buf
    cdef Py_ssize_t size, i
    cdef int read_amt
    cdef char c, end_delim = 0

    if isinstance(value, Vec):
        vec.x = (<Vec>value).val.x
        vec.y = (<Vec>value).val.y
        vec.z = (<Vec>value).val.z
    elif isinstance(value, Angle):
        vec.x = (<Angle>value).val.x
        vec.y = (<Angle>value).val.y
        vec.z = (<Angle>value).val.z
    else:
        buf = PyUnicode_AsUTF8AndSize(value, &size)
        # First, skip through whitespace, and stop after the first
        # <{[( delim, or when any other char is found.
        i = 0
        while i < size:
            c = buf[i]
            if c in b' \t\v\f':
                i += 1
                continue
            elif c == b'<':
                i += 1
                end_delim = b'>'
                break
            elif c == b'{':
                i += 1
                end_delim = b'}'
                break
            elif c == b'[':
                i += 1
                end_delim = b']'
                break
            elif c == b'(':
                i += 1
                end_delim = b')'
                break
            else:
                break

        # Parse all three floats with scanf. < 3, because %n might not be counted?
        if sscanf(buf+i, b"%lf %lf %lf%n", &vec.x, &vec.y, &vec.z, &read_amt) < 3:
            vec.x = x
            vec.y = y
            vec.z = z
            return 0
        # Then check the remaining characters for the end delim.
        for i in range(i+read_amt, size):
            if buf[i] == end_delim and end_delim != 0:
                end_delim = 0  # Only allow it once.
            elif buf[i] not in b' \t\v\f':
                # Illegal char, pretend the scan failed.
                vec.x = x
                vec.y = y
                vec.z = z
                return 0
    return 1


def parse_vec_str(object val, object x=0.0, object y=0.0, object z=0.0):
    """Convert a string in the form '(4 6 -4)' into a set of floats.

    If the string is unparsable, this uses the defaults (x,y,z).
    The string can be surrounded with any of the (), {}, [], <> bracket
    types.

    If the 'string' is actually a Vec, the values will be returned.
    """
    cdef vec_t vec
    cdef tuple tup
    # Don't pass x/y/z to _parse_vec_str, so that we can pass through the objects
    # if it fails.
    if _parse_vec_str(&vec, val, NAN, NAN, NAN) == 1:
        return _make_tuple(vec.x, vec.y, vec.z)
    else:
        tup = (x, y, z)
        if USE_TYPE_INTERNALS:
            Py_INCREF(Vec_tuple)
            (<PyObject *> tup).ob_type = <PyTypeObject *> Vec_tuple
            return tup
        else:  # Not CPython, use more correct but slow method.
            with cython.optimize.unpack_method_calls(False):
                return tuple_new(Vec_tuple, tup)

@cython.cdivision(False)  # ZeroDivisionError is needed.
def lerp(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Linearly interpolate from in to out.

    If both in values are the same, ZeroDivisionError is raised.
    """
    return out_min + ((x - in_min) * (out_max - out_min)) / (in_max - in_min)


cdef inline unsigned char conv_vec(
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

cdef inline unsigned char conv_angles(vec_t *result, object ang) except False:
    """Convert some object to a unified Angle struct. 
    
    If scalar is True, allow int/float to set all axes.
    """
    cdef double x, y, z
    if isinstance(ang, Angle):
        result.x = (<Angle>ang).val.x
        result.y = (<Angle>ang).val.y
        result.z = (<Angle>ang).val.z
    elif isinstance(ang, float) or isinstance(ang, int):
        raise TypeError('Cannot convert scalars to an Angle!')
    elif isinstance(ang, tuple):
        x, y, z = <tuple>ang
        result.x = norm_ang(x)
        result.y = norm_ang(y)
        result.z = norm_ang(z)
    else:
        try:
            result.x = norm_ang(ang.x)
            result.y = norm_ang(ang.y)
            result.z = norm_ang(ang.z)
        except AttributeError:
            raise TypeError(f'{type(ang)} is not an Angle-like object!')
    return True

cdef inline double _vec_mag_sq(vec_t *vec):
    # This is faster if you just need to compare.
    return vec.x**2 + vec.y**2 + vec.z**2

cdef inline double _vec_mag(vec_t *vec):
    return math.sqrt(_vec_mag_sq(vec))

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


cdef inline void mat_mul(mat_t targ, mat_t rot):
    """Rotate target by the rotator matrix."""
    cdef double a, b, c
    cdef int i
    for i in range(3):
        a = targ[i][0]
        b = targ[i][1]
        c = targ[i][2]
        # The source rows only affect that row, so we only need to
        # store a copy of 3 at a time.
        targ[i][0] = a * rot[0][0] + b * rot[1][0] + c * rot[2][0]
        targ[i][1] = a * rot[0][1] + b * rot[1][1] + c * rot[2][1]
        targ[i][2] = a * rot[0][2] + b * rot[1][2] + c * rot[2][2]


cdef inline void vec_rot(vec_t *vec, mat_t mat):
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
    cdef double p = deg_2_rad * angle.x
    cdef double y = deg_2_rad * angle.y
    cdef double r = deg_2_rad * angle.z

    res[0][0] = cos(p) * cos(y)
    res[0][1] = cos(p) * sin(y)
    res[0][2] = -sin(p)

    res[1][0] = sin(p) * sin(r) * cos(y) - cos(r) * sin(y)
    res[1][1] = sin(p) * sin(r) * sin(y) + cos(r) * cos(y)
    res[1][2] = sin(r) * cos(p)

    res[2][0] = sin(p) * cos(r) * cos(y) + sin(r) * sin(y)
    res[2][1] = sin(p) * cos(r) * sin(y) - sin(r) * cos(y)
    res[2][2] = cos(r) * cos(p)


cdef inline void _mat_to_angle(vec_t *ang, mat_t mat):
    # https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/mathlib/mathlib_base.cpp#L208
    cdef double horiz_dist = math.sqrt(mat[0][0]**2 + mat[0][1]**2)
    if horiz_dist > 0.001:
        ang.x = norm_ang(rad_2_deg * math.atan2(-mat[0][2], horiz_dist))
        ang.y = norm_ang(rad_2_deg * math.atan2(mat[0][1], mat[0][0]))
        ang.z = norm_ang(rad_2_deg * math.atan2(mat[1][2], mat[2][2]))
    else:
        # Vertical, gimbal lock (yaw=roll)...
        ang.x = norm_ang(rad_2_deg * math.atan2(-mat[0][2], horiz_dist))
        ang.y = norm_ang(rad_2_deg * math.atan2(-mat[1][0], mat[1][1]))
        ang.z = 0.0  # Can't produce.


cdef bint _mat_from_basis(mat_t mat, Vec x, Vec y, Vec z) except True:
    """Implement the shared parts of Matrix/Angle .from_basis()."""
    cdef vec_t res

    if x is None:
        if y is not None and z is not None:
            _vec_cross(&res, &y.val, &z.val)
        else:
            raise TypeError('At least two vectors must be provided!')
    else:
        res = x.val

    _vec_normalise(&res, &res)
    mat[0] = res.x, res.y, res.z

    if y is None:
        if x is not None and z is not None:
            _vec_cross(&res, &z.val, &x.val)
        else:
            raise TypeError('At least two vectors must be provided!')
    else:
        res = y.val

    _vec_normalise(&res, &res)
    mat[1] = res.x, res.y, res.z

    if z is None:
        if x is not None and y is not None:
            _vec_cross(&res, &x.val, &y.val)
        else:
            raise TypeError('At least two vectors must be provided!')
    else:
        res = z.val

    _vec_normalise(&res, &res)
    mat[2] = res.x, res.y, res.z
    return False


cdef inline void _mat_identity(mat_t matrix):
    """Set the matrix to the identity transform."""
    memset(matrix, 0, sizeof(mat_t))
    matrix[0][0] = 1.0
    matrix[1][1] = 1.0
    matrix[2][2] = 1.0


cdef bint _conv_matrix(mat_t result, object value) except True:
    """Convert various values to a rotation matrix.

    Vectors will be treated as angles, and None as the identity.
    """
    cdef vec_t ang
    if value is None:
        _mat_identity(result)
    elif isinstance(value, Matrix):
        memcpy(result, (<Matrix>value).mat, sizeof(mat_t))
    elif isinstance(value, Angle):
        _mat_from_angle(result, &(<Angle>value).val)
    elif isinstance(value, Vec):
        _mat_from_angle(result, &(<Vec>value).val)
    else:
        [ang.x, ang.y, ang.z] = value
        _mat_from_angle(result, &ang)
    return False


def to_matrix(value) -> Matrix:
    """Convert various values to a rotation matrix.

    Vectors will be treated as angles, and None as the identity.
    """
    cdef Matrix result = Matrix.__new__(Matrix)
    _conv_matrix(result.mat, value)
    return result


@cython.final
@cython.internal
cdef class VecIter:
    """Implements Vec/Angle iteration."""
    cdef uint_fast8_t index
    cdef double a, b, c

    def __cinit__(self):
        self.index = self.a = self.b = self.c = 0

    def __iter__(self) -> VecIter:
        return self

    def __next__(self) -> float:
        if self.index == 3:
            raise StopIteration
        self.index += 1
        if self.index == 1:
            return self.a
        elif self.index == 2:
            return self.b
        elif self.index == 3:
            return self.c


@cython.final
@cython.internal
cdef class VecIterGrid:
    """Implements Vec.iter_grid()."""
    cdef:
        long long start_x
        long long start_y
        long long start_z

        long long stop_x
        long long stop_y
        long long stop_z

        long long cur_x
        long long cur_y
        long long cur_z

        long stride

    def __iter__(self) -> VecIterGrid:
        return self

    def __next__(self) -> Vec:
        cdef Vec vec
        if self.cur_x > self.stop_x:
            raise StopIteration

        vec =_vector(self.cur_x, self.cur_y, self.cur_z)

        self.cur_z += self.stride
        if self.cur_z > self.stop_z:
            self.cur_z = self.start_z
            self.cur_y += self.stride
            if self.cur_y > self.stop_y:
                self.cur_y = self.start_y
                self.cur_x += self.stride
                # If greater, next raises StopIteration.

        return vec


@cython.final
@cython.internal
cdef class VecIterLine:
    """Implements Vec.iter_line()."""
    cdef:
        vec_t start
        vec_t diff
        long stride
        long long cur_off
        long long max
        vec_t end

    def __iter__(self) -> VecIterLine:
        return self

    def __next__(self) -> Vec:
        cdef Vec vec
        if self.cur_off < 0:
            raise StopIteration

        if self.cur_off >= self.max:
            # Be exact here.
            vec = _vector(self.end.x, self.end.y, self.end.z)
            self.cur_off = -1
        else:
            vec =_vector(
                self.start.x + self.cur_off * self.diff.x,
                self.start.y + self.cur_off * self.diff.y,
                self.start.z + self.cur_off * self.diff.z,
            )
            self.cur_off += self.stride

        return vec


@cython.final
@cython.internal
cdef class VecTransform:
    """Implements Vec.transform()."""
    cdef Matrix mat
    cdef Vec vec
    def __cinit__(self, Vec vec not None):
        self.vec = vec
        self.mat = None

    def __enter__(self):
        self.mat = Matrix.__new__(Matrix)
        return self.mat

    def __exit__(self, exc_type, exc_val, exc_tb):
        if (
            self.mat is not None and
            self.vec is not None and
            exc_type is None and
            exc_val is None and
            exc_tb is None
        ):
            vec_rot(&self.vec.val, self.mat.mat)
        return False


@cython.final
@cython.internal
cdef class AngleTransform:
    """Implements Angle.transform()."""
    cdef Matrix mat
    cdef Angle ang
    def __cinit__(self, Angle ang not None):
        self.ang = ang
        self.mat = None

    def __enter__(self):
        self.mat = Matrix.__new__(Matrix)
        _mat_from_angle(self.mat.mat, &self.ang.val)
        return self.mat

    def __exit__(self, exc_type, exc_val, exc_tb):
        if (
            self.mat is not None and
            self.vec is not None and
            exc_type is None and
            exc_val is None and
            exc_tb is None
        ):
            _mat_to_angle(&self.ang.val, self.mat.mat)
        return False


@cython.freelist(16)
@cython.final
cdef class Vec:
    """A 3D Vector. This has most standard Vector functions.

    Many of the functions will accept a 3-tuple for comparison purposes.
    """
    __match_args__ = ('x', 'y', 'z')

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
        """The Z axis of the vector."""
        return self.val.z

    @z.setter
    def z(self, value):
        self.val.z = value

    def __init__(
        self,
        x=0.0,
        y=0.0,
        z=0.0,
    ) -> None:
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

    def __deepcopy__(self, memodict=None):
        """Create a duplicate of this vector."""
        return _vector(self.val.x, self.val.y, self.val.z)

    def __reduce__(self):
        return unpickle_vec, (self.val.x, self.val.y, self.val.z)

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

    @classmethod
    @cython.boundscheck(False)
    def with_axes(cls, *args) -> Vec:
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
    ) -> Vec:
        """Rotate a vector by a Source rotational angle.
        Returns the vector, so you can use it in the form
        val = Vec(0,1,0).rotate(p, y, r)

        If round is True, all values will be rounded to 3 decimals
        (since these calculations always have small inprecision.)

        This is deprecated - use an Angle and the @ operator.
        """
        cdef vec_t angle
        cdef mat_t matrix

        PyErr_WarnEx(DeprecationWarning, "Use vec @ Angle() instead.", 1)

        angle.x = pitch
        angle.y = yaw
        angle.z = roll

        _mat_from_angle(matrix, &angle)
        vec_rot(&self.val, matrix)

        if round_vals:
            self.val.x = round(self.val.x, ROUND_TO)
            self.val.y = round(self.val.y, ROUND_TO)
            self.val.z = round(self.val.z, ROUND_TO)

        return self

    def rotate_by_str(
        self,
        ang,
        double pitch=0.0,
        double yaw=0.0,
        double roll=0.0,
        bint round_vals=True,
    ) -> Vec:
        """Rotate a vector, using a string instead of a vector.

        If the string cannot be parsed, use the passed in values instead.
        This is deprecated - use Angle.from_str and the @ operator.
        """
        PyErr_WarnEx(DeprecationWarning, "Use vec @ Angle.from_str() instead.", 1)
        cdef vec_t angle
        cdef mat_t matrix

        _parse_vec_str(&angle, ang, pitch, yaw, roll)
        _mat_from_angle(matrix, &angle)
        vec_rot(&self.val, matrix)

        if round_vals:
            self.val.x = round(self.val.x, ROUND_TO)
            self.val.y = round(self.val.y, ROUND_TO)
            self.val.z = round(self.val.z, ROUND_TO)

        return self

    @classmethod
    def bbox(cls, *points: Vec) -> 'Tuple[Vec, Vec]':
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

            conv_vec(&bbox_min.val, first, scalar=False)
            bbox_max.val = bbox_min.val

            try:
                while True:
                    point = next(points_iter)
                    conv_vec(&vec, point, scalar=False)

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
            conv_vec(&bbox_min.val, points[0], scalar=False)
            bbox_max.val = bbox_min.val

            for i in range(1, len(points)):
                point = points[i]
                conv_vec(&vec, point, scalar=False)

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


    @classmethod
    def iter_grid(
        cls,
        object min_pos: Union[Vec, Tuple[int, int, int]],
        object max_pos: Union[Vec, Tuple[int, int, int]],
        stride: int = 1,
    ) -> Iterator[Vec]:
        """Loop over points in a bounding box. All coordinates should be integers.

        Both borders will be included.
        """
        cdef VecIterGrid it = VecIterGrid.__new__(VecIterGrid)
        cdef vec_t mins
        cdef vec_t maxs
        conv_vec(&mins, min_pos, scalar=True)
        conv_vec(&maxs, max_pos, scalar=True)

        if maxs.x < mins.x or maxs.y < mins.y or maxs.z < mins.z:
            return EMPTY_ITER

        it.cur_x = it.start_x = llround(mins.x)
        it.cur_y = it.start_y = llround(mins.y)
        it.cur_z = it.start_z = llround(mins.z)

        it.stop_x = llround(maxs.x)
        it.stop_y = llround(maxs.y)
        it.stop_z = llround(maxs.z)

        it.stride = stride

        return it

    def iter_line(self, end: Vec, stride: int=1) -> Iterator[Vec]:
        """Yield points between this point and 'end' (including both endpoints).

        Stride specifies the distance between each point.
        If the distance is less than the stride, only end-points will be yielded.
        If they are the same, that point will be yielded.
        """
        cdef vec_t offset, direction
        cdef double length
        cdef double pos
        cdef VecIterLine it = VecIterLine.__new__(VecIterLine)
        offset.x = end.val.x - self.val.x
        offset.y = end.val.y - self.val.y
        offset.z = end.val.z - self.val.z

        length = _vec_mag(&offset)
        _vec_normalise(&it.diff, &offset)

        it.start = self.val
        it.end = end.val
        it.cur_off = 0
        it.max = llround(length)
        it.stride = int(stride)

        return it

    @classmethod
    @cython.cdivision(True)  # Manually do it once.
    def lerp(cls, x: float, in_min: float, in_max: float, out_min: Vec, out_max: Vec) -> Vec:
        """Linerarly interpolate between two vectors.

        If in_min and in_max are the same, ZeroDivisionError is raised.
        """
        cdef double diff = in_max - in_min
        cdef double off = x - in_min
        if diff == 0.0:
            raise ZeroDivisionError('In values must not be equal!')
        return _vector(
            out_min.val.x + (off * (out_max.val.x - out_min.val.x)) / diff,
            out_min.val.y + (off * (out_max.val.y - out_min.val.y)) / diff,
            out_min.val.z + (off * (out_max.val.z - out_min.val.z)) / diff,
        )

    def axis(self) -> str:
        """For a normal vector, return the axis it is on."""
        cdef bint x, y, z
        # Treat extremely close to zero as zero.
        x = abs(self.val.x) > TOL
        y = abs(self.val.y) > TOL
        z = abs(self.val.z) > TOL
        if x and not y and not z:
            return 'x'
        if not x and y and not z:
            return 'y'
        if not x and not y and z:
            return 'z'
        raise ValueError(
            f'({self.val.x:g}, {self.val.y:g}, {self.val.z:g}) is '
            'not an on-axis vector!'
        )

    @cython.boundscheck(False)
    def other_axes(self, object axis) -> Tuple[float, float]:
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

    def in_bbox(self, a, b):
        """Check if this point is inside the specified bounding box."""
        cdef vec_t avec, bvec
        conv_vec(&avec, a, scalar=False)
        conv_vec(&bvec, b, scalar=False)
        if avec.x > bvec.x:
            avec.x, bvec.x = bvec.x, avec.x
        if avec.y > bvec.y:
            avec.y, bvec.y = bvec.y, avec.y
        if avec.z > bvec.z:
            avec.z, bvec.z = bvec.z, avec.z

        return (
            avec.x <= self.val.x <= bvec.x and
            avec.y <= self.val.y <= bvec.y and
            avec.z <= self.val.z <= bvec.z
        )

    def as_tuple(self) -> Tuple[float, float, float]:
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
            norm_ang(rad_2_deg * math.atan2(-self.val.z, horiz_dist)),
            norm_ang(math.atan2(self.val.y, self.val.x) * rad_2_deg),
            norm_ang(roll),
        )

    def to_angle_roll(self, z_norm: Vec, stride: int=...) -> Angle:
        """Produce a Source Engine angle with roll.

        The z_normal should point in +z, and must be at right angles to this
        vector.
        This is deprecated, use Matrix.from_basis().to_angle().
        Stride is no longer used.
        """
        cdef mat_t mat
        cdef Angle ang
        PyErr_WarnEx(DeprecationWarning, 'Use Matrix.from_basis().to_angle()', 1)
        ang = Angle.__new__(Angle)

        _mat_from_basis(mat, x=self, z=z_norm, y=None)
        _mat_to_angle(&ang.val, mat)
        return ang

    def rotation_around(self, double rot: float=90) -> Vec:
        """For an axis-aligned normal, return the angles which rotate around it.

        This is deprecated, use Matrix.axis_angle().to_angle() which works
        for any orientation and has a consistent direction.
        """
        cdef Angle ang = Angle.__new__(Angle)
        ang.val.x = ang.val.y = ang.val.z = 0.0
        PyErr_WarnEx(DeprecationWarning, 'Use Matrix.axis_angle().to_angle()', 1)

        if self.val.x != 0 and self.val.y == 0 and self.val.z == 0:
            ang.val.z = norm_ang(math.copysign(rot, self.val.x))
        elif self.val.x == 0 and self.val.y != 0 and self.val.z == 0:
            ang.val.x = norm_ang(math.copysign(rot, self.val.y))
        elif self.val.x == 0 and self.val.y == 0 and self.val.z != 0:
            ang.val.y = norm_ang(math.copysign(rot, self.val.z))
        else:
            raise ValueError(
                f'({self.val.x}, {self.val.y}, {self.val.z}) is '
                'not an on-axis vector!'
            )
        return ang

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
            conv_vec(&vec_a, obj_a, scalar=True)
            conv_vec(&vec_b, obj_b, scalar=True)
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
            conv_vec(&vec_a, obj_a, scalar=True)
            conv_vec(&vec_b, obj_b, scalar=True)
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
            conv_vec(&vec.val, obj_b, scalar=False)
            vec.val.x = scalar * vec.val.x
            vec.val.y = scalar * vec.val.y
            vec.val.z = scalar * vec.val.z
        elif isinstance(obj_b, (int, float)):
            # vector * scalar.
            conv_vec(&vec.val, obj_a, scalar=False)
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
            conv_vec(&vec.val, obj_b, scalar=False)
            vec.val.x = scalar / vec.val.x
            vec.val.y = scalar / vec.val.y
            vec.val.z = scalar / vec.val.z
        elif isinstance(obj_b, (int, float)):
            # vector / scalar.
            conv_vec(&vec.val, obj_a, scalar=False)
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
            conv_vec(&vec.val, obj_b, scalar=False)
            vec.val.x = scalar // vec.val.x
            vec.val.y = scalar // vec.val.y
            vec.val.z = scalar // vec.val.z
        elif isinstance(obj_b, (int, float)):
            # vector // scalar.
            conv_vec(&vec.val, obj_a, scalar=False)
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
            conv_vec(&vec.val, obj_b, scalar=False)
            vec.val.x = scalar % vec.val.x
            vec.val.y = scalar % vec.val.y
            vec.val.z = scalar % vec.val.z
        elif isinstance(obj_b, (int, float)):
            # vector % scalar.
            conv_vec(&vec.val, obj_a, scalar=False)
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
                vec_rot(&res.val, temp)
                return res
            elif isinstance(second, Matrix):
                vec_rot(&res.val, (<Matrix>second).mat)
                return res
        return NotImplemented

    # In-place operators. Self is always a Vec.

    def __iadd__(self, other):
        """+= operation.

        Like the normal one except without duplication.
        """
        cdef vec_t vec_other
        try:
            conv_vec(&vec_other, other, scalar=True)
        except (TypeError, ValueError):
            return NotImplemented

        self.val.x += vec_other.x
        self.val.y += vec_other.y
        self.val.z += vec_other.z

        return self

    def __isub__(self, other):
        """-= operation.

        Like the normal one except without duplication.
        """
        cdef vec_t vec_other
        try:
            conv_vec(&vec_other, other, scalar=True)
        except (TypeError, ValueError):
            return NotImplemented

        self.val.x -= vec_other.x
        self.val.y -= vec_other.y
        self.val.z -= vec_other.z

        return self

    def __imul__(self, other):
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

    def __itruediv__(self, other):
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

    def __ifloordiv__(self, other):
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

    def __imod__(self, other):
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

    def __matmul__(self, other):
        """Rotate this vector by an angle or matrix."""
        cdef mat_t temp
        cdef Vec res
        if not isinstance(self, Vec):
            return NotImplemented

        res = Vec.__new__(Vec)
        res.val.x = (<Vec>self).val.x
        res.val.y = (<Vec>self).val.y
        res.val.z = (<Vec>self).val.z
        if isinstance(other, Angle):
            _mat_from_angle(temp, &(<Angle>other).val)
            vec_rot(&res.val, temp)
        elif isinstance(other, Matrix):
            vec_rot(&res.val, (<Matrix>other).mat)
        else:
            return NotImplemented
        return res

    def __imatmul__(self, other):
        """@= operation: rotate the vector by a matrix/angle."""
        cdef mat_t temp
        if isinstance(other, Angle):
            _mat_from_angle(temp, &(<Angle>other).val)
            vec_rot(&self.val, temp)
        elif isinstance(other, Matrix):
            vec_rot(&self.val, (<Matrix>other).mat)
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
            return NotImplemented

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
        """The len() of a vector is always 3."""
        return 3

    # All the comparisons are similar, so we can use richcmp to
    # nicely combine the parsing code.
    def __richcmp__(self, other_obj, int op):
        """Rich Comparisons.

        Two Vectors are compared based on the axes.
        A Vector can be compared with a 3-tuple as if it was a Vector also.
        A tolerance of 1e-6 is accounted for automatically.
        """
        cdef vec_t other
        try:
            conv_vec(&other, other_obj, False)
        except (TypeError, ValueError):
            return NotImplemented

        # 'redundant' == True prevents the individual comparisons from trying
        # to convert the result individually on failure.
        # Use subtraction so that values within TOL are accepted.
        if op == Py_EQ:
            return (
                abs(self.val.x - other.x) <= TOL and
                abs(self.val.y - other.y) <= TOL and
                abs(self.val.z - other.z) <= TOL
            ) == True
        elif op == Py_NE:
            return (
                abs(self.val.x - other.x) > TOL or
                abs(self.val.y - other.y) > TOL or
                abs(self.val.z - other.z) > TOL
            ) == True
        elif op == Py_LT:
            return (
                (other.x - self.val.x) > TOL and
                (other.y - self.val.y) > TOL and
                (other.z - self.val.z) > TOL
            ) == True
        elif op == Py_GT:
            return (
                (self.val.x - other.x) > TOL and
                (self.val.y - other.y) > TOL and
                (self.val.z - other.z) > TOL
            ) == True
        elif op == Py_LE:  # !GT
            return (
                (self.val.x - other.x) <= TOL and
                (self.val.y - other.y) <= TOL and
                (self.val.z - other.z) <= TOL
            ) == True
        elif op == Py_GE: # !LT
            return (
                (other.x - self.val.x) <= TOL and
                (other.y - self.val.y) <= TOL and
                (other.z - self.val.z) <= TOL
            ) == True
        else:
            raise SystemError(f'Unknown operation {op!r}' '!')

    def max(self, other):
        """Set this vector's values to the maximum of the two vectors."""
        cdef vec_t vec
        conv_vec(&vec, other, scalar=False)
        if self.val.x < vec.x:
            self.val.x = vec.x

        if self.val.y < vec.y:
            self.val.y = vec.y

        if self.val.z < vec.z:
            self.val.z = vec.z

    def min(self, other):
        """Set this vector's values to be the minimum of the two vectors."""
        cdef vec_t vec
        conv_vec(&vec, other, scalar=False)
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

    def mag_sq(self):
        """Compute the distance from the vector and the origin."""
        return _vec_mag_sq(&self.val)

    def len_sq(self):
        """Compute the distance from the vector and the origin."""
        return _vec_mag_sq(&self.val)

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

    def norm_mask(self, normal: 'Vec') -> Vec:
        """Subtract the components of this vector not in the direction of the normal.

        If the normal is axis-aligned, this will zero out the other axes.
        If not axis-aligned, it will do the equivalent.
        """
        cdef vec_t norm

        conv_vec(&norm, normal, False)

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

        conv_vec(&oth, other, False)

        return (
            self.val.x * oth.x +
            self.val.y * oth.y +
            self.val.z * oth.z
        )

    def cross(self, other) -> Vec:
        """Return the cross product of both Vectors."""
        cdef vec_t oth
        cdef Vec res

        conv_vec(&oth, other, False)
        res = Vec.__new__(Vec)
        _vec_cross(&res.val, &self.val, &oth)
        return res

    def localise(self, object origin, object angles=None) -> None:
        """Shift this point to be local to the given position and angles.

        This effectively translates local-space offsets to a global location,
        given the parent's origin and angles.
        """
        cdef mat_t matrix
        cdef vec_t offset
        _conv_matrix(matrix, angles)
        conv_vec(&offset, origin, scalar=False)
        vec_rot(&self.val, matrix)
        self.val.x += offset.x
        self.val.y += offset.y
        self.val.z += offset.z


    def join(self, delim: str=', ') -> str:
        """Return a string with all numbers joined by the passed delimiter.

        This strips off the .0 if no decimal portion exists.
        """
        # :g strips the .0 off of floats if it's an integer.
        return f'{self.val.x:g}{delim}{self.val.y:g}{delim}{self.val.z:g}'

    def __str__(self) -> str:
        """Return the values, separated by spaces.

        This is the main format in Valve's file formats.
        This strips off the .0 if no decimal portion exists.
        """
        return f"{self.val.x:g} {self.val.y:g} {self.val.z:g}"

    def __repr__(self) -> str:
        """Code required to reproduce this vector."""
        return f"Vec({self.val.x:g}, {self.val.y:g}, {self.val.z:g})"

    def __format__(self, format_spec: str) -> str:
        """Control how the text is formatted."""
        if not format_spec:
            format_spec = 'g'
        return f"{self.x:{format_spec}} {self.y:{format_spec}} {self.z:{format_spec}}"

    def __iter__(self) -> VecIter:
        cdef VecIter viter = VecIter.__new__(VecIter)
        viter.a = self.val.x
        viter.b = self.val.y
        viter.c = self.val.z
        return viter

    def __reversed__(self):
        cdef VecIter viter = VecIter.__new__(VecIter)
        viter.a = self.val.z
        viter.b = self.val.y
        viter.c = self.val.x
        return viter

    def __getitem__(self, ind_obj) -> float:
        """Allow reading values by index instead of name if desired.

        This accepts either 0,1,2 or 'x','y','z' to read values.
        Useful in conjunction with a loop to apply commands to all values.
        """
        cdef int ind
        cdef Py_UCS4 axis
        if isinstance(ind_obj, int) and ind_obj is not None:
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
        else:
            if isinstance(ind_obj, str) and ind_obj is not None and len(<str>ind_obj) == 1:
                axis = (<str>ind_obj)[0]

                if axis == "x":
                    return self.val.x
                elif axis == "y":
                    return self.val.y
                elif axis == "z":
                    return self.val.z

        raise KeyError(f'Invalid axis {ind_obj!r}' '!')

    def __setitem__(self, ind_obj, double val: float) -> None:
        """Allow editing values by index instead of name if desired.

        This accepts either 0,1,2 or 'x','y','z' to edit values.
        Useful in conjunction with a loop to apply commands to all values.
        """
        cdef int ind
        cdef Py_UCS4 axis
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
        else:
            if isinstance(ind_obj, str) and len(<str>ind_obj) == 1:
                axis = (<str>ind_obj)[0]

                if axis == "x":
                    self.val.x = val
                    return
                elif axis == "y":
                    self.val.y = val
                    return
                elif axis == "z":
                    self.val.z = val
                    return

        raise KeyError(f'Invalid axis {ind_obj!r}' '!')

    def transform(self):
        """Perform rotations on this Vector efficiently.

        Used as a context manager, which returns a matrix.
        When the body is exited safely, the matrix is applied to
        the angle.
        """
        return VecTransform.__new__(VecTransform, self)


@cython.freelist(16)
@cython.final
cdef class Matrix:
    """Represents a matrix via a transformation matrix."""

    def __init__(self) -> None:
        """Create a matrix set to the identity transform."""
        _mat_identity(self.mat)

    def __eq__(self, other: object) -> object:
        if isinstance(other, Matrix):
            # We can just compare the memory buffers.
            return memcmp(self.mat, (<Matrix>other).mat, sizeof(mat_t)) == 0
        return NotImplemented

    def __ne__(self, other: object) -> object:
        if isinstance(other, Matrix):
            return memcmp(self.mat, (<Matrix>other).mat, sizeof(mat_t)) != 0
        return NotImplemented

    def __repr__(self) -> str:
        return (
            '<Matrix '
            f'{self.mat[0][0]:.3} {self.mat[0][1]:.3} {self.mat[0][2]:.3}, '
            f'{self.mat[1][0]:.3} {self.mat[1][1]:.3} {self.mat[1][2]:.3}, '
            f'{self.mat[2][0]:.3} {self.mat[2][1]:.3} {self.mat[2][2]:.3}'
            '>'
        )

    def copy(self) -> 'Matrix':
        """Duplicate this matrix."""
        cdef Matrix copy = Matrix.__new__(Matrix)
        memcpy(copy.mat, self.mat, sizeof(mat_t))
        return copy

    def __copy__(self) -> 'Matrix':
        """Duplicate this matrix."""
        cdef Matrix copy = Matrix.__new__(Matrix)
        memcpy(copy.mat, self.mat, sizeof(mat_t))
        return copy

    def __deepcopy__(self, dict memodict=None) -> 'Matrix':
        """Duplicate this matrix."""
        cdef Matrix copy = Matrix.__new__(Matrix)
        memcpy(copy.mat, self.mat, sizeof(mat_t))
        return copy

    def __reduce__(self):
        return unpickle_mat, (
            self.mat[0][0], self.mat[0][1], self.mat[0][2],
            self.mat[1][0], self.mat[1][1], self.mat[1][2],
            self.mat[2][0], self.mat[2][1], self.mat[2][2],
        )

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
    def from_angle(cls, pitch, yaw=None, roll=None):
        """Return the rotation representing an Euler angle.

        Either an Angle can be passed, or the raw pitch/yaw/roll angles.
        """
        cdef Matrix rot = Matrix.__new__(Matrix)
        cdef vec_t ang
        if isinstance(pitch, Angle):
            ang = (<Angle>pitch).val
        elif yaw is None or roll is None:
            raise TypeError('Matrix.from_angles() accepts a single Angle or 3 floats!')
        else:
            ang.x = float(pitch)
            ang.y = float(yaw)
            ang.z = float(roll)
        _mat_from_angle(rot.mat, &ang)
        return rot


    @classmethod
    def axis_angle(cls, object axis, double angle) -> 'Matrix':
        """Compute the rotation matrix forming a rotation around an axis by a specific angle."""
        cdef vec_t vec_axis
        cdef double sin, cos, icos, x, y, z
        conv_vec(&vec_axis, axis, scalar=False)
        _vec_normalise(&vec_axis, &vec_axis)
        angle *= -deg_2_rad

        cos = math.cos(angle)
        icos = 1 - cos
        sin = math.sin(angle)

        x = vec_axis.x
        y = vec_axis.y
        z = vec_axis.z

        cdef Matrix mat = Matrix.__new__(cls)

        mat.mat[0][0] = x*x * icos + cos
        mat.mat[0][1] = x*y * icos - z*sin
        mat.mat[0][2] = x*z * icos + y*sin

        mat.mat[1][0] = y*x * icos + z*sin
        mat.mat[1][1] = y*y * icos + cos
        mat.mat[1][2] = y*z * icos - x*sin

        mat.mat[2][0] = z*x * icos - y*sin
        mat.mat[2][1] = z*y * icos + x*sin
        mat.mat[2][2] = z*z * icos + cos

        return mat

    def forward(self, mag: float = 1.0):
        """Return a vector with the given magnitude pointing along the X axis."""
        return _vector(mag * self.mat[0][0], mag * self.mat[0][1], mag * self.mat[0][2])

    def left(self, mag: float = 1.0):
        """Return a vector with the given magnitude pointing along the Y axis."""
        return _vector(mag * self.mat[1][0], mag * self.mat[1][1], mag * self.mat[1][2])

    def up(self, mag: float = 1.0):
        """Return a vector with the given magnitude pointing along the Z axis."""
        return _vector(mag * self.mat[2][0], mag * self.mat[2][1], mag * self.mat[2][2])

    def __getitem__(self, item):
        """Retrieve an individual matrix value by x, y position (0-2)."""
        cdef int x, y
        try:
            x, y = item
        except (ValueError, TypeError, OverflowError):
            raise KeyError(f'Invalid coordinate {item!r}' '!')
        if 0 <= x < 3 and 0 <= y < 3:
            return self.mat[x][y]
        else:
            raise KeyError(f'Invalid coordinate {x}, {y}' '!')

    def __setitem__(self, item, double value):
        """Set an individual matrix value by x, y position (0-2)."""
        cdef int x, y
        try:
            x, y = item
        except (ValueError, TypeError, OverflowError):
            raise KeyError(f'Invalid coordinate {item!r}' '!')
        if 0 <= x < 3 and 0 <= y < 3:
            self.mat[x][y] = value
        else:
            raise KeyError(f'Invalid coordinate {x}, {y}' '!')

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
        _mat_from_basis(mat.mat, x, y, z)
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
                mat_mul(mat.mat, (<Matrix>second).mat)
            elif isinstance(second, Angle):
                _mat_from_angle(temp, &(<Angle>second).val)
                mat_mul(mat.mat, temp)
            else:
                return NotImplemented
            return mat
        elif isinstance(second, Matrix):
            if isinstance(first, Vec):
                vec = Vec.__new__(Vec)
                memcpy(&vec.val, &(<Vec>first).val, sizeof(vec_t))
                vec_rot(&vec.val, (<Matrix>second).mat)
                return vec
            elif isinstance(first, tuple):
                vec = Vec.__new__(Vec)
                vec.val.x, vec.val.y, vec.val.z = <tuple>first
                vec_rot(&vec.val, (<Matrix>second).mat)
                return vec
            elif isinstance(first, Angle):
                ang = Angle.__new__(Angle)
                _mat_from_angle(temp, &(<Angle>first).val)
                mat_mul(temp, (<Matrix>second).mat)
                _mat_to_angle(&ang.val, temp)
                return ang
            else:
                return NotImplemented
        else:
            raise SystemError('Neither are Matrices?')

    def __imatmul__(self, other):
        cdef mat_t temp
        if isinstance(other, Matrix):
            mat_mul(self.mat, (<Matrix>other).mat)
            return self
        elif isinstance(other, Angle):
            _mat_from_angle(temp, &(<Angle>other).val)
            mat_mul(self.mat, temp)
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
    __match_args__ = ('pitch', 'yaw', 'roll')

    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0) -> None:
        """Create an Angle.

        All values are converted to Floats automatically.
        If no value is given, that axis will be set to 0.
        An iterable can be passed in (as the pitch argument), which will be
        used for pitch, yaw, and roll. This includes Vectors and other Angles.
        """
        cdef tuple tup
        if isinstance(pitch, float) or isinstance(pitch, int):
            self.val.x = norm_ang(pitch)
            self.val.y = norm_ang(yaw)
            self.val.z = norm_ang(roll)
        elif isinstance(pitch, Angle):
            self.val.x = (<Angle>pitch).val.x
            self.val.y = (<Angle>pitch).val.y
            self.val.z = (<Angle>pitch).val.z
        elif isinstance(pitch, tuple):
            tup = <tuple>pitch
            if len(tup) >= 1:
                self.val.x = norm_ang(tup[0])
            else:
                self.val.x = 0.0

            if len(tup) >= 2:
                self.val.y = norm_ang(tup[1])
            else:
                self.val.y = norm_ang(yaw)

            if len(tup) >= 3:
                self.val.z = norm_ang(tup[2])
            else:
                self.val.z = norm_ang(roll)

        else:
            it = iter(pitch)
            try:
                self.val.x = norm_ang(next(it))
            except StopIteration:
                self.val.x = 0.0
                self.val.y = norm_ang(yaw)
                self.val.z = norm_ang(roll)
                return

            try:
                self.val.y = norm_ang(next(it))
            except StopIteration:
                self.val.y = norm_ang(yaw)
                self.val.z = norm_ang(roll)
                return

            try:
                self.val.z = norm_ang(next(it))
            except StopIteration:
                self.val.z = norm_ang(roll)

    def copy(self) -> 'Angle':
        """Create a duplicate of this angle."""
        return _angle(self.val.x, self.val.y, self.val.z)

    def __copy__(self) -> 'Angle':
        """Create a duplicate of this angle."""
        return _angle(self.val.x, self.val.y, self.val.z)

    def __deepcopy__(self, dict memodict=None) -> 'Angle':
        """Create a duplicate of this angle."""
        return _angle(self.val.x, self.val.y, self.val.z)

    def __reduce__(self):
        return unpickle_ang, (self.val.x, self.val.y, self.val.z)

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
        ang.val.x = norm_ang(ang.val.x)
        ang.val.y = norm_ang(ang.val.y)
        ang.val.z = norm_ang(ang.val.z)
        return ang

    @property
    def pitch(self) -> float:
        """The Y-axis rotation, performed second."""
        return self.val.x

    @pitch.setter
    def pitch(self, double pitch) -> None:
        self.val.x = norm_ang(pitch)

    @property
    def yaw(self) -> float:
        """The Z-axis rotation, performed last."""
        return self.val.y

    @yaw.setter
    def yaw(self, double yaw) -> None:
        self.val.y = norm_ang(yaw)

    @property
    def roll(self) -> float:
        """The X-axis rotation, performed first."""
        return self.val.z

    @roll.setter
    def roll(self, double roll) -> None:
        self.val.z = norm_ang(roll)

    def join(self, delim: str=', ') -> str:
        """Return a string with all numbers joined by the passed delimiter.

        This strips off the .0 if no decimal portion exists.
        """
        return f'{self.val.x:g}{delim}{self.val.y:g}{delim}{self.val.z:g}'

    def __str__(self) -> str:
        """Return the values, separated by spaces.

        This is the main format in Valve's file formats, though identical to
        vectors.
        This strips off the .0 if no decimal portion exists.
        """
        return f"{self.val.x:g} {self.val.y:g} {self.val.z:g}"

    def __repr__(self) -> str:
        return f'Angle({self.val.x:g}, {self.val.y:g}, {self.val.z:g})'

    def __format__(self, format_spec: str) -> str:
        """Control how the text is formatted."""
        if not format_spec:
            format_spec = 'g'
        return f"{self.val.x:{format_spec}} {self.val.y:{format_spec}} {self.val.z:{format_spec}}"

    def as_tuple(self):
        """Return the Angle as a tuple."""
        return Vec_tuple(self.val.x, self.val.y, self.val.z)

    def __len__(self) -> int:
        """The length of an Angle is always 3."""
        return 3

    def __iter__(self) -> VecIter:
        """Iterating over the angles returns each value in turn."""
        cdef VecIter viter = VecIter.__new__(VecIter)
        viter.a = self.val.x
        viter.b = self.val.y
        viter.c = self.val.z
        return viter

    def __reversed__(self) -> VecIter:
        """Iterating over the angles returns each value in turn."""
        cdef VecIter viter = VecIter.__new__(VecIter)
        viter.a = self.val.z
        viter.b = self.val.y
        viter.c = self.val.x
        return viter


    @classmethod
    @cython.boundscheck(False)
    def with_axes(cls, *args):
        """Create an Angle, given a number of axes and corresponding values.

        This is a convenience for doing the following:
            ang = Angle()
            ang[axis1] = val1
            ang[axis2] = val2
            ang[axis3] = val3
        The magnitudes can also be Angles, in which case the matching
        axis will be used from the angle.
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
                    ang.val.x = norm_ang(axis_val)
            elif axis in ('y', 'yaw'):
                if isinstance(axis_val, Angle):
                    ang.val.y = (<Angle>axis_val).val.y
                else:
                    ang.val.y = norm_ang(axis_val)
            elif axis in ('r', 'rol', 'roll'):
                if isinstance(axis_val, Angle):
                    ang.val.z = (<Angle>axis_val).val.z
                else:
                    ang.val.z = norm_ang(axis_val)

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
        cdef mat_t mat
        cdef Angle ang = Angle.__new__(Angle)
        _mat_from_basis(mat, x, y, z)
        _mat_to_angle(&ang.val, mat)
        return ang


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
            if index == 0:
                return self.val.x
            if index == 1:
                return self.val.y
            if index == 2:
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
        val = norm_ang(val)

        if isinstance(pos, int):
            index = <int>pos
            if index == 0:
                self.val.x = val
            if index == 1:
                self.val.y = val
            if index == 2:
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

    def __richcmp__(self, other_obj, int op):
        """Rich Comparisons.

        Angles only support equality, since ordering is nonsensical.
        """
        cdef vec_t other
        try:
            conv_angles(&other, other_obj)
        except (TypeError, ValueError):
            return NotImplemented

        # 'redundant' == True prevents the individual comparisons from
        # trying
        # to convert the result individually on failure.
        # Use subtraction so that values within TOL are accepted.
        if op == Py_EQ:
            return (
                abs(self.val.x - other.x) <= TOL and
                abs(self.val.y - other.y) <= TOL and
                abs(self.val.z - other.z) <= TOL
            ) == True
        elif op == Py_NE:
            return (
                abs(self.val.x - other.x) > TOL or
                abs(self.val.y - other.y) > TOL or
                abs(self.val.z - other.z) > TOL
            ) == True
        elif op in [Py_LT, Py_GT, Py_GE, Py_LE]:
            return NotImplemented
        else:
            raise SystemError(f'Unknown operation {op!r}' '!')

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
            norm_ang(angle.val.x * scalar),
            norm_ang(angle.val.y * scalar),
            norm_ang(angle.val.z * scalar),
        )

    def __matmul__(first, second):
        cdef mat_t temp1, temp2
        if isinstance(first, Angle):
            _mat_from_angle(temp1, &(<Angle>first).val)
            if isinstance(second, Angle):
                _mat_from_angle(temp2, &(<Angle>second).val)
                mat_mul(temp1, temp2)
            elif isinstance(second, Matrix):
                mat_mul(temp1, (<Matrix>second).mat)
            else:
                return NotImplemented
            res = Angle.__new__(Angle)
            _mat_to_angle(&(<Angle>res).val, temp1)
            return res
        elif isinstance(second, Angle):
            _mat_from_angle(temp2, &(<Angle>second).val)
            if isinstance(first, tuple):
                res = Vec.__new__(Vec)
                (<Vec>res).val.x, (<Vec>res).val.y, (<Vec>res).val.z = <tuple>first
                vec_rot(&(<Vec>res).val, temp2)
                return res
            # These classes should do this themselves, but this is here for
            # completeness.
            if isinstance(first, Matrix):
                res = Matrix.__new__(Matrix)
                memcpy((<Matrix>res).mat, (<Matrix>first).mat, sizeof(mat_t))
                mat_mul((<Matrix>res).mat, temp2)
                return res
            elif isinstance(first, Vec):
                res = Vec.__new__(Vec)
                memcpy(&(<Vec>res).val, &(<Vec>first).val, sizeof(vec_t))
                vec_rot(&(<Vec>res).val, temp2)
                return res

        return NotImplemented

    def __imatmul__(self, second):
        cdef mat_t mat_self, temp2
        _mat_from_angle(mat_self, &(<Angle>self).val)
        if isinstance(second, Angle):
            _mat_from_angle(temp2, &(<Angle>second).val)
            mat_mul(mat_self, temp2)
        elif isinstance(second, Matrix):
            mat_mul(mat_self, (<Matrix>second).mat)
        else:
            return NotImplemented
        _mat_to_angle(&self.val, mat_self)
        return self

    def transform(self):
        """Perform transformations on this angle.

        Used as a context manager, which returns a matrix.
        When the body is exited safely, the matrix is applied to
        the angle.
        """
        return AngleTransform.__new__(AngleTransform, self)


def quickhull(vertexes: 'Iterable[Vec]') -> 'list[tuple[Vec, Vec, Vec]]':
    """Use the quickhull algorithm to construct a convex hull around the provided points."""
    cdef size_t v1, v2, v3, ind
    cdef vector[quickhull.Vector3[double]] values = vector[quickhull.Vector3[double]]()
    cdef list vert_list, result
    cdef Vec vecobj
    cdef quickhull.QuickHull[double] qhull = quickhull.QuickHull[double]()

    for vecobj in vertexes:
        values.push_back(quickhull.Vector3[double](vecobj.val.x, vecobj.val.y, vecobj.val.z))

    cdef quickhull.ConvexHull[double] result_hull = qhull.getConvexHull(values, False, False)

    cdef list vectors = [
        _vector(v.x, v.y, v.z)
        for v in result_hull.getVertexBuffer()
    ]
    cdef vector[size_t] indices = result_hull.getIndexBuffer()
    res = []
    for ind in range(0, indices.size(), 3):
        v1 = indices[ind + 0]
        v2 = indices[ind + 1]
        v3 = indices[ind + 2]
        res.append((vectors[v1], vectors[v2], vectors[v3]))
    return res


# Override the class' names to match the public one.
# This fixes all the methods too, though not in exceptions.

from cpython.object cimport PyTypeObject
if USE_TYPE_INTERNALS:
    (<PyTypeObject *>Vec).tp_name = b"srctools.math.Vec"
    (<PyTypeObject *>Angle).tp_name = b"srctools.math.Angle"
    (<PyTypeObject *>Matrix).tp_name = b"srctools.math.Matrix"
    (<PyTypeObject *>VecIter).tp_name = b"srctools.math._Vec_or_Angle_iterator"
    (<PyTypeObject *>VecIterGrid).tp_name = b"srctools.math._Vec_grid_iterator"
    (<PyTypeObject *>VecIterLine).tp_name = b"srctools.math._Vec_line_iterator"
    (<PyTypeObject *>VecTransform).tp_name = b"srctools.math._Vec_transform_cm"
    (<PyTypeObject *>AngleTransform).tp_name = b"srctools.math._Angle_transform_cm"
try:
    parse_vec_str.__module__ = 'srctools.math'
    to_matrix.__module__ = 'srctools.math'
    lerp.__module__ = 'srctools.math'
except Exception:
    pass  # Perfectly fine.

# Drop references.
Tuple = Iterator = Union = None
