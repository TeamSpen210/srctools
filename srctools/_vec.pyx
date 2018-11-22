# cython: language_level=3, embedsignature=True, auto_pickle=False
# """Optimised Vector object."""
from libc cimport math
cimport cython

# Lightweight struct just holding the three values.
# Used for subcalculations.
cdef struct vec_t:
    double x
    double y
    double z

cdef inline Vec _vector(double x, double y, double z):
    """Make a Vector directly."""
    cdef Vec vec = Vec.__new__(Vec)
    vec.val.x = x
    vec.val.y = y
    vec.val.z = z
    return vec

cdef object _vector_make
from srctools.vec import _mk as _vector_make

cdef void _parse_vec_str(vec_t *vec, object value, double x, double y, double z):
    cdef unicode str_x, str_y, str_z

    if isinstance(value, Vec):
        vec.x = (<Vec>value).val.x
        vec.y = (<Vec>value).val.y
        vec.z = (<Vec>value).val.z
        return

    try:
        str_x, str_y, str_z = (<unicode?>value).split(' ')
    except ValueError:
        vec.x = x
        vec.y = y
        vec.z = z
        return

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

cdef inline void _conv_vec(vec_t *result, object vec):
    """Convert some object to a unified Vector struct."""
    if isinstance(vec, Vec):
        result.x = (<Vec>vec).val.x
        result.y = (<Vec>vec).val.y
        result.z = (<Vec>vec).val.z
    elif isinstance(vec, tuple):
        result.x, result.y, result.z = <tuple>vec
    else:
        result.x = vec.x
        result.y = vec.y
        result.z = vec.z

DEF PI = 3.141592653589793238462643383279502884197
# Multiply to convert.
DEF rad_2_deg = 180 / PI
DEF deg_2_rad = PI / 180.0

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
        if self.index == 2:
            return self.vec.val.y
        if self.index == 3:
            # Drop our reference.
            ret = self.vec.val.z
            self.vec = None
            return ret


# Lots of temporaries are expected.
@cython.freelist(16)
cdef class Vec:
    """A 3D Vector. This has most standard Vector functions.

    Many of the functions will accept a 3-tuple for comparison purposes.
    """
    # This is a sub-struct, so we can pass pointers to it to other
    # functions.
    cdef vec_t val

    @property
    def x(self):
        return self.val.x

    @x.setter
    def x(self, value):
        self.val.x = value

    @property
    def y(self):
        return self.val.y

    @y.setter
    def y(self, value):
        self.val.y = value

    @property
    def z(self):
        return self.val.z

    @z.setter
    def z(self, value):
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
                self.val.z = tup[3]
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
        return _vector_make, (self.val.x, self.val.y, self.val.z)

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
        cdef unicode axis
        cdef unsigned char i
        for i in range(0, arg_count, 2):
            axis = args[i]
            axis_val = args[i+1]
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
                raise ValueError(f'Invalid axis {axis!r}!')

        return vec

    @staticmethod
    def bbox(*points: Vec) -> 'Tuple[Vec, Vec]':
        """Compute the bounding box for a set of points.

        Pass either several Vecs, or an iterable of Vecs.
        Returns a (min, max) tuple.
        """
        cdef Vec bbox_min = Vec.__new__(Vec)
        cdef Vec bbox_max = Vec.__new__(Vec)
        cdef Vec sing_vec
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
            points_iter = iter(points)
            try:
                first = next(points_iter)
            except StopIteration:
                raise TypeError('Empty iterator!') from None

            _conv_vec(&bbox_min.val, first)
            bbox_max.val = bbox_min.val

            try:
                while True:
                    point = next(points_iter)
                    bbox_min.min(point)
                    bbox_max.max(point)
            except StopIteration:
                pass
        elif len(points) == 0:
            raise TypeError(
                'Vec.bbox() expected at '
                'least 1 argument, got 0.'
            )
        else:
            # Tuple-specific.
            _conv_vec(&bbox_min.val, points[0])
            bbox_max.val = bbox_min.val

            for i in range(1, len(points)):
                point = points[i]
                bbox_min.min(point)
                bbox_max.max(point)

        return bbox_min, bbox_max
