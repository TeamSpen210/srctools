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
