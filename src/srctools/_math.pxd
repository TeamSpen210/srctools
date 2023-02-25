# cython: language_level=3
cimport cython


# Lightweight struct just holding the three values.
# We can use this for temporaries. It's also used for angles.
cdef struct vec_t:
    double x
    double y
    double z

ctypedef double[3][3] mat_t

#  0: failed to string parse and using x/y/z defaults.
#  1: successfully parsed or is a Vec/Angle
# -1: Exception.
cdef int _parse_vec_str(vec_t *vec, object value, double x, double y, double z) except -1

cdef bint conv_vec(vec_t *result, object vec, bint scalar) except False
cdef bint conv_angles(vec_t *result, object ang) except False

cdef bint mat_mul(mat_t targ, mat_t rot) except False
cdef bint vec_rot(vec_t *vec, mat_t mat) except False

@cython.internal
cdef class VecBase:
    cdef vec_t val

@cython.final
cdef class Vec(VecBase):
    pass

@cython.final
cdef class FrozenVec(VecBase):
    pass

@cython.internal
cdef class MatrixBase:
    cdef mat_t mat

@cython.final
cdef class Matrix(MatrixBase):
    pass

@cython.final
cdef class FrozenMatrix(MatrixBase):
    pass

@cython.internal
cdef class AngleBase:
    cdef vec_t val

@cython.final
cdef class Angle(AngleBase):
    pass

@cython.final
cdef class FrozenAngle(AngleBase):
    pass
