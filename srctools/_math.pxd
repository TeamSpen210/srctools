# cython: language_level=3
cimport cython

# Lightweight struct just holding the three values.
# We can use this for temporaries. It's also used for angles.
cdef struct vec_t:
    double x
    double y
    double z

ctypedef double[3][3] mat_t

cdef unsigned char _parse_vec_str(vec_t *vec, object value, x, y, z) except False

cdef unsigned char conv_vec(vec_t *result, object vec, bint scalar) except False
cdef unsigned char conv_angles(vec_t *result, object ang) except False

cdef void mat_mul(mat_t targ, mat_t rot)
cdef void vec_rot(vec_t *vec, mat_t mat)

@cython.final
cdef class Vec:
    cdef vec_t val

@cython.final
cdef class Matrix:
    cdef mat_t mat

@cython.final
cdef class Angle:
    cdef vec_t val
