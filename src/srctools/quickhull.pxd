"""Cython definitions for the quickhull library."""
from libcpp.vector cimport vector


cdef extern from "Structs/Vector3.hpp" namespace "quickhull":
    cdef cppclass Vector3[T]:
        Vector3()
        Vector3(T x, T y, T z)
        T x
        T y
        T z

cdef extern from "Structs/VertexDataSource.hpp" namespace "quickhull":
    cdef cppclass VertexDataSource[T]:
        VertexDataSource()
        VertexDataSource(const Vector3[T] * ptr, size_t count)
        VertexDataSource(const vector[Vector3[T]]& vec)
        size_t size()
        const Vector3[T]& operator[](size_t index)
        const Vector3[T] * begin()
        const Vector3[T] * end()

cdef extern from "ConvexHull.hpp" namespace "quickhull":
    cdef cppclass ConvexHull[T]:
        ConvexHull()
        vector[size_t]& getIndexBuffer()
        VertexDataSource[T]& getVertexBuffer()

cdef extern from "QuickHull.hpp" namespace "quickhull":
    cdef cppclass QuickHull[FloatType]:
        QuickHull()
        ConvexHull[FloatType] getConvexHull(
            const vector[Vector3[FloatType]]& pointCloud,
            bint CCW,
            bint useOriginalIndices
        )

