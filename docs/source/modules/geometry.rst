srctools.geometry
------------------------

.. module:: srctools.geometry
    :synopsis: Calculates vertices for brushes, then performs operations on them.

In VMF files, brushes are defined solely by the direction of each face, and do not explicitly
specify vertices. This makes manipulation and checks against them difficult. This module calculates
the vertices of a brush, and provides the logic for operations like clipping and carving.

See :sdk-2013:`utils/common/polylib.cpp` for some of the algorithms.

Conversion
==========

First, call `~Geometry.from_brush` to convert a `~srctools.vmf.Solid` into a calculated geometry
object, or `~Geometry.from_points` to construct a convex hull out of a point cloud.

.. automethod:: Geometry.from_brush
.. automethod:: Geometry.from_points

These are composed of `Polygon` objects, which each wrap a `~srctools.vmf.Side` object. After
performing operations, call `Geometry.rebuild` to remake the `~srctools.vmf.Solid` objects, reusing
faces if possible.

.. automethod:: Geometry.rebuild

Alternatively, polygons can be converted to SMD triangles for preview.

.. automethod:: Polygon.to_smd_tris
    :for: triangle

Operations
==========

.. automethod:: Geometry.carve
.. automethod:: Geometry.clip
.. automethod:: Geometry.merge


API
===

.. autoclass:: Geometry

    .. autoattribute:: polys
    .. automethod:: Geometry.raw_carve
    .. automethod:: Geometry.raw_clip
    .. automethod:: Geometry.unshare_faces

.. autoclass:: Polygon

    .. autoattribute:: original
    .. autoattribute:: vertices
    .. autoattribute:: plane

    .. automethod:: build_face
