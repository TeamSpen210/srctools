"""Implements tools for manipulating brush geometry.

See `polylib.cpp <https://github.com/ValveSoftware/source-sdk-2013/blob/0565403b153dfcde602f6f58d8f4d13483696a13/src/utils/common/polylib.cpp>`_ for some of the algorithms.
"""

from typing import Optional

from collections.abc import Iterable, Iterator, MutableMapping
from collections import defaultdict

import attrs

from srctools import EmptyMapping, FrozenVec, Matrix, VMF, Vec
from srctools import vmf as vmf_mod, smd
from srctools.bsp import Plane, PlaneType


__all__ = [
    'Geometry', 'Polygon',
    'Plane',  # Re-export
]


MAX_PLANE = [
    FrozenVec(-100_000, -100_000),
    FrozenVec(100_000, -100_000),
    FrozenVec(100_000, 100_000),
    FrozenVec(-100_000, 100_000),
]
# Determine the order to split brushes when carving.
# We want to prioritise axial brushes to improve results.
PLANE_PRIORITY = {
    PlaneType.Z: 0,
    PlaneType.X: 1,
    PlaneType.Y: 2,
    PlaneType.ANY_X: 3,
    PlaneType.ANY_Y: 4,
    PlaneType.ANY_Z: 5,
}


# noinspection PyProtectedMember
@attrs.define(eq=False)
class Geometry:
    """A group of faces with vertices calculated for geometry operations."""
    #: Faces making up this brush solid.
    polys: list['Polygon']

    @classmethod
    def from_brush(cls, brush: vmf_mod.Solid) -> 'Geometry':
        """Convert a VMF brush into a set of polygons with vertices computed."""
        polys = []
        for side in brush:
            norm = side.normal()
            polys.append(Polygon(side, [], Plane(norm, Vec.dot(side.planes[0], norm))))

        return Geometry([
            poly for poly in polys
            if poly._recalculate(polys)
        ])

    def __iter__(self) -> Iterator['Polygon']:
        yield from self.polys

    def rebuild(self, vmf: vmf_mod.VMF, mat: str) -> vmf_mod.Solid:
        """Rebuild faces and the brush for this geometry.

        :param vmf: The VMF faces will be added to.
        :param mat: If faces were newly created, assign this material to the created face.
        """
        return vmf_mod.Solid(vmf, -1, [
            poly.build_face(vmf, mat)
            for poly in self.polys
        ])

    @classmethod
    def unshare_faces(cls, geo: Iterable['Geometry']) -> dict[int, list[int]]:
        """If faces are reused, duplicate the VMF side to make each unique.

        This makes geometry cut by `raw_clip()`/`raw_carve()` valid to use again.

        :returns: Mapping from side IDs to any copies made.
        """
        # Copy faces only if used by both. If only used by one, preserve it.
        used = set()
        ids = defaultdict(list)
        for brush in geo:
            for poly in brush.polys:
                if poly.original is None:
                    continue
                if poly.original in used:
                    old_id = poly.original.id
                    poly.original = poly.original.copy()
                    ids[old_id].append(poly.original.id)
                else:
                    used.add(poly.original)
        return dict(ids)

    def raw_clip(
        self,
        plane: Plane,
    ) -> tuple[Optional['Geometry'], Optional['Geometry']]:
        """Clip this geometry by the specified plane, without modifying faces.

        New polygons will have their face set to `None`, and duplicated polygons will share faces.
        This is not valid, but allows post-processing to track sides precisely.
        The non-raw version calls `unshare_faces()` afterwards.

        :param plane: The plane to clip along.
        :returns: :pycode:`(front, back)` tuple. The two brushes are `self <typing.Self>` and
            `None` if entirely on one side, otherwise this copies faces and returns two solids.
        """
        front_verts = back_verts = 0
        for poly in self.polys:
            for vert in poly.vertices:
                off = plane.normal.dot(vert) - plane.dist
                if off > 1e-6:
                    front_verts += 1
                elif off < -1e-6:
                    back_verts += 1
        if front_verts and not back_verts:
            return (None, self)
        elif back_verts and not front_verts:
            return (self, None)
        front = self
        # Make a copy of each poly, but share faces.
        back = Geometry([
            Polygon(poly.original, list(poly.vertices), poly.plane)
            for poly in self.polys
        ])
        front.polys.append(Polygon(None, [], ~plane))
        back.polys.append(Polygon(None, [], plane))
        front.polys = [
            poly for poly in front.polys
            if poly._recalculate(front.polys)
        ]
        back.polys = [
            poly for poly in back.polys
            if poly._recalculate(back.polys)
        ]
        return front, back

    @classmethod
    def raw_carve(cls, target: Iterable['Geometry'], subtract: 'Geometry') -> list['Geometry']:
        """Carve a set of brushes by another, without modifying faces.

        New polygons will have their face set to `None`, and duplicated polygons will share faces.
        This is not valid, but allows post-processing to track sides precisely.
        The non-raw version calls `unshare_faces()` afterwards.

        :param target: Brushes to carve.
        :param subtract: Brush to cut into the others. If you want multiple, call `!raw_carve()` again.
        :returns: Result brushes. Brushes are omitted if fully carved, or may have been split.
        """
        # Sort planes to prefer axial splits first.
        planes = sorted(subtract.polys, key=lambda poly: PLANE_PRIORITY[poly.plane.type])

        result = []
        todo = list(target)
        for splitter in planes:
            next_todo = []
            for brush in todo:
                front, back = brush.raw_clip(splitter.plane)
                # Anything in front is outside the subtract brush, so it must be kept.
                if front is not None:
                    result.append(front)
                # Back brushes need to be split further.
                if back is not None:
                    next_todo.append(back)
            todo = next_todo
        # Any brushes that were 'back' for all planes are inside = should be removed.
        return result

    def clip(self, plane: Plane) -> tuple[Optional['Geometry'], Optional['Geometry']]:
        """Clip this geometry by the specified plane, creating a valid brush.

        If polygons are used in both brushes, they will be copied.

        :param plane: The plane to clip along.
        :returns: :pycode:`(front, back)` tuple. The two brushes are `self <typing.Self>` and
            `None` if entirely on one side, otherwise this copies faces and returns two solids.
        """
        tup = self.raw_clip(plane)
        self.unshare_faces(filter(None, tup))
        return tup

    @classmethod
    def carve(cls, target: Iterable['Geometry'], subtract: 'Geometry') -> list['Geometry']:
        """Carve a set of brushes by another.

        :param target: Brushes to carve.
        :param subtract: Brush to cut into the others. If you want multiple, call `!carve()` again.
        :returns: Result brushes. Brushes are omitted if fully carved, or may have been split.
        """
        result = cls.carve(target, subtract)
        cls.unshare_faces(result)
        return result


@attrs.define(eq=False)
class Polygon:
    """A face, including the associated vertices."""
    #: The brush side this was constructed from, or None if this was created fresh.
    original: Optional[vmf_mod.Side]
    #: Vertex loop around this polygon. The start point is not specified.
    vertices: list[FrozenVec]
    #: The plane this face is pointing along.
    plane: Plane

    def build_face(self, vmf: VMF, mat: str) -> vmf_mod.Side:
        """Apply the polygon to the face. If the face is not present, create it.

        Returns the face, since it is known to exist.
        """
        if len(self.vertices) < 3:
            raise ValueError('No verts?')
        if self.original is None:
            orient = Matrix.from_basis(x=self.plane.normal)
            vert = self.vertices[0].thaw()
            self.original = vmf_mod.Side(
                vmf,
                [
                    vert + orient.left(16),
                    vert,
                    vert + orient.up(-16),
                ],
                mat=mat,
            )
            self.original.reset_uv()
        elif Vec.dot(self.plane.normal, self.original.normal()) < 0.99:
            # Not aligned, recalculate.
            self.original.planes = [
                self.vertices[0].thaw(),
                self.vertices[1].thaw(),
                Vec.cross(self.plane.normal, self.vertices[1] - self.vertices[0])
            ]
            self.original.reset_uv()
        return self.original

    def _recalculate(self, polys: list['Polygon']) -> bool:
        """Recalculate vertices by intersecting planes. Returns whether this is still valid."""

        # First, initialise with a massive plane.
        orient = Matrix.from_basis(z=self.plane.normal)
        pos = (self.plane.normal * self.plane.dist).freeze()
        self.vertices = [pos + off @ orient for off in MAX_PLANE]

        for other in polys:
            if other is not self:
                self._clip_plane(other.plane)
        return len(self.vertices) >= 3

    def _clip_plane(self, plane: Plane) -> None:
        """Clip these verts against the provided plane."""
        new_verts = []
        count = len(self.vertices)
        for i, vert in enumerate(self.vertices):
            off = plane.normal.dot(vert) - plane.dist
            if off > -1e-6:  # On safe side.
                new_verts.append(vert)
                continue
            mid = plane.intersect_line(self.vertices[(i - 1) % count], vert)
            if mid is not None:
                new_verts.append(mid)
            mid = plane.intersect_line(vert, self.vertices[(i + 1) % count])
            if mid is not None:
                new_verts.append(mid)
        self.vertices = new_verts

    def to_smd_tris(self, links: list[tuple[smd.Bone, float]]) -> Iterator[smd.Triangle]:
        """Convert to SMD triangles. UVs are not fully correct yet.

        :param links: The bone weights to use.
        """
        face = self.original
        if face is None:
            return
        norm = self.plane.normal
        u = face.uaxis.vec() * face.uaxis.scale
        v = face.vaxis.vec() * face.vaxis.scale
        points = [
            smd.Vertex(
                vert.thaw(), norm,
                # TODO: Calculate correct UV coordinates.
                (u.dot(vert) + face.uaxis.offset) / 512.0,
                (v.dot(vert) + face.vaxis.offset) / 512.0,
                links
            )
            for vert in self.vertices
        ]
        yield smd.Triangle(face.mat, points[0], points[1], points[2])
        for a, b in zip(points[2:], points[3:]):
            yield smd.Triangle(face.mat, points[0], a, b)
