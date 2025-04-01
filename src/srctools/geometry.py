"""Implements tools for manipulating geometry."""
from typing import Optional, Literal

from collections.abc import Iterator

from enum import Flag

import attrs

from srctools import FrozenVec, Matrix, Vec
from srctools import vmf, smd
from srctools.bsp import Plane


class Intersect(Flag):
    """Indicates the result of an intersection."""
    COPLANAR = 0
    FRONT = 1
    BACK = 2
    SPANNING = 3


MAX_PLANE = [
    FrozenVec(-100_000, -100_000),
    FrozenVec(100_000, -100_000),
    FrozenVec(100_000, 100_000),
    FrozenVec(-100_000, 100_000),
]


# noinspection PyProtectedMember
@attrs.define(eq=False)
class Geometry:
    """A group of faces with vertices calculated for geometry operations."""
    polys: list['Polygon']

    @classmethod
    def from_brush(cls, brush: vmf.Solid) -> 'Geometry':
        """Convert a VMF brush into a set of polygons with vertices computed."""
        polys = []
        for side in brush:
            norm = side.normal()
            polys.append(Polygon(side, [], Plane(norm, Vec.dot(side.planes[0], norm))))

        return Geometry([
            poly for poly in polys
            if poly._recalculate(polys)
        ])


@attrs.define(eq=False)
class Polygon:
    """A face, including the associated vertices."""
    original: Optional[vmf.Side]
    vertices: list[FrozenVec]
    plane: Plane

    @property
    def norm(self) -> Vec:
        """Make it easier to access the plane normal."""
        return self.plane.normal

    @property
    def plane_dist(self) -> float:
        """Make it easier to access the plane distance."""
        return self.plane.dist

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
        """Convert to SMD triangles."""
        face = self.original
        if face is None:
            return
        norm = self.norm
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
