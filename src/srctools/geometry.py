"""Implements tools for manipulating geometry."""
from typing import Optional

from collections.abc import Iterable, Iterator, MutableMapping

import attrs

from srctools import EmptyMapping, FrozenVec, Matrix, VMF, Vec
from srctools import vmf as vmf_mod, smd
from srctools.bsp import Plane, PlaneType


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
        """Rebuild faces and the brush for this geometry."""
        return vmf_mod.Solid(vmf, -1, [
            poly.build_face(vmf, mat)
            for poly in self.polys
        ])

    def clip(
        self, plane: Plane,
        side_mapping: MutableMapping[int, int] = EmptyMapping,
    ) -> tuple[Optional['Geometry'], Optional['Geometry']]:
        """Clip this geometry by the specified plane.

        Returns self/None if entirely on one side, otherwise copies the geo and returns two solids.
        If provided, side_mapping will be set to have original -> new side IDs, if copying is
        required.
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
        # Make a copy of each poly, but share faces for now.
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
        # Now, copy faces only if used by both. If only used by one, preserve it.
        front_used = set()
        for poly in front.polys:
            if poly.original is not None:
                front_used.add(poly.original)
        for poly in back.polys:
            if poly.original is not None and poly.original in front_used:
                poly.original = poly.original.copy(side_mapping=side_mapping)
        return front, back

    @classmethod
    def carve(cls, target: Iterable['Geometry'], subtract: 'Geometry') -> list['Geometry']:
        """Carve a set of brushes by another."""
        # Sort planes to prefer axial splits first.
        planes = sorted(subtract.polys, key=lambda poly: PLANE_PRIORITY[poly.plane.type])

        result = []
        todo = list(target)
        for splitter in planes:
            next_todo = []
            for brush in todo:
                front, back = brush.clip(splitter.plane)
                # Anything in front is outside the subtract brush, so it must be kept.
                if front is not None:
                    result.append(front)
                # Back brushes need to be split further.
                if back is not None:
                    next_todo.append(back)
            todo = next_todo
        # Any brushes that were 'back' for all planes are inside = should be removed.
        return result


@attrs.define(eq=False)
class Polygon:
    """A face, including the associated vertices."""
    original: Optional[vmf_mod.Side]
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
                new_verts.append(mid.freeze())
            mid = plane.intersect_line(vert, self.vertices[(i + 1) % count])
            if mid is not None:
                new_verts.append(mid.freeze())
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
