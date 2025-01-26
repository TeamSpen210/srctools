"""Implements tools for manipulating geometry.

This is based on `csg.js <https://github.com/evanw/csg.js/>`_
"""
from typing import Iterable, Optional

from collections.abc import Iterator

from enum import Flag
import collections
import math

import attrs

from srctools import FrozenVec, Vec
from srctools import vmf, smd
from srctools.bsp import Plane


__all__ = ['Polygon', 'VertBrush']


class Intersect(Flag):
    """Indicates the result of an intersection."""
    COPLANAR = 0
    FRONT = 1
    BACK = 2
    SPANNING = 3

EPSILON = 1e-6


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

    def copy(self) -> 'Polygon':
        """Copy the data for this polygon, though the VMF side is kept shared."""
        return Polygon(self.original, self.vertices.copy(), self.plane.copy())

    @classmethod
    def from_brush(cls, brush: vmf.Solid) -> 'VertBrush':
        """Convert a VMF brush into a set of polygons with vertices computed."""
        polys = {}
        for side in brush:
            norm = side.normal()
            polys[side] = cls(side, [], Plane(norm, Vec.dot(side.planes[0], norm)))

        all_verts: dict[FrozenVec, FrozenVec] = {}

        # First compute all the vertices.
        # Poly verts will then hold a vert and the other two planes.
        poly_list = list(polys.values())
        poly_verts: dict[Polygon, set[FrozenVec]] = {poly: set() for poly in poly_list}
        vert_polys: dict[FrozenVec, set[Polygon]] = collections.defaultdict(set)
        for i, poly_a in enumerate(poly_list):
            for j, poly_b in enumerate(poly_list[i:], i):
                for poly_c in poly_list[j:]:
                    divisor = Vec.dot(Vec.cross(poly_a.norm, poly_b.norm), poly_c.norm)
                    if abs(divisor) < 0.001:
                        continue

                    vert = (
                        poly_a.plane_dist * FrozenVec.cross(poly_b.norm, poly_c.norm) +
                        poly_b.plane_dist * FrozenVec.cross(poly_c.norm, poly_a.norm) +
                        poly_c.plane_dist * FrozenVec.cross(poly_a.norm, poly_b.norm)
                    ) / divisor
                    vert = all_verts.setdefault(vert, vert)
                    # We need to now also check it's inside the brush, since three
                    # planes can potentially also intersect outside the brush.
                    for poly in poly_list:
                        if Vec.dot(vert, poly.norm) - poly.plane_dist < -0.01:
                            break
                    else:
                        poly_verts[poly_a].add(vert)
                        poly_verts[poly_b].add(vert)
                        poly_verts[poly_c].add(vert)
                        vert_polys[vert] |= {poly_a, poly_b, poly_c}

        for poly, vert_set in poly_verts.items():
            # Create a pair of basis vectors orthogonal to the normal,
            # then we can use those to sort the vectors by their angle around
            # the normal - putting them in order. Now we have an edge loop.
            poly.vertices = verts = list(vert_set)
            u = (verts[1] - verts[0]).norm()
            v = FrozenVec.cross(u, poly.norm)
            cent = sum((vert for vert in verts), FrozenVec()) / len(verts)
            verts.sort(key=lambda vt: math.atan2(
                Vec.dot(v, vt - cent),
                Vec.dot(u, vt - cent),
            ))
        return VertBrush(poly_list)

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

    def split(
        self, plane: Plane,
        coplanar_front: list['Polygon'], coplanar_back: list['Polygon'],
        front: list['Polygon'], back: list['Polygon'],
    ) -> None:
        """Split this polygon by the plane, putting parts in appropriate lists."""

        poly_type = Intersect.COPLANAR
        vert_types = []
        for vert in self.vertices:
            t = Vec.dot(plane.normal, vert) - plane.dist
            if t < -EPSILON:
                vert_type = Intersect.BACK
            elif t > EPSILON:
                vert_type = Intersect.FRONT
            else:
                vert_type = Intersect.COPLANAR
            poly_type |= vert_type
            vert_types.append(vert_type)

        if poly_type is Intersect.COPLANAR:
            if Vec.dot(self.norm, plane.normal) > 0:
                coplanar_front.append(self)
            else:
                coplanar_back.append(self)
        elif poly_type is Intersect.FRONT:
            front.append(self)
        elif poly_type is Intersect.BACK:
            back.append(self)
        elif poly_type is Intersect.SPANNING:
            front_verts = []
            back_verts = []
            for i, (vert_1, type_1) in enumerate(zip(self.vertices, vert_types)):
                j = (i + 1) % len(self.vertices)
                vert_2 = self.vertices[j]
                type_2 = vert_types[j]
                if type_1 is not Intersect.BACK:
                    front_verts.append(vert_1)
                if type_1 is not Intersect.FRONT:
                    back_verts.append(vert_1)
                if (type_1 | type_2) is Intersect.SPANNING:
                    # Overlap, need to add verts.
                    t = (plane.dist - plane.normal.dot(vert_1)) / plane.normal.dot(vert_2 - vert_1)
                    new_vert = vert_1 + (vert_2 - vert_1) * t
                    front_verts.append(new_vert)
                    back_verts.append(new_vert)
            if len(front_verts) >= 3:
                front.append(Polygon(self.original, front_verts, self.plane))
                back.append(Polygon(self.original, back_verts, self.plane))


@attrs.define(eq=False)
class VertBrush:
    """A set of polygons forming a brush, with vertices calculated.

    CSG operations can be performed on this, before it is converted back to a VMF brush.
    """
    polys: list['Polygon']

    def _copy(self) -> list['Polygon']:
        """Copy our polygons."""
        return [poly.copy() for poly in self.polys]

    def to_vmf(self, vmf: vmf.VMF) -> list[vmf.Solid]:
        """Convert this """

    def __or__(self, other: 'VertBrush') -> 'Node':
        """Union two brushes."""
        node_a = Node()
        node_a.build(self._copy())
        node_b = Node()
        node_b.build(other._copy())
        node_a.clip_to(node_b)
        node_b.clip_to(node_a)
        node_b.invert()
        node_b.clip_to(node_a)
        node_b.invert()
        node_a.build(node_b.iter_polys())
        return node_a
        # return VertBrush(list(node_a.iter_polys()))

    def __sub__(self, other: 'VertBrush') -> 'VertBrush':
        """Carve the other brush out of this one."""
        node_a = Node()
        node_b = Node()
        node_a.build(self._copy())
        node_b.build(other._copy())
        node_a.invert()
        node_a.clip_to(node_b)
        node_b.clip_to(node_a)
        node_b.invert()
        node_b.clip_to(node_a)
        node_b.invert()

        node_a.build(node_b.iter_polys())
        node_a.invert()
        return VertBrush(list(node_a.iter_polys()))

    def __and__(self, other: 'VertBrush') -> 'VertBrush':
        """Intersect two brushes."""
        node_a = Node()
        node_b = Node()
        node_a.build(self._copy())
        node_b.build(other._copy())
        node_a.invert()
        node_b.clip_to(node_a)
        node_b.invert()
        node_a.clip_to(node_b)
        node_b.clip_to(node_a)

        node_a.build(node_b.iter_polys())
        node_a.invert()
        return VertBrush(list(node_a.iter_polys()))


@attrs.define
class Node:
    """A BSP tree constructed out of polygons."""
    plane: Optional[Plane] = None
    front: Optional['Node'] = None
    back: Optional['Node'] = None
    polygons: list[Polygon] = attrs.Factory(list)

    def build(self, polys: Iterable[Polygon]) -> 'Node':
        """Fill the tree using a set of polygons."""
        todo = list(polys)
        if not todo:
            return self

        front = []
        back = []

        if self.plane is None:
            # TODO pick plane to split by?
            self.plane = todo[0].plane
        while todo:
            poly = todo.pop()
            poly.split(self.plane, todo, todo, front, back)
        if front:
            if self.front is None:
                self.front = Node()
            self.front.build(front)
        if back:
            if self.back is None:
                self.back = Node()
            self.back.build(front)

    def copy(self) -> 'Node':
        """Copy this, its children and polygons."""
        copy = Node()
        copy.plane = self.plane.copy() if self.plane is not None else None
        copy.front = self.front.copy() if self.front is not None else None
        copy.back = self.back.copy() if self.back is not None else None
        copy.polygons = [poly.copy() for poly in self.polygons]
        return copy

    def invert(self) -> None:
        """Flip this tree inside out, so solid space is empty and vice versa."""
        # TODO: Copy in the process, make this __invert__?
        for poly in self.polygons:
            poly.vertices.reverse()
            poly.plane = ~poly.plane
        if self.front is not None:
            self.front.invert()
        if self.back is not None:
            self.back.invert()
        self.front, self.back = self.back, self.front

    def iter_polys(self) -> Iterator[Polygon]:
        """Iterate over all polygons in the tree."""
        yield from self.polygons
        if self.front is not None:
            yield from self.front.iter_polys()
        if self.back is not None:
            yield from self.back.iter_polys()

    def clip_polys(self, polygons: Iterable[Polygon]) -> list[Polygon]:
        """Remove polygons from the list that intersect us."""
        if self.plane is None:
            return list(polygons)
        front_polys = []
        back_polys = []
        for poly in polygons:
            poly.split(self.plane, front_polys, back_polys, front_polys, back_polys)
        if self.front is not None:
            front_polys = self.front.clip_polys(front_polys)
        if self.back is not None:
            back_polys = self.back.clip_polys(back_polys)
            return front_polys + back_polys
        else:
            # Why this else??
            return front_polys

    def clip_to(self, tree: 'Node') -> None:
        """Remove polygons from this tree that intersect the other tree."""
        self.polygons = tree.clip_polys(self.polygons)
        if self.front is not None:
            self.front.clip_to(tree)
        if self.back is not None:
            self.back.clip_to(tree)
