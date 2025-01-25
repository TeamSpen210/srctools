"""Implements tools for manipulating geometry."""
from typing import Optional

import collections
import math

import attrs

from srctools import FrozenVec, Vec
from srctools import vmf
from srctools.bsp import Plane


@attrs.define(eq=False)
class Polygon:
    """A face, including the associated vertices."""
    original: Optional[vmf.Side]
    vertices: list[FrozenVec]
    plane: Plane

    @property
    def norm(self) -> Vec:
        return self.plane.normal

    @property
    def plane_dist(self) -> float:
        return self.plane.dist


def from_brush(brush: vmf.Solid) -> list[Polygon]:
    """Convert a VMF brush into a set of polygons with vertices computed."""
    polys = {}
    for side in brush:
        norm = side.normal()
        polys[side] = Polygon(side, [], Plane(norm, Vec.dot(side.planes[0], norm)))

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

    poly_centers = {
        poly: sum((vert for vert in verts), FrozenVec()) / len(verts)
        for poly, verts in poly_verts.items()
    }

    for poly, vert_set in poly_verts.items():
        # Create a pair of basis vectors orthogonal to the normal,
        # then we can use those to sort the vectors by their angle around
        # the normal - putting them in order. Now we have an edge loop.
        poly.vertices = list(vert_set)
        u = (poly.vertices[1] - poly.vertices[0]).norm()
        v = FrozenVec.cross(u, poly.norm)
        cent = poly_centers[poly]
        poly.vertices.sort(key=lambda vt: math.atan2(
            Vec.dot(v, vt.pos - cent),
            Vec.dot(u, vt.pos - cent),
        ))
    return poly_list
