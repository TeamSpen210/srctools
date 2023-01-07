"""Implements tools for manipulating geometry."""
import collections
import math
from typing import Optional, Iterator, List

import attr
from srctools import Vec
from srctools.smd import Vertex
from srctools import vmf


@attr.define(eq=False)
class Face:
    """A face."""
    original: Optional[vmf.Side]
    first_edge: 'HalfEdge'
    norm: Vec
    dist: float


@attr.define(eq=False)
class HalfEdge:
    """A half-edge data structure."""
    end_pos: Vertex
    face: Optional[Face]  # Attached face.
    pair: 'HalfEdge'  # The opposite edge. ``self.pair.pair is self``
    next: 'HalfEdge'  # Edge in front. ``self.next.pair.end_pos is self.end_pos``
    prev: 'HalfEdge'  # Edge behind us. ``self.prev.end_pos is self.pair.end_pos``

    @property
    def start_pos(self) -> Vertex:
        """Our pair's position, which is the 'start' position for us."""
        return self.pair.end_pos

    @property
    def length(self) -> float:
        return (self.pair.end_pos.pos - self.end_pos.pos).mag()

    def iter_around_face(self) -> Iterator['HalfEdge']:
        """Move through the edges neighbouring our face."""
        yield self
        edge = self.next
        while edge is not self:
            yield edge
            edge = edge.next

    def iter_around_vert(self) -> Iterator['HalfEdge']:
        """Move through the edges neighbouring our end vertex."""
        yield self
        # To move around a vertex, we move to the next one, then flip
        # so we reversed and point back at the vert.
        edge = self.next.pair
        while edge is not self:
            yield edge
            edge = edge.next.pair


def from_brush(brush: vmf.Solid) -> List[Face]:
    """Convert a VMF brush into a set of edges."""
    faces = {}
    for side in brush:
        norm = side.normal()
        faces[side] = Face(side, None, norm, Vec.dot(side.planes[0], norm))

    all_verts: dict[tuple[float, float, float], Vertex] = {}
    edges: dict[tuple[Vertex, Vertex], HalfEdge] = {}

    # First compute all the vertices.
    # Face verts will then hold a vert and the other two planes.
    face_list = list(faces.values())
    face_verts: dict[Face, set[Vertex]] = {face: set() for face in face_list}
    vert_faces: dict[Vertex, set[Face]] = collections.defaultdict(set)
    for i, face_a in enumerate(face_list):
        for j, face_b in enumerate(face_list[i:], i):
            for face_c in face_list[j:]:
                divisor = Vec.dot(Vec.cross(face_a.norm, face_b.norm), face_c.norm)
                if abs(divisor) < 0.001:
                    continue

                vert_pos = (
                    face_a.dist * Vec.cross(face_b.norm, face_c.norm) +
                    face_b.dist * Vec.cross(face_c.norm, face_a.norm) +
                    face_c.dist * Vec.cross(face_a.norm, face_b.norm)
                ) / divisor
                try:
                    vert = all_verts[vert_pos.as_tuple()]
                except KeyError:
                    vert = all_verts[vert_pos.as_tuple()] = Vertex(vert_pos, Vec(), 0.0, 0.0, [])
                # We need to now also check it's inside the brush, since three
                # planes can potentially also intersect outside the brush.
                for face in face_list:
                    if Vec.dot(vert.pos, face.norm) - face.dist < -0.01:
                        break
                else:
                    face_verts[face_a].add(vert)
                    face_verts[face_b].add(vert)
                    face_verts[face_c].add(vert)
                    vert_faces[vert] |= {face_a, face_b, face_c}

    face_centers = {
        face: sum((vert.pos for vert in verts), Vec()) / len(verts)
        for face, verts in face_verts.items()
    }

    for face, vert_set in face_verts.items():
        # Create a pair of basis vectors orthogonal to the normal,
        # then we can use those to sort the vectors by their angle around
        # the normal - putting them in order. Now we have an edge loop.
        vert_list = list(vert_set)
        u = (vert_list[1].pos - vert_list[0].pos).norm()
        v = Vec.cross(u, face.norm)
        cent = face_centers[face]
        vert_list.sort(key=lambda vt: math.atan2(
            Vec.dot(v, vt.pos - cent),
            Vec.dot(u, vt.pos - cent),
        ))
        last_vert = vert_list[-1]
        edge_loop: list[HalfEdge] = []

        for i, vert in enumerate(vert_list):
            edge = edges[last_vert, vert] = HalfEdge(vert, face, None, None, None)
            edge_loop.append(edge)
            try:
                edge.pair = edges[vert, last_vert]
                edge.pair.pair = edge
            except KeyError:  # Not constructed yet.
                pass
            last_vert = vert
        for i, edge in enumerate(edge_loop):
            edge.next = edge_loop[(i + 1) % len(edge_loop)]
            edge.prev = edge_loop[(i - 1) % len(edge_loop)]
        face.first_edge = edge_loop[0]
    return face_list
