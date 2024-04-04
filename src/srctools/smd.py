"""Parses SMD model/animation data."""
from typing import (
    Any, ClassVar, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union,
)
from typing_extensions import Protocol
from collections import defaultdict
from copy import deepcopy
from operator import itemgetter
import math
import os
import re
import warnings

from srctools.math import Angle, Matrix, Vec, to_matrix


__all__ = [
    'Mesh', 'Triangle', 'Vertex', 'Bone', 'BoneFrame', 'ParseError',
]


class _BinaryFile(Protocol):
    """The methods on files we use."""
    def write(self, __data: bytes) -> object:
        """Writes to the file."""


class Bone:
    """Represents a single bone."""
    __slots__ = ('name', 'parent')
    name: str
    parent: Optional['Bone']

    def __init__(self, name: str, parent: Optional['Bone']) -> None:
        self.name = name
        self.parent = parent

    def __repr__(self) -> str:
        return '<Bone "{}", parent={}>'.format(
            self.name,
            self.parent.name
            if self.parent else
            'None',
        )

    def __copy__(self) -> 'Bone':
        return Bone(self.name, self.parent)

    def __deepcopy__(self, memodict: Optional[Dict[int, Any]] = None) -> 'Bone':
        return Bone(self.name, deepcopy(self.parent, memodict))

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, Bone):
            return self.name == other.name
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        if isinstance(other, Bone):
            return self.name != other.name
        return NotImplemented

    def __hash__(self) -> int:
        """Allow use as a dict key."""
        return hash(self.name)


class BoneFrame:
    """Represents a single frame of bone animation."""
    __slots__ = ('bone', 'position', 'rotation')
    bone: Bone
    position: Vec
    rotation: Angle

    def __init__(self, bone: Bone, position: Vec, rotation: Angle) -> None:
        self.bone = bone
        self.position = position
        if isinstance(rotation, Vec):
            warnings.warn("Use Angle, not Vec.", DeprecationWarning)
            self.rotation = Angle(rotation)
        else:
            self.rotation = rotation

    def __copy__(self) -> 'BoneFrame':
        return BoneFrame(self.bone, self.position, self.rotation)

    def __deepcopy__(self, memodict: Optional[Dict[int, Any]] = None) -> 'BoneFrame':
        return BoneFrame(
            deepcopy(self.bone, memodict),
            self.position.copy(),
            self.rotation.copy(),
        )


class Vertex:
    """A single vertex."""
    __slots__ = ('pos', 'norm', 'tex_u', 'tex_v', 'links')
    pos: Vec
    norm: Vec
    tex_u: float
    tex_v: float
    links: List[Tuple[Bone, float]]

    def __init__(
        self,
        pos: Vec,
        norm: Vec,
        tex_u: float,
        tex_v: float,
        links: List[Tuple[Bone, float]],
    ) -> None:
        self.pos = pos
        self.norm = norm
        self.links = links
        self.tex_u = tex_u
        self.tex_v = tex_v

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'{self.pos!r}, {self.norm!r}, '
            f'{self.tex_u}, {self.tex_v}, {self.links})'
        )

    def copy(self) -> 'Vertex':
        """Copy the vertex."""
        return Vertex(
            self.pos.copy(),
            self.norm.copy(),
            self.tex_u,
            self.tex_v,
            self.links.copy(),
        )

    __copy__ = copy

    def __deepcopy__(self, memodict: Optional[Dict[int, Any]] = None) -> 'Vertex':
        return Vertex(
            self.pos.copy(),
            self.norm.copy(),
            self.tex_u,
            self.tex_v,
            deepcopy(self.links, memodict),
        )

    def with_pos(self, pos: Vec, norm: Optional[Vec] = None) -> 'Vertex':
        """Copy this vertex, changing the position."""
        if norm is None:
            norm = self.norm
        return Vertex(pos, norm, self.tex_u, self.tex_v, self.links)

    def with_uv(self, u: float, v: float) -> 'Vertex':
        """Copy this vertex, changing the UV."""
        return Vertex(self.pos, self.norm, u, v, self.links)


class Triangle:
    """Represents one triangle."""
    __slots__ = ('mat', 'point1', 'point2', 'point3')
    mat: str
    point1: Vertex
    point2: Vertex
    point3: Vertex

    def __init__(self, mat: str, p1: Vertex, p2: Vertex, p3: Vertex) -> None:
        self.mat = mat
        self.point1 = p1
        self.point2 = p2
        self.point3 = p3

    def __iter__(self) -> Iterator[Vertex]:
        yield self.point1
        yield self.point2
        yield self.point3

    def __len__(self) -> int:
        return 3

    def __getitem__(self, item: int) -> Vertex:
        if item == 0:
            return self.point1
        elif item == 1:
            return self.point2
        elif item == 2:
            return self.point3
        else:
            raise IndexError(item)

    def __setitem__(self, item: int, value: Vertex) -> None:
        if not isinstance(value, Vertex):
            raise ValueError('Points can only be vertices!')

        if item == 0:
            self.point1 = value
        elif item == 1:
            self.point2 = value
        elif item == 2:
            self.point3 = value
        else:
            raise IndexError(item)

    def copy(self) -> 'Triangle':
        """Duplicate this triangle."""
        # Copy the points, they shouldn't be shared.
        return Triangle(
            self.mat,
            self.point1.copy(),
            self.point2.copy(),
            self.point3.copy(),
        )

    __copy__ = copy

    def __deepcopy__(self, memodict: Optional[Dict[int, Any]] = None) -> 'Triangle':
        """Duplicate this triangle."""
        return Triangle(
            self.mat,
            self.point1.copy(),
            self.point2.copy(),
            self.point3.copy(),
        )

    def _norm_and_sa(self) -> Vec:
        """Return the normal of this triangle,

        with the magnitude equal to double the surface area.
        See www.ma.ic.ac.uk/~rn/centroid.pdf.
        """
        return Vec.cross(
            self.point2.pos - self.point1.pos,
            self.point3.pos - self.point1.pos,
        )

    def surface_area(self) -> float:
        """Compute the surface area of this triangle."""
        return self._norm_and_sa().mag() / 2.0

    def normal(self) -> Vec:
        """Compute the normal of this triangle, ignoring vertex normals."""
        return self._norm_and_sa().norm()


class ParseError(Exception):
    """Invalid model format."""
    def __init__(self, line_num: Union[int, str], msg: str, *args: object) -> None:
        super().__init__(f'{line_num}: {msg.format(*args)}')


def _clean_file(file: Iterable[bytes]) -> Iterator[Tuple[int, bytes]]:
    line_num = 0
    for line in file:
        line_num += 1
        if b'//' in line:
            line = line.split(b'//', 1)[0]
        if b'#' in line:
            line = line.split(b'#', 1)[0]
        if b';' in line:
            line = line.split(b';', 1)[0]
        line = line.strip()
        if line:
            yield line_num, line


class Mesh:
    """The contents of an SMD model.

    This contains:
    * A bone tree
    * Animation frames
    * Optionally triangle data.
    """
    def __init__(
        self,
        bones: Dict[str, Bone],
        animation: Dict[int, List[BoneFrame]],
        triangles: List[Triangle]
    ) -> None:
        self.bones = bones
        self.animation = animation
        self.triangles = triangles

    def __copy__(self) -> 'Mesh':
        return Mesh(self.bones, self.animation, self.triangles)

    def __deepcopy__(self, memodict: Optional[Dict[int, Any]] = None) -> 'Mesh':
        return Mesh(
            deepcopy(self.bones, memodict),
            deepcopy(self.animation, memodict),
            deepcopy(self.triangles, memodict),
        )

    def root_bone(self) -> Bone:
        """Return the root of our bone hierachy."""
        for bone in self.bones.values():
            if bone.parent is None:
                return bone
        raise ValueError('No root bone?')

    @staticmethod
    def blank(root_name: str) -> 'Mesh':
        """Create an empty mesh, with a single root bone."""
        root_bone = Bone(root_name, None)
        return Mesh(
            {root_name: root_bone},
            {0: [
                BoneFrame(root_bone, Vec(), Angle())
            ]},
            [],
        )

    @staticmethod
    def parse_smd(file: Iterable[bytes]) -> 'Mesh':
        """Parse a SMD file.

        The file argument should be an iterable of lines.
        It is parsed in binary, since non-ASCII characters are not
        permitted in SMDs.
        """
        file_iter = _clean_file(file)

        bones: Optional[Dict[int, Bone]] = None
        anim: Optional[Dict[int, List[BoneFrame]]] = None
        tri: List[Triangle] = []

        line_num = 1

        for line_num, line in file_iter:
            if line.startswith(b'version'):
                version = line.split(None, 1)[1]
                if version != b'1':
                    raise ParseError(
                        line_num,
                        'Unknown version {}!',
                        version,
                    )
            elif line == b'nodes':
                if bones is not None:
                    raise ParseError(
                        line_num,
                        'Duplicate bone section!',
                    )
                bones = Mesh._parse_smd_bones(file_iter)
            elif line == b'skeleton':
                if anim is not None:
                    raise ParseError(
                        line_num,
                        'Duplicate animation section!',
                    )
                elif bones is None:
                    raise ParseError(
                        line_num,
                        'Animations section before bones section!'
                    )
                anim = Mesh._parse_smd_anim(file_iter, bones)
            elif line == b'triangles':
                if bones is None:
                    raise ParseError(
                        line_num,
                        'Triangles section before bones section!'
                    )
                tri.extend(Mesh._parse_smd_tri(file_iter, bones))

        if bones is None:
            raise ParseError(line_num, 'No bone section!')

        if anim is None:
            raise ParseError(line_num, 'No animation section!')

        return Mesh({
            bone.name: bone
            for bone in
            bones.values()
        }, anim, tri)

    @staticmethod
    def _parse_smd_bones(file_iter: Iterator[Tuple[int, bytes]]) -> Dict[int, Bone]:
        """Parse the 'nodes' section of SMDs."""
        bones: Dict[int, Bone] = {}
        for line_num, line in file_iter:
            if line == b'end':
                return bones
            match = re.fullmatch(
                br'([0-9]+)\s*"([^"]*)"\s*(-?[0-9]+)',
                line,
            )
            if match is None:
                raise ParseError(line_num, 'Invalid line!') from None

            bone_ind_bytes, bone_name, bone_parent_bytes = match.groups()
            try:
                bone_ind = int(bone_ind_bytes)
                bone_parent = int(bone_parent_bytes)
            except ValueError:  # None.groups()
                raise ParseError(line_num, 'Invalid line!') from None
            else:
                if bone_parent == -1:
                    parent_ind = None
                else:
                    try:
                        parent_ind = bones[bone_parent]
                    except KeyError:
                        raise ParseError(
                            line_num,
                            'Undefined parent bone {}!',
                            bone_parent,
                        ) from None
                bones[bone_ind] = Bone(bone_name.decode('ascii'), parent_ind)
        raise ParseError('end', 'No end to nodes section!')

    @staticmethod
    def _parse_smd_anim(
        file_iter: Iterator[Tuple[int, bytes]],
        bones: Dict[int, Bone],
    ) -> Dict[int, List[BoneFrame]]:
        """Parse the 'skeleton' section of SMDs."""
        frames: Dict[int, List[BoneFrame]] = {}
        time: Optional[int] = None
        for line_num, line in file_iter:
            if line.startswith((b'//', b'#', b';')):
                continue
            if line.startswith(b'time'):
                try:
                    time = int(line[4:])
                except ValueError:
                    raise ParseError(line_num, 'Invalid time value!') from None
                if time in frames:
                    raise ParseError(line_num, f'Duplicate frame time {time}!')
                frames[time] = []
            elif line == b'end':
                return frames
            else:  # Bone.
                if time is None:
                    raise ParseError(line_num, 'No time specification!')
                try:
                    byt_ind, byt_x, byt_y, byt_z, byt_pit, byt_yaw, byt_rol = line.split()
                    pos = Vec(float(byt_x), float(byt_y), float(byt_z))
                    rot = Angle(math.degrees(float(byt_pit)), math.degrees(float(byt_yaw)), math.degrees(float(byt_rol)))
                except ValueError:
                    raise ParseError(line_num, 'Invalid line!') from None
                try:
                    bone = bones[int(byt_ind)]
                except KeyError:
                    raise ParseError(line_num, 'Unknown bone index {}!', int(byt_ind)) from None
                frames[time].append(BoneFrame(bone, pos, rot))

        raise ParseError('end', 'No end to skeleton section!')

    @staticmethod
    def _parse_smd_tri(file_iter: Iterator[Tuple[int, bytes]], bones: Dict[int, Bone]) -> List[Triangle]:
        """Parse the 'triangles' section of SMDs."""
        tris: List[Triangle] = []
        # Temporary vertex, which we overwrite in the loop.
        points = [Vertex(Vec(), Vec(), 0.0, 0.0, [])] * 3
        for line_num, line in file_iter:
            if line == b'end':
                return tris
            try:
                mat_name = line.decode('ascii')
            except UnicodeDecodeError as exc:
                raise ParseError(
                    line_num,
                    'Non-ASCII material: {} at position {} - {}',
                    exc.reason,
                    exc.start
                ) from None

            # The file extension is ignored, and we may have extra whitespace.
            mat_name, _ = os.path.splitext(mat_name.rstrip('\\/ \t\b\n\r'))

            # Grab the three lines.
            for i in range(3):
                try:
                    line_num, line = next(file_iter)
                except StopIteration:
                    raise ParseError('end', 'Incomplete triangles!') from None
                try:
                    (
                        byt_parent,
                        x, y, z,
                        nx, ny, nz,
                        byt_tex_u, byt_tex_v,
                        *links_raw,
                    ) = line.split()
                except ValueError:
                    raise ParseError(line_num, 'Not enough values!') from None
                try:
                    pos = Vec(float(x), float(y), float(z))
                    norm = Vec(float(nx), float(ny), float(nz))
                except ValueError:
                    raise ParseError(line_num, 'Invalid normal or position!') from None
                try:
                    tex_u = float(byt_tex_u)
                    tex_v = float(byt_tex_v)
                except ValueError:
                    raise ParseError(
                        line_num, 'Invalid texture UV: ({}, {})',
                        byt_tex_u, byt_tex_v,
                    ) from None

                try:
                    parent = bones[int(byt_parent)]
                except KeyError:
                    raise ParseError(line_num, 'Invalid bone {}!', int(byt_parent)) from None

                if links_raw:
                    link_count = int(links_raw[0])

                    if (link_count * 2 + 1) != len(links_raw):
                        raise ParseError(line_num, 'Extra weight number: {}', links_raw)

                    links = []
                    for off in range(1, len(links_raw), 2):
                        try:
                            bone = bones[int(links_raw[off])]
                        except KeyError:
                            raise ParseError(line_num, 'Unknown bone {}!', links_raw[off]) from None
                        links.append((bone, float(links_raw[off+1])))
                    if not links:
                        # Okay, there's no links set here, use the first index.
                        links = [(parent, 1.0)]
                else:
                    links = [(parent, 1.0)]

                points[i] = Vertex(pos, norm, tex_u, tex_v, links)
            tris.append(Triangle(mat_name, *points))

        raise ParseError('end', 'No end to triangles section!')

    def export(self, file: _BinaryFile) -> None:
        """Write out the SMD to the given file."""
        file.write(b"version 1\nnodes\n")

        # Deconstruct the tree into indexes, with parents before children.
        bone_indexes: Dict[Bone, int] = {}
        next_ind = 0
        todo: Set[Bone] = set(self.bones.values())
        while todo:
            changed = False
            for bone in list(todo):
                if not bone.parent or bone.parent in bone_indexes:
                    bone_indexes[bone] = next_ind
                    if bone.parent is None:
                        parent_ind = -1
                    else:
                        parent_ind = bone_indexes[bone.parent]  # or KeyError.
                    file.write(b'%i "%s" %i\n' % (
                        next_ind,
                        bone.name.encode('ascii'),
                        parent_ind,
                    ))
                    next_ind += 1
                    todo.remove(bone)
                    changed = True
            if not changed:
                # Every bone had a parent, so it must be a loop somewhere!
                raise ValueError('Loop in bone parenting! Remaining: {}', list(todo))

        file.write(b'end\nskeleton\n')
        for time, frame in sorted(self.animation.items(), key=itemgetter(0)):
            file.write(b'time %i\n' % time)
            for bone_pose in frame:
                x, y, z = bone_pose.position
                pit, yaw, rol = bone_pose.rotation
                file.write(b'%i %.6f %.6f %.6f  %.6f %.6f %.6f\n' % (
                    bone_indexes[bone_pose.bone],
                    x, y, z,
                    math.radians(pit), math.radians(yaw), math.radians(rol),
                ))
        file.write(b'end\n')
        if self.triangles:
            file.write(b'triangles\n')
            for tri in self.triangles:
                file.write(tri.mat.encode('ascii') + b'\n')
                for vert in tri:
                    # If there's only one bone, it's the first value.
                    # Otherwise they're appended to the end and the first is
                    # ignored (but must be valid).
                    assert len(vert.links) > 0
                    file.write(
                        b'%i\t%.6f %.6f %.6f\t'  # bone index, position XYZ
                        b'%.6f %.6f %.6f\t'  # Normal XYZ
                        b'%.6f %.6f' % (  # UV
                            bone_indexes[vert.links[0][0]],
                            vert.pos.x, vert.pos.y, vert.pos.z,
                            vert.norm.x, vert.norm.y, vert.norm.z,
                            vert.tex_u, vert.tex_v,
                        )
                    )
                    if len(vert.links) > 1:
                        file.write(b'%i' % (len(vert.links), ))
                        for bone, weight in vert.links:
                            file.write(b' %i %.6f' % (bone_indexes[bone], weight))
                    file.write(b'\n')
            file.write(b'end\n')

    def append_model(
        self,
        mdl: 'Mesh',
        rotation: Union[Angle, Matrix, Vec, None] = None,
        offset: Optional[Vec] = None,
        scale: Union[float, Vec] = 1.0,
    ) -> None:
        """Append another model's geometry onto this one.

        All geometry is attached to the root bone.
        """
        if not mdl.triangles:
            # Nothing to add.
            return
        if offset is None:
            offset = Vec()

        scaling: Vec = Vec(1.0, 1.0, 1.0)
        if isinstance(scale, float):
            scaling = Vec(scale, scale, scale)
        elif isinstance(scale, Vec):
            scaling = scale

        bone_link = [(self.root_bone(), 1.0)]

        matrix = Matrix()

        # Set the scale
        matrix[0, 0] = scaling[0]
        matrix[1, 1] = scaling[1]
        matrix[2, 2] = scaling[2]

        # Rotate the matrix
        matrix @= to_matrix(rotation)

        # Secondary matrix for the normals
        inv = matrix.inverse()
        itm = inv.transpose()

        for orig_tri in mdl.triangles:
            new_tri = orig_tri.copy()
            for vert in new_tri:
                vert.links[:] = bone_link

                # Transform the vertex
                vert.norm @= itm
                vert.norm = vert.norm.norm()
                vert.pos @= matrix
                vert.pos += offset

            self.triangles.append(new_tri)

    # The triangles required for a prism.
    # Each sublist is a triangle.
    # The tuples are (x, y, z, u, v).
    _BBOX_MESH_DATA: ClassVar[Sequence[Sequence[Tuple[int, int, int, float, float]]]] = [
        [
            (-1, -1, -1, 0.0, 0.0),
            (-1, +1, +1, 1.0, 1.0),
            (-1, +1, -1, 0.0, 1.0),
        ],
        [
            (-1, +1, -1, 0.0, 0.0),
            (+1, +1, +1, 1.0, 1.0),
            (+1, +1, -1, 0.0, 1.0),
        ],
        [
            (+1, +1, -1, 0.0, 0.0),
            (+1, -1, +1, 1.0, 1.0),
            (+1, -1, -1, 0.0, 1.0),
        ],
        [
            (+1, -1, -1, 0.0, 0.0),
            (-1, -1, +1, 1.0, 1.0),
            (-1, -1, -1, 0.0, 1.0),
        ],
        [
            (-1, +1, -1, 0.0, 0.0),
            (+1, -1, -1, 1.0, 1.0),
            (-1, -1, -1, 0.0, 1.0),
        ],
        [
            (+1, +1, +1, 0.0, 0.0),
            (-1, -1, +1, 1.0, 1.0),
            (+1, -1, +1, 0.0, 1.0),
        ],
        [
            (-1, -1, -1, 0.0, 0.0),
            (-1, -1, +1, 1.0, 0.0),
            (-1, +1, +1, 1.0, 1.0),
        ],
        [
            (-1, +1, -1, 0.0, 0.0),
            (-1, +1, +1, 1.0, 0.0),
            (+1, +1, +1, 1.0, 1.0),
        ],
        [
            (+1, +1, -1, 0.0, 0.0),
            (+1, +1, +1, 1.0, 0.0),
            (+1, -1, +1, 1.0, 1.0),
        ],
        [
            (+1, -1, -1, 0.0, 0.0),
            (+1, -1, +1, 1.0, 0.0),
            (-1, -1, +1, 1.0, 1.0),
        ],
        [
            (-1, +1, -1, 0.0, 0.0),
            (+1, +1, -1, 1.0, 0.0),
            (+1, -1, -1, 1.0, 1.0),
        ],
        [
            (+1, +1, +1, 0.0, 0.0),
            (-1, +1, +1, 1.0, 0.0),
            (-1, -1, +1, 1.0, 1.0),
        ],
    ]

    @classmethod
    def build_bbox(cls, root_bone: str, mat: str, bbox_min: Vec, bbox_max: Vec) -> 'Mesh':
        """Construct a mesh for a bounding box."""
        mesh = cls.blank(root_bone)
        [root] = mesh.bones.values()
        links = [(root, 1.0)]

        bbox_min, bbox_max = Vec.bbox(bbox_min, bbox_max)

        for tri_def in cls._BBOX_MESH_DATA:
            tri = Triangle(mat, *[
                Vertex(
                    Vec(
                        bbox_max.x if x > 0 else bbox_min.x,
                        bbox_max.y if y > 0 else bbox_min.y,
                        bbox_max.z if z > 0 else bbox_min.z,
                    ), Vec(x, y, z).norm(),
                    u, v, links.copy()
                )
                for x, y, z, u, v in tri_def
            ])
            mesh.triangles.append(tri)
        return mesh

    def weld_vertexes(self, dist_tol: float = 1e-5, normal_tol: float = 0.999) -> None:
        """Run through all vertexes in the triangles, 'welding' close ones together.

        This will result in adjacent faces sharing vertex objects.
        The shared vertexes should have approximately the same position as well
        as normal. This can be accomplished using a mesh with smoothed normals
        as with most studioMDL collision models, or by giving each section the
        same unique normal.
        """
        # pos -> list of vertexes close to here.
        weld_table: Dict[Tuple[float, float, float], List[Vertex]] = {}
        for tri in self.triangles:
            vert: Vertex
            for i, vert in enumerate(tri):
                key = vert.pos.x // 2.0, vert.pos.y // 2.0, vert.pos.z // 2.0
                try:
                    existing = weld_table[key]
                except KeyError:
                    weld_table[key] = [vert]
                    continue
                for other_vert in existing:
                    if (vert.pos - other_vert.pos).mag() < dist_tol and Vec.dot(vert.norm, other_vert.norm) > normal_tol:
                        tri[i] = other_vert
                        break
                else:
                    existing.append(vert)

    def split_collision(self) -> List['Mesh']:
        """Partition a concave collision mesh into each convex volume.

        This will first 'weld' the vertexes, so each convex volume will share
        vertex objects.
        """
        self.weld_vertexes()
        vert_to_tris: Dict[Vertex, List[Triangle]] = defaultdict(list)
        for tri in self.triangles:
            for vert in tri:
                vert_to_tris[vert].append(tri)

        groups: List[Set[Triangle]] = []
        todo = set(self.triangles)
        # To group, we have to recursively go through the verts.
        # We use the id() of vertexes to match, since they're the same now.
        while todo:
            start = todo.pop()
            unchecked = {start}
            group = {start}
            verts: Set[Vertex] = set()
            groups.append(group)
            while unchecked:
                tri = unchecked.pop()
                for vert in tri:
                    if vert in verts:
                        continue
                    verts.add(vert)
                    group.update(vert_to_tris[vert])
                    unchecked.update(vert_to_tris[vert])
                unchecked.discard(tri)  # The above update() will add it back.
            todo -= group
        return [
            Mesh(self.bones, self.animation, list(group))
            for group in groups
        ]

    def compute_volume(self) -> float:
        """Compute the volume of this mesh. It does not need to be convex.

        See www.ma.ic.ac.uk/~rn/centroid.pdf.
        """
        # noinspection PyProtectedMember
        return sum(
            Vec.dot(tri.point1.pos, tri._norm_and_sa())
            for tri in self.triangles
        ) / 6.0

    def smooth_normals(self) -> None:
        """Replace all normals with ones smoothing adjacient faces."""
        vert_to_tris: Dict[
            Tuple[float, float, float],
            Tuple[List[Triangle], List[Vertex]]
        ] = defaultdict(lambda: ([], []))
        tri_to_normal: Dict[Triangle, Vec] = {}
        for tri in self.triangles:
            for vert in tri:
                tris, verts = vert_to_tris[vert.pos.as_tuple()]
                tris.append(tri)
                verts.append(vert)
            tri_to_normal[tri] = tri.normal()
        for tris, verts in vert_to_tris.values():
            normal = sum(map(tri_to_normal.__getitem__, tris), Vec()).norm()
            for vert in verts:
                vert.norm = normal

    def flatten_normals(self) -> None:
        """Replace all vertex normals with the triangle normal."""
        for tri in self.triangles:
            norm = tri.normal()
            for vert in tri:
                vert.norm = norm
