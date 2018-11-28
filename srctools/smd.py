"""Parses SMD model/animation data."""
import re
import math
from typing import List, Optional, Dict, Tuple, Iterator, Iterable, Union

from srctools import Vec


class Bone:
    """Represents a single bone."""
    __slots__ = ('name', 'parent')

    def __init__(self, name: str, parent: Optional['Bone']):
        self.name = name
        self.parent = parent

    def __repr__(self) -> str:
        return '<Bone "{}", parent={}>'.format(
            self.name,
            self.parent.name
            if self.parent else
            'None',
        )


class BoneFrame:
    """Represents a single frame of bone animation."""
    __slots__ = ('bone', 'position', 'rotation')

    def __init__(self, bone: Bone, position: Vec, rotation: Vec):
        self.bone = bone
        self.position = position
        self.rotation = rotation


class Vertex:
    """A single vertex."""
    __slots__ = ('pos', 'norm', 'tex_u', 'tex_v', 'links')
    def __init__(
        self,
        pos: Vec,
        norm: Vec,
        tex_u: float,
        tex_v: float,
        links: List[Tuple[Bone, float]],
    ):
        self.pos = pos
        self.norm = norm
        self.links = links
        self.tex_u = tex_u
        self.tex_v = tex_v

    def __repr__(self) -> str:
        return 'Vertex({!r}, {!r}, {}, {}, {})'.format(
            self.pos, self.norm, self.tex_u, self.tex_v, self.links
        )


class Triangle:
    """Represents one triangle."""
    __slots__ = ('mat', 'point1', 'point2', 'point3')

    def __init__(self, mat: str, p1: Vertex, p2: Vertex, p3: Vertex):
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


class ParseError(Exception):
    """Invalid model format."""
    def __init__(self, line_num: Union[int, str], msg: str, *args: object):
        super(ParseError, self).__init__('{}: {}'.format(
            line_num,
            msg.format(*args),
        ))


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


class Model:
    """The contents of an SMD model.

    This contains:
    * A bone tree
    * Animation frames
    * Optionally triangle data.
    """
    def __init__(
        self,
        bones: List[Bone],
        animation: Dict[int, List[BoneFrame]],
        triangles: List[Triangle]
    ):
        self.bones = bones
        self.animation = animation
        self.triangles = triangles

    @staticmethod
    def parse_smd(file: Iterable[bytes]):
        """Parse a SMD file.

        The file argument should be an iterable of lines.
        It is parsed in binary, since non-ASCII characters are not
        permitted in SMDs.
        """
        file_iter = _clean_file(file)

        bones = None
        anim = None
        tri = None

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
                bones = Model._parse_smd_bones(file_iter)
            elif line == b'skeleton':
                if anim is not None:
                    raise ParseError(
                        line_num,
                        'Duplicate animation section!',
                    )
                anim = Model._parse_smd_anim(file_iter, bones)
            elif line == b'triangles':
                if tri is not None:
                    raise ParseError(
                        line_num,
                        'Duplicate triangle section!',
                    )
                elif bones is None:
                    raise ParseError(
                        line_num,
                        'Triangles section before bones section!'
                    )
                tri = Model._parse_smd_tri(file_iter, bones)

        if bones is None:
            raise ParseError(line_num, 'No bone section!')

        if anim is None:
            raise ParseError(line_num, 'No animation section!')

        if tri is None:
            tri = []

        return Model(list(bones.values()), anim, tri)

    @staticmethod
    def _parse_smd_bones(file_iter: Iterator[Tuple[int, bytes]]) -> Dict[int, Bone]:
        """Parse the 'nodes' section of SMDs."""
        bones = {}
        for line_num, line in file_iter:
            if line == b'end':
                return bones
            try:
                bone_ind, bone_name, bone_parent = re.fullmatch(
                    br'([0-9]+)\s*"([^"]*)"\s*(-?[0-9]+)',
                    line,
                ).groups()
                bone_ind = int(bone_ind)
                bone_parent = int(bone_parent)
            except (ValueError, AttributeError):  # None.groups()
                raise ParseError(line_num, 'Invalid line!') from None
            else:
                if bone_parent == -1:
                    bones[bone_ind] = Bone(bone_name, None)
                else:
                    try:
                        bones[bone_ind] = Bone(bone_name, bones[bone_parent])
                    except KeyError:
                        raise ParseError(
                            line_num,
                            'Undefined parent bone {}!',
                            bone_parent,
                        ) from None
        raise ParseError('end', 'No end to nodes section!')

    @staticmethod
    def _parse_smd_anim(file_iter: Iterator[Tuple[int, bytes]], bones: Dict[int, Bone]):
        """Parse the 'skeleton' section of SMDs."""
        frames = {}
        time = -999999999
        for line_num, line in file_iter:
            if line.startswith((b'//', b'#', b';')):
                continue
            if line.startswith(b'time'):
                try:
                    time = int(line[4:])
                except ValueError:
                    raise ParseError(line_num, 'Invalid time value!') from None
                if time in frames:
                    raise ValueError(line_num, 'Duplicate frame time {}!', time)
                frames[time] = []
            elif line == b'end':
                return frames
            else:  # Bone.
                try:
                    byt_ind, byt_x, byt_y, byt_z, byt_pit, byt_yaw, byt_rol = line.split()
                    pos = Vec(float(byt_x), float(byt_y), float(byt_z))
                    rot = Vec(float(byt_pit), float(byt_yaw), float(byt_rol))
                except ValueError:
                    raise ParseError(line_num, 'Invalid line!') from None
                try:
                    bone = bones[int(byt_ind)]
                except KeyError:
                    raise ParseError(line_num, 'Unknown bone index {}!', int(byt_ind))
                frames[time].append(BoneFrame(bone, pos, rot))

        raise ParseError('end', 'No end to skeleton section!')

    @staticmethod
    def _parse_smd_tri(file_iter: Iterator[Tuple[int, bytes]], bones: Dict[int, Bone]):
        """Parse the 'triangles' section of SMDs."""
        tris = []
        points = [None, None, None]
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
            # Grab the three lines.
            for i in range(3):
                try:
                    line_num, line = next(file_iter)
                except StopIteration:
                    raise ParseError('end', 'Incomplete triangles!')
                try:
                    (
                        byt_parent,
                        x, y, z,
                        nx, ny, nz,
                        byt_tex_u, byt_tex_v,
                        *links_raw,
                    ) = line.split()
                except ValueError:
                    raise ParseError(line_num, 'Not enough values!')
                try:
                    pos = Vec(float(x), float(y), float(z))
                    norm = Vec(float(nx), float(ny), float(nz))
                except ValueError:
                    raise ParseError(line_num, 'Invalid normal or position!')
                try:
                    tex_u = float(byt_tex_u)
                    tex_v = float(byt_tex_v)
                except ValueError:
                    raise ParseError(line_num, 'Invalid texture UV!')

                try:
                    parent = bones[int(byt_parent)]
                except KeyError:
                    raise ParseError(line_num, 'Invalid bone {}!', int(byt_parent))

                if links_raw:
                    link_count = int(links_raw[0])

                    if (link_count * 2 + 1) != len(links_raw):
                        raise ParseError(line_num, 'Extra weight number: {}', links_raw)

                    links = []
                    for off in range(1, len(links_raw), 2):
                        try:
                            bone = bones[int(links_raw[off])]
                        except KeyError:
                            raise ParseError(line_num, 'Unknown bone {}!', links_raw[off])
                        links.append((bone, float(links_raw[off+1])))
                    remainder = 1.0 - math.fsum(weight for bone, weight in links)
                    if remainder:
                        links.append((parent, remainder))
                else:
                    links = [(parent, 1.0)]

                points[i] = Vertex(pos, norm, tex_u, tex_v, links)
            tris.append(Triangle(mat_name, *points))

        raise ParseError('end', 'No end to triangles section!')
