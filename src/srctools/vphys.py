"""Code for parsing VPhysics solids. These form .phy files, but are also generated for brushes."""
from __future__ import annotations
from typing import IO
from struct import Struct

import attrs
from srctools import Keyvalues, Vec, binformat


ST_LEDGE_NODE = Struct('<ii4f3sx')


@attrs.define(kw_only=True)
class LedgeBase:
    """compactledge_t is referenced by a terminal node, this stores the data for that."""
    center: Vec
    radius: float
    box_sizes: bytes  # 3 chars?


@attrs.define(kw_only=True)
class Ledge(LedgeBase):
    """Some sort of hull?"""
    _point_offset: int
    client_data: bytes
    has_children: int
    is_compact: int
    size_div_16: int
    tri_count: int
    future_data: int


@attrs.define(kw_only=True)
class LedgeNode(LedgeBase):
    left: LedgeBase
    right: LedgeBase

    @classmethod
    def parse_tree(cls, f: IO[bytes], memo: dict[int, LedgeBase], pos: int) -> LedgeBase:
        """Parse the tree of nodes."""
        if pos in memo:
            return memo[pos]
        print(f'parse_tree({pos})')
        f.seek(pos)
        (
            off_right, off_ledge,
            cent_x, cent_y, cent_z,
            radius,
            box_sizes,
        ) = binformat.struct_read(ST_LEDGE_NODE, f)
        center = Vec(cent_x, cent_y, cent_z)

        if off_right:
            left = LedgeNode.parse_tree(f, memo, pos + ST_LEDGE_NODE.size)
            right = LedgeNode.parse_tree(f, memo, pos + off_right)
            memo[pos] = node = LedgeNode(
                center=center, box_sizes=box_sizes, radius=radius,
                left=left, right=right
            )
            return node
        f.seek(pos + off_ledge)
        (
            point_off,
            client_data,
            flags,
            size_bytes,
            tri_count,
            future
        ) = binformat.struct_read('<ii B3s hh', f)
        has_children = flags >> 30
        is_compact = (flags >> 28) & 0b11
        memo[pos] = ledge = Ledge(
            center=center, box_sizes=box_sizes, radius=radius,
            point_offset=point_off,
            has_children=has_children,
            is_compact=is_compact,
            size_div_16=int.from_bytes(size_bytes, 'little'),
            tri_count=tri_count,
            future_data=future,
            client_data=client_data,
        )
        return ledge


@attrs.define
class Collision:
    # collideheader_t
    vphy_id: bytes
    version: int
    mdl_type: int
    # compactsurfaceheader_t:
    surf_size: int
    drag_axis_areas: Vec
    axis_map_size: int
    # compactsurface_t
    mass_center: Vec
    rotation_intertia: Vec
    upper_limit_radius: float
    max_factor_surface_deviation: int
    byte_size: int
    ledgetree: LedgeBase

    @classmethod
    def parse(cls, f: IO[bytes], pos: int, vphys_id: bytes) -> Collision:
        if vphys_id != b'VPHY':
            # TODO handle.
            raise ValueError(f'Legacy vphysics mesh {vphys_id}')

        (
            version,
            mdl_type,
        ) = binformat.struct_read('<hh', f)  # collideheader_t
        (
            surf_size,
            drag_ax_x,
            drag_ax_y,
            drag_ax_z,
            axis_size,
        ) = binformat.struct_read('<i 3f i', f)  # compactsurfaceheader_t
        surf_pos = f.tell()
        (
            mass_cent_x, mass_cent_y, mass_cent_z,
            rot_inert_x, rot_inert_y, rot_inert_z,
            upper_limit_radius,
            surface_dev,
            size,
            ledgetree_root_off,
        ) = binformat.struct_read('<3f 3f f B3s i 12x', f)  # compactsurface_t

        if mdl_type != 0:
            raise ValueError(f'Unknown solid type {mdl_type:02x}?')

        ledgetree = LedgeNode.parse_tree(f, {}, surf_pos + ledgetree_root_off)

        return cls(
            vphys_id,
            version,
            mdl_type,
            surf_size,
            Vec(drag_ax_x, drag_ax_y, drag_ax_z),
            axis_size,
            Vec(mass_cent_x, mass_cent_y, mass_cent_z),
            Vec(rot_inert_x, rot_inert_y, rot_inert_z),
            upper_limit_radius,
            surface_dev,
            int.from_bytes(size, 'little'),
            ledgetree,
        )


@attrs.define
class PhysFile:
    """The data in a .phy file."""
    phy_id: int
    collide: list[Collision]
    keyvalues: Keyvalues

    @classmethod
    def parse_phy(cls, f: IO[bytes]) -> 'PhysFile':
        """Parse a .phy file."""
        (
            header_size,
            phy_id,
            collide_count,
            checksum,
        ) = binformat.struct_read('3il', f)

        # Should have ended this now.
        assert f.tell() == header_size, 'Wrong header size?'

        collide: list[Collision] = []

        for _ in range(collide_count):
            start_pos = f.tell()
            (
                data_size,
                vphys_id,
            ) = binformat.struct_read('i 4s', f)
            collide.append(Collision.parse(f, start_pos, vphys_id))
            f.seek(start_pos + data_size)

        # Random 4 bytes...
        value = binformat.struct_read('I', f)

        keyvalues = Keyvalues.parse(
            binformat.read_nullstr(f),
            getattr(f, "name", ".phy") + ":keyvalues",
            allow_escapes=False,
            single_line=True,
        )
        return cls(phy_id, collide, keyvalues)
