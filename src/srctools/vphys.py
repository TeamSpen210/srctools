"""Code for parsing VPhysics solids. These form .phy files, but are also generated for brushes."""
import attrs
from typing import IO

from srctools import Keyvalues, Vec, binformat


@attrs.define
class Collision:
    vphy_id: bytes
    version: int
    mdl_type: int
    surf_size: int
    drag_axis_areas: Vec
    axis_map_size: int
    _data: bytes


@attrs.define
class PhysFile:
    """The data in a .phy file."""
    phy_id: int
    collide: list[Collision]
    keyvalues: Keyvalues

    @classmethod
    def parse_phy(cls, f: IO[bytes]) -> 'PhysFile':
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
                version,
                mdl_type,
            ) = binformat.struct_read('i 4s hh', f)

            (
                surf_size,
                drag_ax_x,
                drag_ax_y,
                drag_ax_z,
                axis_size,
            ) = binformat.struct_read('i 3f i', f)

            # The data size includes the header.
            # This = sizeof('i 4s hh i 3f i')
            # data_size -= 4 * 2 + 2*2 + 4 + 3*4 + 4
            data_size -= f.tell() - start_pos

            collide.append(Collision(
                vphys_id,
                version,
                mdl_type,
                surf_size,
                Vec(drag_ax_x, drag_ax_y, drag_ax_z),
                axis_size,
                f.read(data_size)
            ))

        # Random 4 bytes...
        value = binformat.struct_read('I', f)

        keyvalues = Keyvalues.parse(
            binformat.read_nullstr(f),
            getattr(f, "name", ".phy") + ":keyvalues",
            allow_escapes=False,
            single_line=True,
        )
        return cls(phy_id, collide, keyvalues)
