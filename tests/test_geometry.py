"""Test the various geometry operations."""
from pathlib import Path

from pytest_regressions.file_regression import FileRegressionFixture
import pytest

from srctools import Vec, FrozenVec, VMF, Keyvalues
from srctools.bsp import Plane
from srctools.geometry import Geometry


AXES = {
    Vec.N: 'north',
    Vec.S: 'south',
    Vec.E: 'east',
    Vec.W: 'west',
    Vec.T: 'up',
    Vec.B: 'down',
}


@pytest.mark.parametrize("axis", AXES, ids=AXES.get)
def test_cube_clipping(axis: FrozenVec, file_regression: FileRegressionFixture) -> None:
    """Test clipping a cube in many different directions."""
    vmf = VMF()
    for i, dist in enumerate((-128, -32, 0, 32, 128)):
        brush = Geometry.from_brush(vmf.make_prism(
            Vec(-64, -64, -64), Vec(64, 64, 64),
            'tools/toolsnodraw',
        ).solid)
        front, back = brush.clip(Plane(axis.thaw(), dist))
        name = str(dist).replace('-', 'n')

        offset = Vec(192, 0, 0) * i
        if front is not None:
            solid = front.rebuild(vmf, 'tools/toolsclip')
            solid.localise(offset)
            vmf.create_ent('func_brush', targetname=f'{name}_front').solids.append(solid)
        if back is not None:
            solid = back.rebuild(vmf, 'tools/toolsclip')
            solid.localise(offset)
            vmf.create_ent('func_brush', targetname=f'{name}_back').solids.append(solid)

    file_regression.check(
        vmf.export(),
        basename=f'cube_clip_{AXES[axis]}',
        extension='.vmf',
    )


def test_carve(datadir: Path, file_regression: FileRegressionFixture) -> None:
    """Carve various objects to test the functionality.

    We read a VMF, then for each ent subtract toolsclip from the rest.
    """
    with open(datadir / 'carve_tests.vmf') as f:
        vmf = VMF.parse(Keyvalues.parse(f))
    for ent in list(vmf.by_class['func_brush']):
        ent.remove()
        carvers = [
            brush for brush in ent.solids if
            any(face.mat.casefold() == 'tools/toolsclip' for face in brush.sides)
        ]
        try:
            [carver] = carvers
        except ValueError:
            raise pytest.fail(f'{carvers=}, {ent=}')
        ent.solids.remove(carver)
        result = Geometry.carve(
            [Geometry.from_brush(solid) for solid in ent.solids],
            Geometry.from_brush(carver),
        )
        if result:
            vmf.create_ent('func_brush', targetname=ent['targetname']).solids = [
                brush.rebuild(vmf, 'tools/toolsskip')
                for brush in result
            ]
        else:
            vmf.create_ent('info_null', targetname=ent['targetname'])

    file_regression.check(vmf.export(), extension='.vmf')
