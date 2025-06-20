"""Test the various geometry operations."""
from pathlib import Path
from random import Random

from pytest_regressions.file_regression import FileRegressionFixture
import pytest

from srctools import Vec, FrozenVec, VMF, Keyvalues, Matrix
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


def reset_ids(vmf: VMF) -> None:
    """To prevent GC timing from affecting IDs, reset all of them."""
    for ent in vmf.entities:
        ent.id = 1
        for brush in ent.solids:
            brush.id = 1
            for side in brush.sides:
                side.id = 1


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

    reset_ids(vmf)
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
    for ent in sorted(vmf.by_class['func_brush'], key=lambda ent: ent['targetname']):
        ent.remove()
        carvers = [
            brush for brush in ent.solids if
            any(face.mat.casefold() == 'tools/toolsclip' for face in brush.sides)
        ]
        try:
            [carver] = carvers
        except ValueError as exc:
            raise pytest.fail(f'{carvers=}, {ent=}') from exc
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

    reset_ids(vmf)
    file_regression.check(vmf.export(), extension='.vmf')


def test_merge(file_regression: FileRegressionFixture) -> None:
    """Test merging brushes. We clip some cubes randomly, then check they re-merge.

    We swap materials a few times to make visualisation easier. skip is brush A, clip is brush B,
    and hint is the clip plane.
    """
    vmf = VMF()
    cube_template = vmf.make_prism(
        Vec(-64, -64, -64), Vec(64, 64, 64), 'tools/toolsskip',
    ).solid

    rng = Random(48)
    for i in range(64):
        copy = cube_template.copy()
        copy.localise(Vec(0, 0, 0), Matrix.from_angle(
            rng.randrange(0, 360, 15),
            rng.randrange(0, 360, 15),
            0,
        ))
        brush = Geometry.from_brush(copy)
        clip = Plane(
            Matrix.from_angle(
                rng.randrange(0, 360, 5),
                rng.randrange(0, 360, 5),
                0,
            ).forward(),
            rng.randrange(-16, 16),
        )
        brush_a, brush_b = brush.clip(clip)
        assert brush_a is not None
        assert brush_b is not None

        ent = vmf.create_ent('func_brush')
        ent.comments = repr(clip)
        ent.solids = [
            brush_a.rebuild(vmf, 'tools/toolshint').copy(),
            brush_b.rebuild(vmf, 'tools/toolshint').copy(),
        ]
        # Mark the second brush, for easier visualisation in Hammer.
        for face in ent.solids[1].sides:
            if face.mat == 'tools/toolsnodraw':
                face.mat = 'tools/toolsclip'

        merged = Geometry.merge(brush_a, brush_b)
        if merged is not None:
            ent.solids.append(solid := merged.rebuild(vmf, 'tools/toolsnodraw'))
            solid.localise(Vec(0, 0, 192))
        offset = Vec(divmod(i, 8)) * 192
        for solid in ent.solids:
            solid.localise(offset)

    reset_ids(vmf)
    file_regression.check(vmf.export(), extension='.vmf')
