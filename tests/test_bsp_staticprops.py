"""Tests for the BSP static props lump."""
import pytest

from srctools import Angle, Vec
from srctools.bsp import (
    BrushContents, StaticProp, StaticPropFlags, StaticPropVersion, VisLeaf, VisLeafFlags,
)
from test_bsp import make_dummy


@pytest.mark.parametrize('version', [
    vers for vers in StaticPropVersion
    if vers.name not in ('DEFAULT', 'UNKNOWN')
], ids=lambda ver: ver.name.lower())
def test_export(file_regression, version: StaticPropVersion) -> None:
    """Test each prop version is exported the same way."""
    bsp = make_dummy()
    # Dummy visleaf, we don't really care.
    visleaf = VisLeaf(
        contents=BrushContents.EMPTY,
        cluster_id=0,
        area=15,
        flags=VisLeafFlags.NONE,
        mins=Vec(-32, -32, -32),
        maxes=Vec(32, 32, 32),
        faces=[],
        brushes=[],
        water_id=-1,
    )
    bsp.nodes = [visleaf]

    bsp.static_prop_version = version
    bsp.props = [StaticProp(
        model='models/props_testing/some_model001.mdl',
        origin=Vec(289.289, 3782.187, -28.821),
        angles=Angle(12.5, 270.0, 0),
        scaling=1.25,
        visleafs={visleaf},
        solidity=6,
        flags=StaticPropFlags.NONE,
        skin=1,
        min_fade=8.2,
        max_fade=12.9,
        lighting=Vec(0.38, -1.8278, -5.2981),
        fade_scale=-1.0,
        min_dx_level=1,
        max_dx_level=3,
        min_cpu_level=2,
        max_cpu_level=4,
        min_gpu_level=3,
        max_gpu_level=6,
        tint=Vec(192, 255, 64),
        renderfx=128,
        disable_on_xbox=False,
        lightmap_x=48,
        lightmap_y=32,
    )]

    data = bsp._lmp_write_props(bsp.props)
    file_regression.check(data, extension='.lmp', binary=True)
