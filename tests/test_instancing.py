"""Test handling of instance collapsing."""
from typing import cast

import pytest

from srctools import FrozenMatrix, Matrix, VMF, Vec, FrozenVec, Angle, instancing
from srctools.fgd import ValueTypes


MAT_CCW = FrozenMatrix.from_yaw(270)
MAT_CW = FrozenMatrix.from_yaw(90)
MAT_FLIP = FrozenMatrix.from_yaw(180)
MAT_IDENT = FrozenMatrix()
MAT_UP = FrozenMatrix.from_pitch(-90)
MAT_DN = FrozenMatrix.from_pitch(90)

MAT_ORDER = [MAT_IDENT, MAT_CW, MAT_FLIP, MAT_CCW, MAT_UP, MAT_DN]
MAT_ORDER_ID = ['ident', 'cw', 'flip', 'ccw', 'up', 'dn']
VECS = [
    FrozenVec(x, y, z)
    for x in [-128, 0, 128]
    for y in [-128, 0, 128]
    for z in [-128, 0, 128]
]
VEC_IDS = [
    f'{x}{y}{z}'
    for x in 'n0p'
    for y in 'n0p'
    for z in 'n0p'
]


@pytest.mark.parametrize('origin', VECS, ids=VEC_IDS)
@pytest.mark.parametrize('orient', MAT_ORDER, ids=MAT_ORDER_ID)
def test_kv_unaffected(origin: FrozenVec, orient: FrozenMatrix) -> None:
    """Test keyvalue types which do not change when collapsed."""
    inst = instancing.Instance('test_inst', '', origin.thaw(), orient.thaw())
    vmf = VMF()
    assert inst.fixup_key(vmf, [], ValueTypes.STRING, 'blank') == 'blank'
    assert inst.fixup_key(vmf, [], ValueTypes.BOOL, '0') == '0'
    assert inst.fixup_key(vmf, [], ValueTypes.BOOL, '1') == '1'
    assert inst.fixup_key(vmf, [], ValueTypes.INT, '42') == '42'
    assert inst.fixup_key(vmf, [], ValueTypes.FLOAT, '-163.84') == '-163.84'
    assert inst.fixup_key(vmf, [], ValueTypes.COLOR_1, '0.8 1.2 0.5') == '0.8 1.2 0.5'
    assert inst.fixup_key(vmf, [], ValueTypes.COLOR_1, '0.8 1.2 0.5 1.0') == '0.8 1.2 0.5 1.0'
    assert inst.fixup_key(vmf, [], ValueTypes.COLOR_255, '128 192 255') == '128 192 255'
    assert inst.fixup_key(vmf, [], ValueTypes.COLOR_255, '128 192 255 64') == '128 192 255 64'
    assert inst.fixup_key(vmf, [], ValueTypes.EXT_VEC_LOCAL, '12 43 -68') == '12 43 -68'
    assert inst.fixup_key(vmf, [], ValueTypes.EXT_ANGLES_LOCAL, '45 90 12.5') == '45 90 12.5'
    assert inst.fixup_key(vmf, [], ValueTypes.STR_SCENE, 'scenes/breencast/welcome.vcd') == 'scenes/breencast/welcome.vcd'
    assert inst.fixup_key(vmf, [], ValueTypes.STR_SOUND, 'Weapon_Crowbar.Single') == 'Weapon_Crowbar.Single'
    assert inst.fixup_key(vmf, [], ValueTypes.STR_PARTICLE, 'striderbuster_attach') == 'striderbuster_attach'
    assert inst.fixup_key(vmf, [], ValueTypes.STR_SPRITE, 'sprites/orangecore1.vmt') == 'sprites/orangecore1.vmt'
    assert inst.fixup_key(vmf, [], ValueTypes.EXT_STR_TEXTURE, 'effects/flashlight001') == 'effects/flashlight001'
    assert inst.fixup_key(vmf, [], ValueTypes.STR_DECAL, 'decals/lambdaspray_2a.vmt') == 'decals/lambdaspray_2a.vmt'
    assert inst.fixup_key(vmf, [], ValueTypes.STR_MATERIAL, 'dev/dev_measuregeneric01') == 'dev/dev_measuregeneric01'
    assert inst.fixup_key(vmf, [], ValueTypes.STR_MODEL, 'models/antlion.mdl') == 'models/antlion.mdl'
    assert inst.fixup_key(vmf, [], ValueTypes.STR_VSCRIPT_SINGLE, 'lib/math.nut') == 'lib/math.nut'
    assert inst.fixup_key(vmf, [], ValueTypes.STR_VSCRIPT, 'lib/math.nut dice.nut') == 'lib/math.nut dice.nut'
    # Instance types would control fixup, but not when the func_instance itself is being positioned.
    assert inst.fixup_key(vmf, [], ValueTypes.INST_FILE, 'instances/gameplay/cube_dropper.vmf') == 'instances/gameplay/cube_dropper.vmf'
    assert inst.fixup_key(vmf, [], ValueTypes.INST_VAR_DEF, '$skin integer 0') == '$skin integer 0'
    assert inst.fixup_key(vmf, [], ValueTypes.INST_VAR_REP, '$skin 3') == '$skin 3'


@pytest.mark.parametrize('origin', VECS, ids=VEC_IDS)
@pytest.mark.parametrize('orient', MAT_ORDER, ids=MAT_ORDER_ID)
def test_kv_vector(origin: FrozenVec, orient: FrozenMatrix) -> None:
    """Test vector keyvalues."""
    inst = instancing.Instance('test_inst', '', origin.thaw(), orient.thaw())
    vmf = VMF()
    assert inst.fixup_key(vmf, [], ValueTypes.VEC, '12 43 -68') == str(Vec(12, 43, -68) @ orient + origin)
    assert inst.fixup_key(vmf, [], ValueTypes.VEC_ORIGIN, '12 43 -68') == str(Vec(12, 43, -68) @ orient + origin)
    assert inst.fixup_key(vmf, [], ValueTypes.VEC_LINE, '12 43 -68') == str(Vec(12, 43, -68) @ orient + origin)
    # No offset!
    assert inst.fixup_key(vmf, [], ValueTypes.EXT_VEC_DIRECTION, '12 43 -68') == str(Vec(12, 43, -68) @ orient)
    assert inst.fixup_key(
        vmf, [], ValueTypes.VEC_AXIS,
        '45 -62 35.82, 12 3.8 0',
    ) == f'{Vec(45, -62, 35.82) @ orient + origin}, {Vec(12, 3.8, 0) @ orient + origin}'
    assert inst.fixup_key(vmf, [], ValueTypes.ANGLES, '45 270.5 -12.5') == str(Angle(45, 270.5, -12.5) @ orient)


def test_kv_sides() -> None:
    """Test the 'sidelist' keyvalue type."""
    inst = instancing.Instance('test_inst', '', Vec(), Matrix())
    vmf = VMF()
    # Pretend we had collapsed brushes with these values.
    cast(dict, inst.face_ids).update({45: 68, 12: 12, 85: 92, 86: 93})

    # Invalid ints and missing keys are ignored.
    assert inst.fixup_key(vmf, [], ValueTypes.SIDE_LIST, '45 ardvark 6 12') == '12 68'
    assert inst.fixup_key(vmf, [], ValueTypes.SIDE_LIST, '85 86 45 45 12 34 34') == '12 68 92 93'
    # Empty string is fine.
    assert inst.fixup_key(vmf, [], ValueTypes.SIDE_LIST, '') == ''
    assert inst.fixup_key(vmf, [], ValueTypes.SIDE_LIST, '86') == '93'


def test_pitch_errors() -> None:
    """Pitch keyvalues can't be fixed up individually."""
    inst = instancing.Instance('test_inst', '', Vec(), Matrix())
    vmf = VMF()

    with pytest.raises(ValueError):
        inst.fixup_key(vmf, [], ValueTypes.ANGLE_NEG_PITCH, '45')
    with pytest.raises(ValueError):
        inst.fixup_key(vmf, [], ValueTypes.EXT_ANGLE_PITCH, '45')
