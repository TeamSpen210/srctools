"""Test VMT parsing."""
from srctools.vmt import Material, VarType, get_parm_type


def test_vartypes() -> None:
    """Test a bunch of common variable types to ensure they are correct."""
    assert get_parm_type('$basetexture') is VarType.TEXTURE
    assert get_parm_type('$NoCull') is VarType.FLAG
    # Omitting $ is allowed.
    assert get_parm_type('surfaceprop') is VarType.STR
    assert get_parm_type('$detailblendmode') is VarType.INT
    assert get_parm_type('$bottommaterial') is VarType.MATERIAL
    assert get_parm_type('$bumptransform') is VarType.MATRIX
    # % must be specified
    assert get_parm_type('%compilenodraw') is VarType.FLAG
    assert get_parm_type('compilesky', 'missing') == 'missing'
    assert get_parm_type('%compilesky', 45) is VarType.FLAG
