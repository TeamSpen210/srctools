"""Test VMT parsing."""
import io

from pytest_regressions.file_regression import FileRegressionFixture

from srctools import Keyvalues
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
    # ? flags are ignored.
    assert get_parm_type('hdr?$ssbump') is VarType.INT
    assert get_parm_type('!360?cheapwaterenddistance') is VarType.FLOAT
    # % must be specified
    assert get_parm_type('%compilenodraw') is VarType.FLAG
    assert get_parm_type('compilesky', 'missing') == 'missing'
    assert get_parm_type('%compilesky', 45) is VarType.FLAG


def test_parse() -> None:
    """Test parsing an example material."""

    mat = Material.parse(r'''VertexLitGeneric {
    $basetexture "some\base\texture.tga"
    $vector "[0 1 .5]"
    %noportal 1
    $alpha .3
    $surfaceprop metal
    Proxies
    {
        SomeProxy // another comment
        {
            mins "$vector"
            resultVar $alpha
        }
        AnotherProxy
            { // comment
            value "42.5"
            }
    }
}
''')
    assert mat.shader == "VertexLitGeneric"
    assert len(mat) == 5
    assert mat['$baseTexture'] == 'some\\base\\texture.tga'
    assert mat['%noportal'] == '1'
    assert mat['$alpha'] == '.3'
    assert mat['$surfaceprop'] == 'metal'

    assert len(mat.proxies) == 2
    assert mat.proxies == [
        Keyvalues('SomeProxy', [
            Keyvalues('mins', '$vector'),
            Keyvalues('resultVar', '$alpha'),
        ]),
        Keyvalues('AnotherProxy', [
            Keyvalues('value', '42.5'),
        ])
    ]


def test_export(file_regression: FileRegressionFixture) -> None:
    """Test exporting materials."""
    mat = Material('LightmappedGeneric')
    mat['$basetexture'] = 'tools/toolsskybox'
    mat['$surfaceprop'] = 'dirt'
    mat['$selfillum'] = '1'
    mat['$reflectivity'] = '[.4 .8 .12]'
    mat.proxies.append(Keyvalues('Sine', [
        Keyvalues('min', '0'),
        Keyvalues('max', '1'),
        Keyvalues('period', '2.3'),
        Keyvalues('resultVar', '$selfillumscale[0]'),
    ]))

    buf = io.StringIO()
    mat.export(buf)

    file_regression.check(buf.getvalue(), extension='.vmt')
