"""Import all modules, to ensure at least imports work."""
import pytest
import importlib
from pathlib import Path

package_loc = Path(__file__, '..', '..', 'src', 'srctools')


@pytest.mark.parametrize('mod_name', [
    fname.stem
    for fname in package_loc.glob('*.py')
    if fname.stem != '__init__'
] + ['_class_resources'])
def test_smoke(mod_name: str) -> None:
    """Ensure every module is importable."""
    importlib.import_module('srctools.' + mod_name)


def test_entclass_resources() -> None:
    """Test the class resources database can be loaded."""
    from srctools.packlist import entclass_resources
    assert list(entclass_resources('info_target')) == []


def test_vmt_types() -> None:
    """Test the VMT types database can be loaded."""
    from srctools.vmt import get_parm_type, VarType
    assert get_parm_type('$basetexture') is VarType.TEXTURE


def test_engine_fgd() -> None:
    """Test the FGD database can be loaded."""
    from srctools.fgd import FGD
    assert 'env_beam' in FGD.engine_dbase().entities
