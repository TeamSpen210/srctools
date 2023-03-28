"""Import all modules, to ensure at least imports work."""
from pathlib import Path
import importlib

import pytest


package_loc = Path(__file__, '..', '..', 'src', 'srctools')


@pytest.mark.parametrize('mod_name', [
    fname.stem
    for fname in package_loc.glob('*.py')
    # Don't import the package init or deprecated modules.
    if fname.stem not in ['__init__', 'property_parser', 'vec']
] + ['_cy_vtf_readwrite', '_math', '_tokenizer'])
def test_smoke(mod_name: str) -> None:
    """Ensure every module is importable."""
    importlib.import_module('srctools.' + mod_name)


def test_vmt_types() -> None:
    """Test the VMT types database can be loaded."""
    from srctools.vmt import VarType, get_parm_type
    assert get_parm_type('$basetexture') is VarType.TEXTURE


def test_engine_fgd() -> None:
    """Test the FGD database can be loaded."""
    from srctools.fgd import EntityDef
    assert 'env_beam' in EntityDef.engine_classes()
    assert 'wait' in EntityDef.engine_def('trigger_multiple').kv
