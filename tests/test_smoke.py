"""Import all modules, to ensure at least imports work."""
import pytest
import importlib
from pathlib import Path

package_loc = Path(__file__, '..', '..', 'src', 'srctools')


@pytest.mark.parametrize('mod_name', [
    fname.stem
    for fname in package_loc.glob('*.py')
    if fname.stem != '__init__'
])
def test_smoke(mod_name: str) -> None:
    """Ensure every module is importable."""
    importlib.import_module('srctools.' + mod_name)
