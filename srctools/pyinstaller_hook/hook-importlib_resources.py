"""Pyinstaller hooks for importlib_resources."""
from PyInstaller.utils.hooks import get_module_file_attribute
from PyInstaller.compat import is_py2, is_py3, is_py37
import os

res_loc = os.path.dirname(get_module_file_attribute('importlib_resources'))

datas = [
    (os.path.join(res_loc, 'version.txt'), 'importlib_resources'),
]

# Replicate the module's version checks to exclude unused modules.
if is_py37:
    # Stdlib now has the implmentation of this,
    # so the backports aren't used.
    excludedmodules = [
        'importlib_resources._py2',
        'importlib_resources._py3',
    ]
elif is_py3:
    excludedmodules = ['importlib_resources._py2']
elif is_py2:
    excludedmodules = ['importlib_resources._py3']
else:
    excludedmodules = []
