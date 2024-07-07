"""Pyinstaller hooks for the main module."""
from pathlib import Path

from PyInstaller.utils.hooks import get_module_file_attribute  # pyright: ignore


srctools_init = get_module_file_attribute('srctools')
assert srctools_init is not None
srctools_loc = Path(srctools_init).parent

datas = [
    # Add our FGD database.
    (str(Path(srctools_loc, 'fgd.lzma').resolve()), 'srctools'),
]

excludedimports = [
    'PIL',  # Pillow is optional for VTF, the user will import if required.
    'tkinter',  # Same for Tkinter.
    'wx',  # And wxWidgets.
]
