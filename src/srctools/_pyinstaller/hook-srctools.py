"""Pyinstaller hooks for the main module."""
import os

from PyInstaller.utils.hooks import get_module_file_attribute  # type: ignore


srctools_init = get_module_file_attribute('srctools')
assert srctools_init is not None
srctools_loc = os.path.dirname(srctools_init)

datas = [
    # Add our FGD database.
    (os.path.join(srctools_loc, 'fgd.lzma'), 'srctools'),
]

excludedimports = [
    'PIL',  # Pillow is optional for VTF, the user will import if required.
    'tkinter',  # Same for Tkinter.
    'wx',  # And wxWidgets.
]
