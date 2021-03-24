"""Pyinstaller hooks for the main module."""
from PyInstaller.utils.hooks import get_module_file_attribute
import os

srctools_loc = os.path.dirname(get_module_file_attribute('srctools'))

datas = [
    # Add our FGD database and our srctools custom FGD.
    (os.path.join(srctools_loc, 'fgd.lzma'), 'srctools'),
    (os.path.join(srctools_loc, 'srctools.fgd'), 'srctools'),
]

excludedimports = [
    'PIL',  # Pillow is optional for VTF, the user will import if required.
    'tkinter',  # Same for Tkinter.
    'wx',  # And wxWidgets.
]
