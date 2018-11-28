from cx_Freeze import setup, Executable

import srctools
import os

zip_includes = [
    # Add the FGD data for us.
    (os.path.join(srctools.__path__[0], 'fgd.lzma'), 'srctools/fgd.lzma'),
]

# We need to include this version data.
try:
    import importlib_resources
    zip_includes.append(
        (
            os.path.join(importlib_resources.__path__[0], 'version.txt'),
            'importlib_resources/version.txt',
         )
    )
except ImportError:
    pass


setup(
    name='srctools PostCompiler',
    version='1.0',
    options={
        'build_exe': {
            'build_exe': 'build/',
            # Include all modules in the zip..
            'zip_include_packages': '*',
            'zip_exclude_packages': '',
            'zip_includes': zip_includes,
            'excludes': ['tkinter', 'PIL'],
        },
    },
    requires=['setuptools', 'cx_freeze'],
    executables=[
        Executable(
            'srctools/scripts/postcompiler.py',
            base='Console',
            targetName='postcompiler.exe',
        ),
    ]
)

