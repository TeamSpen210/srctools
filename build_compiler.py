from cx_Freeze import setup, Executable

setup(
    name='VBSP_VRAD',
    options={
        'build_exe': {
            'build_exe': 'build/',
            # Include all modules in the zip..
            'zip_include_packages': '*',
            'zip_exclude_packages': '',
            'zip_includes': [
                ('seecompiler/fgd.lzma', 'seecompiler/fgd.lzma')
            ],
        },
    },
    requires=['srctools', 'setuptools', 'cx_freeze'],
    executables=[
        # Executable(
        #     'seecompiler/vbsp.py',
        #     base='Console',
        #     targetName='vbsp.exe',
        # ),
        Executable(
            'seecompiler/vrad.py',
            base='Console',
            targetName='vrad.exe',
        ),
    ]
)

