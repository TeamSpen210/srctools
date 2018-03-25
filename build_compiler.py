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
                ('srctools/fgd.lzma', 'srctools/fgd.lzma')
            ],
        },
    },
    requires=['setuptools', 'cx_freeze'],
    executables=[
        # Executable(
        #     'srctools/scripts/vbsp.py',
        #     base='Console',
        #     targetName='vbsp.exe',
        # ),
        Executable(
            'srctools/scripts/vrad.py',
            base='Console',
            targetName='vrad.exe',
        ),
    ]
)

