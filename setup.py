from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
except ImportError:
    print('Cython not installed, not compiling Cython modules.')
    modules = []
    cythonize = None

setup(
    name='srctools',
    version='1.2.0',
    description="Modules for working with Valve's Source Engine file formats.",
    url='https://github.com/TeamSpen210/srctools',

    author='TeamSpen210',
    author_email='spencerb21@live.com',
    license='unlicense',

    keywords='',
    classifiers=[
        'License :: Public Domain',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],
    packages=[
        'srctools',
        'srctools.scripts',
        'srctools.test',
        'srctools.bsp_transform',
        'srctools.pyinstaller_hook',
        'srctools._class_resources',
    ],
    # Setuptools automatically runs Cython, if available.
    ext_modules=cythonize([
        Extension(
            "srctools._tokenizer",
            sources=["srctools/_tokenizer.pyx"],
            # extra_compile_args=['/FAs'],  # MS ASM dump
        ),
        Extension(
            "srctools._cy_vtf_readwrite",
            sources=["srctools/_cy_vtf_readwrite.pyx"],
            # extra_compile_args=['/FAs'],  # MS ASM dump
        ),
    ]),

    package_data={'srctools': [
        'fgd.lzma',
        'srctools.fgd',
        'py.typed',
    ]},

    entry_points={
        'console_scripts': [
            'srctools_dump_parms = srctools.scripts.dump_parms:main',
            'srctools_diff = srctools.scripts.diff:main',
        ],
    },
    python_requires='>=3.6, <4',
    install_requires=[
        'PyInstaller',
        'importlib_resources',
    ],
)
