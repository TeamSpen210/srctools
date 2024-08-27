"""Build the Srctools package."""
from setuptools import setup, Extension
import sys
import os
import setuptools

WIN = sys.platform.startswith('win')
MAC = sys.platform.startswith('darwin')
root = os.path.dirname(__file__)

print(f'Srctools, {WIN=}, {MAC=}, setuptools={getattr(setuptools, "__version__", "???")}')

# Mandatory in CI!
optional_ext = os.environ.get('CIBUILDWHEEL', '0') != '1'

if WIN:
    openmp = ['/openmp']
    openmp_link = []
elif MAC:
    openmp = []  # Not supported by system Clang.
    openmp_link = []
else:
    openmp = ['-fopenmp']
    openmp_link = ['-fopenmp']

extensions = [
    Extension(
        "srctools._tokenizer",
        sources=["src/srctools/_tokenizer.pyx"],
        optional=optional_ext,
        extra_compile_args=[
            # '/FAs',  # MS ASM dump
        ],
    ),
    Extension(
        "srctools._cy_vtf_readwrite",
        include_dirs=[os.path.abspath(os.path.join(root, "src", "libsquish"))],
        language='c++',
        optional=optional_ext,
        sources=[
            "src/srctools/_cy_vtf_readwrite.pyx",
            "src/libsquish/alpha.cpp",
            "src/libsquish/clusterfit.cpp",
            "src/libsquish/colourblock.cpp",
            "src/libsquish/colourfit.cpp",
            "src/libsquish/colourset.cpp",
            "src/libsquish/maths.cpp",
            "src/libsquish/rangefit.cpp",
            "src/libsquish/singlecolourfit.cpp",
            "src/libsquish/squish.cpp",
        ],
        extra_compile_args=[
            # '/FAs',  # MS ASM dump
            *openmp,
        ],
        extra_link_args=openmp_link,
    ),
    Extension(
        "srctools._math",
        include_dirs=[os.path.abspath(os.path.join(root, "src", "quickhull/"))],
        language='c++',
        optional=optional_ext,
        sources=["src/srctools/_math.pyx", "src/srctools/_math_matrix.cpp", "src/quickhull/QuickHull.cpp"],
        extra_compile_args=[
            # '/FAs',  # MS ASM dump
        ] if WIN else [
            '-std=c++11',  # Needed for Mac to work
        ],
    ),
]

# Don't build extension modules for docs, they're not useful.
if 'READTHEDOCS' in os.environ:
    setup()
else:
    setup(ext_modules=extensions)
