"""Build the Srctools package."""
from setuptools import setup, Extension, find_packages
import sys
import os

WIN = sys.platform.startswith('win')
MAC = sys.platform.startswith('darwin')

SQUISH_CPP = [
    'libsquish/alpha.cpp',
    'libsquish/clusterfit.cpp',
    'libsquish/colourblock.cpp',
    'libsquish/colourfit.cpp',
    'libsquish/colourset.cpp',
    'libsquish/maths.cpp',
    'libsquish/rangefit.cpp',
    'libsquish/singlecolourfit.cpp',
    'libsquish/squish.cpp',
]
# Mandatory in CI!
optional_ext = os.environ.get('CIBUILDWHEEL', '0') != '1'

if WIN:
    openmp = ['/openmp']
elif MAC:
    openmp = []  # Not supported by system Clang.
else:
    openmp = ['-fopenmp']

setup(
    packages=find_packages(include=['srctools', 'srctools.*']),
    # Setuptools automatically runs Cython, if available.
    ext_modules=[
        Extension(
            "srctools._tokenizer",
            sources=["srctools/_tokenizer.pyx"],
            optional=optional_ext,
            extra_compile_args=[
                # '/FAs',  # MS ASM dump
            ],
        ),
        Extension(
            "srctools._cy_vtf_readwrite",
            include_dirs=[os.path.abspath("libsquish/")],
            language='c++',
            optional=optional_ext,
            sources=[
                "srctools/_cy_vtf_readwrite.pyx",
            ] + SQUISH_CPP,
            extra_compile_args=[
                # '/FAs',  # MS ASM dump
            ] + openmp,
            extra_link_args=openmp,
        ),
        Extension(
            "srctools._math",
            include_dirs=[os.path.abspath("quickhull/")],
            language='c++',
            optional=optional_ext,
            sources=["srctools/_math.pyx", "quickhull/QuickHull.cpp"],
            extra_compile_args=[
                # '/FAs',  # MS ASM dump
            ] if WIN else [
                '-std=c++11',  # Needed for Mac to work
            ],
        ),
    ],
)
