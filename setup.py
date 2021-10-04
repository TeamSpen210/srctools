"""Build the Srctools package."""
from setuptools import setup, Extension, find_packages
import sys
import os

WIN = sys.platform.startswith('win')

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

setup(
    packages=find_packages(include=['srctools', 'srctools.*']),
    # Setuptools automatically runs Cython, if available.
    ext_modules=[
        Extension(
            "srctools._tokenizer",
            sources=["srctools/_tokenizer.pyx"],
            optional=True,
            extra_compile_args=[
                # '/FAs',  # MS ASM dump
            ],
        ),
        Extension(
            "srctools._cy_vtf_readwrite",
            include_dirs=[os.path.abspath("libsquish/")],
            language='c++',
            optional=True,
            sources=[
                "srctools/_cy_vtf_readwrite.pyx",
            ] + SQUISH_CPP,
            extra_compile_args=[
                '/openmp',
                # '/FAs',  # MS ASM dump
            ] if WIN else [
                '-fopenmp',
            ],
            extra_link_args=[] if WIN else ['-fopenmp'],
        ),
        Extension(
            "srctools._math",
            include_dirs=[os.path.abspath("quickhull/")],
            language='c++',
            optional=True,
            sources=["srctools/_math.pyx", "quickhull/QuickHull.cpp"],
            extra_compile_args=[
                # '/FAs',  # MS ASM dump
            ] if WIN else [

            ],
        ),
    ],
)
