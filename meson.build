project(
    'srctools',
    'cython', 'c', 'cpp',
    version: '2.4.2',
    default_options : ['c_std=c11', 'cpp_std=c++11'],
    )

fs = import('fs')
py = import('python').find_installation(pure: false)
dep_py = py.dependency()


# Build quickhull.
inc_quickhull = include_directories('src' / 'quickhull')
lib_quickhull = static_library(
    'quickhull',
    files(
        'src' / 'quickhull' / 'QuickHull.cpp',
    ),
    include_directories: inc_quickhull,
)

# Build libsquish
inc_libsquish = include_directories('src' / 'libsquish')
lib_squish = static_library(
    'libsquish',
    files(
        'src' / 'libsquish' / 'alpha.cpp',
        'src' / 'libsquish' / 'clusterfit.cpp',
        'src' / 'libsquish' / 'colourblock.cpp',
        'src' / 'libsquish' / 'colourfit.cpp',
        'src' / 'libsquish' / 'colourset.cpp',
        'src' / 'libsquish' / 'maths.cpp',
        'src' / 'libsquish' / 'rangefit.cpp',
        'src' / 'libsquish' / 'singlecolourfit.cpp',
        'src' / 'libsquish' / 'squish.cpp',
    ),
    include_directories: inc_libsquish,
)

subdir('src' / 'srctools')
