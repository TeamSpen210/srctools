project(
    'srctools',
    'cython', 'c', 'cpp',
    version: '2.3.14',
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

subdir('src' / 'srctools')
