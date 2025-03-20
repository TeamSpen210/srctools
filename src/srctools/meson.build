pure_python = files(
    '__init__.py',
    '_class_resources.py',
    '_engine_db.py',
    '_fgd_helpers.py',
    '_py_vtf_readwrite.py',
    '_shaderdb.py',
    'binformat.py',
    'bsp.py',
    'choreo.py',
    'cmdseq.py',
    'const.py',
    'dmx.py',
    'fgd.py',
    'filesys.py',
    'game.py',
    'instancing.py',
    'keyvalues.py',
    'logger.py',
    'math.py',
    'mdl.py',
    'packlist.py',
    'particles.py',
    'property_parser.py',
    'run.py',
    'smd.py',
    'sndscript.py',
    'steam.py',
    'surfaceprop.py',
    'tokenizer.py',
    'vec.py',
    'vmf.py',
    'vmt.py',
    'vphys.py',
    'vpk.py',
    'vtf.py',
)
pure_python += files(
    'py.typed',
    'fgd.lzma',
)

py.install_sources(pure_python, subdir: 'srctools')

subdir('_pyinstaller')
subdir('scripts')

if get_option('build_extensions')
    py.extension_module(
        '_math',
        files(
            '_math.pyx',
            '_math_matrix.cpp',
        ),
        override_options: ['cython_language=cpp'],
        link_with: [lib_quickhull],
        include_directories: inc_quickhull,

        dependencies: dep_py,
        install: true,
        subdir: 'srctools',
    )

    py.extension_module(
        '_tokenizer',
        files(
            '_tokenizer.pyx',
        ),
        install: true,
        subdir: 'srctools',
    )

    py.extension_module(
        '_cy_vtf_readwrite',
        files(
            '_cy_vtf_readwrite.pyx',
        ),
        override_options: ['cython_language=cpp'],
        link_with: [lib_squish],
        include_directories: inc_libsquish,

        dependencies: dep_py,
        install: true,
        subdir: 'srctools',
    )
endif
