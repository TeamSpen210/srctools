pure_python = files(
    '__init__.py',
    '_class_resources.py',
    '_engine_db.py',
    '_fgd_helpers.py',
    '_py_vtf_readwrite.py',
    '_shaderdb.py',
    'binformat.py',
    'bsp.py',
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
    'surfaceprop.py',
    'tokenizer.py',
    'vec.py',
    'vmf.py',
    'vmt.py',
    'vpk.py',
    'vtf.py',
)
pure_python += files(
    'py.typed',
    'fgd.lzma',
)

vcs_tag(
    fallback: '<dev>',
    input: '_version.input.py',
    output: '_version.py',
)
pure_python += files('_version.py')

py.install_sources(pure_python, subdir: 'srctools')

subdir('_pyinstaller')
