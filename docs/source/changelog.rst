Changelog
=========

.. contents::
	:local:
	:backlinks: none

----------
Dev branch
----------

* Expand on documentation, build into explicit docs files.
* Fix :py:meth:`srctools.logging.LoggerAdapter.log` being invalid in Python 3.7.
* Make :py:mod:`srctools.fgd` work when reloaded.
* Remove blank ``srctools.choreo`` module.
* Disable iterating on :py:class:`srctools.math.Matrix`.
* Add iterable parameter to :py:meth:`srctools.dmx.Attribute.array()`, for constructing arrays with values.
* Remove useless ``header_len`` attribute from :py:class:`srctools.vpk.VPK`.
* Rename ``srctools.property_parser.Property`` to :py:class:`srctools.keyvalues.Keyvalues`. as well as :py:class:`~srctools.keyvalues.NoKeyError` and :py:class:`~srctools.keyvalues.KeyValError`.

-------------
Version 2.3.4
-------------

* Add public submodules to ``__all__``.
* Disable escapes when parsing gameinfo files.
* Add unprefixed ``vtx`` files to :py:data:`srctools.mdl.MDL_EXTS`.
* Skip empty folder/extension dicts when writing VPK files.
* Clean up VPK fileinfo dicts when deleting files.
* Default :py:class:`srctools.fgd.IODef` to :py:attr:`srctools.fgd.ValueTypes.VOID`.
* Sort tags when exporting FGDs, to make it determinstic.

-------------
Version 2.3.3
-------------

* Writing out soundscript/particle cache can be non-atomic.
* Vendor code from deprecated ``chunk.Chunk`` standard library class.
* Fix bad use of builtin generics.

-------------
Version 2.3.2
-------------

* Make particle systems use a cache file for the manifest too.
* Make :py:meth:`srctools.fgd.FGD.engine_db()` actually cache and copy the database.
* Automatically add the ``update`` folder to searchpath precedence, fixing TeamSpen210/HammerAddons#164.
* Make DMX scalar type deduction more strict (removing iterable -> vec support), making it typesafe.
* Add :py:data:`srctools.filesys.CACHE_KEY_INVALID`.
* Add :py:meth:`srctools.math.Matrix.from_angstr()`.

-------------
Version 2.3.1
-------------

* Fix :py:meth:`srctools.vmf.Output.combine` not handling ``times`` correctly.
* :py:func:`srctools.math.quickhull()` is now public.
* Add :py:meth:`srctools.bsp.BSP.is_cordoned_heuristic()`.
* Restrict :py:attr:`srctools.bsp.Overlay.min_cpu`, :py:attr:`~srctools.bsp.Overlay.max_cpu`, :py:attr:`~srctools.bsp.Overlay.min_gpu` and :py:attr:`~srctools.bsp.Overlay.max_ppu` to valid values.
* Test against Python 3.11.
* Read/write the :py:attr:`~srctools.bsp.BSP_LUMPS.LEAFMINDISTTOWATER` lump data into :py:attr:`srctools.bsp.VisLeaf.min_water_dist`.
* Read/write the :py:attr:`~srctools.bsp.BSP_LUMPS.LEAFWATERDATA` lump.
* Copy flags when copying :py:class:`srctools.bsp.TexInfo` from an existing source.
* :py:class:`srctools.tokenizer.Tokenizer` now handles universal newlines conversion.
* Disallow newlines in keyvalues keys when parsing by default. This catches syntax errors earlier.
* More :py:class:`srctools.game.Game` ``gameinfo.txt`` fields are now optional.

-------------
Version 2.3.0
-------------

* __Postcompiler code has been moved to HammerAddons.__
* Fix raw sound filenames not stripping special characters from the start when packing.
* Allow :py:class:`srctools.dmx.Color` to omit alpha when parsed from strings, and roound/clamp values.
* Handle INFRA's altered :py:class:`srctools.bsp.Primitive` lump.
* Read soundscripts and breakable chunk files with code page 1252.
* Handle TF2's LZMA compressed lumps.
* Detect various alternate versions of :py:class:`srctools.bsp.StaticProp` lumps, and parse them.
* :py:class:`srctools.vmf.Entity` now directly implements :external:py:class:`collections.abc.MutableMapping`. Direct access to the ``Entity.keys`` :external:py:class:`dict` is deprecated.
* Correctly handle proxy blocks in :py:class:`~srctools.vmt.VMT` patch shaders.
* DMX stub and null elements use an immutable subclass, instead of having elements be None-able.
* Disallow entities to have a blank classname.
* Elide long arrays in element reprs.
* Add some additional logs when finding propcombine models fails.
* Clean up :py:meth:`srctools.Property.build()` API.
* Make error messages more clear when :py:meth:`srctools.tokenizer.Tokenizer.error()` is used directly with a :py:class:`~srctools.tokenizer.Token`.
* Include potential variables in :external:py:class:`KeyError` from :py:meth:`srctools.vmf.EntityFixup.substitute()`.
* Remove support for deprecated ``imghdr`` module.
* Upgrade plugin finding logic to ensure each source is mounted under a persistent ID.
* Add missing :py:attr:`srctools.bsp.Primitive.dynamic_shadows`.
* Deprecate :py:class:`srctools.Atomicwriter`, use the ``atomicwrites`` module.
* :py:mod:`srctools._class_resources` is now only imported when required.
* Use Cython when building, instead of including sources.
* :py:attr:`srctools.vmf.Entity.fixup` will instantiate the :py:class:`~srctools.vmf.EntityFixup` object only when actually required.


-------------
Version 2.2.5
-------------

* Restore :py:meth:`srctools.dmx.Attribute.iter_str()` etc method's ability to iterate scalars.
* Suppress warnings in :py:meth:`srtools.Property.copy()`.


-------------
Version 2.2.4
-------------

* Fix behaviour of :py:meth:`srctools.Property.__getitem__()` and :py:meth:`~srctools.Property.__setitem__()`.
* Improve performance of :py:class:`~srctools.vpk.VPK` parsing.
* Add support for Portal Revolution's :py:class:`~srctools.fgd.FGD` helper tweaks.
* Add option to collapse and remove IO proxies entirely.
* Fix ``ModelCompiler`` creating directories with relative paths.
* Pass through unknown model flag bits unchanged.
* Fix VPK ascii check.
* Handle VMF groups correctly.
* Add :py:meth:`srctools.math.Vec.bbox_intersect`.
* Allow indexing :py:class:`~srctools.vmf.PrismFace` objects by a normal to get a :py:class:`~srctools.vmf.Side`.
* Add :py:meth:`srctools.dmx.Attribute.iter_str()` etc methods for iterating converted values. Directly iterating the :py:class:`~srctools.dmx.Attribute` is deprecated.
* Add :py:meth:`srctools.dmx.Attribute.append()`, :py:meth:`~srctools.dmx.Attribute.extend()` and :py:meth:`~srctools.dmx.Attribute.clear_array()` methods.
* Fix corruption from mistaken deduplication of :py:class:`srctools.bsp.VisLeaf` and :py:class:`~srctools.bsp.Primitive` lumps.

-------------
Version 2.2.3
-------------

* Fix use of builtin generics.

-------------
Version 2.2.2
-------------

* Document some known particle manifest paths.
* Handle double-spacing in animation particle options.
* Improve type hints in :py:mod:`srctools.smd`.


-------------
Version 2.2.1
-------------

* Missing particles is now an warning, not an error.
* Particles are now case-insensitive.
* py:meth:`srctools.vmf.EntityFixup.keys()`, :py:meth:`~srctools.vmf.EntityFixup.values()` and :py:meth:`~srctools.vmf.EntityFixup.items()` are now full mapping views.
* Fix incompatibility with some Python versions.

-------------
Version 2.2.0
-------------

* Make ``srctools.compiler.mdl_compiler`` generic, to allow typechecking results.
* Add :py:mod:`srctools.particles`.
* DMX attributes may now be copied using the :external:py:mod:`copy` module, and also tested for equality.
* :py:class:`srctools.sndscript.Sound` now lazily creates operator stack keyvalue objects.
* :py:class:`srctools.packlist.Packlist` now can pack particle systems, and generate particle manifests.
* Animation events which spawn particles are also detected.

-------------
Version 2.1.0
-------------

* Fix ``%``-formatted logs breaking when :py:mod:`srctools.logger` is used.
* Add :py:meth:`Property.extend()`, instead of using ``+`` or :py:meth:`Property.append()` with a block. That usage is deprecated.
* Deprecate creating root properties with ``name=None``.
* :py:class:`srctools.filesys.FileSystemChain` is no longer generic, this is not useful.
* Add functions which embed a Keyvalues1 tree in a DMX tree.
