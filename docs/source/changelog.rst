Changelog
=========

.. contents::
    :local:
    :backlinks: none

-------------
Version (dev)
-------------
* Add :mod:`srctools.sndscape`, for parsing soundscape scripts.
* Changed `PackList.load_choreo_scenes() <srctools.packlist.PackList.load_choreo_scenes>` to warn if ``scenes.image`` is not found.
* Add support for ``orientedwidthheight()``, ``orientedwidthheighthalf()`` and ``clusteredlight()`` Strata FGD helpers.
* Added ``noinherit()`` FGD helper, for indicating values to avoid inheriting from base classes.

-------------
Version 2.6.2
-------------

* Fix unknown bits sometimes not round tripping through enum bitfields.
* Fix incorrect parsing of the deprecated ``bbox()`` FGD helper.
* Add support for saving/loading VTF hotspot data.
* Don't backslash-escape single quote characters.
* Consult the various app IDs when looking for folders for searchpaths.
* Deprecate :meth:`UVAxis.rotate() <srctools.vmf.UVAxis.rotate>`.

-------------
Version 2.6.1
-------------

* Add :class:`srctools.math.PointsMap`, which fuzzily matches vector keys.
* Add :meth:`Geometry.from_points() <srctools.geometry.Geometry.from_points>`, creating a brush from a point cloud.
* Add support for exporting the old FGD 'report' syntax (``name(*type)``).
* Fix :func:`srctools.math.quickhull()` not functioning, allow any vector as input.
* Suppress the custom FGD editor-only flag if custom syntax is disabled.
* Support various shader parameters added by Strata Source.
* Properly implement detail prop packing logic, only pack sprites/models that
  are used by the BSP.
* Add :mod:`srctools.types`, which contains common types and classes.
* Switch type hints for all functions taking files to more precise protocols.
* Add source tracking for packed files, allows logging the cause for each file.
* Don't resave package/soundscript caches for packlists if not modified.
* Fix issue with locating Steam folders on non-Windows platforms.

-------------
Version 2.6.0
-------------

* Rewrite FGD 'helper' parsing to more closely match the actual parser.
    * Spaces between arguments, not the commas are used to separate them. For most helpers commas are meaningless.
    * Make it clear that :class:`srctools.fgd.UnknownHelper` should not be relied on -
      expect proper implementation to be added in future versions.
    * RGB colours now are ``0-255`` integers, not floats.
    * Helpers now all support tags, except for some extension helpers.
    * Deprecate ``bbox()`` helper, this doesn't actually exist.
    * Backward-incompatible change - renamed `HelperExtAppliesTo.tag <srctools.fgd.HelperExtAppliesTo.applies>`
      to ``.applies``.
* Add :mod:`srctools.geometry`, which implements clipping and carving brushes.
* Add extension ``editor`` (-only) flag to FGD keyvalue definitions.
* :src-issue:`38`: Preserve entity/node/brush/etc IDs when parsing VMFs and BSP entities.
* Treat ``ogg`` files as valid sounds in various places.
* Consistently raise :class:`TypeError` if invalid types are sent to a :class:`~srctools.tokenizer.Tokenizer`.
* Properly add ``base_plat_train`` class resource function and fix ``item_teamflag``.
* :bee-pack-issue:`4299`: Fix sprites and decals not being detected properly for packing.
* Added ``periodic_callback`` option to `~srctools.tokenizer.Tokenizer`, and allow that to be passed
  through `Keyvalues.parse() <srctools.keyvalues.Keyvalues.parse>` and
  `FileSystem.read_kv1() <srctools.filesys.FileSystem.read_kv1>`.
* Fix `BSP.static_prop_models() <srctools.bsp.BSP.static_prop_models>` breaking if props were already parsed.
* Make toplevel aliases like ``srctools.VPK`` lazy import.
* Add support for detecting sounds out of choreo scenes when packing.

-------------
Version 2.5.0
-------------
* Drop support for Python 3.8.
* Switch to Meson for building the package.
* :src-issue:`35`: Only use character escapes in FGD custom syntax mode. The original parser only allows ``\n``.
* Add :attr:`Token.PAREN_OPEN <srctools.tokenizer.Token.PAREN_OPEN>`/:attr:`~srctools.tokenizer.Token.PAREN_CLOSE`, enabled by :attr:`Tokenizer.string_parens <srctools.tokenizer.Tokenizer.string_parens>`.
* Avoid escaping ``/`` and ``?``, these are unambiguous. BSPs also allow ``\n`` in values again.
* Add ``multiline`` parameter to :py:func:`~srctools.tokenizer.escape_text`, to allow preserving ``\n``.
* Add :attr:`ValueTypes.EXT_SOUNDSCAPE <srctools.fgd.ValueTypes.EXT_SOUNDSCAPE>`.
* Support exporting FGDs with unevaulated string bases.
* Add suppport for reading non-directory VPK files, where contents are all in one file.
* FGDs may optionally permit unknown value types to be parsed/exported in string form.
* Fix some options being incorrectly handled in Strata game mounts definitions.
* Allow entities to delete multiple keyvalues at once.
* Add support for ``|appid_XXX|`` syntax in ``gameinfo.txt`` files.
* Fix silent buttons trying to pack invalid ``Buttons.snd0`` soundscripts.
* Handle entities being added/removed during iteration of :py:meth:`VMF.search() <srctools.vmf.VMF.search>`.
* Share common strings in the engine database to save some space.
* Fix saving the ``PHYSCOLLIDE`` BSP lump.
* In log files, increase size limits for :class:`ExceptionGroup` tracebacks.
* Add :class:`srctools.sndscript.SoundChars`, allowing easier parsing of sound characters.
* Add :attr:`srctools.vtf.ImageFormats.is_transparent`.
* Add :meth:`srctools.dmx.Element.get_wrap()`, allowing handling defaults more conveniently.
* Make :attr:`EntityDef.kv <srctools.fgd.EntityDef.kv>`, :attr:`.inp <srctools.fgd.EntityDef.inp>`
  and :attr:`.out <srctools.fgd.EntityDef.out>` views settable, improve behaviour.

-------------
Version 2.4.1
-------------
* Add :mod:`srctools.steam`, written by :gh-user:`Thenderek0`.
  This allows locating Steam games based on their app ID. Support was also added for parsing Strata
  mount definitions in gameinfo.txt.
* Add ``header_only`` option for :py:meth:`VTF.read() <srctools.vtf.VTF.read>`, allowing reading only metadata if the image is not required.
* Fix casing not being preserved for names of FGD keyvalues during parsing.
* Fix :meth:`PackList.write_soundscript_manifest() <srctools.packlist.PackList.write_soundscript_manifest>`,
  :meth:`~srctools.packlist.PackList.write_particles_manifest` and :meth:`~srctools.packlist.PackList.write_manifest` trying to write to a closed file.
* Handle string/int/float subclasses being assigned to VMF keys.
* Add ``single_block`` argument to :py:meth:`Keyvalues.parse() <srctools.keyvalues.Keyvalues.parse>`,
  allowing parsing blocks in the middle of a document.
* Allow disabling the "spawnflag labelling" FGD feature.
* :mod:`srctools.logger` log files will now always be written as UTF-8.
* Add a ``custom_syntax`` option to :py:meth:`FGD.export() <srctools.fgd.FGD.export>`, disabling
  export of custom syntax. Resources can now be exported.
* Correctly produce an error if a FGD entity definition is missing its closing bracket.
* Escape all characters :sdk-2013:`utlbuffer.cpp <tier1/utlbuffer.cpp#L57-L69>` does - ``\n``, ``\t``, ``\v``, ``\b``, ``\r``, ``\f``, ``\a``, ``\``, ``?``, ``'``, ``"``.
* Unconditionally enable support for escaping characters in DMX Keyvalues2, since Valve's parser can handle it. Binary formats never needed escaping.
* Correctly look up types for conditional shader parameters (``ldr?$bumpmap``).
* Parse FGDs correctly which have multiline strings with the plus on the second line.

-------------
Version 2.4.0
-------------
* Added :py:mod:`srctools.choreo`, for parsing choreographed scenes.
* Allow passing :py:class:`~srctools.math.FrozenVec` to :py:meth:`VMF.make_prism() <srctools.vmf.VMF.make_prism>`/:py:meth:`~srctools.vmf.VMF.make_hollow`.
* Fix bare strings on the end of CRLF lines eating the ``\r``.
* Escape characters like ``"`` when exporting VMFs and BSPs. This isn't supported by regular Source, but can be added by other branches.
* Added :py:attr:`Keyvalues.line_num <srctools.keyvalues.Keyvalues.line_num>`, to
  allow reporting the source location in exceptions.
* :py:meth:`Keyvalues.export() <srctools.keyvalues.Keyvalues.export>` is now deprecated, use :py:meth:`serialise() <srctools.keyvalues.Keyvalues.serialise>` instead.
* Allow passing VMF settings via keyword arguments - the ``map_info`` dictionary parameter is now deprecated.
* Allow directly passing enums to set VMF keyvalues and fixups, if the ``value`` is itself a valid value.
* Parse Strata Source's other VMF additions - viewport configuration, brush face vertices and instance visibility.
* Add :py:attr:`Tokenizer.plus_operator <srctools.tokenizer.Tokenizer.plus_operator>`, allowing
  ``+`` to be parsed as an operator for FGDs but still be valid inside bare strings elsewhere.
  These are common in ``gameinfo.txt``.
* Add :py:attr:`Solid.is_cordon <srctools.vmf.Solid.is_cordon>` to replace
  :py:attr:`cordon_solid <srctools.vmf.Solid.is_cordon>`, better representing its boolean nature.
* :src-issue:`29`: Fix ``0x03`` characters causing an early EOF in the tokeniser.
* Preserve passed in key casing in :py:meth:`~srctools.keyvalues.Keyvalues.find_key`/:py:meth:`~srctools.keyvalues.Keyvalues.find_block`'s return values.

--------------
Version 2.3.17
--------------
* Added :py:meth:`Keyvalues.serialise() <srctools.keyvalues.Keyvalues.serialise>`, a replacement for :py:meth:`~srctools.keyvalues.Keyvalues.export`.
* Fix ``+`` and ``=`` being parsed as part of a bare string if not the first character.
* Fix keyvalue-type snippets causing a parse error for code coming after them.
* Include filename/line number in missing snippet errors.

--------------
Version 2.3.16
--------------

* Fix entity keyvalues being lowercased when parsed from files.
* Add "snippets" to FGD parsing, allowing reuse of descriptions and other small pieces of data.
* Allow VMTs to use ``/* */`` comments.
* :src-issue:`24`: Fix incorrect :py:func:`matrix.inverse() <srctools.math.MatrixBase.inverse>` being calculated. PR by :gh-user:`Ozxybox`.
* Allow omitting file/line number parameters for TokenSyntaxError.
* Allow passing :py:class:`~srctools.vmf.PrismFace` to :py:class:`VMF.add_brush() <srctools.vmf.VMF.add_brush>`.
* Parse Strata Source's VMF displacement data.
* Remove negative zeros when formatting vector and angle values.
* Expand :py:meth:`Angle <srctools.math.AngleBase.from_basis>`/:py:meth:`Matrix.from_basis() <srctools.math.MatrixBase.from_basis>` to pick the orientation if less than 2 vectors are provided.
* Add :py:meth:`vmf.Side.from_normal() <srctools.vmf.Side.from_plane>`, which generates a VMF face pointing in an arbitary direction.
* Add :py:meth:`vmf.Solid.point_inside() <srctools.vmf.Solid.point_inside>`, which checks if a point is inside or outside a brush.

--------------
Version 2.3.15
--------------
* :ha-issue:`237`: Ensure FGD model helpers will override each other.
* :src-issue:`20`: Fix VTF.compute_mipmaps() not working for cubemaps.
* Correctly handle :file:`.vvd`/:file:`.vtx` etc files if packed as `~srctools.const.FileType.MODEL`.
* Improve performance of pure-Python VTF save/loading code.
* Add :py:meth:`Vec.clamped() <srctools.math.VecBase.clamped>`, for applying min/max bounds to a vector.
* Fix :py:meth:`Entity.make_unique() <srctools.vmf.Entity.make_unique>` renaming entities with numeric suffixes which were already unique.

--------------
Version 2.3.14
--------------
* Drop support for Python 3.7.
* Fix VMT parsing not braces on the same line as the block name, like ``Proxies {``.
* Add Cythonised versions of :py:func:`~srctools.conv_int`, :py:func:`~srctools.conv_float` and :py:func:`~srctools.conv_bool`.
* Added a :func:`repr` for :py:class:`srctools.vmf.Entity`.
* Automatically clean up up empty sets when removing entities from :py:attr:`VMF.by_class <srctools.vmf.VMF.by_class>` and :py:attr:`.by_target <srctools.vmf.VMF.by_target>`.
* Fixed saving/loading issues with a number of VTF formats.

--------------
Version 2.3.13
--------------
* Renamed :py:attr:`!NO_FLASHLIGHT` in :py:attr:`bsp.StaticPropFlags <srctools.bsp.StaticPropFlags>` to 
  :py:attr:`NO_SHADOW_DEPTH <srctools.bsp.StaticPropFlags.NO_SHADOW_DEPTH>` to reflect the actual 
  behaviour of the flag, added the real :py:attr:`NO_FLASHLIGHT <srctools.bsp.StaticPropFlags.NO_FLASHLIGHT>` define.
* Add :py:attr:`Tokenizer.preserve_comments <srctools.tokenizer.Tokenizer.preserve_comments>`, which
  produces :py:const:`COMMENT <srctools.tokenizer.Token.COMMENT>` tokens instead of discarding them.
* :src-issue:`18`: Fix incorrect module/function names in logging messages (via :gh-user:`ENDERZOMBI102`).
* Fix :py:meth:`srctools.vmt.Material.apply_patches()` not applying material proxies from the parent.
* Use ``surrogateescape`` when encoding/decoding BSP data, to allow values to round-trip.

--------------
Version 2.3.12
--------------
* Handle the special ``$gender`` "variable" in WAV filenames.
* Add ``prop_door_rotating`` class resource function.
* Remove ``weapon_script`` class resource function, instead use a direct resource in the FGD.
* Use :py:func:`!typing_extensions.deprecated` to mark functions and methods which should not be used.

--------------
Version 2.3.11
--------------
* Include the docs and tests in the source distribution.
* Add support for detecting and packing weapon scripts.
* Make custom model gibs inherit skinset when packing.
* Add :py:meth:`srctools.bsp.BModel.clear_physics()`, to delete physics data for a brush model.
* Add :py:class:`srctools.keyvalues.LeafKeyvalueError`, raised when block-only operations are
  attempted on leaf keyvalues. This improves the messages raised and makes them consistent.
* Fix :py:class:`srctools.vtf.Frame` indexing behaviour. It would access totally incorrect pixels.
* Correctly read/write L4D2's BSP format.

--------------
Version 2.3.10
--------------

* Fix :py:meth:`srctools.vtf.Frame.copy_from()` not clearing cached unparsed file data. If the VTF
  was parsed from a file, this could case changes to be overwritten with the original data.
* Add :py:meth:`srctools.vtf.Frame.fill()`, for filling a frame with a constant colour.
* :src-pull:`17`: Add support for Strata's non-uniform static prop scaling (by :gh-user:`Ozxybox`).
* Correctly handle non-float numeric values being passed to various :py:mod:`srctools.math` operations.
* Compute the total vertex count for parsed models.

-------------
Version 2.3.9
-------------

* Fix Cython version of :py:meth:`Vec.join() <srctools.math.VecBase.join>` using a default of
  :samp:`{x} {y} {z}`, not :samp:`{x}, {y}, {z}`.
* Added support for the `Strata BSP format <strata_bsp_>`_ (by :gh-user:`Ozxybox`).
* Improve internal FGD database format to allow parsing entities as they are required. For best
  efficiency, use :py:meth:`EntityDef.engine_def() <srctools.fgd.EntityDef.engine_def>` instead of
  :py:meth:`FGD.engine_dbase() <srctools.fgd.FGD.engine_dbase()>` if possible.
* Fix a few bugs with instance collapsing.

.. _strata_bsp: https://wiki.stratasource.org/modding/overview/bsp-v25

-------------
Version 2.3.8
-------------

* Fix :py:mod:`srctools.logger` discarding :external:py:class:`!trio.MultiError` (or its backport) if it
  bubbles up to the toplevel.
* Tweak VMF :py:meth:`localise() <srctools.vmf.Solid.localise>` and
  :py:meth:`translate()  <srctools.vmf.Solid.translate>` type hints to allow
  :py:class:`~srctools.math.FrozenVec` as the origin.
* Make movement and rotation of displacements work correctly.
* Handle pitch keyvalues correctly when instancing, only rotating if it is a specific type.
* Changed :py:func:`srctools.instancing.collapse_one()` to use the entclass database directly,
  deprecating the ``fgd`` parameter as a result.
* Fix :py:attr:`BSP.surfedges <srctools.bsp.BSP.surfedges>` incorrectly using edge ``0``, which may
  cause a single invisible triangle in maps.

-------------
Version 2.3.7
-------------

* Removed some unusable constructor parameters from :py:class:`srctools.vmf.VMF`, since they
  required passing in an object which requires the not-yet-constructed
  :py:class:`~srctools.vmf.VMF` as a parameter.
* Renamed ``srctools.fgd.KeyValues`` to ``KVDef``, so it is not confused with KV1 trees.
* Replace ``on_error`` callback in :py:meth:`srctools.logger.init_logging()` with ``error``, which
  now takes just an :external:py:class:`BaseException`.
* :py:class:`~srctools.surfaceprop.SurfaceProp` has been rewritten to use ``attrs`` to simplify code.
* Add :py:func:`srctools.run.send_engine_command()`, which executes console commands in a running
  Source game.
* :py:class:`~srctools.math.Vec` and :py:class:`~srctools.math.FrozenVec` no longer inherits from
  :external:py:class:`typing.SupportsRound`, since
  `typeshed updated <https://github.com/python/typeshed/pull/9151>`_ the overloads for
  :external:py:func:`round()` to permit zero-arg calls to return a non-:external:py:class:`int` type.
* Permit VMFs to accept frozen math classes directly as keyvalues.
* Fix multiplying vectors and :py:meth:`Vec.norm_mask() <srctools.math.VecBase.norm_mask()>` not producing
  :py:class:`~srctools.math.FrozenVec`.
* Parse errors in ``BSP.ents`` are more informative and verbose.
* Add an additional callback parameter to :py:meth:`PackList.pack_into_zip() <srctools.packlist.PackList.pack_into_zip()>` to
  finely control which files are packed.
* Implement vector and angle stringification manually, to ensure ``.0`` prefixes are always removed.
* Use :py:class:`~srctools.math.FrozenVec` and :py:class:`~srctools.math.FrozenAngle` in the
  :py:class:`~srctools.dmx` module instead of :external:py:func:`~collections.namedtuple` subclasses.
* Upgrade :py:class:`srctools.dmx.Time` to a full class instead of a :external:py:class:`typing.NewType`.
* Fix packlist logic inadvertently discarding ``skinset`` keyvalue hints when packing models.
* Change behaviour of DMX ``name`` and ``id`` attributes to match game logic. ``name`` is actually a
  regular attribute, but the uuid has a unique type and so can coexist with an attribute of the same name.
* Add support for Black Mesa's static prop format.
* Support integer values for soundscript channels, instead of just ``CHAN_`` constants.
* Add a distinct exception (:py:class:`~srctools.filesys.RootEscapeError`) for when :file:`../` paths
  go above the root of a filesystem.

-------------
Version 2.3.6
-------------

* Add ability to specify resources used in entities to the FGD file, move internal class resource
  definitions to the Hammer Addons repository.
* Added new :py:meth:`srctools.fgd.EntityDef.get_resources()` method, replacing ``fgd.entclass_*()``
  methods.
* When parsing VMF outputs, assume extraneous commas are part of the parameter.
* Add :py:class:`~srctools.math.FrozenVec`, :py:class:`~srctools.math.FrozenAngle` and
  :py:class:`~srctools.math.FrozenMatrix` - immutable versions of the existing classes. This is a
  far better version of ``Vec_tuple``, which is now deprecated.
* Build Python 3.11 wheels.
* Drop dependency on ``atomicwrites``, it is no longer being maintained.

-------------
Version 2.3.5
-------------

* Expand on documentation, build into explicit docs files.
* Fix :py:meth:`!srctools.logging.LoggerAdapter.log` being invalid in Python 3.7.
* Make :py:mod:`srctools.fgd` work when reloaded.
* Remove blank ``srctools.choreo`` module.
* Disable iterating on :py:class:`srctools.math.Matrix`, this is not useful.
* Add iterable parameter to :py:meth:`srctools.dmx.Attribute.array()`, for constructing arrays
  with values.
* Fix DMX :external:py:class:`bool` to :external:py:class:`float` conversions mistakenly returning
  :external:py:class:`int` instead.
* Remove useless ``header_len`` attribute from :py:class:`srctools.vpk.VPK`.
* Rename ``srctools.property_parser.Property`` to :py:class:`srctools.keyvalues.Keyvalues`,
  as well as :py:class:`~srctools.keyvalues.NoKeyError` and
  :py:class:`~srctools.keyvalues.KeyValError`.
* Allow parsing :py:class:`srctools.fgd.IODef` types which normally are not allowed for I/O.
  This will be substituted when exporting.
* Use ``__class__.__name__`` in reprs, to better support subclasses.
* :src-issue:`14`: Disable some size checks on LZMA decompression, so more TF2 maps can be parsed.

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
* Make :py:meth:`srctools.fgd.FGD.engine_dbase()` actually cache and copy the database.
* Automatically add the ``update`` folder to searchpath precedence, fixing :ha-issue:`164`.
* Make DMX scalar type deduction more strict (removing iterable -> vec support), making it typesafe.
* Add :py:data:`srctools.filesys.CACHE_KEY_INVALID`.
* Add :py:func:`Matrix.from_angstr() <srctools.math.MatrixBase.from_angstr>`.

-------------
Version 2.3.1
-------------

* Fix :py:meth:`srctools.vmf.Output.combine` not handling ``times`` correctly.
* :py:func:`srctools.math.quickhull()` is now public.
* Add :py:meth:`srctools.bsp.BSP.is_cordoned_heuristic()`.
* Restrict :py:attr:`srctools.bsp.Overlay.min_cpu`, :py:attr:`~srctools.bsp.Overlay.max_cpu`,
  :py:attr:`~srctools.bsp.Overlay.min_gpu` and :py:attr:`~srctools.bsp.Overlay.max_gpu` to valid values.
* Test against Python 3.11.
* Read/write the :py:attr:`~srctools.bsp.BSP_LUMPS.LEAFMINDISTTOWATER` lump data into
  :py:attr:`srctools.bsp.VisLeaf.min_water_dist`.
* Read/write the :py:attr:`~srctools.bsp.BSP_LUMPS.LEAFWATERDATA` lump.
* Copy flags when copying :py:class:`srctools.bsp.TexInfo` from an existing source.
* :py:class:`srctools.tokenizer.Tokenizer` now handles universal newlines conversion.
* Disallow newlines in keyvalues keys when parsing by default. This catches syntax errors earlier.
* More :py:class:`srctools.game.Game` ``gameinfo.txt`` fields are now optional.

-------------
Version 2.3.0
-------------

* **Postcompiler code has been moved to HammerAddons.**
* Fix raw sound filenames not stripping special characters from the start when packing.
* Allow :py:class:`srctools.dmx.Color` to omit alpha when parsed from strings, and roound/clamp values.
* Handle INFRA's altered :py:class:`srctools.bsp.Primitive` lump.
* Read soundscripts and breakable chunk files with code page 1252.
* Handle TF2's LZMA compressed lumps.
* Detect various alternate versions of :py:class:`srctools.bsp.StaticProp` lumps, and parse them.
* :py:class:`srctools.vmf.Entity` now directly implements
  :external:py:class:`collections.abc.MutableMapping`. Direct access to the ``Entity.keys``
  :external:py:class:`dict` is deprecated.
* Correctly handle proxy blocks in :py:class:`~srctools.vmt.Material` patch shaders.
* DMX stub and null elements use an immutable subclass, instead of having elements be None-able.
* Disallow entities to have a blank classname.
* Elide long arrays in element reprs.
* Add some additional logs when finding propcombine models fails.
* Clean up :py:meth:`!srctools.Property.build()` API.
* Make error messages more clear when :py:meth:`Tokenizer.error() <srctools.tokenizer.BaseTokenizer.error()>` is used
  directly with a :py:class:`~srctools.tokenizer.Token`.
* Include potential variables in :external:py:class:`KeyError` from
  :py:meth:`srctools.vmf.EntityFixup.substitute()`.
* Remove support for deprecated ``imghdr`` module.
* Upgrade plugin finding logic to ensure each source is mounted under a persistent ID.
* Add missing :py:attr:`srctools.bsp.Face.dynamic_shadows`.
* Deprecate :py:class:`srctools.AtomicWriter`, use the ``atomicwrites`` module.
* :py:mod:`!srctools._class_resources` is now only imported when required.
* Use Cython when building, instead of including sources.
* :py:attr:`srctools.vmf.Entity.fixup` will instantiate the :py:class:`~srctools.vmf.EntityFixup`
  object only when actually required.


-------------
Version 2.2.5
-------------

* Restore :py:meth:`srctools.dmx.Attribute.iter_str()` etc method's ability to iterate scalars.
* Suppress warnings in :py:meth:`Property.copy() <srctools.keyvalues.Keyvalues.copy>`.


-------------
Version 2.2.4
-------------

* Fix behaviour of :py:meth:`Property.__getitem__() <srctools.keyvalues.Keyvalues.__getitem__()>` and :py:meth:`Property.__setitem__() <srctools.keyvalues.Keyvalues.__setitem__()>`.
* Improve performance of :py:class:`~srctools.vpk.VPK` parsing.
* Add support for Portal Revolution's :py:class:`~srctools.fgd.FGD` helper tweaks.
* Add option to collapse and remove IO proxies entirely.
* Fix ``ModelCompiler`` creating directories with relative paths.
* Pass through unknown model flag bits unchanged.
* Fix VPK ascii check.
* Handle VMF groups correctly.
* Add :py:meth:`Vec.bbox_intersect() <srctools.math.VecBase.bbox_intersect>`.
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
* py:meth:`!srctools.vmf.EntityFixup.keys()`, :py:meth:`~srctools.vmf.EntityFixup.values()` and :py:meth:`~srctools.vmf.EntityFixup.items()` are now full mapping views.
* Fix incompatibility with some Python versions.

-------------
Version 2.2.0
-------------

* Make ``srctools.compiler.mdl_compiler`` generic, to allow typechecking results.
* Add :py:mod:`srctools.particles`.
* DMX attributes may now be copied using the :external:py:mod:`copy` module, and also tested for equality.
* :py:class:`srctools.sndscript.Sound` now lazily creates operator stack keyvalue objects.
* :py:class:`srctools.packlist.PackList` now can pack particle systems, and generate particle manifests.
* Animation events which spawn particles are also detected.

-------------
Version 2.1.0
-------------

* Fix ``%``-formatted logs breaking when :py:mod:`srctools.logger` is used.
* Add :py:meth:`Property.extend() <srctools.keyvalues.Keyvalues.extend>`, instead of using ``+`` or :py:meth:`<Property.append() <srctools.keyvalues.Keyvalues.append>` with a block. That usage is deprecated.
* Deprecate creating root properties with ``name=None``.
* :py:class:`srctools.filesys.FileSystemChain` is no longer generic, this is not useful.
* Add functions which embed a Keyvalues1 tree in a DMX tree.
