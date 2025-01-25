############
srctools.bsp
############

.. module:: srctools.bsp
	:synopsis: Reads and write compiled BSP maps.

The BSP module allows reading and writing compiled maps.
Note that the file format is composed of a large number of individual "lumps", each independent from one another.
Since there is a large amount of data, parsing occurs lazily - the first time an attribute is accessed,
it will be parsed into proper data structures along with any dependent lumps. When saved, parsed
lumps will be reconstructed, while unparsed lumps will be resaved unchanged.

In Team Fortress 2, support was added for compressed BSP files. Support for this is somewhat incomplete.

The data structures closely match the underlying file format, though some details are automatically
handled. For full details, consult the `VDC article on the format <https://developer.valvesoftware.com/wiki/BSP_(Source)>`_.


General Functionality
=====================
To load a BSP, simply construct :py:class:`BSP`, passing the filename as a parameter. This will
load the full file into memory, but only parse the main headers. Due to the complexity of BSP files,
currently it is not possible to create one from scratch. Call ::py:meth:`BSP.save()` to save.

.. py:class:: BSP

	.. py:method:: __init__(filename: StringPath, version: Union[VERSIONS, GameVersion, None] = None)

		Create and load a BSP.

		:param filename: The filename to read.
		:param version: Specify the expected file version, causing an error if the BSP does not match.

	.. py:method:: read(self, expected_version: VERSIONS | GameVersion | None = None) -> None

		Reload the BSP file from disk.

		:param expected_version: Specify the expected file version, causing an error if the BSP does not match.

	.. py:method:: save(self, filename: str | None = None) -> None:

		Write the BSP back into the given file.

		:param filename: If specified, overrides the originally read filename.


Versions
--------

A number of attributes are available describing the versions. Known versions are stored as enums.

.. autoattribute:: BSP.version

.. autoattribute:: BSP.game_ver

.. autoattribute:: BSP.map_revision

.. autosrcenum:: VERSIONS
	:members:
	:undoc-members:
	:member-order: bysource

.. autosrcenum:: GameVersion
	:members:
	:undoc-members:
	:member-order: bysource


Lumps
=====

Each BSP lump is exposed via a number of properties. When first accessed, the data is parsed, and
when saving, parsed data is re-exported. There are two types of lumps - the regular ones are always
present and defined by their order, while "game lumps" are optional and use an associated 4-byte ID.

Many lumps contain arrays of data, which are indexed into by other lumps. This is mostly handled
automatically - when parsing, the indexes are resolved into the referenced object. When saving,
the object is added to the array automatically if not present, so there is no need to update the
original lump. However, it is possible to clear that to remove unused values, if all lumps using
the data have been parsed so they can refill the array.

In addition to using parsed objects, raw data for both lump types can be accessed with the following methods:

.. py:attribute:: BSP.lumps

	Maps a lump ID to the stored lump.

	:type: dict[BSP_LUMPS, Lump]

.. py:attribute:: BSP.game_lumps

	Maps a lump ID to the stored game lump. The key should be 4 characters long.

	:type: dict[bytes, Lump]

.. automethod:: BSP.get_lump

.. automethod:: BSP.get_game_lump

.. autosrcenum:: BSP_LUMPS
	:members:
	:undoc-members:
	:member-order: bysource

.. autoclass:: Lump

	:members:

.. autoclass:: GameLump

	:members:


Entity Lump
-----------

.. py:attribute:: BSP.ents

	:type: ~srctools.vmf.VMF

	The entity lump stores all entities and their keyvalues. When parsed, this is exposed as a
	:py:class:`srctools.vmf.VMF` object, since the entities function identically. However, all brushes
	are stored elsewhere, so :py:attr:`~srctools.vmf.Entity.solids` attributes will all be blank and
	are ignored. See the :py:attr:`BSP.bmodels` attribute to locate the compiled brush data for an entity.

	Aside from the entities, the rest of the VMF object is ignored.

Pakfile
-------

.. py:attribute:: BSP.pakfile
	:type: zipfile.ZipFile

	The pakfile is an internal archive containing resources packed into the BSP for the level to use.
	By default, cubemaps are included here as well as patched brush materials VBSP generates for various
	purposes. Any file can be stored here, but only the :py:attr:`zipfile.ZIP_STORED` (no compression) format is allowed.

	The returned zipfile wraps an :py:class:`io.BytesIO` object, and will be closed/finalised automatically. Do not close it yourself.


Textures
--------

Textures (really materials, the name is a GoldSource leftover) are stored as `TexInfo` objects, which contain the material as well
as S/T positioning, lightmap and other data. This allows brushes with matching data to be shared. Overlays also
use texinfo, but ignore the S/T positioning. VBSP stores some data derived from the
material's `$basetexture` in the BSP - its size and reflectivity. This means that creating `TexInfo`
requires either providing this extra data manually, or supplying a
:py:class:`~srctools.fsys.FileSystem` to automatically read this data from the material/texture files.

.. py:attribute:: BSP.textures
	:type: list[str]

	The raw list of materials used. Automatically appended to by ``texinfo``.

.. py:attribute:: BSP.texinfo
    :type: list[TexInfo]

    Materials along with their positioning information.

To allow reusing existing values, the following method
should be used to create each entry:

.. automethod:: BSP.create_texinfo

.. Hide the constructor, create_texinfo should always be used.
.. autoclass:: TexInfo()
    :members:
    :undoc-members:


.. Lumps TODO:
    cubemaps: ParsedLump[list[Cubemap]] = ParsedLump(BSP_LUMPS.CUBEMAPS)
    overlays: ParsedLump[list[Overlay]] = ParsedLump(
        BSP_LUMPS.OVERLAYS,
        BSP_LUMPS.OVERLAY_FADES, BSP_LUMPS.OVERLAY_SYSTEM_LEVELS,
    )
    bmodels: ParsedLump['WeakKeyDictionary[Entity, BModel]'] = ParsedLump(
        BSP_LUMPS.MODELS,
        BSP_LUMPS.PHYSCOLLIDE,
    )
    brushes: ParsedLump[list[Brush]] = ParsedLump(BSP_LUMPS.BRUSHES, BSP_LUMPS.BRUSHSIDES)
    visleafs: ParsedLump[list[VisLeaf]] = ParsedLump(
        BSP_LUMPS.LEAFS,
        BSP_LUMPS.LEAFFACES, BSP_LUMPS.LEAFBRUSHES, BSP_LUMPS.LEAFMINDISTTOWATER,
    )
    water_leaf_info: ParsedLump[list[LeafWaterInfo]] = ParsedLump(BSP_LUMPS.LEAFWATERDATA)
    nodes: ParsedLump[list[VisTree]] = ParsedLump(BSP_LUMPS.NODES)
    # This is None if VVIS has not been run.
    visibility: ParsedLump[Optional[Visibility]] = ParsedLump(BSP_LUMPS.VISIBILITY)
    vertexes: ParsedLump[list[Vec]] = ParsedLump(BSP_LUMPS.VERTEXES)
    surfedges: ParsedLump[list[Edge]] = ParsedLump(BSP_LUMPS.SURFEDGES, BSP_LUMPS.EDGES)
    planes: ParsedLump[list[Plane]] = ParsedLump(BSP_LUMPS.PLANES)
    faces: ParsedLump[list[Face]] = ParsedLump(BSP_LUMPS.FACES)
    orig_faces: ParsedLump[list[Face]] = ParsedLump(BSP_LUMPS.ORIGINALFACES)
    hdr_faces: ParsedLump[list[Face]] = ParsedLump(BSP_LUMPS.FACES_HDR)
    primitives: ParsedLump[list[Primitive]] = ParsedLump(
        BSP_LUMPS.PRIMITIVES,
        BSP_LUMPS.PRIMINDICES, BSP_LUMPS.PRIMVERTS,
    )
    # Game lumps
    props: ParsedLump[list['StaticProp']] = ParsedLump(LMP_ID_STATIC_PROPS)
    detail_props: ParsedLump[list['DetailProp']] = ParsedLump(LMP_ID_DETAIL_PROPS)

Miscellaneous
=============

.. automethod:: BSP.is_cordoned_heuristic


Deprecated Functionality
========================

These methods have been replaced by others, and should not be used.

.. automethod:: BSP.read_header
	:deprecated: No longer has functionality.

.. automethod:: BSP.read_game_lumps
	:deprecated: No longer has functionality.

.. automethod:: BSP.replace_lump
	:deprecated: Use the relevant property directly, or set :py:attribute:`Lump.data`.

.. Classes TODO:
'StaticProp', 'StaticPropFlags',
'DetailProp', 'DetailPropModel', 'DetailPropOrientation', 'DetailPropShape', 'DetailPropSprite',
'TexData', 'TexInfo',
'Cubemap', 'Overlay',
'VisTree', 'VisLeaf', 'VisLeafFlags', 'LeafWaterInfo',
'Visibility',
'BModel', 'Plane', 'PlaneType',
'Primitive', 'Face', 'Edge', 'RevEdge',
'Brush', 'BrushSide', 'BrushContents',
