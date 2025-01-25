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
The raw data for both can be accessed with the following methods:

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


Textures
--------

.. automethod:: BSP.create_texinfo


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
