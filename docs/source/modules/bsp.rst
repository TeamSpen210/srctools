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
handled. For full details, consult the :vdc:`VDC article on the format <BSP_(Source)>`.


General Functionality
=====================
To load a BSP, simply construct :py:class:`BSP`, passing the filename as a parameter. This will
load the full file into memory, but only parse the main headers. Due to the complexity of BSP files,
currently it is not possible to create one from scratch. Call ::py:meth:`BSP.save()` to save.

.. py:class:: BSP

    .. py:method:: __init__(filename: str | os.PathLike[str], version: Union[VERSIONS, GameVersion, None] = None)

        Create and load a BSP.

        :param filename: The filename to read.
        :param version: Specify the expected file version, causing an error if the BSP does not match.

    .. py:method:: read(self, expected_version: VERSIONS | GameVersion | None = None) -> None

        Reload the BSP file from disk.

        :param expected_version: Specify the expected file version, causing an error if the BSP does not match.

    .. py:method:: save(self, filename: str | None = None) -> None

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

.. py:attribute:: BSP.bmodels
    :type: ~weakref.WeakKeyDictionary[Entity, BModel]

    For each brush entity (including worldspawn) this defines the compiled brush data. Since this
    is a weak dictionary, removing entities will automatically clear their brush data. In engine,
    entities are linked to their brushes with a "model name" like ``*42``. Any entity in this
    dictionary will have its ``model`` keyvalue overwritten to point to the specified model.

.. py:class:: BModel

    .. py:attribute:: mins
        :type: ~srctools.math.Vec

    .. py:attribute:: maxes
        :type: ~srctools.math.Vec

        Axial bounding box surrounding the brush model.

    .. py:attribute:: origin
        :type: ~srctools.math.Vec

        Original position of the brushes.

    .. py:attribute:: node
        :type: VisTree

        References the root node for the visleaf tree for this brush model.

    .. py:attribute:: faces
        :type: list[Face]

        All faces in this brush model, unsorted.

    .. py:attribute:: phys_keyvalues
        :type: ~srctools.keyvalues.Keyvalues | None

        If this brush model is solid, this contains the VPhysics data. It is very similar to that
        in ``.phy`` model files. For each brush, this stores metadata like mass, surfaceprop, etc.

    .. automethod:: clear_physics

Pakfile
-------

.. py:attribute:: BSP.pakfile
    :type: zipfile.ZipFile

    The pakfile is an internal archive containing resources packed into the BSP for the level to use.
    By default, cubemaps are included here as well as patched brush materials VBSP generates for various
    purposes. Any file can be stored here, but only the :py:data:`zipfile.ZIP_STORED` (no compression) format is allowed.

    The returned zipfile wraps an :py:class:`io.BytesIO` object, and will be closed/finalised automatically. Do not close it yourself.


Textures
--------

Textures (really materials, the name is a GoldSource leftover) are stored as `TexInfo` objects, which contain the material as well
as S/T positioning, lightmap and other data. This allows brushes with matching data to be shared. Overlays also
use texinfo, but ignore the S/T positioning. VBSP stores some data derived from the
material's ``$basetexture`` in the BSP - its size and reflectivity. This means that creating `TexInfo`
requires either providing this extra data manually, or supplying a
:py:class:`~srctools.filesys.FileSystem` to automatically read this data from the material/texture files.

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

Static Props
------------

.. py:attribute:: BSP.props
    :type: list[StaticProp]

    ``prop_static`` entities are specially handled by VBSP, and stored in their own game lump (with ID ``sprp``).
    This lump has undergone significant changes in various games, so many attributes may do nothing.

.. automethod:: BSP.static_prop_models
    :for: model_name

.. py:attribute:: BSP.static_prop_version
    :type: StaticPropVersion
    :value: StaticPropVersion.UNKNOWN

    The version number for static props is unreliable, with incompatible games using overlapping versions.
    The byte size of each prop structure is used to determine the precise version used. This stores
    the version detected to allow saving correctly. This can also be set manually before ``props``
    is accessed to override the automatic detection.

.. autoclass:: StaticProp
    :members:
    :undoc-members:

.. autosrcenum:: StaticPropFlags
    :members:
    :undoc-members:
    :member-order: bysource

.. autosrcenum:: StaticPropVersion
    :members:
    :undoc-members:
    :member-order: bysource

Detail Props
------------

.. py:attribute:: BSP.detail_props
    :type: list[DetailProp]

    When compiling, VBSP places detail props onto all brushes, and stores them in their
    own game lump (with ID ``dprp``). There are three types of prop - model, simple sprite,
    and 'shape' (formed of multiple sprites in a particular pattern).

.. autosrcenum:: DetailPropOrientation
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: DetailProp()
    :members:
    :undoc-members:

.. autoclass:: DetailPropModel
    :members:
    :undoc-members:

.. autoclass:: DetailPropSprite
    :members:
    :undoc-members:

.. autoclass:: DetailPropShape
    :members:
    :undoc-members:


Cubemaps
--------

.. py:attribute:: BSP.cubemaps
    :type: list[Cubemap]

    ``env_cubemap`` entities are parsed out of the VMF and stored in their own lump.

.. autoclass:: Cubemap
    :members:

Overlays
--------

.. py:attribute:: BSP.overlays
    :type: list[Overlay]

    ``info_overlay`` s are stored in this special lump.

.. autoclass:: Overlay
    :members:
    :undoc-members:


Visibility
----------

.. py:attribute:: BSP.visibility
    :type: Visibility | None

    If VVIS has run, this contains the computed visibility calculations between all leaves.

.. autoclass:: Visibility
    :members:

.. automethod:: BSP.is_potentially_visible

.. automethod:: BSP.set_potentially_visible


Visleafs
--------

The visleaf structure is composed of a tree either of nodes or vis-leafs. Note these are always
calculated, even without VVIS - the tree is fundamental to the BSP structure.

.. attribute:: BSP.visleafs
    :type: list[VisLeaf]

    The array of all visleafs.

.. attribute:: BSP.nodes
    :type: list[VisTree]

    The array of nodes, each splitting the parent into two children.

.. automethod:: BSP.vis_tree

.. autoclass:: VisTree

    .. autoattribute:: plane

    .. autoattribute:: mins

    .. autoattribute:: maxes

    .. autoattribute:: child_neg

    .. autoattribute:: child_pos

    .. autoattribute:: area_ind

    .. autoattribute:: faces

    .. automethod:: test_point

    .. automethod:: VisTree.iter_leafs
        :for: visleaf

.. autoclass:: VisLeaf
    :members:
    :undoc-members:

.. autosrcenum:: VisLeafFlags
    :members:
    :undoc-members:
    :member-order: bysource


.. py:attribute:: BSP.water_leaf_info

    For each water brush in the map, this structure stores some additional information.

.. autoclass:: LeafWaterInfo
    :members:
    :undoc-members:


Planes
------
.. TODO Descriptions.

.. py:attribute:: BSP.planes
    :type: list[Plane]

.. autoclass:: Plane
    :members:
    :undoc-members:

.. autosrcenum:: PlaneType

    .. automethod:: PlaneType.from_normal


Brushes
-------

The brushes lump contains a copy of all the original brushes in the map. This is mainly used for
collision.

.. py:attribute:: BSP.brushes
    :type: list[Brush]

.. autoclass:: Brush
    :members:
    :undoc-members:

.. autoclass:: BrushSide
    :members:
    :undoc-members:

Faces
-----
.. TODO description

.. py:attribute:: BSP.faces
    :type: list[Face]

.. py:attribute:: BSP.orig_faces
    :type: list[Face]

    These faces more closely match those in Hammer.

.. py:attribute:: BSP.hdr_faces
    :type: list[Face]

    Face data used in HDR mode.

.. py:attribute:: BSP.primitives

    Primitive surfaces, also known as 't-junctions' or 'waterverts' are generated to
    stitch together T-junction faces. This fixes potential seams.

.. py:attribute:: BSP.surfedges
    :type: list[Edge]

.. py:attribute:: BSP.vertexes
    :type: list[Vec]

.. autoclass:: Face
    :members:
    :undoc-members:

.. autoclass:: Primitive
    :members:
    :undoc-members:

.. autoclass:: Edge
    :members:
    :undoc-members:

.. autoclass:: RevEdge
    :members:
    :undoc-members:

Miscellaneous
=============

.. automethod:: BSP.is_cordoned_heuristic


.. py:type:: BrushContents
    :canonical: srctools.consts.BSPContents

    This enum specifies collision types for brushes. It is reimported in the BSP module for convenience.

.. TODO: Where to put this alias in the file?
