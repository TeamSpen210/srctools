srctools
--------

.. module:: srctools
	:synopsis: Items in the toplevel package.

A number of utilities are present in the toplevel of the ``srctools`` package.

=============
File Handling
=============

.. autofunction:: srctools.clean_line

.. autofunction:: srctools.is_plain_text

.. autofunction:: escape_quote_split

.. autoclass:: AtomicWriter

===============
Type Conversion
===============

.. autofunction:: bool_as_int

.. autofunction:: conv_int

.. autofunction:: conv_float

.. autofunction:: conv_bool

.. py:data:: BOOL_LOOKUP
	:type: typing.Mapping[str, bool]

``BOOL_LOOKUP`` is a constant mapping string values to the appropriate boolean result, the same way :py:func:`conv_bool` functions.

.. py:data:: EmptyMapping
	:type: collections.abc.MutableMapping[Any, Any]

This is a constant mapping which behaves as if it is always empty. It is intended for use in default parameter values, and other fallbacks. It also supports writing operations, which simply do nothing.


=======
Aliases
=======

Several classes from other modules are automatically imported, allowing for more convenient access.


.. py:class:: Vec
	:canonical: srctools.math.Vec

.. py:class:: Angle
	:canonical: srctools.math.Angle

.. py:class:: Matrix
	:canonical: srctools.math.Matrix

.. py:class:: Vec_tuple
	:canonical: srctools.math.Vec_tuple

.. py:function:: parse_vec_str
	:canonical: srctools.math.parse_vec_str

.. py:function:: lerp
	:canonical: srctools.math.lerp


These are found in :py:mod:`srctools.math`.

-------------------------------------

.. py:class:: NoKeyError
	:canonical: srctools.keyvalues.NoKeyError

.. py:class:: KeyValError
	:canonical: srctools.keyvalues.KeyValError

.. py:class:: Keyvalues
	:canonical: srctools.keyvalues.Keyvalues

These are found in :py:mod:`srctools.keyvalues`.

-----------------------------------------------

.. py:class:: FileSystem
	:canonical: srctools.filesys.FileSystem

.. py:class:: FileSystemChain
	:canonical: srctools.filesys.FileSystemChain

.. py:function:: get_filesystem
	:canonical: srctools.filesys.get_filesystem

These are found in :py:mod:`srctools.filesys`.

----------------------------------------------------

.. py:class:: VMF
	:canonical: srctools.vmf.VMF

.. py:class:: Entity
	:canonical: srctools.vmf.Entity

.. py:class:: Solid
	:canonical: srctools.vmf.Solid

.. py:class:: Side
	:canonical: srctools.vmf.Side

.. py:class:: Output
	:canonical: srctools.vmf.Output

.. py:class:: UVAxis
	:canonical: srctools.vmf.UVAxis

These are found in :py:mod:`srctools.vmf`.

----------------------------------------------------------

.. py:class:: SurfaceProp
	:canonical: srctools.surfaceprop.SurfaceProp

.. py:class:: SurfChar
	:canonical: srctools.surfaceprop.SurfChar

These are found in :py:mod:`srctools.surfaceprop`.

---------------------------------------------------------

.. py:class:: VPK
	:canonical: srctools.vpk.VPK

This is :py:class:`srctools.vpk.VPK`.

.. py:class:: FGD
	:canonical: srctools.fgd.FGD

This is :py:class:`srctools.fgd.FGD`.

.. py:class:: GameID
	:canonical: srctools.const.GameID

This is :py:class:`srctools.const.GameID`.

.. py:class:: VTF
	:canonical: srctools.vtf.VTF

This is :py:class:`srctools.vtf.VTF`.
