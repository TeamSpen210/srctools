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
	:type: collections.abc.MutableMapping

This is a constant mapping which behaves as if it is always empty. It is intended for use in default parameter values, and other fallbacks. It also supports writing operations, which simply do nothing.


=======
Aliases
=======

Several classes from other modules are automatically imported, allowing for more convenient access.


.. py:class:: Vec

	:canonical: :py:class:`srctools.math.Vec`

.. py:class:: Angle

	:canonical: :py:class:`srctools.math.Angle`

.. py:class:: Matrix

	:canonical: :py:class:`srctools.math.Matrix`

.. py:class:: Vec_tuple

	:canonical: :py:class:`srctools.math.Vec_tuple`

.. py:function:: parse_vec_str

	:canonical: :py:func:`srctools.math.parse_vec_str()`

.. py:function:: lerp

	:canonical: :py:func:`srctools.math.lerp()`

-------------------------------------

.. py:class:: NoKeyError

	:canonical: :py:class:`srctools.keyvalues.NoKeyError`

.. py:class:: KeyValError

	:canonical: :py:class:`srctools.keyvalues.KeyValError`

.. py:class:: Keyvalues

	:canonical: :py:class:`srctools.keyvalues.Keyvalues`

-----------------------------------------------

.. py:class:: FileSystem

	:canonical: :py:class:`srctools.filesys.FileSystem`

.. py:class:: FileSystemChain

	:canonical: :py:class:`srctools.filesys.FileSystemChain`

.. py:function:: get_filesystem

	:canonical: :py:func:`srctools.filesys.get_filesystem()`

----------------------------------------------------

.. py:class:: VMF

	:canonical: :py:class:`srctools.vmf.VMF`

.. py:class:: Entity

	:canonical: :py:class:`srctools.vmf.Entity`

.. py:class:: Solid

	:canonical: :py:class:`srctools.vmf.Solid`

.. py:class:: Side

	:canonical: :py:class:`srctools.vmf.Side`

.. py:class:: Output

	:canonical: :py:class:`srctools.vmf.Output`

.. py:class:: UVAxis

	:canonical: :py:class:`srctools.vmf.UVAxis`

----------------------------------------------------------

.. py:class:: SurfaceProp

	:canonical: :py:class:`srctools.surfaceprop.SurfaceProp`
.. py:class:: SurfChar

	:canonical: :py:class:`srctools.surfaceprop.SurfChar`

---------------------------------------------------------

.. py:class:: VPK

	:canonical: :py:class:`srctools.vpk.VPK`

.. py:class:: FGD

	:canonical: :py:class:`srctools.fgd.FGD`

.. py:class:: GameID

	:canonical: :py:class:`srctools.const.GameID`

.. py:class:: VTF

	:canonical: :py:class:`srctools.vtf.VTF`
