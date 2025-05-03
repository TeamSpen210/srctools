############
Deprecations
############

Several features and modules are deprecated, renamed to something else or removed to allow other features.


srctools.bsp
------------

.. py:currentmodule:: srctools.bsp

.. automethod:: BSP.read_header

.. automethod:: BSP.read_game_lumps

.. automethod:: BSP.replace_lump

.. automethod:: BSP.read_ent_data

.. automethod:: BSP.write_ent_data(vmf: VMF, use_comma_sep: bool = ...)

.. automethod:: BSP.read_texture_names

.. automethod:: BSP.packfile
	:with: zipfile

.. automethod:: BSP.static_props

.. autoproperty:: VisTree.plane_norm

.. autoproperty:: VisTree.plane_dist


srctools.dmx
------------

.. py:currentmodule:: srctools.dmx

.. py:data:: STUB

	Call :py:meth:`srctools.dmx.StubElement.stub()` instead.

.. py:class:: AngleTup
	:no-index:
	:canonical: srctools.math.FrozenAngle

	Was a named tuple, use the frozen class instead.

.. py:class:: Vec3
	:no-index:
	:canonical: srctools.math.FrozenVec

	Was a named tuple, use the frozen class instead.


srctools.filesys
----------------

.. py:currentmodule:: srctools.filesys

.. automethod:: FileSystem.read_prop

.. automethod:: FileSystem._check_open

.. automethod:: FileSystem.__enter__

.. automethod:: FileSystem.__exit__

.. automethod:: FileSystem.open_ref

.. automethod:: FileSystem.close_ref


srctools.fgd
------------

.. py:class:: srctools.fgd.Keyvalues
	:no-index:
	:canonical: srctools.fgd.KVDef

	This was renamed so it is not confused with Keyvalues1 trees.



srctools.property_parser
------------------------

.. py:module:: srctools.property_parser
	:deprecated:
	:synopsis: Moved to srctools.keyvalues.

Deprecated original location of the :py:mod:`srctools.keyvalues` Keyvalues1 parser.

.. py:class:: Property
	:canonical: srctools.keyvalues.Keyvalues

Deprecated original name of :py:class:`srctools.keyvalues.Keyvalues`.


.. py:class:: KeyValError
	:no-index:
	:canonical: srctools.keyvalues.KeyValError

Deprecated original name of :py:class:`srctools.keyvalues.KeyValError`.


.. py:class:: NoKeyError
	:no-index:
	:canonical: srctools.keyvalues.NoKeyError

Deprecated original name of :py:class:`srctools.keyvalues.NoKeyError`.

srctools.vec
------------

.. py:module:: srctools.vec
	:deprecated:
	:synopsis: Moved to srctools.math.

Deprecated original location of :py:mod:`srctools.math` vector code.

.. py:class:: Vec
	:no-index:
	:canonical: srctools.math.Vec

Deprecated original name of :py:class:`srctools.math.Vec`.

.. py:class:: Angle
	:no-index:
	:canonical: srctools.math.Angle

Deprecated original name of :py:class:`srctools.math.Angle`.

.. py:class:: Matrix
	:no-index:
	:canonical: srctools.math.Matrix

Deprecated original name of :py:class:`srctools.math.Matrix`.

.. py:class:: Vec_tuple
	:no-index:
	:canonical: srctools.math.Vec_tuple

Deprecated original name of :py:class:`srctools.math.Vec_tuple`.

.. py:function:: srctools.vec.parse_vec_str
	:no-index:
	:canonical: srctools.math.parse_vec_str

Deprecated original name of :py:class:`srctools.math.parse_vec_str`.

.. py:function:: srctools.vec.to_matrix
	:no-index:
	:canonical: srctools.math.to_matrix

Deprecated original name of :py:class:`srctools.math.to_matrix`.

.. py:function:: srctools.vec.lerp
	:no-index:
	:canonical: srctools.math.lerp

Deprecated original name of :py:class:`srctools.math.lerp`.
