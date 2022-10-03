srctools.binformat
------------------

.. automodule:: srctools.binformat
	:synopsis: Common code for handling binary formats.

=========
Constants
=========

.. py:data:: ST_VEC
	:type: struct.Struct
	:value: Struct('fff')

	A :external:class:`struct.Struct` with three floats, for unpacking :py:class:`~srctools.math.Vec`, :py:class:`~srctools.math.Angle`, etc.
.. py:data:: SIZES
	:type: dict[str, struct.Struct]

	A dict mapping each fixed-size number format character (``i``, ``L``, ``h``, etc) to the size of the data.

-------------

.. py:data:: SIZE_CHAR
	:value: 1

.. py:data:: SIZE_SHORT
	:value: 2

.. py:data:: SIZE_INT
	:value: 4

.. py:data:: SIZE_LONG
	:value: 8

.. py:data:: SIZE_FLOAT
	:value: 4

.. py:data:: SIZE_DOUBLE
	:value: 8

	The size of each of these standard numeric types.

===============
Structure tools
===============

.. autoclass:: DeferredWrites
	:members:

-------------

.. autofunction:: struct_read

-------------

.. autofunction:: str_readvec

-------------

.. autofunction:: read_array

.. autofunction:: write_array

-------------

.. autofunction:: read_nullstr

-------------

.. autofunction:: read_nullstr_array

-------------

.. autofunction:: read_offset_array


============
Checksumming
============

.. autofunction:: checksum

.. py:data:: EMPTY_CHECKSUM
	:value: 0

	The checksum value of an empty bytes buffer (``b""``).


====================
LZMA (de)compression
====================

.. autofunction:: decompress_lzma

.. autofunction:: compress_lzma
