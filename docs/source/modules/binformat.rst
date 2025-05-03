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


===============
Array Insertion
===============

In files like BSPs, data is often stored in a large array, then other sections refer to specific
indexes. These functions help in constructing such an array, while reusing entries to avoid
unnecessary duplication. `find_or_insert` inserts a single item, while `find_or_extend` inserts
a subsequence, returning the offset.

.. autofunction:: find_or_insert[T](item_list: list[T], key_func: Callable[[T], Hashable] = id) -> Callable[[T], int]:

.. autofunction:: find_or_extend[T](item_list: list[T], key_func: Callable[[T], Hashable] = id) -> Callable[[list[T]], int]:


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
