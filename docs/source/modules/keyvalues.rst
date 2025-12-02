srctools.keyvalues
------------------------

.. module:: srctools.keyvalues
	:synopsis: Reads and writes KeyValues 1 files.

:vdc:`Keyvalues 1 <KeyValues>` files are very common in Source 1.
VMFs, soundscripts, VMT and many other files use the syntax.
A keyvalues file is a tree structure composed of keyvalue nodes. Each has a name,
and either a single string value or any number of children. Duplicate names are allowed.

===============
Core Behaviours
===============

Keyvalues are represented with :py:class:`Keyvalues` objects, which have three forms:

* "Leaf" keyvalues have a name, and a single string :py:attr:`~Keyvalues.value` but no children.
* "Block" keyvalues have a name, and a list of child keyvalues.
* "Root" keyvalues are a special block type, with no name. When exported, its children are written
  directly into the file with no nesting. This is returned by :py:meth:`~Keyvalues.parse` to allow
  use of the Keyvalues API on these root blocks, while preserving the structure.

.. autoclass:: Keyvalues

	.. automethod:: root

	.. autoattribute:: name

	.. autoattribute:: real_name

	.. autoattribute:: value

	.. autoattribute:: line_num

	The following two methods allow identifying the type of a keyvalue.

	.. automethod:: has_children

	.. automethod:: is_root

If an operation is performed on a leaf keyvalue which requires children, the following exception
is raised:

.. autoclass:: LeafKeyvalueError
	:show-inheritance:


===================
Reading and Writing
===================
To read and write keyvalues files, use :py:meth:`~Keyvalues.parse` and :py:meth:`~Keyvalues.serialise`::

	with open('filename.txt', 'r') as read_file:
		kv = Keyvalues.parse(read_file, 'filename.txt')
	with open('filename_2.txt', 'w') as write_file:
		kv.serialise(write_file)

.. automethod:: Keyvalues.parse
.. automethod:: Keyvalues.serialise

:py:meth:`~Keyvalues.serialise` is also available with the spelling :py:meth:`~Keyvalues.serialize`.

.. automethod:: Keyvalues.export
	:for: line

.. autoclass:: KeyValError

.. todo: Document FLAGS_DEFAULT?

=========
Searching
=========

A number of different methods are available to search a tree and locate specific keyvalues. When
searching by name, all methods are case-insensitive, and return the *last* matching value, not the
first.

The simplest search is to find a matching key as a direct child. :py:meth:`~Keyvalues.find_key` finds
any child, while :py:meth:`~Keyvalues.find_block` only finds blocks. Keyvalues can also be indexed
to retrieve a string value::

	kv['value']  # Returns the contents, or raises
	kv['value', 'default'] # Returns default value if not found.


.. automethod:: Keyvalues.find_key

.. automethod:: Keyvalues.find_block

This exception is raised if a lookup fails. Currently direct indexing raises :py:exc:`IndexError`,
but this will change. Prefer catching :py:exc:`LookupError`, which will catch both.

.. autoclass:: NoKeyError
	:show-inheritance:

As a convenience, :py:meth:`Keyvalues.int`, :py:meth:`Keyvalues.float`, :py:meth:`~Keyvalues.bool`
and :py:meth:`~Keyvalues.vec` will search for a key, parse to the specified type, returning a default
if the parse fails or the key is missing.

.. automethod:: Keyvalues.bool

.. automethod:: Keyvalues.float

.. automethod:: Keyvalues.int

.. automethod:: Keyvalues.vec

For searching deeper in trees, the following iterators are available:

.. automethod:: Keyvalues.find_all
	:for: kv

.. automethod:: Keyvalues.find_children
	:for: child

.. automethod:: Keyvalues.iter_tree
	:for: kv

In many keyvalues files, arrays of data are defined as a block of leaf values, where the name of each
is identical or irrelevant. The following helper function makes parsing these easier:

.. automethod:: Keyvalues.as_array

Block keyvalues are also sequences, and support iteration and :py:func:`len()` as normal.

.. automethod:: Keyvalues.__iter__

.. automethod:: Keyvalues.__len__

=======
Editing
=======
.. todo: Organise these.

.. automethod:: Keyvalues.append

.. automethod:: Keyvalues.extend

.. automethod:: Keyvalues.build

.. automethod:: Keyvalues.ensure_exists

.. automethod:: Keyvalues.merge_children

.. automethod:: Keyvalues.set_key

===============
Special methods
===============

.. automethod:: Keyvalues.__repr__

.. automethod:: Keyvalues.__str__

.. automethod:: Keyvalues.__ne__

.. automethod:: Keyvalues.__eq__

.. automethod:: Keyvalues.__add__

.. automethod:: Keyvalues.__iadd__

.. automethod:: Keyvalues.__bool__

.. automethod:: Keyvalues.__contains__

.. automethod:: Keyvalues.__getitem__

.. automethod:: Keyvalues.__setitem__

.. automethod:: Keyvalues.__delitem__


=============
Miscellaneous
=============

.. automethod:: Keyvalues.as_dict() -> dict

.. automethod:: Keyvalues.clear

.. automethod:: Keyvalues.copy

.. automethod:: Keyvalues.edit
