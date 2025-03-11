srctools.keyvalues
------------------------

.. module:: srctools.keyvalues
	:synopsis: Reads and writes KeyValues 1 files.

`Keyvalues 1 <https://developer.valvesoftware.com/wiki/KeyValues>`_ files are very common in Source 1.
VMFs, soundscripts, VMT and many other files use the syntax.
A keyvalues file is a tree structure composed of keyvalue nodes. Each has a name,
and either a single string value or any number of children. Duplicate names are allowed.

===============
Core Attributes
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

	.. automethod:: is_root

	.. automethod:: has_children


===================
Reading and Writing
===================
To read and write keyvalues files, use :py:meth:`~Keyvalues.parse` and :py:meth:`~Keyvalues.serialise`::

	with open('filename.txt', 'r') as read_file:
		kv = Keyvalues.parse(read_file, 'filename.txt')
	with open('filename_2.txt', 'w') as write_file:
		kv.serialise(write_file)

.. automethod:: Keyvalues.parse

.. method:: Keyvalues.serialise(*, indent: str='\t', indent_braces: bool=True, start_indent: str='') -> str
	:no-index:
.. automethod:: Keyvalues.serialise(file: io.TextIOBase, /, *, indent: str='\t', indent_braces: bool=True, start_indent: str='') -> None

.. py:function:: Keyvalues.serialize()
	:no-typesetting:

:py:meth:`~Keyvalues.serialise` is also available with the spelling :py:meth:`~Keyvalues.serialize`.

If an error occurs, the following exception is raised:

.. autoclass:: KeyValError

.. autodata:: FLAGS_DEFAULT

=========
Searching
=========
.. todo: Organise these.

.. automethod:: Keyvalues.find_all

.. automethod:: Keyvalues.find_key

.. automethod:: Keyvalues.find_block

.. automethod:: Keyvalues.find_children

.. autoclass:: NoKeyError
	:show-inheritance:

.. automethod:: Keyvalues.iter_tree

.. automethod:: Keyvalues.__iter__

.. automethod:: Keyvalues.bool

.. automethod:: Keyvalues.float

.. automethod:: Keyvalues.int

.. automethod:: Keyvalues.vec

.. automethod:: Keyvalues.append

=======
Editing
=======
.. todo: Organise these.

.. automethod:: Keyvalues.append

.. automethod:: Keyvalues.extend

.. automethod:: Keyvalues.as_array

.. automethod:: Keyvalues.as_dict

.. automethod:: Keyvalues.build

.. automethod:: Keyvalues.clear

.. automethod:: Keyvalues.copy

.. automethod:: Keyvalues.edit

.. automethod:: Keyvalues.ensure_exists

.. automethod:: Keyvalues.export

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

.. automethod:: Keyvalues.__iter__

.. automethod:: Keyvalues.__len__

.. automethod:: Keyvalues.__getitem__

.. automethod:: Keyvalues.__setitem__

.. automethod:: Keyvalues.__delitem__
