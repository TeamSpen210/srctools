srctools.sndscape
------------------

.. module:: srctools.sndscape
	:synopsis: Reads and writes Soundscape files.


The `!sndscape` module reads and writes soundscape files. Basic usage involves first parsing the
file into a :py:class:`~srctools.keyvalues.Keyvalues` tree, then parsing into soundscapes::

	with open('soundscape.txt', 'r') as f:
		kv = Keyvalues.parse(f)
	soundscapes = Soundscape.parse(kv)

The opposite is done by simply calling `Soundscape.export()` on each soundscape to append to a file.

Enumerations
============

.. autosrcenum:: PosType
	:members:
	:undoc-members:
	:member-order: bysource

Classes
=======


.. autoclass:: Soundscape
	:members:
	:undoc-members:

.. autoclass:: SubScape
	:members:
	:undoc-members:

.. py:class:: SoundRule()
	:abstract:

	These attributes are common to `RandSound` and `LoopSound`.

	.. autoattribute:: position
	.. autoattribute:: volume
	.. autoattribute:: pitch
	.. autoattribute:: level
	.. autoattribute:: no_restore

.. autoclass:: RandSound
	:show-inheritance:
	:members:
	:undoc-members:

.. autoclass:: LoopSound
	:show-inheritance:
	:members:
	:undoc-members:
