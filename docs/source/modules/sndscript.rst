srctools.sndscript
------------------

.. module:: srctools.sndscript
	:synopsis: Reads and writes Soundscript files.


Enumerations
============
Several options have a number of predefined names.

.. autosrcenum:: Pitch
	:members:
	:undoc-members:
	:member-order: bysource

.. autosrcenum:: Channel
	:members:
	:undoc-members:
	:member-order: bysource

.. autosrcenum:: Level
	:members:
	:undoc-members:
	:member-order: bysource

.. py:class:: VOLUME

.. py:data:: VOL_NORM

	Volumes have only one value available, which produces a full volume (:pycode:`1.0`).


Sound
=====

.. autoclass:: Sound
	:members:
	:undoc-members:

Sound Characters
================


.. py:data:: SND_CHARS

	String with some operator characters. Deprecated, use :py:class:`SoundChars` instead.

.. autosrcenum:: SoundChars
	:members:
	:undoc-members:
	:member-order: bysource


Utilities
=========

.. autofunction:: parse_split_float

.. autofunction:: split_float

.. autofunction:: join_float

.. autofunction:: wav_is_looped

.. autofunction:: atten_to_level
