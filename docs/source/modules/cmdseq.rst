srctools.cmdseq
---------------

.. module:: srctools.cmdseq
    :synopsis: Reads and writes Hammer's .wc expert compile options

:file:`CmdSeq.wc` is Hammer's configuration format for "expert" compiles, storing the various sequences of executables to compile maps with.
This module allows for reading and writing this file format. Valve's is a binary format,
while both Strata Source and Hammer++ have alternate keyvalue formats which are human-readable.

Classes
=========

.. autosrcenum:: SpecialCommand
   :members:

.. autoclass:: Command
   :members:


Binary Format
=============

Valve uses a binary file format, which can be read and written by these two functions.

.. autofunction:: parse

.. autofunction:: write

Keyvalues Format
=======================

Strata Source and Hammer++ both independently switched over to using a keyvalues-based format
instead, which is human-editable. The `parse` function detects and reads this automatically,
but these functions are available to directly parse from keyvalues, and produce the tree to resave:

.. autofunction:: parse_keyvalues

.. autofunction:: build_keyvalues
