srctools.types
------------------------

.. module:: srctools.types
	:synopsis: Common types and concrete classes.

This module contains various shared type definitions used by other modules.

==============
File Protocols
==============

Most operations acting on file objects only need a subset of the possible methods.
These protocols indicate the current expected API.

.. py:class:: FileR[AnyStr]
	:abstract:

	A readable file. Context manager support is expected.

	.. py:method:: read(self, data: AnyStr, /) -> bytes

.. py:class:: FileWBinary
	:abstract:

	A writable binary file.

	.. py:method:: write(self, data: bytes | bytearray, /) -> ...

.. py:class:: FileWText
	:abstract:

	A writable text file.

	.. py:method:: write(self, data: str, /) -> ...

.. py:class:: FileSeek
	:abstract:

	A seekable file.

	.. py:method:: seek(self, pos: int, whence: 0 | 1 | 2 = 0, /) -> ...

		The whence values correspond to `os.SEEK_SET`, `~os.SEEK_CUR` and `~os.SEEK_END`

	.. py:method:: tell(self, /) -> int

These protocols then combine the above with seekability:

.. py:class:: FileRSeek[AnyStr](FileR[AnyStr], FileSeek)
	:abstract:

	Combination of `FileR` and `FileSeek`.

.. py:class:: FileWBinarySeek(FileWBinary, FileSeek)
	:abstract:

	Combination of `FileWBinary` and `FileSeek`.

.. py:class:: FileWTextSeek(FileWText, FileSeek)
	:abstract:

	Combination of `FileWText` and `FileSeek`.
