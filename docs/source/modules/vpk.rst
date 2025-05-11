srctools.vpk
------------

.. module:: srctools.vpk
	:synopsis: Reads and writes VPK archives.

This module reads and writes VPK archives, used to store content for Source games.
These archives are uncompressed, and consist of two sorts of files:

* The "directory" (usually named :samp:`{archive}_dir.vpk`). The directory contains the list of
  all files, and optionally part or whole sections of files.
* One or more data files (named like :samp:`{archive}_032.vpk`).This has no structure, and simply
  consists of file data concatenated together.

Alternatively, a "singular" VPK archive is also possible, where all contents must be stored in the
directory file. Note that removing data from a data file is not possible - changing a file's contents
or deleting it will change the directory, but leave the old data. This does have the advantage that
updating a shipped game only requires downloading new data files for the additional content, but it
means old data stays around.

Opening and closing
===================

VPKs can be opened in three modes, like regular files. Changes to the directory are only applied
once :py:meth:`~VPK.write_dirfile` is called, but writes to data files occur immediately.
The VPK can be used like a context manager to automatically save, if no exception was raised.

.. autosrcenum:: OpenModes
	:members:
	:undoc-members:
	:member-order: bysource

.. autoclass:: VPK

	.. automethod:: load_dirfile

	.. automethod:: write_dirfile


Attributes
==========

	.. autoattribute:: VPK.folder

	.. autoattribute:: VPK.path

	.. autoattribute:: VPK.mode

	.. autoattribute:: VPK.version

	.. autoattribute:: VPK.file_prefix

	.. autoattribute:: VPK.dir_limit

	.. autoattribute:: VPK.footer_data

File Objects
============

.. autoclass:: FileInfo()

	These have no public constructor, and represent a file in a VPK. All attributes should be
	considered readonly - call methods instead.

	.. attribute:: name

	.. autoattribute:: filename

	.. autoattribute:: dir

	.. autoattribute:: ext

	.. autoattribute:: size

	.. autoattribute:: arch_index

	.. autoattribute:: arch_len

	.. autoattribute:: vpk

	.. autoattribute:: offset

	.. autoattribute:: start_data

Locating Files
==============

The structure of a VPK organises files by folder and extension, making iteration efficient.
To locate a specific file, the simplest way is to index the VPK in one of three ways::

	vpk['folders/name.ext']
	vpk['folders', 'name.ext']
	vpk['folders', 'name', 'ext']

If the extension or folders are known, this avoids needing to split the filename.
``in``, and :py:func:`len` checks also work as expected.

If the filename is not known, iterating the VPK gives each file in turn. Alternatively, the following
methods can be used:

.. automethod:: VPK.filenames
	:for: filename

.. automethod:: VPK.folders
	:for: folder

.. automethod:: VPK.fileinfos
	:for: file

Once found, call `~FileInfo.read` to read the contents:

.. automethod:: FileInfo.read

Writing
=======

To write to a VPK, create a new file info entry, then call `~FileInfo.write`:

.. automethod:: VPK.new_file

.. automethod:: FileInfo.write

Alternatively, call these methods to do both at once.

.. automethod:: VPK.add_file

.. automethod:: VPK.add_folder

You can also do :pycode:`del vpk['some/filename']` to remove a file.

Miscellaneous
=============

.. automethod:: VPK.extract_all

.. automethod:: FileInfo.verify

.. automethod:: VPK.verify_all

.. autodata:: VPK_SIG

.. autodata:: DIR_ARCH_INDEX

.. autofunction:: get_arch_filename

.. Don't document script_write(), should be command line only?
