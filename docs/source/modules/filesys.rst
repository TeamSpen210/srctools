srctools.filesys
----------------

.. automodule:: srctools.filesys
	:synopsis: Provides a unified API for reading from different sources, like Source does.

.. autofunction:: get_filesystem

.. autoclass:: FileSystem

	.. autoattribute:: path

	.. automethod:: walk_folder
		:for: file

	.. automethod:: open_bin

	.. automethod:: open_str

	.. automethod:: read_kv1

	.. automethod:: __eq__

	.. automethod:: __iter__

	.. automethod:: __getitem__

	.. automethod:: __contains__

.. autoclass:: File()

	.. autoattribute:: path

	.. autoattribute:: sys

	.. automethod:: open_bin

	.. automethod:: open_str

	.. automethod:: cache_key

	.. automethod:: __fspath__

.. py:data:: CACHE_KEY_INVALID
   :value: -1

   This is returned from :py:meth:`File.cache_key` to indicate no key could be computed.

.. autoexception:: RootEscapeError
   :members:

========================
Concrete implementations
========================

The specific implementations of base ``FileSystem`` methods have been omitted.

.. autoclass:: FileSystemChain

	.. autoattribute:: systems

	.. automethod:: get_system

	.. automethod:: add_sys

	.. automethod:: walk_folder_repeat
		:for: file

.. autoclass:: RawFileSystem

	.. autoattribute:: constrain_path

.. autoclass:: VirtualFileSystem

	.. autoattribute:: bytes_encoding

	.. py:attribute:: path
		:value: "<virtual>"

		Always constant.


.. autoclass:: ZipFileSystem

	.. autoattribute:: zip

.. autoclass:: VPKFileSystem

	.. autoattribute:: vpk

===============================
Implementing Custom FileSystems
===============================
Custom filesystems can be implemented by subclassing :py:class`FileSystem`. File objects store a
"data" value, which is private to the filesystem implementation, and can hold things like a
file-info object. Specify it as the `_FileDataT` typevar.
These are the methods that should be overridden:

* :py:meth:`FileSystem.walk_folder`
* :py:meth:`FileSystem.open_str`
* :py:meth:`FileSystem.open_bin`

.. automethod:: FileSystem._file_exists
	:abstractmethod:

.. automethod:: FileSystem._get_file
	:abstractmethod:

.. automethod:: FileSystem._get_cache_key
	:abstractmethod:

These methods can be called by subclasses:

.. automethod:: File.__init__

.. automethod:: FileSystem._get_data
