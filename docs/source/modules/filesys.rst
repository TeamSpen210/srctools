srctools.filesys
----------------

.. automodule:: srctools.filesys
	:synopsis: Provides a unified API for reading from different sources, like Source does.

.. autoexception:: RootEscapeError
   :members:

.. autofunction:: get_filesystem

.. py:data:: CACHE_KEY_INVALID
   :value: -1

   This is returned from :py:meth:`FileSystem.cache_key()` to indicate no key could be computed.

.. autoclass:: FileSystem
   :members:
   :private-members:
   :special-members:

.. autoclass:: File
   :members:
   :private-members:
   :special-members:

========================
Concrete implementations
========================

.. autoclass:: FileSystemChain
   :members:

.. autoclass:: RawFileSystem
   :members:

.. autoclass:: VirtualFileSystem
   :members:

.. autoclass:: ZipFileSystem
   :members:

.. autoclass:: VPKFileSystem
   :members:
