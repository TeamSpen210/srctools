srctools.packlist
-----------------

.. py:module:: srctools.packlist
    :synopsis: Analyses files to determine additional dependencies

The packlist module collects a list of resources, then analyses them further to discover their dependencies.
As the name implies this was originally intended for packing into BSPs, but it can be used for a variety of purposes.


PackList
========

.. autoclass:: PackFile
	:members:
	:undoc-members:

.. autoclass:: PackList
	:members:
	:undoc-members:


"Manifest" Files
================

.. autosrcenum:: FileMode
	:members:

.. autoclass:: ManifestedFiles
	:members:
	:undoc-members:

Functions
=========

.. autofunction:: unify_path
.. autofunction:: strip_extension
