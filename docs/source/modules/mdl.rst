srctools.mdl
------------

.. module:: srctools.mdl
	:synopsis: Parses metadata in Source MDL models.

.. autoclass:: Model
   :members:
   :private-members:
   :special-members:

Constants
=========

.. :py:data:: MDL_EXTS
   :value: ['.mdl', '.phy', ...]

   A sequence of the file extensions used for the various model files.

.. autosrcenum:: AnimEventTypes()
	:undoc-members:
	:members:
	:member-order: bysource

.. py:data:: CL
   :value: AnimEventTypes.CLIENT

   Alias for the CLIENT event type.

.. py:data:: SV
   :value: AnimEventTypes.SERVER

   Alias for the Server event type.

.. autosrcenum:: AnimEvents()
	:members:
	:undoc-members:
	:member-order: bysource

.. py:data:: ANIM_EVENT_BY_INDEX
   :type: dict[int, AnimEvents]

   Maps ordinal indexes to :py:class:`AnimEvents`.

.. py:data:: ANIM_EVENT_BY_NAME
   :type: dict[str, AnimEvents]

   Maps string names to :py:class:`AnimEvents`.

Components
==========

.. autoclass:: IncludedMDL
   :members:
   :undoc-members:

.. autoclass:: SeqEvent
   :members:
   :undoc-members:

.. autoclass:: Sequence
   :members:
   :undoc-members:
