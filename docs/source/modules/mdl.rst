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

.. py:data:: MDL_EXTS
    :type: ~collections.abc.Sequence[str]
    :value: ['.mdl', '.vvd', '.dx90.vtx', '.phy', ...]

    A sequence of the file extensions used for the various model files.

.. py:data:: MDL_EXTS_EXTRA
    :type: ~collections.abc.Sequence[str]
    :value: ['.vvd', '.dx90.vtx', '.phy', ...]

    A sequence of the file extensions used for the various model files, excluding ``.mdl``.

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
