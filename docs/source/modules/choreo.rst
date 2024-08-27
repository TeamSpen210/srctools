srctools.choreo
------------------------

.. module:: srctools.choreo
    :synopsis: Reads and writes choreographed scenes, in binary, text and ``scenes.image`` format.

Text format is the `.vcd` format saved by Faceposer, and includes a number of additional attributes to save the editor configuration.
Binary format is contained in the ``scenes.image`` file, only storing the data required to actually play the scene.

Note that scenes from ``scenes.image`` are not named - they are looked up by a hash of the ``VCD`` filename, so the filenames must already be known to identify an entry.

===========
Basic usage
===========

An example for reading and writing VCDs and ``scenes.image``::

    from srctools.choreo import *
    from srctools.tokenizer import Tokenizer

    with open('some_scene.vcd') as file:
        scene = Scene.parse_text(Tokenizer(file))
    with open('some_scene_copy.vcd', 'w') as file:
        scene.export_text(file)

    with open('scenes.image', 'rb') as file:
        image = parse_scenes_image(file)
    with open('new_scenes.image', 'wb') as file:
        save_scenes_image_sync(file, image)

=================
Scenes.image
=================

Entry instances represent the additional metadata contained in an image file.

.. autoclass:: Entry
    :members:
    :undoc-members:

.. autofunction:: checksum_filename

.. autofunction:: parse_scenes_image

.. autofunction:: save_scenes_image_sync



==========
Scene Tree
==========

Scenes consist of many :py:class:`Events <Event>`, organised into :py:class:`Channels <Channel>` which themselves are directed at specific :py:class:`Actor` NPCs.

.. autoclass:: Scene
    :members:
    :undoc-members:

.. autoclass:: Actor
    :members:
    :undoc-members:

.. autoclass:: Channel
    :members:
    :undoc-members:


======
Events
======

There are many :py:class:`types <EventType>` of events. Most share common configuration and use the base :py:class:`Event` class, but some have additional options and require a specific subclass.

.. autosrcenum:: EventType()
    :members:
    :undoc-members:
	:member-order: bysource

.. autosrcenum:: EventFlags()
    :members:
    :undoc-members:
	:member-order: bysource

.. autoclass:: Event
    :members:
    :undoc-members:

.. autoclass:: GestureEvent
    :members:
    :undoc-members:

.. autoclass:: LoopEvent
    :members:
    :undoc-members:

.. autoclass:: SpeakEvent
    :members:
    :undoc-members:

=====================
Miscellaneous Classes
=====================

.. autoclass:: Interpolation
    :members:
    :undoc-members:

.. autoclass:: CurveType
    :members:
    :undoc-members:

.. autosrcenum:: CaptionType()
	:members:
	:member-order: bysource

.. autoclass:: ExpressionSample
    :members:
    :undoc-members:

.. autoclass:: TimingTag
    :members:
    :undoc-members:

.. autoclass:: AbsoluteTag
    :members:
    :undoc-members:

.. autoclass:: CurveEdge
    :members:
    :undoc-members:

.. autoclass:: Curve
    :members:
    :undoc-members:

.. autoclass:: FlexAnimTrack
    :members:
    :undoc-members:
