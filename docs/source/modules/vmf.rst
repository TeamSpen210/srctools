srctools.vmf
------------

.. automodule:: srctools.vmf
    :synopsis: Reads and writes VMF maps.


Core Functionality
==================

.. autoclass:: VMF

    .. automethod:: VMF.parse

    .. automethod:: VMF.export

    .. autoattribute:: is_prefab
    .. autoattribute:: cordon_enabled
    .. autoattribute:: map_ver
    .. autoattribute:: format_ver
    .. autoattribute:: hammer_ver
    .. autoattribute:: hammer_build
    .. autoattribute:: show_grid
    .. autoattribute:: show_3d_grid
    .. autoattribute:: snap_grid
    .. autoattribute:: show_logic_grid
    .. autoattribute:: grid_spacing

Common behaviours
=================

Automatic conversion
++++++++++++++++++++
VMFs store string values in several locations, which are commonly treated as other data types,
parsed as necessary. To make this easier to deal with, several data types are automatically converted
when passed to relevant functions.

.. py:type:: ValidKVs

    Parameters with this type automatically convert values to a string.

.. autofunction:: conv_kv

Locations with automatic conversion include entity values, instance fixups, and output parameters.

ID management
+++++++++++++
The VMF class records IDs used by brushes, faces, entities, groups, visgroups and AI nodes. When
creating new objects, they will automatically get assigned a new ID. IDs are 'owned' while the
relevant object is alive - removing them from the VMF will still keep it reserved.

.. autoattribute:: VMF.solid_id
.. autoattribute:: VMF.face_id
.. autoattribute:: VMF.ent_id
.. autoattribute:: VMF.group_id
.. autoattribute:: VMF.vis_id
.. autoattribute:: VMF.node_id

.. autoclass:: IDMan()

    .. autoattribute:: allow_duplicates
    .. automethod:: get_id

.. automethod:: VMF.allow_duplicate_ids(self)
    :with:

.. autoattribute:: Solid.id
.. autoattribute:: Side.id
.. autoattribute:: VisGroup.id
.. autoattribute:: VisGroup.id
.. autoattribute:: Entity.id
.. autoattribute:: EntityGroup.id

Brushes
=======

.. autoattribute:: VMF.brushes
.. automethod:: VMF.add_brush
.. automethod:: VMF.add_brushes
.. automethod:: VMF.remove_brush

.. autoclass:: Solid

    .. autoattribute:: editor_color
    .. autoattribute:: group_id
    .. autoattribute:: hidden
    .. autoattribute:: sides
    .. autoattribute:: vis_auto_shown
    .. autoattribute:: vis_shown
    .. autoattribute:: visgroup_ids
    .. autoattribute:: is_cordon

    .. automethod:: parse
    .. automethod:: export
    .. automethod:: copy
    .. automethod:: remove
    .. automethod:: get_bbox
    .. automethod:: get_origin
    .. automethod:: translate
    .. automethod:: localise
    .. automethod:: point_inside

.. todo srctools.vmf.Solid.map


.. autoclass:: Side

    .. autoattribute:: planes
    .. autoattribute:: mat
    .. autoattribute:: smooth
    .. autoattribute:: ham_rot
    .. autoattribute:: lightmap
    .. autoattribute:: uaxis
    .. autoattribute:: vaxis

    .. autoproperty:: scale
    .. autoproperty:: offset

    .. automethod:: parse
    .. automethod:: export
    .. automethod:: from_plane
    .. automethod:: copy
    .. automethod:: get_bbox
    .. automethod:: get_origin
    .. automethod:: translate
    .. automethod:: localise
    .. automethod:: normal

    .. todo: srctools.vmf.Side.map

.. autoclass:: UVAxis

    .. autoattribute:: offset
    .. autoattribute:: scale
    .. autoattribute:: x
    .. autoattribute:: y
    .. autoattribute:: z

    .. automethod:: parse
    .. automethod:: copy

    .. automethod:: vec
    .. automethod:: rotate
    .. automethod:: localise

Creation
++++++++

.. automethod:: VMF.make_prism

.. automethod:: VMF.make_hollow

.. autoclass:: PrismFace

    .. autoattribute:: north
    .. autoattribute:: south
    .. autoattribute:: east
    .. autoattribute:: west
    .. autoattribute:: top
    .. autoattribute:: bottom
    .. autoattribute:: solid

Displacements
+++++++++++++

Brush faces converted to displacements add a number of additional properties to the face.

.. autoattribute:: Side.disp_power
.. autoproperty:: Side.is_disp
.. autoattribute:: Side.disp_size
.. autoattribute:: Side.disp_allowed_vert
.. autoattribute:: Side.disp_elevation
.. autoattribute:: Side.disp_flags
.. autoattribute:: Side.disp_pos

.. automethod:: Side.disp_get_tri_verts

.. autoclass:: Vec4
    :members:
    :undoc-members:

.. autosrcenum:: DispFlag
    :members:
    :undoc-members:
    :member-order: bysource

.. autosrcenum:: TriangleTag
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: DispVertex
    :members:
    :undoc-members:

Entities
========

.. autoattribute:: VMF.entities
.. autoattribute:: VMF.spawn
.. automethod:: VMF.add_ent
.. automethod:: VMF.remove_ent
.. automethod:: VMF.add_ents
.. automethod:: VMF.create_ent

.. autoclass:: Entity

    .. automethod:: parse
    .. automethod:: export
    .. automethod:: copy
    .. automethod:: remove
    .. automethod:: make_unique

    .. autoattribute:: hidden
    .. autoattribute:: visgroup_ids
    .. autoattribute:: vis_shown
    .. autoattribute:: vis_auto_shown
    .. autoattribute:: groups
    .. autoattribute:: editor_color
    .. autoattribute:: logical_pos
    .. autoattribute:: comments


    .. autoattribute:: solids
    .. automethod:: is_brush
    .. automethod:: get_bbox
    .. automethod:: get_origin
    .. automethod:: sides
        :for: face

.. include? clear_keys, get_key
.. debug: keys

Outputs
+++++++

.. autoattribute:: Entity.outputs
.. automethod:: Entity.add_out
.. automethod:: Entity.output_targets
.. automethod:: VMF.iter_inputs
    :for: output

.. autoclass:: Output

    .. autoattribute:: input
    .. autoattribute:: target
    .. autoattribute:: output
    .. autoattribute:: params
    .. autoattribute:: delay
    .. autoattribute:: times
    .. autoproperty:: only_once
    .. autoattribute:: comma_sep
    .. autoattribute:: inst_in
    .. autoattribute:: inst_out

    ---------------

    .. automethod:: parse
    .. automethod:: export
    .. automethod:: as_keyvalue

    .. automethod:: combine
    .. automethod:: parse_name
    .. automethod:: exp_out
    .. automethod:: exp_in
    .. automethod:: copy
    .. automethod:: gen_addoutput

.. data:: SEP
.. autoattribute:: Output.SEP

Instances
+++++++++

``func_instance`` entities can have fixup variables defined on them, which are stored in a number
of :samp:`replace{01}` keys. To avoid having to deal with the details, these are automatically parsed
and stored under a `fixup <Entity.fixup>` attribute.

.. autoattribute:: Entity.fixup

.. autoclass:: EntityFixup

    .. automethod:: get
    .. automethod:: copy_values
    .. automethod:: clear
    .. automethod:: setdefault
    .. automethod:: items
    .. automethod:: values
    .. automethod:: export
    .. automethod:: substitute
    .. automethod:: int
    .. automethod:: float
    .. automethod:: bool
    .. automethod:: vec

.. py:type:: FixupValue

    Opaque value storing a key/value pair plus the index.
    Can be passed to fixups to attempt to preserve the index.

Finding Entities
++++++++++++++++

Several features are provided to help locate entities in a similar way to the game.

.. attribute:: VMF.by_class
    :type: MutableMapping[str, AbstractSet[str]]

.. attribute:: VMF.target
    :type: MutableMapping[str, AbstractSet[str]]

.. automethod:: VMF.search
    :for: entity

.. automethod:: VMF.iter_ents
    :for: entity

.. automethod:: VMF.iter_ents_tags
    :for: entity

Cordons
=======

.. autoattribute:: VMF.cordons

.. autoclass:: Cordon

    .. automethod:: parse
    .. automethod:: export
    .. automethod:: copy
    .. automethod:: remove


Strata Source Extensions
========================

Strata Source adds a few additional values to preserve vertices and store settings.

.. autoattribute:: Side.strata_points

.. autoattribute:: VMF.strata_viewports

.. autoclass:: Strata2DViewport
    :members:
    :undoc-members:

.. autoclass:: Strata3DViewport
    :members:
    :undoc-members:

.. autoattribute:: VMF.strata_instance_vis

.. autosrcenum:: StrataInstanceVisibility
    :members:
    :undoc-members:
    :member-order: bysource

.. extra
    srctools.vmf.CURRENT_HAMMER_BUILD
    srctools.vmf.CURRENT_HAMMER_VERSION
    srctools.vmf.Axis
    srctools.vmf.overlay_bounds
    srctools.vmf.make_overlay
    srctools.vmf.localise_overlay
    srctools.vmf.VMF.create_visgroup
    srctools.vmf.VMF.iter_wbrushes
    srctools.vmf.VMF.iter_wfaces
    srctools.vmf.VMF.cameras
    srctools.vmf.VMF.vis_tree
    srctools.vmf.VMF.groups
    srctools.vmf.VMF.active_cam
    srctools.vmf.VMF.quickhide_count
    srctools.vmf.Camera
    srctools.vmf.Camera.targ_ent
    srctools.vmf.Camera.is_active
    srctools.vmf.Camera.set_active
    srctools.vmf.Camera.set_inactive_all
    srctools.vmf.Camera.parse
    srctools.vmf.Camera.copy
    srctools.vmf.Camera.remove
    srctools.vmf.Camera.export
    srctools.vmf.Camera.pos
    srctools.vmf.Camera.target
    srctools.vmf.Camera.map
    srctools.vmf.EntityGroup
    srctools.vmf.EntityGroup.parse
    srctools.vmf.EntityGroup.copy
    srctools.vmf.EntityGroup.export
    srctools.vmf.EntityGroup.auto_shown
    srctools.vmf.EntityGroup.color
    srctools.vmf.EntityGroup.shown
    srctools.vmf.EntityGroup.vmf
    srctools.vmf.VisGroup
    srctools.vmf.VisGroup.parse
    srctools.vmf.VisGroup.export
    srctools.vmf.VisGroup.set_visible
    srctools.vmf.VisGroup.child_ents
    srctools.vmf.VisGroup.child_solids
    srctools.vmf.VisGroup.copy
    srctools.vmf.VisGroup.child_groups
    srctools.vmf.VisGroup.color
    srctools.vmf.VisGroup.name
    srctools.vmf.VisGroup.vmf
