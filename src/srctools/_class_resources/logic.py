"""Logic entities. Not much resources here."""
from srctools._class_resources import *

res('logic_achievement')
res('logic_active_autosave')
res('logic_auto')
res('logic_autosave')
res('logic_branch')
res('logic_branch_listener')
res('logic_case')
res('logic_collision_pair')
res('logic_console')
res('logic_compare')
res('logic_coop_manager')
res('logic_datadesc_accessor')
res('logic_format')
res('logic_keyfield')
res('logic_lineto')
res('logic_measure_direction')
res('logic_measure_movement')
res('logic_mirror_movement')
res('logic_modelinfo')
res('logic_multicompare')
res('logic_navigation')
res('logic_parent')
res('logic_playerinfo')


@cls_func
def logic_playmovie(pack: PackList, ent: Entity):
    """Mark the BIK movie as being used, though it can't be packed."""
    pack.pack_file('media/' + ent['MovieFilename'])


res('logic_playerproxy')
res('logic_random_outputs')
res('logic_register_activator')
res('logic_relay')
res('logic_relay_queue')
res('logic_scene_list_manager')
res('logic_script')
res('logic_sequence')
res('logic_skill')
res('logic_timescale')
res('logic_timer')

res('math_counter')
res('math_counter_advanced')
res('math_generate')
res('math_lightpattern')
res('math_mod')
res('math_remap')
res('math_vector')
res('math_colorblend')

