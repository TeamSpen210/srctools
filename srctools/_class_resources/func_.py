"""func_ entities."""
from srctools._class_resources import *

@cls_func
def func_breakable_surf(pack: PackList, ent: Entity):
    """Additional materials required for func_breakable_surf."""
    pack.pack_file('models/brokenglass_piece.mdl', FileType.MODEL)

    surf_type = conv_int(ent['surfacetype'])

    if surf_type == 1:  # Tile
        mat_type = 'tile'
    elif surf_type == 0:  # Glass
        mat_type = 'glass'
        pack.pack_file('materials/models/brokenglass/glassbroken_solid.vmt', FileType.MATERIAL)
    else:
        # Unknown
        return

    for num in '123':
        for letter in 'abcd':
            pack.pack_file(
                'materials/models/broken{0}/'
                '{0}broken_0{1}{2}.vmt'.format(mat_type, num, letter),
                FileType.MATERIAL,
            )

res('func_dust',
    'materials/particle/sparkles.vmt',
    )
res('func_movelinear')
res('func_portal_bumper')
res('func_portal_detector')
res('func_portal_orientation')
res('func_portalled')

res('func_tankchange', sound('FuncTrackChange.Blocking'))
res('func_recharge',
    sound('SuitRecharge.Deny'),
    sound('SuitRecharge.Start'),
    sound('SuitRecharge.ChargingLoop'),
    )
