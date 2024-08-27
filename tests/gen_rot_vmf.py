"""Generates VMFs for testing rotation values."""
import os

from srctools import VMF, Vec


NUMS = [-1, 0, 1]
VECS = [
    Vec(x=1),
    Vec(y=1),
    Vec(z=1),
    Vec(x=-1),
    Vec(y=-1),
    Vec(z=-1),
]

os.makedirs('instances/', exist_ok=True)

inst_vmf = VMF()

inst_vmf.create_ent(
    'func_instance_parms',
    origin='0 0 0',
    parm1='$rot_pitch integer 0',
    parm2='$rot_roll integer 0',
    parm3='$rot_yaw integer 0',
)

for pos in VECS:
    inst_vmf.create_ent(
        'info_target',
        origin=pos * 128,
        targetname='offset',
        angles='0 0 0',
        rot_pitch='$rot_pitch',
        rot_yaw='$rot_yaw',
        rot_roll='$rot_roll',
        local_x=pos.x,
        local_y=pos.y,
        local_z=pos.z
    )

main_vmf = VMF()

main_vmf.add_brushes(main_vmf.make_hollow(
    Vec(-256, -256, -256),
    Vec(+256, +256, +256),
))

# We can't be more accurate, we'll blow the ent limit.
for pitch in range(0, 360, 45):
    for yaw in range(0, 360, 45):
        for roll in range(0, 360, 45):
            inst = main_vmf.create_ent(
                'func_instance',
                file='instances/rot_inst.vmf',
                angles=Vec(pitch, yaw, roll),
                origin='0 0 0',
            )
            inst.fixup['$rot_pitch'] = pitch
            inst.fixup['$rot_yaw'] = yaw
            inst.fixup['$rot_roll'] = roll

with open('instances/rot_inst.vmf', 'w', encoding='utf8') as f:
    inst_vmf.export(f)

with open('rot_main.vmf', 'w', encoding='utf8') as f:
    main_vmf.export(f)

print('Done.')
