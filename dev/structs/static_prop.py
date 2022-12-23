from structs import *

prop_v4 = Struct(
    origin=type_vector,
    angles=type_angles,
    proptype=type_u16,
    firstLeaf=type_u16,
    leafCount=type_u16,
    solid=type_u8,
    flags_byt=type_u8,
    skin=type_int,
    fademin=type_float,
    fademax=type_float,
    lighting=type_vector,
)

prop_v5 = Struct(base=prop_v4, forcedFadeScale=type_float)
prop_v5_ship = Struct(base=prop_v5, targetname=Fixed(128, 'str'))

prop_v6 = Struct(base=prop_v5, minDX=type_ushort, maxDX=type_ushort)
prop_v6_bloody = Struct(base=prop_v6, targetname=Fixed(128, 'str'))

prop_v7_l4d = Struct(base=prop_v6, rendercolor=type_color32)

prop_v7_zeno_clash = Struct(base=prop_v6, unknown=type_u32)

prop_v8 = Struct(
    base=prop_v5,
    minCPU=type_u8,
    maxCPU=type_u8,
    minGPU=type_u8,
    maxGPU=type_u8,
    rendercolor=type_color32,
)

prop_v9 = Struct(
    base=prop_v8,
    disableX360=type_u8,
    padding=Fixed(3, 'pad'),
)

# Not really relevant?
# prop_v9_d_est = Struct(
#     origin=type_vector,
#     angles=type_angles,
#     proptype=type_u16,
#     firstLeaf=type_u16,
#     leafCount=type_u16,
#     solid=type_u8,
#     flags=type_u8,
#     _1=Fixed(4, 'unk'),
#     skin=type_int,
#     fademin=type_float,
#     fademax=type_float,
#     _2=Fixed(12, 'unk'),
#     forcedFade=type_float,
#     minCPU=type_u8,
#     maxCPU=type_u8,
#     minGPU=type_u8,
#     maxGPU=type_u8,
#     _3=type_u8,
#     rendercolor=type_color32,
#     _4=Fixed(3, 'unk'),
# )

prop_v10 = Struct(
    base=prop_v6,
    flags=type_uint,  # This replaces the single byte!
    lightmap=Struct(x=type_ushort, y=type_ushort),
)

prop_v10_csgo = Struct(
    base=prop_v9,
    # unknown=Fixed(4, 'unk')
    flags2=type_uint,
)

prop_v11_lite = Struct(
    base=prop_v10,
    rendercolor=type_color32,
)

prop_v11 = Struct(
    base=prop_v11_lite,
    flags2=type_uint,
)

prop_v11_csgo = Struct(
    base=prop_v10_csgo,
    scale=type_float,
)

display_all(globals())
