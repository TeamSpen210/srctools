"""Test parsing and exporting entity lumps."""
from pathlib import Path

import unittest.mock

from srctools.bsp import BSP
from srctools.vmf import Entity


def make_dummy() -> BSP:
    """Create a totally empty BSP object, so lump functions can be called."""
    with unittest.mock.patch('srctools.bsp.BSP.read'):
        return BSP('<dummy>')


def test_read(datadir: Path) -> None:
    """Test reading entity lumps."""
    bsp = make_dummy()
    # Note: \n endings, null byte at the end.
    with open(datadir / 'ent_lump.lmp', 'rb') as f:
        # vmf: VMF = BSP.ents._read(bsp, f.read())
        vmf = bsp._lmp_read_ents(f.read())
    assert vmf.map_ver == 3928

    assert vmf.spawn['world_mins'] == '-2048 -2048 0'
    assert vmf.spawn['world_maxs'] == '2048 2048 256'

    assert len(vmf.entities) == 2
    relay: Entity
    target: Entity
    [relay, target] = vmf.entities

    assert relay['classname'] == 'logic_relay'
    assert target['targetname'] == 'particle_targ_38'

    # Check these are initialised
    assert relay in vmf.by_class['logic_relay']
    assert target in vmf.by_target['particle_targ_38']

    assert len(relay.outputs) == 2, relay.outputs
    # Test both old comma format, and new ESC format. Parameter has exactly 4 commas, to test that
    # the priority is the correct order.
    assert relay.outputs[0].output == 'ontrigger'
    assert relay.outputs[0].target == 'the_target'
    assert relay.outputs[0].input == 'Skin'
    assert relay.outputs[0].params == '2'
    assert relay.outputs[0].delay == 0.5
    assert relay.outputs[0].comma_sep is True
    assert relay.outputs[0].only_once is False

    assert relay.outputs[1].output == 'OnSpawn'
    assert relay.outputs[1].target == '!player'
    assert relay.outputs[1].input == 'RunScriptCode'
    assert relay.outputs[1].params == 'Setup(45,32,28,96,false)'
    assert relay.outputs[1].delay == 2.5
    assert relay.outputs[1].comma_sep is False
    assert relay.outputs[1].only_once is True

    assert target['angles'] == '-189.8792 27.2189 1.91'
    assert target['multiline_key'] == 'This is a key which happens\nto have multiple lines.'
