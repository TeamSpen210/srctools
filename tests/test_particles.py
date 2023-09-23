"""Test particle system parsing."""
from pathlib import Path

from srctools.dmx import Attribute, Element
from srctools.particles import FORMAT_NAME, Operator, Particle


def test_parsing(datadir: Path) -> None:
    """Parse a sample particle, and verify the values are correct."""
    with (datadir / 'sample.pcf').open('rb') as f:
        root, fmt_name, fmt_version = Element.parse(f)
    assert fmt_name == 'pcf' == FORMAT_NAME
    assert fmt_version in [1, 2]

    particles = Particle.parse(root, fmt_version)
    assert len(particles) == 1
    part = particles['test_part']
    assert part.name == 'test_part'

    # Test a few options to ensure they're valid.
    assert part.options['material'] == Attribute.string('material', 'editor\\ai_goal_police.vmt')
    assert part.options['normal'] == Attribute.vec3('normal', 0, 0, -1)
    assert part.options['max_particles'] == Attribute.int('max_particles', 52)

    assert len(part.renderers) == 2
    rend_spr: Operator = part.renderers[0]
    assert rend_spr.name == 'sprite_anim'
    assert rend_spr.function == 'render_animated_sprites'
    assert rend_spr.options['animation rate'] == Attribute.float('animation rate', 0.75)
    assert rend_spr.options['orientation_type'] == Attribute.int('orientation_type', 1)

    rend_mdl: Operator = part.renderers[1]
    assert rend_mdl.name == 'mdl_error'
    assert rend_mdl.function == 'Render models'
    assert rend_mdl.options['sequence 0 model'] == Attribute.string('sequence 0 model', 'editor\\cone_helper.mdl')
    assert rend_mdl.options['orient model z to normal'] == Attribute.bool('orient model z to normal', False)
