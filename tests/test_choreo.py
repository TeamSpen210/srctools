"""Test choreographed scene parsing."""
from pathlib import Path
import io

from pytest_regressions.file_regression import FileRegressionFixture
import dirty_equals

from srctools.choreo import (
    CaptionType, Curve, CurveEdge, CurveType, Event, EventFlags, EventType, GestureEvent,
    Interpolation, LoopEvent, Scene, SpeakEvent, Tag,
)
from srctools.tokenizer import Tokenizer


def test_parse_vcd(datadir: Path) -> None:
    """Test parsing a sample VCD file."""
    with open(datadir / 'sample.vcd', encoding='utf8') as f:
        scene = Scene.parse_text(Tokenizer(f))
    assert scene.fps == 60
    assert scene.scale_settings == {
        'CChoreoView': '100',
        'ExpressionTool': '100',
        'GestureTool': '100',
        'RampTool': '100',
        'SceneRampTool': '100',
    }
    curve_type = CurveType(Interpolation.DEFAULT, Interpolation.DEFAULT)
    curve_edge = CurveEdge(active=False, zero_pos=0.0, curve_type=curve_type)

    assert len(scene.events) == 3
    assert scene.events[0] == dirty_equals.IsInstance(LoopEvent) & dirty_equals.HasAttributes(
        absolute_playback_tags=[],
        absolute_shifted_tags=[],
        default_curve_type=curve_type,
        dist_to_targ=0,
        flags=EventFlags.Active,
        flex_anim_tracks=[],
        loop_count=8,
        name='some loop',
        parameters=(dirty_equals.IsStr(regex=r'0(\.0*)?'), '', ''),
        pitch=0,
        ramp=Curve(ramp=[], left=curve_edge, right=curve_edge),
        relative_tags=[],
        start_time=dirty_equals.IsFloat(approx=4.40667),
        end_time=-1.0,
        tag_name=None,
        tag_wav_name=None,
        timing_tags=[],
        yaw=0,
    )
    assert scene.events[1] == dirty_equals.IsInstance(Event) & dirty_equals.HasAttributes(
        absolute_playback_tags=[],
        absolute_shifted_tags=[],
        default_curve_type=curve_type,
        dist_to_targ=0,
        flags=EventFlags.Active,
        flex_anim_tracks=[],
        name='puase',
        parameters=('noaction', '', ''),
        pitch=0,
        ramp=Curve(ramp=[], left=curve_edge, right=curve_edge),
        relative_tags=[],
        start_time=dirty_equals.IsFloat(approx=3.90667),
        end_time=-1.0,
        tag_name=None,
        tag_wav_name=None,
        timing_tags=[],
        type=EventType.Section,
        yaw=0,
    )
    assert scene.events[2] == dirty_equals.IsInstance(Event) & dirty_equals.HasAttributes(
        absolute_playback_tags=[],
        absolute_shifted_tags=[],
        default_curve_type=curve_type,
        dist_to_targ=0,
        end_time=-1.0,
        flags=EventFlags.Active,
        flex_anim_tracks=[],
        name='a_fire',
        parameters=('noaction', '', ''),
        pitch=0,
        ramp=Curve(ramp=[], left=curve_edge, right=curve_edge),
        relative_tags=[],
        start_time=dirty_equals.IsFloat(approx=-0.746667),
        tag_name=None,
        tag_wav_name=None,
        timing_tags=[],
        type=EventType.StopPoint,
        yaw=0,
    )

    [actor] = scene.actors
    assert actor.name == 'an_Actor'
    assert actor.active
    [channel] = actor.channels
    assert channel.name == 'first_channel'
    assert len(channel.events) == 6
    assert channel.events[0] == dirty_equals.IsInstance(SpeakEvent) & dirty_equals.HasAttributes(
            name='npc_gman.welcome',
            caption_type=CaptionType.Master,
            cc_token='',
            flags=EventFlags.FixedLength | EventFlags.Active,
            parameters=('npc_gman.welcome', '', ''),
            start_time=dirty_equals.IsFloat(approx=-0.213333),
            ramp=Curve(ramp=[], left=curve_edge, right=curve_edge),
            suppress_caption_attenuation=False,
    )
    assert channel.events[1] == dirty_equals.IsInstance(Event) & dirty_equals.HasAttributes(
        name='looking',
        flags=EventFlags.Active,
        parameters=('!enemy', '', ''),
        ramp=Curve(ramp=[], left=curve_edge, right=curve_edge),
        start_time=dirty_equals.IsFloat(approx=0.806667),
        end_time=dirty_equals.IsFloat(approx=1.806667),
        type=EventType.LookAt,
        pitch=61,
        yaw=-47,
    )
    assert channel.events[2] == dirty_equals.IsInstance(Event) & dirty_equals.HasAttributes(
        name='fire_trig',
        flags=EventFlags.Active,
        parameters=('7', '', ''),
        ramp=Curve(ramp=[], left=curve_edge, right=curve_edge),
        start_time=dirty_equals.IsFloat(approx=2.213333),
        end_time=-1.0,
        type=EventType.FireTrigger,
    )
    assert channel.events[3] == dirty_equals.IsInstance(GestureEvent) & dirty_equals.HasAttributes(
        name='a_gesture',
        flags=EventFlags.Active,
        parameters=('circle', '', ''),
        ramp=Curve(ramp=[], left=curve_edge, right=curve_edge),
        start_time=dirty_equals.IsFloat(approx=2.26),
        end_time=dirty_equals.IsFloat(approx=4.16),
    )
    assert channel.events[4] == dirty_equals.IsInstance(Event) & dirty_equals.HasAttributes(
        name='mover',
        flags=EventFlags.Active | EventFlags.ResumeCondition | EventFlags.ForceShortMovement,
        parameters=('!friend', 'Run', '!target2'),
        ramp=Curve(ramp=[], left=curve_edge, right=curve_edge),
        start_time=dirty_equals.IsFloat(approx=3.2),
        end_time=dirty_equals.IsFloat(approx=3.453333),
        dist_to_targ=dirty_equals.IsFloat(approx=59.0),
        type=EventType.MoveTo,
    )
    assert channel.events[5] == dirty_equals.IsInstance(SpeakEvent) & dirty_equals.HasAttributes(
        name='barn.ditchcar',
        flags=EventFlags.FixedLength | EventFlags.Active,
        parameters=('barn.ditchcar', '0.8', ''),
        start_time=dirty_equals.IsFloat(approx=0.78),
        end_time=dirty_equals.IsFloat(approx=3.322585),
        ramp=Curve(ramp=[], left=curve_edge, right=curve_edge),
        relative_tags=dirty_equals.IsList(
            dirty_equals.IsInstance(Tag) & dirty_equals.HasAttributes(
                name='a_tag',
                value=dirty_equals.IsFloat(approx=0.138743),
            ),
        ),
        caption_type=CaptionType.Master,
        cc_token='',
        suppress_caption_attenuation=True,
    )


def test_save_text(datadir: Path, file_regression: FileRegressionFixture) -> None:
    """Test resaving a text VCD file."""
    with open(datadir / 'sample.vcd', encoding='utf8') as f:
        scene = Scene.parse_text(Tokenizer(f))
    buf = io.StringIO()
    scene.export_text(buf)
    file_regression.check(buf.getvalue(), extension='.vcd')


def test_save_binary(datadir: Path, file_regression: FileRegressionFixture) -> None:
    """Test resaving a binary VCD file."""
    import json
    with open(datadir / 'sample.vcd', encoding='utf8') as f:
        scene = Scene.parse_text(Tokenizer(f))

    pool = []

    def pool_func(value: str) -> int:
        """Simulate the string pool."""
        try:
            return pool.index(value)
        except ValueError:
            pos = len(pool)
            pool.append(value)
            return pos

    result = scene.export_binary(pool_func)
    # Add the pool to the start, so we check both.
    result = json.dumps(pool).encode('utf8') + result
    file_regression.check(result, extension='.bvcd', binary=True)
