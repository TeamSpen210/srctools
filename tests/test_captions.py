import pytest

from srctools.captions import ColourTag, DelayTag, Message, PlayerColourTag, TAG_BOLD, TAG_ITALIC


def test_parse_text_basic() -> None:
    """Parse basic tags."""
    msg = Message.parse_text('Basic text.')
    assert not msg.is_sfx
    assert msg.message == ['Basic text.']

    msg = Message.parse_text('normal<b>bold<i>italicbold<b><i>normal')
    assert not msg.is_sfx
    assert msg.message == [
        'normal', TAG_BOLD, 'bold', TAG_ITALIC,
        'italicbold', TAG_BOLD, TAG_ITALIC, 'normal',
    ]

    msg = Message.parse_text('<sfx>This is a sound effect<delay:4.75>Another.')
    assert msg.is_sfx
    assert msg.message == [
        'This is a sound effect',
        DelayTag(4.75),
        'Another.',
    ]

    msg = Message.parse_text('Non-repeating<norepeat:38>')
    assert not msg.is_sfx
    assert msg.no_repeat == 38
    assert msg.message == ['Non-repeating']


def test_parse_text_colour() -> None:
    """Test parsing colours."""
    msg = Message.parse_text('A<clr:10,255,10>blue line.')
    assert msg.message == [
        'A', ColourTag(red=10, green=255, blue=10), 'blue line.'
    ]
    msg = Message.parse_text('A<playerclr:82,10,6:12,34,56>multicolour line.')
    assert msg.message == [
        'A', PlayerColourTag(
            player_red=82, player_green=10, player_blue=6,
            red=12, green=34, blue=56,
        ), 'multicolour line.'
    ]


def test_invalid_text_basic() -> None:
    """Test invalid parsing of basic tags."""
    with pytest.raises(ValueError, match=r'^Unknown tag <invalid>!$'):
        Message.parse_text('An <invalid> tag')
    with pytest.raises(ValueError, match=r'^Unknown tag <arg>!$'):
        Message.parse_text('An <arg:3,48> tag')
    with pytest.raises(ValueError, match=r'^Tag <b> got parameter "ahh45", but accepts none!$'):
        Message.parse_text('Extra <b:ahh45> args')
    with pytest.raises(ValueError, match=r'^Invalid <delay> parameter "hello", must be a number!$'):
        Message.parse_text('Some<delay:hello>bad')
    with pytest.raises(ValueError, match=r'^Invalid <norepeat> parameter " ", must be a number!$'):
        Message.parse_text('<norepeat: >bad')


def test_invalid_text_colour() -> None:
    """Test invalid parsing for <clr>."""
    with pytest.raises(ValueError, match=r'^<clr> tag expected 3 args, got 1 args$'):
        Message.parse_text('<clr:4>')
    with pytest.raises(
        ValueError,
        match=r'^Invalid color value "hi", must be a number within the range 0-255!$'
    ):
        Message.parse_text('<clr:hi,2,15>')

    with pytest.raises(
        ValueError,
        match=r'^Invalid color value "-15", must be within the range 0-255!$',
    ):
        Message.parse_text('<clr:4,8,-15>')

    with pytest.raises(
        ValueError,
        match=r'^Invalid color value "256", must be within the range 0-255!$',
    ):
        Message.parse_text('<clr:4,256,15>')


def test_invalid_text_player_colour() -> None:
    """Test invalid parsing for <playerclr>."""
    with pytest.raises(
        ValueError,
        match=r'^<playerclr> tag expected "R,G,B:R,G,B" colour pairs, got',
    ):
        Message.parse_text('<playerclr:>')
    with pytest.raises(
        ValueError,
        match=r'^<playerclr> tag expected "R,G,B:R,G,B" colour pairs, got',
    ):
        Message.parse_text('<playerclr:1,2,3:4,5,6:7,8,9>')
    with pytest.raises(
        ValueError,
        match=r'^<playerclr> tag expected "R,G,B:R,G,B" colour pairs, got',
    ):
        Message.parse_text('<playerclr:1,3:4,5,6>')
    with pytest.raises(
        ValueError,
        match=r'^<playerclr> tag expected "R,G,B:R,G,B" colour pairs, got',
    ):
        Message.parse_text('<playerclr:1,2,3:4,5,6,7>')
    with pytest.raises(
        ValueError,
        match=r'^Invalid color value "256", must be within the range 0-255!$',
    ):
        Message.parse_text('<playerclr:4,256,15:1,2,3>')
