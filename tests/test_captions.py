from srctools.captions import Message, TAG_BOLD, TAG_ITALIC


def test_caption_parse_text_basic() -> None:
    """Parse basic tags."""
    msg = Message.parse_text('Basic text.')
    assert msg.message == ['Basic text.']

    msg = Message.parse_text('normal<b>bold<i>italicbold<b><i>normal')
    assert msg.message == [
        'normal', TAG_BOLD, 'bold', TAG_ITALIC,
        'italicbold', TAG_BOLD, TAG_ITALIC, 'normal',
    ]
