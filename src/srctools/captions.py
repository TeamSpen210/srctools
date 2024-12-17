"""Parses caption/subtitle files."""
from enum import Enum
from typing import List, Optional, Self, Union
import abc
import io
import re

import attrs
from pyanalyze.value import concrete_values_from_iterable

from srctools.math import format_float


TAG_REGEX = re.compile(r'<([a-zA-Z]+)(?::([0-9,:\s]*))?>')


def parse_single_float(tag_name: str, args: str) -> float:
    """Parse a single float, raising an approriate error."""
    try:
        return float(args)
    except (ValueError, TypeError):
        raise ValueError(
            f'Invalid <{tag_name}> parameter "{args}", must be a number!'
        ) from None


def parse_int_byte(value: str) -> int:
    """Parse a 0-255 integer."""
    try:
        colour = int(value)
    except (ValueError, TypeError):
        raise ValueError(f'Invalid color value "{value}", must be a number in 0-255!')
    if 0 <= colour <= 255:
        return colour
    else:
        raise ValueError(f'Invalid color value "{value}", must be a number in 0-255!')


class Tag:
    """Base class for tags that can be interspersed in captions text."""
    def export(self) -> str:
        """Produce the string form for this."""
        raise NotImplementedError


class SimpleTag(Tag, Enum):
    """Represents tags with no parameter."""
    #: Toggles whether the following text is bold.
    BOLD = 'b'
    #: Toggles whether the following text is italic.
    ITALIC = 'i'
    #: Inserts a newline between text.
    NEWLINE = 'cr'

    def export(self) -> str:
        return f'<{self.value}>'


TAG_BOLD = SimpleTag.BOLD
TAG_ITALIC = SimpleTag.ITALIC
TAG_NEWLINE = SimpleTag.NEWLINE


@attrs.frozen
class ColourTag(Tag):
    """Sets colours used for the following text."""
    red: int
    green: int
    blue: int

    @classmethod
    def parse(cls, args: str) -> Self:
        """Parse a colour tag."""
        try:
            r, g, b = args.split(',')
        except ValueError:
            raise ValueError(f'<clr> tag expected R,G,B args, got {args.count(",")} args') from None
        else:
            return cls(parse_int_byte(r), parse_int_byte(g), parse_int_byte(b))

    def export(self) -> str:
        return f'<clr:{self.red},{self.green},{self.blue}>'


@attrs.frozen
class PlayerColourTag(ColourTag):
    """Overrides the colours when emitted by the current player."""
    player_red: int
    player_green: int
    player_blue: int

    @classmethod
    def parse(cls, args: str) -> Self:
        """Parse a player colour tag."""
        try:
            player_r, player_g, player_b, other_r, other_g, other_b = args.split(',')
        except ValueError:
            raise ValueError(f'<clr> tag expected R,G,B args, got {args.count(",")} args') from None
        else:
            return cls(
                parse_int_byte(other_r), parse_int_byte(other_g), parse_int_byte(other_b),
                parse_int_byte(player_r), parse_int_byte(player_g), parse_int_byte(player_b),
            )

    def export(self) -> str:
        return (
            f'<clr:{self.player_red},{self.player_green},{self.player_blue}'
            f':{self.red},{self.green},{self.blue}>'
        )


@attrs.frozen
class DelayTag(Tag):
    """Splits the caption, adding a delay between sections and resetting formatting."""
    duration: float

    def export(self) -> str:
        return f'<delay:{format_float(self.duration)}>'


@attrs.frozen
class Message:
    """A caption or subtitle, along with parameters that affect the whole caption."""
    #: Raw text, interspersed with tag values. Tags change the style of text that follows them.
    message: List[Union[Tag, str]]
    #: If set, the message is hidden if subtitles-only mode is enabled.
    is_sfx: bool = attrs.field(default=False, kw_only=True)
    #: If set, the line cannot be shown again for this time.
    no_repeat: Optional[float] = attrs.field(default=None, kw_only=True)
    #: If set, overrides the duration of the line.
    length: Optional[float] = attrs.field(default=None, kw_only=True)

    @classmethod
    def parse_text(cls, text: str) -> Self:
        """Parse text into the appropriate tags."""
        pos = 0
        message: List[Union[Tag, str]] = []
        is_sfx = False
        no_repeat: Optional[float] = None
        length: Optional[float] = None

        while True:
            match = TAG_REGEX.search(text, pos)
            if match is None:
                # No more tags.
                if pos != len(text):
                    message.append(text[pos:])
                break
            if pos - match.start():
                message.append(text[pos:match.start()])
            # Skip past the tag.
            pos = match.end()

            tag_name, args = match.groups()
            tag_name = tag_name.casefold()
            try:
                tag = SimpleTag(tag_name)
            except ValueError:
                pass
            else:
                if args is not None:
                    raise ValueError(
                        f'Tag "{tag_name}" got parameter "{args}", but accpets none!'
                    ) from None
                message.append(tag)
                continue
            # Complex tags
            if tag_name == 'sfx':
                is_sfx = True
            elif tag_name == 'norepeat':
                no_repeat = parse_single_float(tag_name, args)
            elif tag_name == 'length':
                length = parse_single_float(tag_name, args)
            elif tag_name == 'clr':
                message.append(ColourTag.parse(args))
            elif tag_name == 'playerclr':
                message.append(PlayerColourTag.parse(args))
            elif tag_name == 'delay':
                message.append(DelayTag(parse_single_float(tag_name, args)))
            else:
                raise ValueError(f'Unknown tag <{tag_name}>!')

        return cls(message, is_sfx=is_sfx, no_repeat=no_repeat, length=length)

    def export_text(self) -> str:
        """Produce the source for this message."""
        buf = io.StringIO()
        if self.is_sfx:
            buf.write('<sfx>')
        if self.no_repeat is not None:
            buf.write(f'<norepeat:{format_float(self.no_repeat)}>')
        if self.length is not None:
            buf.write(f'<len:{format_float(self.length)}>')
        for segment in self.message:
            if isinstance(segment, str):
                buf.write(segment)
            else:
                buf.write(segment.export())
        return buf.getvalue()
