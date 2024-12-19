"""Parses caption/subtitle files."""
import types
from enum import Enum
from typing_extensions import Self
from typing import ClassVar, Dict, IO, List, Optional, Type, Union, final
import io
import re

import attrs

from srctools import Keyvalues
from srctools._crc_map import ChecksumMap
from srctools.binformat import struct_read
from srctools.math import format_float

__all__ = [
    'Tag', 'SimpleTag', 'ColourTag', 'PlayerColourTag', 'DelayTag',
    'TAG_BOLD', 'TAG_ITALIC', 'TAG_NEWLINE',
    'Message', 'CaptionsMap',
]
TAG_REGEX = re.compile(r'<([a-zA-Z]+)(?::([^>]*))?>')


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
        raise ValueError(f'Invalid color value "{value}", must be a number within the range 0-255!')
    if 0 <= colour <= 255:
        return colour
    else:
        raise ValueError(f'Invalid color value "{value}", must be within the range 0-255!')


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
class BlockRef:
    """Location of a caption in the binary file"""
    checksum: int
    block: int
    offset: int
    size: int


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
            raise ValueError(f'<clr> tag expected 3 args, got {args.count(",")+1} args') from None
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
            player, other = args.split(':')
            player_r, player_g, player_b = player.split(',')
            other_r, other_g, other_b = other.split(',')
        except ValueError:
            raise ValueError(f'<playerclr> tag expected "R,G,B:R,G,B" colour pairs, got "{args}"') from None
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


@final
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
                        f'Tag <{tag_name}> got parameter "{args}", but accepts none!'
                    ) from None
                message.append(tag)
                continue
            # Tags with parameters.
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


@final
class CaptionsMap(ChecksumMap[Message]):
    """A mapping of caption keys to the associated messages.

    If parsed from text keyvalues or when messages are stored programmatically, this behaves
    like any mapping. If parsed from a binary file, this has some special behaviour.

    Binary files use a checksum of the token to identify messages, so it is not possible to initially
    iterate over tokens when first parsed. This mapping will store the associated key during
    lookup, allowing iteration.

    The binary format also allows lazy parsing. This will keep the file open, parsing blocks
    only when required.
    """
    # Checksum -> block number. We always parse the whole block each time.
    _unparsed: Dict[int, int]
    # For each block, the data and a list of captions present there.
    _blocks: Dict[int, List[BlockRef]]
    _block_size: int  # Size of each block.
    _data_offset: int  # Offset to data blocks.
    # If blocks are present, the open file.
    file: Optional[IO[bytes]]

    def __init__(self) -> None:
        super().__init__()
        self._blocks = {}
        self._unparsed = {}
        self._block_size = 512
        self._data_offset = 0
        self.file = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[types.TracebackType],
    ) -> None:
        if self.file is not None:
            self.file.close()

    @classmethod
    def parse_binary(cls, file: IO[bytes]) -> Self:
        """Parse a binary captions file.

        This is lazy - only the directory of checksums is parsed, each block is parsed individually.
        """
        (magic, version) = struct_read('<4sI', file)
        if magic != b'VCCD':
            raise ValueError(f'File is not a captions file, got magic={magic!r}.')
        if version != 1:
            raise ValueError(f'Invalid captions version, got {version}, only v1 is valid.')
        mapping = cls()
        (
            block_count,
            mapping._block_size,
            dir_size,
            mapping._data_offset,
        ) = struct_read('4i', file)
        for _ in range(dir_size):
            (check, block_num, offset, size) = struct_read('<IiHH', file)
            ref = BlockRef(check, block_num, offset, size)
            mapping._unparsed[check] = block_num
            try:
                mapping._blocks[block_num].append(ref)
            except KeyError:
                mapping._blocks[block_num] = [ref]

        mapping.file = file
        return mapping

    def clear(self) -> None:
        """Clear all values."""
        super().clear()
        self._unparsed.clear()
        if self.file is not None:
            self.file.close()
            self.file = None

    def parse_all_blocks(self) -> None:
        """Immediately parse all of the binary captions file, then close the file."""
        for index in list(self._blocks):
            self._parse_block(index)
        self.file.close()
        self.file = None

    def _try_parse(self, key: str, check: int) -> Message:
        """Hook to allow lazily parsing values.

        This should parse this value, store it then return, or raise KeyError if not found.
        """
        try:
            block = self._unparsed[check]
        except KeyError:
            raise KeyError(key) from None
        if self._parse_block(block):
            # These were parsed with no key known, store the key
            # we do know.
            blank_key, value = self._values[check]
            if blank_key is None:
                self._values[check] = key, value
            return value
        else:
            # Block already parsed.
            raise KeyError(key)

    def _clear_unparsed(self, check: int) -> bool:
        """If a lazily parsed value with this checksum is present, clear it."""
        return False

    def _parse_block(self, block: int) -> bool:
        """Parse this block, storing values, or return False if not present."""
        if self.file is None:
            raise ValueError(f'Trying to parse block {block}, but no open file!')
        try:
            refs = self._blocks.pop(block)
        except KeyError:
            return False
        self.file.seek(self._data_offset + block * self._block_size)
        data = self.file.read(self._block_size)
        for ref in refs:
            line = data[ref.offset:ref.offset+ref.size].decode(self.encoding)
            # If already present, keep existing values.
            self._values.setdefault(ref.checksum, (None, Message.parse_text(line)))
            self._unparsed.pop(ref.checksum, None)
        return True
