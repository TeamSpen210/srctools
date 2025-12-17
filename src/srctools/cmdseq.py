"""Reads and writes Hammer's compilation configs.

These set the arguments to the compile tools.
"""
from typing import Union, Final, Optional, IO

from collections.abc import Sequence, Mapping
from enum import Enum
from struct import Struct, pack, unpack
import io

import attrs

from . import conv_int
from .keyvalues import Keyvalues
from .tokenizer import Tokenizer, Token
from .types import FileWBinary


ST_COMMAND = Struct('Bi260s260sii260sii')
ST_COMMAND_PRE_V2 = Struct('Bi260s260sii260si')

# We have two formats here, Valve's binary form, and Strata Source's keyvalues form.
# For keyvalues, we enforce that the root name must start immediately, so we can parse the
# initial bytes. Since it's got a space we don't have to worry about non-quoted forms.
SEQ_HEADER_BINARY: Final[bytes] = b'Worldcraft Command Sequences\r\n\x1a'
SEQ_HEADER_KV: Final[bytes] = b'"command sequences"'
# This is longer, chop so we can compare.
SEQ_HEADER_BINARY_A: Final[bytes] = SEQ_HEADER_BINARY[:len(SEQ_HEADER_KV)]
SEQ_HEADER_BINARY_B: Final[bytes] = SEQ_HEADER_BINARY[len(SEQ_HEADER_KV):]

__all__ = ['SpecialCommand', 'Command', 'parse', 'write']


class SpecialCommand(Enum):
    """Special commands to run instead of an executable.

    Depending on the command, these either take a single :samp:`{filename}` or a :samp:`{source} {destination}` pair.
    """
    #: Change the working directory to the specified folder.
    CHANGE_DIR = 256
    #: Copies the :file:`source` file to the :file:`destination` filename.
    COPY_FILE = 257
    #: Deletes the specified :file:`filename`.
    DELETE_FILE = 258
    #: Renames the :file:`source` file to the :file:`destination` filename.
    RENAME_FILE = 259
    #: Strata Source addition. If :file:`source` exists, copies it to the :file:`destination` filename.
    STRATA_COPY_FILE_IF_EXISTS = 261


# When exporting, the exe field still exists in the binary format. The GUI shows these names, fill
# in what Hammer shows.
SPECIAL_NAMES = {
    SpecialCommand.CHANGE_DIR: 'Change Directory',
    SpecialCommand.COPY_FILE: 'Copy File',
    SpecialCommand.DELETE_FILE: 'Delete File',
    SpecialCommand.RENAME_FILE: 'Rename File',
    SpecialCommand.STRATA_COPY_FILE_IF_EXISTS: 'Copy File if it Exists',
}
NAME_TO_SPECIAL: Mapping[str, Optional[SpecialCommand]] = {
    'none': None,
    'change_dir': SpecialCommand.CHANGE_DIR,
    'copy_file': SpecialCommand.COPY_FILE,
    'delete_file': SpecialCommand.DELETE_FILE,
    'rename_file': SpecialCommand.RENAME_FILE,
    'copy_file_if_exists': SpecialCommand.STRATA_COPY_FILE_IF_EXISTS,
}


def strip_cstring(data: bytes) -> str:
    """Strip strings to the first null, and convert to ascii.

    The CmdSeq files appear to often have junk data in the unused
    sections after the null byte, where C code doesn't touch.
    """
    if b'\0' in data:
        return data[:data.index(b'\0')].decode('ascii')
    else:
        return data.decode('ascii')


def pad_string(text: str, length: int) -> bytes:
    """Pad the string to the specified length and convert."""
    if len(text) > length:
        raise ValueError(f'{text!r} is longer than {length}!')
    return text.encode('ascii') + b'\0' * (length - len(text))


@attrs.define
class Command:
    """A command to run."""
    #: Either the path to the executable, or one of several special commands.
    exe: Union[str, SpecialCommand]
    args: str  #: Parameters to pass to the executable or command.

    #: Whether this command is checked and should run.
    enabled: bool = attrs.field(default=True, kw_only=True)
    #: If non-None, the command should fail if this file doesn't exist after it runs.
    ensure_file: Union[str, None] = attrs.field(default=None, kw_only=True)
    #: Determines if the command should be executed directly, or captured in the 'run process'
    #: window. Obsolete for Hammer versions with ``hammer_run_map_launcher.exe``.
    use_proc_win: bool = attrs.field(default=True, kw_only=True)
    #: Indicates whether Hammer should wait for the command to finish before proceeding. Seems nonfunctional.
    no_wait: bool = attrs.field(default=False, kw_only=True)

    @classmethod
    def parse(
        cls,
        is_enabled: int,
        is_special: int,
        executable: bytes,
        args: bytes,
        is_long_filename: int,  # Unused
        ensure_check: bytes,
        ensure_file: bytes,
        use_proc_win: int,
        no_wait: int = 0,
    ) -> 'Command':
        """Parse the command from the structure in the file."""
        exe: Union[str, SpecialCommand]
        ensure: Union[str, None]
        if is_special:
            exe = SpecialCommand(is_special)
        else:
            exe = strip_cstring(executable)
        if ensure_check:
            ensure = strip_cstring(ensure_file)
        else:
            ensure = None

        return cls(
            exe,
            strip_cstring(args),
            ensure_file=ensure,
            use_proc_win=bool(use_proc_win),
            no_wait=bool(no_wait),
            enabled=bool(is_enabled),
        )

    def __bool__(self) -> bool:
        return self.enabled


# IO[bytes] and not a specific protocol, TextIOWrapper calls most of the API.
def parse(file: IO[bytes]) -> dict[str, list[Command]]:
    """Read a list of sequences from a file.

    This returns a dict mapping names to a list of sequences.
    The file may either be in the Valve binary format, or Strata's keyvalues format.
    """
    header_a = file.read(len(SEQ_HEADER_KV))
    if header_a.lower() == SEQ_HEADER_KV:
        # This is a keyvalues file, switch to parsing as that.
        # The header is a single token, we can push that onto the tokenizer immediately.
        tok = Tokenizer(
            io.TextIOWrapper(file),
            string_bracket=True, allow_escapes=True,
        )
        tok.push_back(Token.STRING, 'command sequences')
        return parse_strata_keyvalues(Keyvalues.parse(tok, getattr(file, 'name', 'CmdSeq.wc')))

    header_b = file.read(len(SEQ_HEADER_BINARY_B))
    if header_a != SEQ_HEADER_BINARY_A or header_b != SEQ_HEADER_BINARY_B:
        raise ValueError(f'Invalid header: {header_a + header_b!r}')

    [version] = unpack('f', file.read(4))

    # Read a command
    if version < 0.2:
        cmd_struct = ST_COMMAND_PRE_V2
    else:
        cmd_struct = ST_COMMAND

    [seq_count] = unpack('I', file.read(4))
    sequences: dict[str, list[Command]] = {}
    for _ in range(seq_count):
        seq_name = strip_cstring(file.read(128))
        [cmd_count] = unpack('I', file.read(4))
        sequences[seq_name] = [
            Command.parse(
                *cmd_struct.unpack(file.read(cmd_struct.size)),
            )
            for i in range(cmd_count)
        ]
    return sequences


def write(sequences: Mapping[str, Sequence[Command]], file: FileWBinary) -> None:
    """Write commands back to a file."""
    file.write(SEQ_HEADER_BINARY)
    file.write(pack('f', 0.2))

    file.write(pack('I', len(sequences)))

    for name, commands in sequences.items():
        file.write(pad_string(name, 128))
        file.write(pack('I', len(commands)))
        for cmd in commands:
            if isinstance(cmd.exe, SpecialCommand):
                special = cmd.exe.value
                exe = SPECIAL_NAMES[cmd.exe]
            else:
                special = 0
                exe = cmd.exe

            if cmd.ensure_file is not None:
                ensure_file = pad_string(cmd.ensure_file, 260)
                has_ensure_file = 1
            else:
                ensure_file = bytes(260)
                has_ensure_file = 0

            file.write(ST_COMMAND.pack(
                cmd.enabled,
                special,
                pad_string(exe, 260),
                pad_string(cmd.args, 260),
                True,  # is_long_filename
                has_ensure_file,
                ensure_file,
                cmd.use_proc_win,
                cmd.no_wait,
            ))


def parse_strata_keyvalues(kv: Keyvalues) -> dict[str, list[Command]]:
    """Parse Strata Source's alternate Keyvalues file format."""
    sequences: dict[str, list[Command]] = {}
    for seq_kv in kv.find_children('Command Sequences'):
        seq_list = sequences.setdefault(seq_kv.real_name, [])
        # These are named sequentially, if it's not numeric though just use the file order.
        for command_kv in sorted(seq_kv, key=lambda child: conv_int(child.name, -1)):
            # This can be a string name, or
            special_str = command_kv['special_cmd', 'none']
            special_cmd: Optional[SpecialCommand]
            if special_str.isdigit():
                special_num = int(special_str)
                special_cmd = None if special_num == 0 else SpecialCommand(special_num)
            else:
                special_cmd = NAME_TO_SPECIAL[special_str.casefold()]
            executable = special_cmd if special_cmd is not None else command_kv['run']
            if command_kv.bool('ensure_check'):
                ensure_file = command_kv['ensure_fn', '']
            else:
                ensure_file = None

            seq_list.append(Command(
                enabled=command_kv.bool('enabled', True),
                exe=executable,
                args=command_kv['params', ''],
                ensure_file=ensure_file,
                no_wait=command_kv.bool('no_wait'),
                use_proc_win=command_kv.bool('use_process_wnd', True),
            ))
    return sequences
