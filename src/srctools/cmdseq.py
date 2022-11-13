"""Reads and writes Hammer's compilation configs.

These set the arguments to the compile tools.
"""
from typing import IO, Dict, List, Optional, Sequence, Union
from collections import OrderedDict
from enum import Enum
from struct import Struct, pack, unpack


ST_COMMAND = Struct('Bi260s260sii260sii')
ST_COMMAND_PRE_V2 = Struct('Bi260s260sii260si')

SEQ_HEADER = b'Worldcraft Command Sequences\r\n\x1a'

__all__ = ['SpecialCommand', 'Command', 'parse', 'write']


class SpecialCommand(Enum):
    """Special commands to run instead of the exe."""
    CHANGE_DIR = 256
    COPY_FILE = 257
    DELETE_FILE = 258
    RENAME_FILE = 259

SPECIAL_NAMES = {
    SpecialCommand.CHANGE_DIR: 'Change Directory',
    SpecialCommand.COPY_FILE: 'Copy File',
    SpecialCommand.DELETE_FILE: 'Delete File',
    SpecialCommand.RENAME_FILE: 'Rename File',
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
        raise ValueError('{!r} is longer than {}!'.format(text, length))
    return text.encode('ascii') + b'\0' * (length - len(text))


class Command:
    """A command to run."""
    def __init__(
        self,
        executable: Union[str, SpecialCommand],
        args: str,
        *,
        enabled: bool = True,
        ensure_file: Optional[str] = None,
        use_proc_win: bool = True,
        no_wait: bool = False,
    ) -> None:
        self.exe = executable
        self.enabled = enabled
        self.args = args
        self.ensure_file = ensure_file
        self.use_proc_win = use_proc_win
        self.no_wait = no_wait

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
        no_wait: int=0,
    ) -> 'Command':
        """Parse the command from the structure in the file."""
        exe: Union[str, SpecialCommand]
        ensure: Optional[str]
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

    def __repr__(self) -> str:
        return repr(vars(self))


def parse(file: IO[bytes]) -> Dict[str, List[Command]]:
    """Read a list of sequences from a file.

    This returns a dict mapping names to a list of sequences.
    """
    header = file.read(len(SEQ_HEADER))
    if header != SEQ_HEADER:
        raise ValueError('Wrong header: ', header)

    [version] = unpack('f', file.read(4))

    # Read a command
    if version < 0.2:
        cmd_struct = ST_COMMAND_PRE_V2
    else:
        cmd_struct = ST_COMMAND

    [seq_count] = unpack('I', file.read(4))
    sequences = OrderedDict()  # type: Dict[str, List[Command]]
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


def write(sequences: Dict[str, Sequence[Command]], file: IO[bytes]) -> None:
    """Write commands back to a file."""
    file.write(SEQ_HEADER)
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
