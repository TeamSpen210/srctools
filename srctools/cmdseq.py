"""Reads and writes Hammer's compilation configs.

These set the arguments to the compile tools.
"""
from struct import Struct, unpack, pack
from enum import Enum
from collections import OrderedDict

ST_HEADER = Struct('')

ST_COMMAND = Struct('ci260s260sii260sii')
ST_COMMAND_PRE_V2 = Struct('ci260s260sii260si')

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

def strip_cstring(data: bytes):
    """Strip strings to the first null, and convert to ascii.
    
    The CmdSeq files appear to often have junk data in the unused
    sections after the null byte, where C code doesn't touch.
    """
    if b'\0' in data:
        return data[:data.index(b'\0')].decode('ascii')
    else:
        return data.decode('ascii')
    
def pad_string(text: str, length: int):
    """Pad the string to the specified length and convert."""
    if len(text) > length:
        raise ValueError('{!r} is longer than {}!'.format(text, length))
    return text.encode('ascii') + b'\0' * (length - len(text))

class Command:
    """A command to run."""
    def __init__(
        self,
        enabled: bool,
        # If it's a special command like 'Copy File'
        special: int,
        executable: str,
        args: str,
        # If enabled, ensure file exists...
        ensure_check: bool,
        ensure_file: str,
        use_proc_win: bool,
        no_wait: bool
    ):
        self.enabled = enabled
        if special:
            self.exe = SpecialCommand(special)
        else:
            self.exe = executable
        self.args = args
        self.ensure_check = ensure_check
        self.ensure_file = ensure_file
        self.use_proc_win = use_proc_win
        self.no_wait = no_wait
        
    @classmethod
    def parse(
        cls, 
        is_enabled,
        is_special,
        executable,
        args,
        is_long_filename, # Unused
        ensure_check,
        ensure_file,
        use_proc_win,
        no_wait=False, 
    ):
        return cls(
            bool(is_enabled),
            is_special,
            strip_cstring(executable),
            strip_cstring(args),
            bool(ensure_check),
            strip_cstring(ensure_file),
            use_proc_win,
            bool(no_wait),
        )
        
    def __bool__(self): 
        return self.enabled
        
    def __repr__(self):
        return repr(vars(self))
        
def parse(file):
    """Read a list of sequences from a file.
    
    This returns a dict mapping names to a list of sequences.
    """
    header = file.read(len(SEQ_HEADER))
    if header != SEQ_HEADER:
        raise ValueError('Wrong header!')
        
    [version] = unpack('f', file.read(4))
        
    # Read a command
    if version < 0.2:
        cmd_struct = ST_COMMAND_PRE_V2
    else:
        cmd_struct = ST_COMMAND
        
    [seq_count] = unpack('I', file.read(4))
    sequences = OrderedDict()
    for _ in range(seq_count):
        seq_name = strip_cstring(file.read(128))
        [cmd_count] = unpack('I', file.read(4))
        sequences[seq_name] = cmd_list = [None] * cmd_count
        
        for i in range(cmd_count):  
            cmd_list[i] = Command.parse(
                *cmd_struct.unpack(file.read(cmd_struct.size)),
            )
    return sequences
    
def write(sequences: dict, file):
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
            file.write(ST_COMMAND.pack(
                bytes(cmd.enabled),
                special,
                pad_string(exe, 260),
                pad_string(cmd.args, 260),
                True, # is_long_filename
                cmd.ensure_check,
                pad_string(cmd.ensure_file, 260),
                cmd.use_proc_win,
                cmd.no_wait,
            ))
