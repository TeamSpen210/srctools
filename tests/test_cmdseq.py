"""Test cmdseq parsing."""
import io

from pytest_datadir.plugin import LazyDataDir
from pytest_regressions.file_regression import FileRegressionFixture

from srctools.cmdseq import SpecialCommand, Command, parse, write


def test_parse_binary_v2(lazy_datadir: LazyDataDir) -> None:
    """Parse the standard binary format, version 2.

    Both the 'no-wait' flag and 'use proc_win' is always unset.
    """
    with open(lazy_datadir / 'sequence_binary_v2.wc', 'rb') as f:
        sequences = parse(f)
    assert sequences == {
        'Test Config': [
            Command('C:/executable.exe', 'parm1 parm2', enabled=True, ensure_file=None, use_proc_win=False),
            Command(SpecialCommand.CHANGE_DIR, 'destination', enabled=True, ensure_file='post_exists', use_proc_win=False),
            Command(SpecialCommand.COPY_FILE, 'src dest', enabled=True, ensure_file=None, use_proc_win=False),
            Command(SpecialCommand.DELETE_FILE, 'deleted', enabled=True, ensure_file='post/file', use_proc_win=False),
            Command(SpecialCommand.RENAME_FILE, 'first second', enabled=True, ensure_file=None, use_proc_win=False),
            Command('$bsp_exe', 'vbsp', enabled=False, use_proc_win=False),
            Command('$vis_exe', 'vvis', enabled=False, use_proc_win=False),
            Command('$light_exe', 'vrad', enabled=False, use_proc_win=False),
            Command('$game_exe', 'hl2', enabled=False, use_proc_win=False, no_wait=True),
        ],
        'Second Config': [
            Command('$bsp_exe', '$path/$file with spaces', enabled=False, use_proc_win=False),
        ],
    }


def test_export_binary(file_regression: FileRegressionFixture) -> None:
    """Test exporting the binary format."""
    sequences = {
        'FirstSeq': [
            Command('C:/executable.exe', 'parm1 parm2', enabled=True, ensure_file=None, use_proc_win=False, no_wait=False),
            Command(SpecialCommand.CHANGE_DIR, 'destination', enabled=True, ensure_file='post_exists', use_proc_win=False, no_wait=True),
            Command(SpecialCommand.COPY_FILE, 'src dest', enabled=True, ensure_file=None, use_proc_win=False, no_wait=False),
        ], 'Second Seq': [
            Command(SpecialCommand.DELETE_FILE, 'deleted', enabled=True, ensure_file='post/file', use_proc_win=False, no_wait=True),
            Command(SpecialCommand.RENAME_FILE, 'first second', enabled=True, ensure_file=None, use_proc_win=False, no_wait=False),
        ], 'Third': [
            Command('$bsp_exe', 'vbsp', enabled=False, use_proc_win=True, no_wait=False),
            Command('$vis_exe', 'vvis', enabled=False, use_proc_win=True, no_wait=True),
            Command('$light_exe', 'vrad', enabled=False, use_proc_win=False, no_wait=False),
            Command('$game_exe', 'hl2', enabled=False, use_proc_win=False, no_wait=True),
        ]
    }
    buf = io.BytesIO()
    write(sequences, buf)
    file_regression.check(buf.getvalue(), binary=True, extension='.wc')
