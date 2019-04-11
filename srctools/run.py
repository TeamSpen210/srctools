"""Code for running VBSP and VRAD."""
from typing import List, IO

import os.path
import sys
import subprocess
import logging

from srctools.logger import get_logger

LOGGER = get_logger(__name__)


def quote(txt: str) -> str:
    """Add quotes to text if needed."""
    if ' ' in txt:
        return '"' + txt + '"'
    return txt


def get_compiler_name(program: str) -> str:
    """Get the real executable name for VBSP or VRAD."""
    if 'darwin' in sys.platform:
        name = program + '_osx_original'
    elif 'win' in sys.platform:
        name = program + '_original.exe'
    else:
        name = program + '_linux_original'
    return os.path.abspath(name)


def run_compiler(
    name: str,
    args: List[str],
    logger: logging.Logger,
) -> int:
    """
    Execute the original vbsp, vvis or vrad.

    The provided logger will be given the output from the compiler.
    The process exit code is returned.
    """
    logger.info("Calling original {}...", name.upper())
    logger.info('Args: {}', ', '.join(map(repr, args)))

    buf_out = bytearray()
    buf_err = bytearray()

    comp_name = get_compiler_name(name)

    # On Windows, calling this will pop open a console window. This suppresses
    # that.
    if sys.platform == 'win32':
        startup_info = subprocess.STARTUPINFO()
        startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup_info.wShowWindow = subprocess.SW_HIDE
    else:
        startup_info = None

    with subprocess.Popen(
        args=[comp_name] + args,
        executable=comp_name,
        bufsize=0,  # No buffering at all.
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        startupinfo=startup_info,
    ) as proc:
        # Loop reading data from the subprocess, until it's dead.
        stdout = proc.stdout  # type: IO[bytes]
        stderr = proc.stderr  # type: IO[bytes]
        while proc.poll() is None:  # Loop until dead.
            buf_out.extend(stdout.read())
            buf_err.extend(stderr.read())

            _pop_lines(logging.ERROR, name, buf_err)
            _pop_lines(logging.INFO, name, buf_out)

        # Grab any extra lines still in the pipe.
        buf_out.extend(stdout.read())
        buf_err.extend(stderr.read())
        _pop_lines(logging.ERROR, name, buf_err)
        _pop_lines(logging.INFO, name, buf_out)

    return proc.returncode


def _pop_lines(loglevel: int, compname: str, buf: bytearray) -> None:
    """Pull off as many complete lines as possible, logging them."""
    if b'\r' in buf:
        buf[:] = buf.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    while True:
        try:
            newline_off = buf.index(b'\n')
        except ValueError:
            return
        else:
            # Discard any invalid ASCII - we don't really care.
            line = buf[:newline_off].decode('ascii', 'ignore')
            del buf[:newline_off + 1]

            # Generate a logging record directly, so the logs appear to "come"
            # from the compiler itself.
            LOGGER.handle(LOGGER.makeRecord(
                "subproc",
                loglevel,
                "<valve>",
                0,  # Line number.
                line,
                (),
                None,
                compname,
            ))
