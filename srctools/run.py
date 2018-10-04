"""Code for running VBSP and VRAD."""
import io
from typing import List

import os.path
import sys
import subprocess
import logging

from srctools.logger import get_logger

LOGGER = get_logger(__name__)


def quote(txt):
    """Add quotes to text if needed."""
    if ' ' in txt:
        return '"' + txt + '"'
    return txt


def get_compiler_name(program: str):
    """Get the real executable name for VBSP or VRAD."""
    if 'win' in sys.platform:
        name = program + '_original.exe'
    elif 'darwin' in sys.platform:
        name = program + '_osx_original'
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
        universal_newlines=False,
        bufsize=0,  # No buffering at all.
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        startupinfo=startup_info,
    ) as proc:
        # Loop reading data from the subprocess, until it's dead.
        stdout = proc.stdout  # type: io.FileIO
        stderr = proc.stderr  # type: io.FileIO
        while proc.poll() is None:  # Loop until dead.
            buf_out.extend(stdout.read(64))
            buf_err.extend(stderr.read(64))

            if b'\r' in buf_err:
                buf_err = buf_err.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
            if b'\r' in buf_out:
                buf_out = buf_out.replace(b'\r\n', b'\n').replace(b'\r', b'\n')

            try:
                newline_off = buf_err.index(b'\n')
            except ValueError:
                pass
            else:
                # Discard any invalid ASCII - we don't really care.
                logger.error(buf_err[:newline_off].decode('ascii', 'ignore'))
                buf_err = buf_err[newline_off+1:]

            try:
                newline_off = buf_out.index(b'\n')
            except ValueError:
                pass
            else:
                # Discard any invalid ASCII - we don't really care.
                logger.info(buf_out[:newline_off].decode('ascii', 'ignore'))
                buf_out = buf_out[newline_off+1:]

    return proc.returncode
