"""Code for running VBSP and VRAD."""
from typing import List, IO

import os.path
import sys
import subprocess
import logging
import threading

from srctools.logger import get_logger

__all__ = ['run_compiler', 'get_compiler_name']


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
    logger: logging.Logger=get_logger('<compiler>'),
    change_name: bool=True,
) -> int:
    """
    Execute the original vbsp, vvis or vrad.

    The provided logger will be given the output from the compiler.
    The process exit code is returned.
    """
    log_name = os.path.basename(name)
    logger.info("Calling original {}...", log_name.upper())
    logger.info('Args: {}', ', '.join(map(repr, args)))

    comp_name = get_compiler_name(name) if change_name else name

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
        cwd=os.path.dirname(comp_name) or None,
    ) as proc:
        # Run both threads, then wait for it to die.
        thread_out = threading.Thread(
            target=_daemon,
            args=(logger, logging.INFO, log_name, proc.stdout),
        )
        thread_err = threading.Thread(
            target=_daemon,
            args=(logger, logging.ERROR, log_name,  proc.stderr),
        )
        thread_out.daemon = thread_err.daemon = True  # These should be killed on shutdown.
        thread_err.start()
        thread_out.start()
        proc.wait()
    # Exiting the with: causes the streams to be closed, so these should terminate.
    thread_out.join()
    thread_err.join()

    return proc.returncode


def _daemon(logger: logging.Logger, loglevel: int, compname: str, stream: IO[bytes]) -> None:
    """Handles logging each stream.

    Continuously pull off data, then log whenever they're complete.
    """
    buf = bytearray()
    while not stream.closed:
        try:
            buf.extend(stream.read(1))
        except ValueError:
            # Closed file.
            break
        # If it's \r, we need to see if the next is \n so we can skip.
        if buf.endswith(b'\r'):
            continue

        if b'\r' in buf:
            buf[:] = buf.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
        while True:
            try:
                newline_off = buf.index(b'\n')
            except ValueError:
                break
            else:
                # Discard any invalid ASCII - we don't really care.
                line = buf[:newline_off].decode('ascii', 'ignore')
                del buf[:newline_off + 1]

                # Generate a logging record directly, so the logs appear to
                # "come"
                # from the compiler itself.
                logger.handle(logger.makeRecord(
                    "subproc",
                    loglevel,
                    "<valve>",
                    0,  # Line number.
                    line,
                    (),
                    None,
                    compname,
                ))
    if buf:
        # There's still data we haven't emitted, so just dump it all
        # out at the end.
        buf[:] = buf.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
        logger.handle(logger.makeRecord(
            "subproc",
            loglevel,
            "<valve>",
            0,  # Line number.
            buf.decode('ascii', 'ignore'),
            (),
            None,
            compname,
        ))