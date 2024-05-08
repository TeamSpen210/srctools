"""Code which handles executing and interfacing with game binaries."""
from __future__ import annotations
from typing import IO
import logging
import os.path
import subprocess
import sys
import threading

from srctools.logger import get_logger


__all__ = ['run_compiler', 'get_compiler_name', 'send_engine_command']


def quote(txt: str) -> str:
    """Add quotes to text if needed."""
    if ' ' in txt:
        return f'"{txt}"'
    return txt


if sys.platform == 'win32':
    import ctypes

    _FindWindowW = ctypes.windll.user32.FindWindowW
    _SendMessageW = ctypes.windll.user32.SendMessageW

    _FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
    _FindWindowW.restype = ctypes.c_void_p  # HWND
    _is_64bit = sys.maxsize > 2 ** 33
    _uint_ptr = ctypes.POINTER(ctypes.c_ulonglong if _is_64bit else ctypes.c_uint)
    _ulong_ptr = ctypes.POINTER(ctypes.c_ulonglong if _is_64bit else ctypes.c_ulong)
    _long_ptr = ctypes.POINTER(ctypes.c_longlong if _is_64bit else ctypes.c_long)
    _SendMessageW.argtypes = [
        ctypes.c_void_p,  # HWND hWnd
        ctypes.c_uint,  # Msg
        _uint_ptr,  # WPARAM wParam
        _long_ptr,  # LPARAM lParam
    ]
    _SendMessageW.restype = _long_ptr

    WM_COPYDATA = 0x4A
    del _is_64bit

    class CopyDataStruct(ctypes.Structure):
        """Data to pass for WM_COPYDATA."""
        _fields_ = (
            ('dWData', _ulong_ptr),
            ('cbData', ctypes.c_uint32),
            ('lpData', ctypes.c_void_p),
        )

    def _send_cmd(command: bytes, classname: str) -> None:
        """Send the command to a game."""
        window: ctypes.c_void_p | None = _FindWindowW(classname, None)
        if window is None:
            raise LookupError('No window found!')
        buf = ctypes.create_string_buffer(command)
        data = CopyDataStruct(
            dWData=_ulong_ptr(),
            cbData=len(command) + 1,
            lpData=ctypes.cast(buf, ctypes.c_void_p),
        )
        if not _SendMessageW(window, WM_COPYDATA, _uint_ptr(), ctypes.cast(ctypes.byref(data), _long_ptr)):
            raise ValueError("Failed to send command.")
else:
    def _send_cmd(command: bytes, classname: str) -> None:
        """Not available on this OS."""
        raise OSError('Functions not available!')


def send_engine_command(command: bytes, *, classname: str = 'Valve001') -> None:
    """Send a command to a Source Engine game, using the mechanism :option:`!-hijack` uses.

    If a Source Engine game is currently running, the string will be executed as if it was typed
    into the console.
    This is only functional on Windows, since it uses the `WM_COPYDATA`_ mechanism.

    .. _WM_COPYDATA: https://learn.microsoft.com/en-us/windows/win32/dataxchg/wm-copydata

    :param command: The command to execute. For multiple commands use ``;`` as a separator.
    :param classname: The classname of the window to look for. This should be ``Valve001``, but \
        it is possible other branches may have changed this.
    :raises LookupError: If the game could not be found.
    :raises OSError: If this is not running on Windows.
    :raises ValueError: If the command failed to execute.
    """
    _send_cmd(command, classname)


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
    args: list[str],
    logger: logging.Logger = get_logger('<compiler>'),
    change_name: bool = True,
) -> int:
    """
    Execute the original vbsp, vvis or vrad.

    The provided logger will be given the output from the compiler.
    The process exit code is returned.
    """
    log_name = os.path.basename(name)
    logger.info("Calling original {}...", log_name.upper())
    logger.info('Args: {}', ', '.join([repr(arg) for arg in args]))

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
        args=[comp_name, *args],
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
            args=(logger, logging.ERROR, log_name, proc.stderr),
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
