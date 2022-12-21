"""
Wrapper around logging to provide our own functionality.

This adds the ability to log using str.format() instead of %.
"""
from typing import (
    Any, Callable, Dict, Generator, Iterable, List, Mapping, Optional, TextIO, Tuple, Type,
    Union, cast, overload,
)
from io import StringIO
from types import TracebackType
import contextlib
import contextvars
import itertools
import logging
import os
import sys
import traceback

from srctools import StringPath


__all__ = ['LoggerAdapter', 'get_handler', 'get_logger', 'init_logging', 'context']
# Only generic in stubs!
CTX_STACK: 'contextvars.ContextVar[List[str]]' = contextvars.ContextVar('srctools_logger')


class LogMessage:
    """Allow using str.format() in logging messages.

    The __str__() method performs the joining.
    """
    fmt: str
    args: Tuple[object, ...]
    kwargs: Dict[str, object]
    has_args: bool
    def __init__(
        self,
        fmt: str,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> None:
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs
        self.has_args = bool(kwargs or args)

    def format_msg(self) -> str:
        """Format using str.format."""
        # Only format if we have arguments!
        # That way { or } can be used in regular messages.
        if self.has_args:
            f = self.fmt = str(self.fmt).format(*self.args, **self.kwargs)

            # Don't repeat the formatting, and don't keep refs to the args.
            del self.args, self.kwargs
            self.has_args = False
            return f
        else:
            return str(self.fmt)

    def __str__(self) -> str:
        """Format the string, and add an ASCII indent."""
        msg = self.format_msg()

        if '\n' not in msg:
            return msg

        # For multi-line messages, add an indent so they're associated
        # with the logging tag.
        lines = msg.split('\n')
        if lines[-1].isspace():
            # Strip last line if it's blank
            del lines[-1]
        # [I] A multi-line message looks like the following:
        # | blah
        # | blah
        # |___
        #
        return '\n | '.join(lines[:]) + '\n |___\n'

_SysExcInfoType = Union[
    Tuple[Type[BaseException], BaseException, Optional[TracebackType]],
    Tuple[None, None, None]
]


class LoggerAdapter(logging.LoggerAdapter):  # type: ignore[type-arg]  # Only generic in stubs.
    """Fix loggers to use str.format().

    """
    def __init__(self, logger: logging.Logger, alias: Optional[str] = None) -> None:
        # Alias is a replacement module name for log messages.
        self.alias = alias
        self.logger = logger
        logging.LoggerAdapter.__init__(self, logger, extra={})

    def log(
        self,
        level: int,
        msg: Any,
        *args: Any,
        exc_info: Union[
            None, bool,
            Union[
                Tuple[type, BaseException, Optional[TracebackType]],
                Tuple[None, None, None]
            ],
            BaseException
        ]=None,
        stack_info: bool=False,
        extra: Optional[Mapping[str, object]] = None,
        **kwargs: Any,
    ) -> None:
        """This version of :external:py:meth:`~logging.Logger.log()` is for :external:py:meth:`str.format()` compatibility.

        The message is wrapped in a :py:class:`LogMessage` object, which is given the
        ``args`` and ``kwargs``.
        ``stacklevel`` is supported, but is silently ignored in Python 3.7.
        """
        if self.isEnabledFor(level):
            try:
                ctx = ', '.join(CTX_STACK.get())
            except LookupError:
                ctx = ''

            new_extra = {} if extra is None else dict(extra)
            new_extra['alias'] = self.alias
            new_extra['context'] = f' ({ctx})' if ctx else ''

            # Pull these arguments out of kwargs, so they can be set..
            log_args = {
                'extra': new_extra,
                'exc_info': exc_info,
                'stack_info': stack_info,
            }
            # Not present in 3.7, silently discard.
            if sys.version_info >= (3, 8) and 'stacklevel' in kwargs:
                log_args['stacklevel'] = kwargs.pop('stacklevel')

            # noinspection PyProtectedMember
            self.logger._log(
                level,
                LogMessage(str(msg), args, kwargs),
                (),  # No positional arguments, we do the formatting through
                # LogMessage..
                **log_args,
            )

    def __getattr__(self, attr: str) -> Any:
        """Delegate unknown methods to the logger."""
        return getattr(self.logger, attr)


class Formatter(logging.Formatter):
    """Override exception handling."""
    SKIP_LIBS = ['importlib', 'cx_freeze', 'PyInstaller']
    def formatException(self, ei: Union[
        Tuple[Type[BaseException], BaseException, Optional[TracebackType]],
        Tuple[None, None, None],
    ]) -> str:
        """Ignore importlib, cx_freeze and PyInstaller."""
        exc_type, exc_value, exc_tb = ei
        buffer = StringIO()

        trace: Optional[TracebackType] = exc_tb

        try:
            while trace is not None:
                filename = trace.tb_frame.f_code.co_filename.casefold()
                if all(keyword not in filename for keyword in self.SKIP_LIBS):
                    break
                trace = trace.tb_next

            if trace is None:
                # It's to do with these libraries themselves? Show the full
                # traceback.
                trace = exc_tb
        except Exception:  # Something failed, keep the original.
            trace = exc_tb

        # TODO: Stubs are wrong, this accepts None for the args and does nothing.
        for line in traceback.TracebackException(exc_type, exc_value, trace).format():  # type: ignore
            buffer.write(line)

        return buffer.getvalue().rstrip('\n')

    def format(self, record: logging.LogRecord) -> str:
        """Ensure a default context is set in the record."""
        record.__dict__.setdefault('context', '')
        return super().format(record)


def get_handler(filename: 'str | os.PathLike[str]') -> logging.FileHandler:
    """Cycle log files, then give the required file handler."""
    name, ext = os.path.splitext(filename)

    suffixes = ('.5', '.4', '.3', '.2', '.1', '')

    try:
        # Remove the oldest one.
        try:
            os.remove(name + suffixes[0] + ext)
        except FileNotFoundError:
            pass

        # Go in reverse, moving each file over to give us space.
        for frm, to in zip(suffixes[1:], suffixes):
            try:
                os.rename(name + frm + ext, name + to + ext)
            except FileNotFoundError:
                pass

        try:
            return logging.FileHandler(filename, mode='x')
        except FileExistsError:
            pass
    except PermissionError:
        pass

    # On windows, we can't touch files opened by other programs (ourselves).
    # If another copy of us is open, it'll hold access.
    # In that case, just keep trying suffixes until we find an empty file.
    for ind in itertools.count(start=1):
        try:
            return logging.FileHandler(
                '{}.{}{}'.format(name, ind, ext),
                mode='x',
            )
        except (FileExistsError, PermissionError):
            pass
    else:
        raise AssertionError  # Never terminates


class NullStream(TextIO):
    """A stream object that discards all data.

    This is needed for multiprocessing, since it tries to flush stdout.
    That'll fail if it is None.
    """
    def __enter__(self) -> 'NullStream':
        return self

    def __exit__(
        self,
        typ: Optional[Type[BaseException]],
        value: Optional[BaseException],
        tback: Optional[TracebackType],
    ) -> None:
        return None

    def __next__(self) -> str:
        raise StopIteration

    def __iter__(self) -> 'NullStream':
        return self

    def close(self) -> None:
        """Closing does nothing."""

    def isatty(self) -> bool:
        """We are not a TTY."""
        return False

    def readable(self) -> bool:
        """Pretend we are readable."""
        return True

    def seekable(self) -> bool:
        """Pretend we are seekable."""
        return True

    def writable(self) -> bool:
        """Pretend we are writable."""
        return True

    def fileno(self) -> int:
        """We do not have a file number."""
        raise OSError('No file number.')

    def flush(self) -> None:
        """Flushing does nothing."""

    def read(self, size: Optional[int] = None) -> str:
        """We never have data."""
        return ''

    def readline(self, limit: int = -1) -> str:
        """We never have data."""
        return ''

    def readlines(self, hint: int = -1) -> List[str]:
        """We never have data."""
        return []

    def write(self, text: str) -> int:
        """Write nothing to the file."""
        return 0

    def writelines(self, lines: Iterable[str]) -> None:
        """Write nothing to the file."""
        pass

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seeking does nothing."""
        return 0

    def tell(self) -> int:
        """We are always at the start position."""
        return 0

    def truncate(self, size: Optional[int]=None) -> int:
        """Truncation does nothing."""
        return 0


class NewLogRecord(logging.LogRecord):
    """Allow passing an alias and context for log modules."""
    alias: Optional[str] = None

    def getMessage(self) -> str:
        """We have to hook here to change the value of .module.

        It's called just before the formatting call is made.
        """
        if self.alias is not None:
            self.module = self.alias
        # If this is one of our logs it {}-formats, otherwise it %-formats.
        return super().getMessage()

@overload
def init_logging(
    filename: Optional[StringPath] = None,
    main_logger: str='',
    on_error: Optional[Callable[
        [Type[BaseException], BaseException, Optional[TracebackType]],
        None,
    ]] = None,
) -> logging.Logger: ...
@overload
def init_logging(
    filename: Optional[StringPath] = None,
    main_logger: str='',
    *,
    error: Callable[[Exception], object],
) -> logging.Logger: ...
def init_logging(
    filename: Optional[StringPath] = None,
    main_logger: str='',
    on_error: Optional[Callable[
        [Type[BaseException], BaseException, Optional[TracebackType]],
        None,
    ]] = None,
    *,
    error: Optional[Callable[[Exception], object]] = None,
) -> logging.Logger:
    """Set up the logger and logging handlers.

    This also sets :py:func:`sys.excepthook`, so uncaught exceptions are captured.

    If an unhandled :external:py:class:`BaseException` is raised, this will not log or call the
    callback.

    :param filename: If this is set, all logs will be written to this file as well.
    :param error: should be a function to call when uncaught exceptions are thrown.
    :param on_error: Deprecated version of ``error`` with the old-style exception tuple \
      representation from Python 3.10 and below.
    :param main_logger: Specify the name of the logger to produce under the `srctools` hierachy.
    """
    if logging.getLogRecordFactory() is not logging.LogRecord:
        raise ValueError('Unknown record factory: ', logging.getLogRecordFactory())
    logging.setLogRecordFactory(NewLogRecord)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if on_error is not None:
        if error is not None:
            raise TypeError('Cannot pass both on_error and error!')
        def error_closure(exc: Exception) -> None:
            """Call the old error handler function."""
            if on_error is not None:  # Mypy can't infer this is constant.
                on_error(type(exc), exc, exc.__traceback__)

        error = error_closure

    # Put more info in the log file, since it's not onscreen.
    long_log_format = Formatter(
        '[{levelname}]{context} {module}.{funcName}(): {message}',
        style='{',
    )
    # Console messages, etc.
    short_log_format = Formatter(
        # One letter for level name
        '[{levelname[0]}]{context} {module}.{funcName}(): {message}',
        style='{',
    )

    if filename is not None:
        # Make the directories the logs are in, if needed.
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # The log contains DEBUG and above logs.
        log_handler = get_handler(filename)
        log_handler.setLevel(logging.DEBUG)
        log_handler.setFormatter(long_log_format)
        logger.addHandler(log_handler)

    if sys.stdout:
        stdout_loghandler = logging.StreamHandler(sys.stdout)
        stdout_loghandler.setLevel(
            logging.DEBUG
            if os.environ.get('SRCTOOLS_DEBUG', 0) == '1' else
            logging.INFO
        )
        stdout_loghandler.setFormatter(short_log_format)
        logger.addHandler(stdout_loghandler)

        if sys.stderr:
            def ignore_warnings(record: logging.LogRecord) -> bool:
                """Filter out messages higher than WARNING.

                Those are handled by stdError, and we don't want duplicates.
                """
                return record.levelno < logging.WARNING
            stdout_loghandler.addFilter(ignore_warnings)
    else:
        sys.stdout = NullStream()

    if sys.stderr:
        stderr_loghandler = logging.StreamHandler(sys.stderr)
        stderr_loghandler.setLevel(logging.WARNING)
        stderr_loghandler.setFormatter(short_log_format)
        logger.addHandler(stderr_loghandler)
    else:
        sys.stderr = NullStream()

    # Use the exception hook to report uncaught exceptions, and finalise the
    # logging system.
    old_except_handler = sys.excepthook

    def except_handler(
        exc_type: Type[BaseException],
        exc_value: BaseException,
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Log uncaught exceptions."""
        if not isinstance(exc_value, Exception):
            # It's subclassing BaseException (KeyboardInterrupt, SystemExit),
            # so we should quit without messages.
            return

        logger.error(
            'Uncaught Exception:',
            exc_info=(exc_type, exc_value, exc_tb),
        )
        if error is not None:
            error(exc_value)
        # Call the original handler - that prints to the normal console.
        if old_except_handler is not sys.__excepthook__:
            old_except_handler(exc_type, exc_value, exc_tb)

    sys.excepthook = except_handler

    if main_logger:
        return get_logger(main_logger)
    else:
        return cast(logging.Logger, LoggerAdapter(logger))


def get_logger(name: str='', alias: Optional[str] = None) -> logging.Logger:
    """Get the named logger object.

    This puts the logger into the ``srctools`` namespace, and wraps it to
    use :external:py:meth:`str.format()` instead of ``%`` formatting.
    If set, ``alias`` is the name to show for the module.
    """
    if name:
        log = logging.getLogger('srctools.' + name)
    else:  # Allow retrieving the main logger.
        log = logging.getLogger('srctools')
    return cast(logging.Logger, LoggerAdapter(log, alias))


@contextlib.contextmanager
def context(name: str) -> Generator[str, None, None]:
    """Context manager to allow specifying additional information for any logs contained in this block.

    The specified string gets included in the log messages.
    """
    try:
        stack = CTX_STACK.get()
    except LookupError:
        stack = []
        CTX_STACK.set(stack)
    stack.append(name)
    try:
        yield name
    finally:
        popped = stack.pop()
        assert popped is name, f'Popped incorrect value: pop({popped!r}) != ctx({name!r})!'
