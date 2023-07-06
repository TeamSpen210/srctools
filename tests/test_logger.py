"""Test the logging system."""
from logging import Logger, getLogger as stdlib_getlogger
import sys


def function(logger: Logger) -> None:
    """Test detecting different methods."""
    logger.info('Starting other function')
    logger.warning('Used wrong logic')
    logger.info('Finishing.')


def test_logging_output(capsys, file_regression, monkeypatch) -> None:
    """Test the output of logging to the console."""
    from srctools.logger import context, get_logger, init_logging

    # Init_logging modifies sys.excepthook, so ensure we undo that.
    monkeypatch.setattr(sys, 'excepthook', sys.excepthook)
    monkeypatch.setattr(stdlib_getlogger(), 'handlers', [])

    root = init_logging()
    root.info('hello there')
    root.error('Root error!:\n- Something failed.')
    get_logger('another').warning('A problem: {}', 45)
    function(root)
    with context('First'):
        root.info('Message')
        with context('Second'):
            root.info('More messages')
        root.warning('A warning.')

    out, err = capsys.readouterr()
    # Merge both files, so if the regression fails the result of both parts is visible.
    combined = f'{out}\n<<<<< STDOUT | STDERR >>>>>\n{err}'
    file_regression.check(combined, encoding='utf8')
