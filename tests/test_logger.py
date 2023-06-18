"""Import all modules, to ensure at least imports work."""
import sys
from io import StringIO


def test_logging_output() -> None:
    """Test the output of logging to the console."""
    from srctools.logger import init_logging, get_logger
    old, sys.stdout = sys.stdout, StringIO()
    init_logging().info('hello there')
    output, sys.stdout = sys.stdout, old
    output.seek(0)
    assert output.read() == '[I] test_logger.test_logging_output(): hello there\n'
