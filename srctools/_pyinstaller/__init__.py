"""Tell PyInstaller to use the hook module."""
import os.path
from typing import List


def get_hook_dirs() -> List[str]:
    """Our only hooks location is this package."""
    return [os.path.dirname(__file__)]
