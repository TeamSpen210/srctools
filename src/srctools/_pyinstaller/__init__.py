"""Tell PyInstaller to use the hook module."""
from typing import List
import os.path


def get_hook_dirs() -> List[str]:
    """Our only hooks location is this package."""
    return [os.path.dirname(__file__)]
