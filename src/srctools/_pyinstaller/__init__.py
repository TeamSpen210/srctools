"""Tell PyInstaller to use the hook module."""
from typing import List
from pathlib import Path


def get_hook_dirs() -> List[str]:
    """Our only hooks location is this package."""
    return [str(Path(__file__).parent)]
