"""For older engines, HLMV requires the actual folder path to exist in order to read VPKs.

This generates those."""
from typing import List
from pathlib import Path
import argparse
import sys

from srctools.vpk import VPK


def main(args: List[str]) -> None:
    """Create empty folders for all models in a VPK."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-f", "--folder",
        default='',
        help="If set, place the folders at this location."
    )

    parser.add_argument(
        "vpk",
        help="The path to the VPK to read.",
    )
    result = parser.parse_args(args)

    dest_folder = Path(result.folder).resolve()
    with VPK(result.vpk) as vpk:
        for folder in vpk.folders(ext='mdl'):
            full_path = (dest_folder / folder)
            print(str(full_path).replace('\\', '/') + '/')
            full_path.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    main(sys.argv[1:])
