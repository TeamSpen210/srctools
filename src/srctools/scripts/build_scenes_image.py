"""Builds a ``scenes.image`` file. Unlike the original this allows merging into an existing image."""
from __future__ import annotations
from typing import Literal, cast
from pathlib import Path
import argparse
import sys

from srctools.binformat import checksum
from srctools.choreo import CRC, Entry, Scene, parse_scenes_image, save_scenes_image_sync
from srctools.tokenizer import Tokenizer


parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
    "input",
    help="Sources to use for scenes. If a directory, look for VCDs in a scenes/ subfolder. "
         "Otherwise this should be an existing scenes.image.",
    nargs="*",
    type=Path,
)
parser.add_argument(
    '--version',
    help="Scenes.image version to produce. Must be 2 or 3",
    type=int,
    choices=[2, 3],
    default=3,
)
parser.add_argument(
    "-o", "--out",
    help="Specify the destination filename.",
    dest='output',
    type=Path,
)
parser.add_argument(
    '--duplicates',
    help="If passed, permit duplicate filenames instead of producing an error. "
         "In this case, later definitions override earlier ones.",
    action="store_false",
    dest='require_unique',
)
parser.add_argument(
    '--encoding',
    help="The encoding to use for storing strings. This isn't really considered by the file format. "
         "Conservatively, this defaults to ASCII.",
    default="ascii",
)


class ParsedArgs:
    """Result of the parser."""
    input: list[Path]
    version: Literal[2, 3]
    output: Path
    require_unique: bool
    encoding: str


def load_scene(root: Path, filename: Path, encoding: str) -> Entry:
    """Load a single scene."""
    data = filename.read_bytes()
    scene = Scene.parse_text(Tokenizer(data.decode(encoding), filename))
    scene.text_crc = checksum(data)
    entry = Entry.from_scene(filename.relative_to(root).as_posix(), scene)
    return entry


def main(args: list[str]) -> None:
    """Main script."""
    opts = cast(ParsedArgs, parser.parse_args(args))
    scenes: dict[CRC, Entry] = {}

    for filename in opts.input:
        if filename.is_dir():
            for scene_file in filename.rglob("scenes/**/*.vcd"):
                scene = load_scene(filename, scene_file, opts.encoding)
                print(f'Adding VCD {scene.filename}')
                if opts.require_unique and scene.checksum in scenes:
                    raise ValueError(f'Duplicate copy of "{scene.filename}" provided: {filename}')
                scenes[scene.checksum] = scene
        elif filename.suffix.casefold() == '.image':
            print(f'Merging image {filename}')
            try:
                with open(filename, 'rb') as f1:
                    existing_image = parse_scenes_image(f1)
                for scene in existing_image.values():
                    if scene.checksum in scenes and opts.require_unique:
                        raise ValueError(
                            f'Duplicate copy of "{scene.filename}" '
                            f'provided via image file {filename}'
                        )
                    scenes[scene.checksum] = scene
            except Exception as exc:
                raise ValueError(f'Failed to parse {filename}:') from exc
        else:
            raise ValueError(f'Unrecognised file "{filename}"')

    print(f'{len(scenes)} scenes. Writing {opts.output}')
    with opts.output.open('wb') as f2:
        save_scenes_image_sync(
            f2, scenes,
            version=opts.version,
            encoding=opts.encoding,
        )

if __name__ == '__main__':
    main(sys.argv[1:])
