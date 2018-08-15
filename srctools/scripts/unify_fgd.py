"""Implements "unified" FGD files.

This allows sharing definitions among different engine versions.
"""
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Set, FrozenSet

from srctools.fgd import FGD, validate_tags, EntityDef, EntityTypes
from srctools.filesys import RawFileSystem


# Chronological order of games.
# If 'since_hl2' etc is used in FGD, all future games also include it.

GAMES = [
    ('HLS',  'Half-Life: Source'),
    ('DODS', 'Day of Defeat: Source'),
    ('CSS',  'Counter-Strike: Source'),
    ('HL2',  'Half-Life 2'),
    ('EP1',  'Half-Life 2 Episode 1'),
    ('EP2',  'Half-Life 2 Episode 2'),
    ('TF2',  'Team Fortress 2'),
    ('P1', 'Portal'),
    ('L4D', 'Left 4 Dead'),
    ('L4D2', 'Left 4 Dead 2'),
    ('ASW', 'Alien Swam'),
    ('P2', 'Portal 2'),
    ('CSGO', 'Counter-Strike Global Offensive'),
    ('SFM', 'Source Filmmaker'),
    ('DOTA2', 'Dota 2'),
    ('PUNT', 'PUNT'),
    ('P2DES', 'Portal 2: Desolation'),
]  # type: List[Tuple[str, str]]

GAME_ORDER = [game for game, desc in GAMES]
GAME_NAME = dict(GAMES)

# Specific features that are backported to various games.

FEATURES = {
    'TF2': ('prop_scaling', ),
    'CSGO': ('prop_scaling', ),
    'P2DES': ('prop_scaling', )
}

ALL_TAGS = set()  # type: Set[str]
ALL_TAGS.update(GAME_ORDER)
ALL_TAGS.update(*FEATURES.values())


def ent_path(ent: EntityDef) -> str:
    """Return the path in the database this entity should be found at."""
    # Very special entity.
    if ent.classname == 'worldspawn':
        return 'worldspawn.fgd'

    if ent.type is EntityTypes.BASE:
        folder = 'bases'
    elif ent.type is EntityTypes.BRUSH:
        folder = 'brush'
    else:
        folder = 'point/'

    # if '_' in ent.classname:
    #     folder += '/' + ent.classname.split('_', 1)[0]

    return '{}/{}.fgd'.format(folder, ent.classname)


def load_database(dbase: Path) -> FGD:
    """Load the entire database from disk."""
    print('Loading database...')
    fgd = FGD()

    fgd.map_size_min = -16384
    fgd.map_size_max = 16384

    with RawFileSystem(str(dbase)) as fsys:
        for file in dbase.rglob("*.fgd"):
            fgd.parse_file(fsys, fsys[str(file.relative_to(dbase))])
            print('.', end='')
    print('\nDone!')
    return fgd


def action_import(
    dbase: Path,
    tags: FrozenSet[str],
    fgd_paths: List[Path],
) -> None:
    """Import an FGD file, adding differences to the unified files."""


def action_export(
    dbase: Path,
    tags: FrozenSet[str],
    output_path: Path,
) -> None:
    """Create an FGD file using the given tags."""


def main(args: List[str]=None):
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Manage a set of unified FGDs, sharing configs "
                    "between engine versions.",

    )
    parser.add_argument(
        "-d", "--database",
        default="../../fgds/",
        help="The folder to write the FGD files to or from."
    )
    subparsers = parser.add_subparsers(dest="mode")

    parser_exp = subparsers.add_parser(
        "export",
        help=action_export.__doc__,
        aliases=["exp", "e"],
    )

    parser_exp.add_argument(
        "-o", "--output",
        default="output.fgd",
        nargs="+",
        help="Destination FGD filename."
    )
    parser_exp.add_argument(
        "tags",
        choices=ALL_TAGS,
        nargs="+",
        help="Tags to include in the output.",
    )

    parser_imp = subparsers.add_parser(
        "import",
        help=action_import.__doc__,
        aliases=["imp", "i"],
    )
    parser_imp.add_argument(
        "-t", "--tag",
        required=True,
        action="append",
        choices=ALL_TAGS,
        help="Tag to associate this FGD set with.",
        dest="tags",
    )
    parser_imp.add_argument(
        "fgd",
        nargs="+",
        type=Path,
        help="The FGD files to import. "
    )

    result = parser.parse_args(args)

    if result.mode is None:
        parser.print_help()
        return

    dbase = Path(result.database).resolve()
    dbase.mkdir(parents=True, exist_ok=True)

    tags = validate_tags(result.tags)

    if result.mode == "import":
        action_import(
            dbase,
            tags,
            result.fgd,
        )
    elif result.mode == "export":
        action_export(
            dbase,
            tags,
            result.output,
        )
    else:
        raise AssertionError("Unknown mode!")


if __name__ == '__main__':
    main(sys.argv[1:])
