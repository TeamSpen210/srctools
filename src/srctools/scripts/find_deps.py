"""Finds dependencies used by resources or maps."""
from typing import List
import argparse
import os
import sys

from srctools.bsp import BSP
from srctools.fgd import FGD
from srctools.filesys import FileSystemChain, RawFileSystem
from srctools.game import Game
from srctools.keyvalues import Keyvalues
from srctools.packlist import PackList
from srctools.vmf import VMF


fgd = FGD.engine_dbase()


def main(args: List[str]) -> None:
    """Main script."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-f", "--filter",
        help="filter output to only display resources in this subfolder. "
             "This can be used multiple times.",
        type=str.casefold,
        action='append',
        metavar='folder',
        dest='filters',
    )
    parser.add_argument(
        "-u", "--unused",
        help="Instead of showing depenencies, show files in the filtered "
             "folders that are unused.",
        action='store_true',
    )
    parser.add_argument(
        "game",
        help="either location of a gameinfo.txt file, or any root folder.",
    )
    parser.add_argument(
        "path",
        help="the files to load. The path can have a single * in the "
             "filename to match files with specific extensions and a prefix.",
    )

    result = parser.parse_args(args)

    if result.unused and not result.filters:
        raise ValueError('At least one filter must be provided in "unused" mode.')

    if result.game:
        try:
            fsys = Game(result.game).get_filesystem()
        except FileNotFoundError:
            fsys = FileSystemChain(RawFileSystem(result.game))
    else:
        fsys = FileSystemChain()

    packlist = PackList(fsys)

    file_path: str = result.path
    print('Finding files...')
    if '*' in file_path:  # Multiple files
        if file_path.count('*') > 1:
            raise ValueError('Multiple * in path!')
        prefix, suffix = file_path.split('*')
        folder, prefix = os.path.split(prefix)
        prefix = prefix.casefold()
        suffix = suffix.casefold()
        print(f'Prefix: {prefix!r}, suffix: {suffix!r}')
        print(f'Searching folder {folder}...')

        files = []
        for file in fsys.walk_folder(folder):
            file_path = file.path.casefold()
            if not os.path.basename(file_path).startswith(prefix):
                continue
            if file_path.endswith(suffix):
                print(' ' + file.path)
                files.append(file)
    else:  # Single file
        files = [fsys[file_path]]
    for file in files:
        ext = file.path[-4:].casefold()
        if ext == '.vmf':
            with file.open_str() as f:
                vmf_props = Keyvalues.parse(f)
                vmf = VMF.parse(vmf_props)
            packlist.pack_fgd(vmf, fgd)
            del vmf, vmf_props  # Hefty, don't want to keep.
        elif ext == '.bsp':
            child_sys = fsys.get_system(file)
            if not isinstance(child_sys, RawFileSystem):
                raise ValueError('Cannot inspect BSPs in VPKs!')
            bsp = BSP(os.path.join(child_sys.path, file.path))
            packlist.pack_from_bsp(bsp)
            packlist.pack_fgd(bsp.ents, fgd)
            del bsp
        else:
            packlist.pack_file(file.path)
    print('Evaluating dependencies...')
    packlist.eval_dependencies()
    print('Done.')

    if result.unused:
        print('Unused files:')
        used = set(packlist.filenames())
        for folder in result.filters:
            for file in fsys.walk_folder(folder):
                if file.path.casefold() not in used:
                    print(' ' + file.path)
    else:
        print('Dependencies:')
        for filename in packlist.filenames():
            if not result.filters or any(map(filename.casefold().startswith, result.filters)):
                print(' ' + filename)

if __name__ == '__main__':
    main(sys.argv[1:])
