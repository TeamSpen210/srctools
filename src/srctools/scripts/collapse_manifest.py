"""Collapses the submaps of a manifest map into a single VMF."""
from typing import Dict, List, Union
from pathlib import Path
import argparse
import sys

from srctools.fgd import EntityDef
from srctools.filesys import RawFileSystem
from srctools.instancing import InstanceFile, Manifest, collapse_one
from srctools.keyvalues import Keyvalues
from srctools.vmf import VMF, VisGroup


def main(args: List[str]) -> None:
    """Main script."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "inp",
        help="The manifest to collapse.",
    )
    parser.add_argument(
        "-o", "--out",
        help="Specify the destination filename. If not set, it will use "
             "the same filename as the input but with a VMF extension.",
        default="",
    )

    result = parser.parse_args(args)
    source = Path(result.inp)
    if result.out:
        dest = Path(result.out)
    else:
        dest = source.with_suffix('.vmf')

    with source.open() as f:
        submaps = Manifest.parse(Keyvalues.parse(f))
    fsys = RawFileSystem(source.with_suffix(''))

    vmf = VMF()
    engine_cache: Dict[str, EntityDef] = {}

    for submap in submaps:
        print(f'Collapsing "{submap.name}"...')

        with fsys[submap.filename].open_str() as f:
            sub_file = InstanceFile(VMF.parse(Keyvalues.parse(f)))

        visgroup: Union[bool, VisGroup]
        if submap.is_toplevel:
            vmf.spawn.update(sub_file.vmf.spawn)
            visgroup = False
        else:
            visgroup = vmf.create_visgroup(submap.name)
        collapse_one(vmf, submap, sub_file, visgroup=visgroup, engine_cache=engine_cache)

    print(f'Writing {dest}...')
    with dest.open('w') as f:
        vmf.export(f)


if __name__ == '__main__':
    main(sys.argv[1:])
