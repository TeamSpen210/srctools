"""Collapses the submaps of a manifest map into a single VMF."""
import argparse
import sys
from pathlib import Path
from typing import List, Union

from srctools import FGD, Property, Vec, VMF
from srctools.instancing import Manifest, InstanceFile, collapse_one
from srctools.filesys import RawFileSystem
from srctools.vmf import VisGroup


def main(args: List[str]) -> None:
    """Main script."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-f", "--fgd",
        help="Path to a FGD file to use to collapse instances. "
             "If not set a builtin file will be used.",
        type=str.casefold,
        action='append',
    )
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

    if result.fgd:
        fgd = FGD()
        fsys = RawFileSystem('.', constrain_path=False)
        with fsys:
            for path in result.fgd:
                fgd.parse_file(fsys, fsys[path])
    else:
        fgd = FGD.engine_dbase()

    with source.open() as f:
        submaps = Manifest.parse(Property.parse(f))
    fsys = RawFileSystem(source.with_suffix(''))
    fsys.open_ref()

    vmf = VMF()

    for submap in submaps:
        print(f'Collapsing "{submap.name}"...')

        with fsys[submap.filename].open_str() as f:
            sub_file = InstanceFile(VMF.parse(Property.parse(f)))

        visgroup: Union[bool, VisGroup]
        if submap.is_toplevel:
            vmf.spawn.keys.update(sub_file.vmf.spawn.keys)
            visgroup = False
        else:
            visgroup = vmf.create_visgroup(submap.name)
        collapse_one(vmf, submap, sub_file, fgd, visgroup)

    print(f'Writing {dest}...')
    with dest.open('w') as f:
        vmf.export(f)

if __name__ == '__main__':
    main(sys.argv[1:])
