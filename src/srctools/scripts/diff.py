"""Compute diffs between files that srctools handles."""
from typing import List
from pathlib import Path
import sys

from srctools.vpk import VPK


def diff_vpk(path1: Path, path2: Path) -> None:
    """Compute the diff of two VPK files."""
    vpk1 = VPK(path1)
    vpk2 = VPK(path2)

    files = set(vpk1.filenames()) | set(vpk2.filenames())

    table = []

    for filename in files:
        change = 'M'

        try:
            file1 = vpk1[filename]
        except KeyError:
            file1_size = 0
            file1_crc = None
            change = '+'
        else:
            file1_size = file1.size
            file1_crc = file1.crc

        try:
            file2 = vpk2[filename]
        except KeyError:
            file2_size = 0
            file2_crc = None
            change = '-'
        else:
            file2_size = file2.size
            file2_crc = file2.crc

        if file1_crc == file2_crc:
            continue  # Identical or both missing (latter should never happen).

        table.append((change, filename, file2_size - file1_size))

    table.sort()

    # Figure out the longest name so we can format a table.
    max_filename = max(len(t[1]) for t in table)

    header = '  | {0:^{1}} | Length'.format('Filename', max_filename)
    print(header)
    print('-' * len(header))

    for change, filename, diff in table:
        print('{type} | {file:<{size}} | {diff:+d}'.format(
            type=change,
            file=filename,
            size=max_filename,
            diff=diff,
        ))


def main(args: List[str]) -> None:
    ext = path = None
    if len(args) == 3:
        ext, fname1, fname2 = args
    elif len(args) == 7:
        # Git diffing.
        path, fname1, hex1, mode1, fname2, hex2, mode2 = args
        ext = Path(path).suffix
    elif len(args) == 2:
        fname1, fname2 = args
    else:
        print('''Usage:
        diff.py file1 file2
        diff.py ext file1 file2
        diff.py path old-file old-hex old-mode new-file new-hex new-mode
    ''')
        return

    if path is None:
        path = fname1

    file1 = Path(fname1)
    file2 = Path(fname2)

    if not file1.is_file() or file1.stat().st_size == 0:
        print('Create ' + str(path))
        return
    if not file2.is_file() or file2.stat().st_size == 0:
        print('Delete ' + str(path))
        return

    if ext is None:
        # Guess extension.
        if file1.suffix.casefold() != file2.suffix.casefold():
            print('Extensions do not match!')
            return
        ext = file1.suffix

    try:
        func = globals()['diff_' + ext.casefold()]
    except KeyError:
        print(f'Unknown extension "{ext}"!')
        return

    func(file1, file2)


if __name__ == '__main__':
    main(sys.argv[1:])
