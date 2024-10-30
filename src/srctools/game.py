"""Reads the GameInfo file to determine where Source game data is stored."""
from typing import Final, List, Optional, Tuple, Union
from pathlib import Path
import itertools
import os
import sys

from srctools.filesys import FileSystemChain, RawFileSystem, VPKFileSystem
from srctools.keyvalues import Keyvalues
from srctools.steam import find_app

__all__ = ['GINFO', 'Game', 'find_gameinfo']
GINFO: Final = 'gameinfo.txt'
MOUNTSKV: Final = 'cfg/mounts.kv'


class Game:
    """Represents the data in GameInfo."""
    path: Path
    game_name: Optional[str]
    app_id: Optional[str]
    tools_id: Optional[str]
    additional_content: Optional[str]
    fgd_loc: Optional[str]
    search_paths: List[Path]
    #: Mount configurations used in Strata Source.
    #: Allows loading searchpaths from other games, based on appid.
    strata_mounts: List[Keyvalues]

    def __init__(self, path: Union[str, Path], encoding: str = 'utf8') -> None:
        """Parse a game from a folder."""
        if isinstance(path, Path):
            self.path = path
        else:
            self.path = Path(path)
        with open(self.path / GINFO, encoding=encoding) as f:
            gameinfo = Keyvalues.parse(
                f,
                allow_escapes=False,  # Allow backslashes in paths.
            ).find_key('GameInfo')
        fsystems = gameinfo.find_key('Filesystem', or_blank=True)

        self.game_name = gameinfo['Game', None]
        self.app_id = fsystems['SteamAppId', None]
        self.tools_id = fsystems['ToolsAppId', None]
        self.additional_content = fsystems['AdditionalContentId', None]
        self.fgd_loc = gameinfo['GameData', None]
        self.search_paths = []
        self.strata_mounts = list(gameinfo.find_children("mount"))

        # Note: the behaviour of Source can be examined via the "path" command.
        for search_path in fsystems.find_children('SearchPaths'):
            exp_path = self.parse_search_path(search_path)
            # Expand /* if at the end of paths.
            if exp_path.name == '*':
                try:
                    self.search_paths.extend(
                        filter(Path.is_dir, exp_path.parent.iterdir())
                    )
                except FileNotFoundError:
                    pass
            # Handle folder_* too.
            elif exp_path.name.endswith('*'):
                exp_path = exp_path.with_name(exp_path.name[:-1])
                self.search_paths.extend(
                    filter(Path.is_dir, exp_path.glob(exp_path.name))
                )
            else:
                self.search_paths.append(exp_path)

        # Add DLC folders based on the first/bin folder.
        try:
            first_search = self.search_paths[0]
        except IndexError:
            pass
        else:
            folder = first_search.parent
            stem = first_search.name + '_dlc'
            for ind in itertools.count(1):
                path = folder / (stem + str(ind))
                if path.exists():
                    self.search_paths.insert(0, path)
                else:
                    break

            # Force including 'platform', for Hammer assets.
            self.search_paths.append(self.path.parent / 'platform')
            # Update goes in front of everything.
            path = folder / 'update'
            if path.exists():
                self.search_paths.insert(0, path)
        
        mounts_path = self.path / MOUNTSKV  # Process mounts.kv file if exists
        if mounts_path.is_file():
            with open(mounts_path, encoding=encoding) as m:
                mountskv = Keyvalues.parse(m)

            self.strata_mounts.extend(mountskv.find_children("Mounts"))

    @property
    def root(self) -> Path:
        """Return the game's root folder."""
        return self.path.parent

    # Strata mount compatibility, both in gameinfo and in mounts.kv
    def parse_strata_mounts(self) -> Tuple[List[Path], List[Path]]:
        """
        Parses the mounts in self.strata_mounts and returns two lists of paths.
        The first should take priority over gameinfo, the second comes after.
        """
        parsed_mounts: List[Path] = []
        parsed_mounts_heads: List[Path] = []
        
        def vpk_patch(p: Path) -> Path:
            """Determine the correct filename for a VPK."""
            v = p.with_suffix(".vpk")
            if v.exists():
                return v
            else:
                v_2 = p.parent / (p.stem + "_dir.vpk")
                if v_2.exists():
                    return v_2
                else:
                    return v

        for mount in self.strata_mounts:
            try:
                appid = int(mount.name)
            except KeyError:
                raise ValueError(f"Invalid appid declaration {mount.name}") from None
            
            target_list = parsed_mounts_heads if mount.bool("head", False) else parsed_mounts
            required = mount.bool("required", False)
            # Selects if we want to mount the "mod_folder" key as a folder or omit
            mountmoddir = mount.bool("mountmoddir", True)
            
            try:
                app_info = find_app(appid)
            except KeyError:
                if required:
                    raise ValueError(
                        f"Required mount of app-id {appid} specified, "
                        "but app is not installed!"
                    ) from None
                continue

            for child in mount:
                if child.name in ("head", "required", "mountmoddir"):
                    continue  # Ignore

                # Else we're working with a mod folder
                this_path = app_info.path / child.real_name

                for local_mount in child.iter_tree():
                    local_path = this_path / local_mount.value
                    
                    if local_mount.name == "vpk":  # We're mounting a VPK
                        target_list.append(vpk_patch(local_path))
                    elif local_mount.name == "dir":  # We're mounting a mod inside a mod
                        target_list.append(local_path)
                    else:
                        raise ValueError(f'Unknown mount type "{local_mount.real_name}"!')

                # Considered last, as per docs of strata
                if mountmoddir:
                    target_list.append(this_path)

        return parsed_mounts_heads, parsed_mounts

    def parse_search_path(self, prop: Keyvalues) -> Path:
        """Evaluate options like :code:`|gameinfo_path|`."""
        if prop.value.casefold().startswith('|gameinfo_path|'):
            return (self.path / prop.value[15:]).absolute()

        # We should have to figure out which of the possible paths this is.
        # But, the game (public/filesystem_init.cpp) doesn't actually, it
        # assumes Steam has included the needed VPKs.
        if prop.value.casefold().startswith('|all_source_engine_paths|'):
            return (self.root / prop.value[25:]).absolute()

        return (self.root / prop.value).absolute()

    def get_filesystem(self) -> FileSystemChain:
        """Build a chained filesystem from the search paths."""
        vpks = []
        raw_folders = []

        mounts_head, mounts = self.parse_strata_mounts()
        fsys = FileSystemChain()

        for path in reversed(mounts_head):  # We need to reverse the list here
            if path.suffix == ".vpk" and path.is_file():
                fsys.add_sys(VPKFileSystem(path), priority=True)
            else:
                fsys.add_sys(RawFileSystem(path), priority=True)

        for path in self.search_paths:
            if path.is_dir():
                raw_folders.append(path)
                for ind in itertools.count(1):
                    vpk = (path / f'pak{ind:02}_dir.vpk')
                    if vpk.is_file():
                        vpks.append(vpk)
                    else:
                        break
                continue

            if not path.suffix:
                path = path.with_suffix('.vpk')
            if not path.name.endswith('_dir.vpk'):
                path = path.with_name(path.name[:-4] + '_dir.vpk')

            if path.is_file() and path.suffix == '.vpk':
                vpks.append(path)

        for path in vpks:
            fsys.add_sys(VPKFileSystem(path))
        for path in raw_folders:
            fsys.add_sys(RawFileSystem(path))
        
        for path in mounts:
            if path.suffix == ".vpk" and path.is_file():
                fsys.add_sys(VPKFileSystem(path))
            else:
                fsys.add_sys(RawFileSystem(path))

        return fsys

    def bin_folder(self) -> Path:
        """Retrieve the location of the :file:`bin/` folder."""
        folder = self.path.parent / 'bin'

        # Engine branches supporting 64-bit have binaries in win64/win32
        # subfolders. So on Windows check for those.
        if sys.platform.startswith('win'):
            # "..W6432" is the key containing the REAL processor if we're
            # in WOW64 mode.
            machine = os.environ.get(
                "PROCESSOR_ARCHITEW6432",
                os.environ.get('PROCESSOR_ARCHITECTURE', '')
            )
            if machine.endswith('64'):
                bit_folder = folder / 'win64'
                if bit_folder.exists():
                    return bit_folder
            bit_folder = folder / 'win32'
            if bit_folder.is_dir():
                return bit_folder
        return folder


def find_gameinfo(argv: Optional[List[str]] = None) -> Game:
    """Locate the game we're in, if launched as a compiler.

    This checks the following:

    * :option:`!-vproject`
    * :option:`!-game`
    * The :envvar:`!VPROJECT` environment variable.
    * The current folder and all parents.
    """
    if argv is None:
        argv = sys.argv

    for i, value in enumerate(argv):
        if value.casefold() in ('-vproject', '-game'):
            try:
                path = argv[i+1]
            except IndexError:
                raise ValueError(f'"{value}" argument has no value!') from None
            if Path(path, GINFO).exists():
                return Game(path)

    # Check VPROJECT
    if 'VPROJECT' in os.environ:
        path = os.environ['VPROJECT']
        if Path(path, GINFO).exists():
            return Game(path)
    else:
        workdir = Path.cwd()
        if Path(workdir, GINFO).exists():
            return Game(workdir)

        for folder in workdir.parents:
            if Path(folder / GINFO).exists():
                return Game(folder)

    raise ValueError("Couldn't find gameinfo.txt!")
