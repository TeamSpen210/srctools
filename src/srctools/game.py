"""Reads the GameInfo file to determine where Source game data is stored."""
from typing import Final, List, Optional, Union
from pathlib import Path
import itertools
import os
import sys

from srctools.filesys import FileSystemChain, RawFileSystem, VPKFileSystem
from srctools.keyvalues import Keyvalues
from srctools.game_mount import GetAppInstallPath

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
        self.mounts: list[Keyvalues] = list(gameinfo.find_children("mount"))

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
        
        mounts_path = self.path / MOUNTSKV # Process mounts.kv file if exists
        if mounts_path.is_file():
            with open(mounts_path, encoding=encoding) as m:
                mountskv = Keyvalues.parse(m)

            mountskv = list(mountskv.find_children("Mounts"))
            self.mounts.extend(mountskv)

    @property
    def root(self) -> Path:
        """Return the game's root folder."""
        return self.path.parent
    
    def parse_mounts(self) -> tuple: # Strata mount compatibility, both in gameinfo and in mounts.kv
        """Parses the mounts in self.mounts and returns two lists of paths. One is the head list that should be merged above"""
        parsed_mounts = []
        parsed_mounts_heads = []

        def add_to_list(p, head_) -> None: # Local helper func
            if head_:
                parsed_mounts_heads.append(p)
            else:
                parsed_mounts.append(p)
        
        def vpk_patch(p: Path) -> Path: #Patches vpk paths
            v = p.with_suffix(".vpk")
            if v.exists():
                return v
            else:
                v_2 = p.parent / (p.stem + "_dir.vpk")
                if v_2.exists():
                    return v_2
                else:
                    return v


        for mount in self.mounts:
            try:
                appid = int(mount.get_key())
            except ValueError:
                raise RuntimeError(f"Invalid appid declaration {mount.get_key()}")
            
            head =          mount.bool("head", False)
            required =      mount.bool("required", False)
            mountmoddir =   mount.bool("mountmoddir", True) # Selects if we want to mount the "mod_folder" key as a folder or omit
            
            app_path: Path = GetAppInstallPath(appid) # type: ignore
            if required and app_path == None:
                raise RuntimeError(f"Required mount of app-id {appid} specified, but app is not installed!")
            
            
            for child in mount.get_child_keys():
                if child == "head" or child == "required":
                    continue #Ignore

                #Else we're working with a mod folder
                this_path = app_path / child

                for local_mount in mount.find_key(child).iter_tree():
                    local_path = local_mount.value
                    local_path = this_path / local_path
                    
                    if local_mount.get_key() == "vpk": # We're mounting a VPK
                        add_to_list(vpk_patch(local_path), head)
                        continue

                    if local_mount.get_key() == "dir": # We're mounting a mod inside a mod
                        add_to_list(local_path, head)
                    


                #Considered last, as per docs of strata
                if mountmoddir:
                    add_to_list(this_path, head)
        
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

        mounts_head, mounts = self.parse_mounts()
        fsys = FileSystemChain()


        for path in mounts_head[::-1]: # We need to reverse the list here
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
