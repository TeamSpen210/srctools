"""Parse Steam configuration files to locate apps by their ID."""
from typing import Collection, Dict, List, Mapping, Optional

from pathlib import Path
import sys

import attrs

from srctools import Keyvalues


REG_STEAM = "SOFTWARE\\WOW6432Node\\Valve\\Steam"
# Known Steam install locations.
_STEAM_LOCS = [
    Path('C:/Program Files (x86)/Steam/'),  # Win64
    Path('C:/Program Files/Steam/'),  # Win32
    Path('~/Library/Application Support/Steam/'),  # OS X
    Path('~/.local/share/Steam/'),  # Linux
    Path('~/.steam/steam/'),  # Linux, older.
]


@attrs.frozen
class AppInfo:
    """Information about a Steam app, parsed from the ACF file."""
    id: int  #: Steam appid.
    name: str  #: Display name for the app.
    path: Path  #: Install directory.

    @classmethod
    def parse(cls, folder: Path, filepath: Path) -> 'AppInfo':
        """Parse from an ACF file."""
        with open(filepath) as acf_f:
            acf = Keyvalues.parse(acf_f)
        acf = acf.find_key("AppState")

        return cls(
            acf.int("appid"),
            acf["name"],
            Path(folder, acf["installdir"])
        )

# Locations of Steam folders.
_library_folders: Optional[List[Path]] = None
# Already parsed games.
_parsed_games: Dict[int, AppInfo] = {}
_parsed_all: bool = False


def clear_caches() -> None:
    """Clear cached data, so they will be parsed again."""
    global _library_folders, _parsed_all
    _library_folders = None
    _parsed_games.clear()
    _parsed_all = False


def get_steam_install_path() -> Path:
    """Retrieve the installation path of Steam.

    :raises FileNotFoundError: If no Steam installation could be located.
    """
    if sys.platform == "win32":
        # The registry is very reliable.
        from winreg import OpenKeyEx, QueryValueEx, HKEY_LOCAL_MACHINE
        try:
            with OpenKeyEx(HKEY_LOCAL_MACHINE, REG_STEAM) as key:
                return Path(QueryValueEx(key, "InstallPath")[0])
        except OSError:
            pass  # Try default locations.
    for possible_loc in _STEAM_LOCS:
        if possible_loc.exists():
            return possible_loc
    raise FileNotFoundError("No known Steam locations for this platform.")


def get_libraries(steam_installpath: Path) -> Collection[Path]:
    """Locate all Steam library folders."""
    global _library_folders
    if _library_folders is not None:
        return _library_folders

    # This file contains information on the directories where games are installed
    fpath = steam_installpath.joinpath("steamapps/libraryfolders.vdf")

    try:
        with open(fpath) as libraryfolders:
            lf = Keyvalues.parse(libraryfolders)
    except FileNotFoundError:
        return ()
    
    lf = lf.find_key("libraryfolders")
    _library_folders = []

    for block in lf:
        path = Path(block["path"], "steamapps")
        if path.exists():
            _library_folders.append(path)
    return _library_folders


def find_all_apps() -> Mapping[int, AppInfo]:
    """Parse all apps, then return a appid -> info mapping."""
    global _parsed_all
    if _parsed_all:
        return _parsed_games

    steam_path = get_steam_install_path()
    for libfolder in get_libraries(steam_path):
        local_path = libfolder.joinpath("common")
        for acf_filename in libfolder.rglob("appmanifest_*.acf"):
            try:
                appinfo = AppInfo.parse(local_path, acf_filename)
            except FileNotFoundError:
                continue
            _parsed_games.setdefault(appinfo.id, appinfo)

    _parsed_all = True
    return _parsed_games


def find_app(app_id: int) -> AppInfo:
    """Locate information for the specified app ID.

    :raises KeyError: if the app could not be found.
    """
    try:
        return _parsed_games[app_id]
    except KeyError:
        # find_all_apps() was called, it's definitely not there.
        if _parsed_all:
            raise

    steam_path = get_steam_install_path()
    for libfolder in get_libraries(steam_path):
        local_path = libfolder.joinpath("common")
        # Try to nail where the file could be
        acf_filename = libfolder.joinpath(f"appmanifest_{app_id}.acf")
        try:
            appinfo = AppInfo.parse(local_path, acf_filename)
        except FileNotFoundError:
            continue
        _parsed_games.setdefault(appinfo.id, appinfo)
        return appinfo

    raise KeyError(app_id)
