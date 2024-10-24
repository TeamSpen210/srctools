from typing import Dict, Iterable, Iterator, Mapping

from pathlib import Path
import sys

import attrs

from srctools import Keyvalues


REG_STEAM = "SOFTWARE\\WOW6432Node\\Valve\\Steam"


@attrs.frozen
class AppInfo:
    """Information about a Steam app, parsed from the ACF file."""
    id: int  #: Steam appid
    name: str
    path: Path  #: Install directory.

    @classmethod
    def parse(cls, folder: Path, filepath: Path) -> 'AppInfo':
        """Parse from an ACF file."""
        with open(filepath, "r") as acf_f:
            acf = Keyvalues.parse(acf_f)
        acf = acf.find_key("AppState")

        return cls(
            acf.int("appid"),
            acf["name"],
            Path(folder, acf["installdir"])
        )

GAMES_DICTIONARY: Dict[int, AppInfo] = {}


def get_steam_install_path() -> Path:
    """Retrieve the installation path of Steam.

    :raises FileNotFoundError: If no Steam installation could be located.
    """
    if sys.platform == "linux" or sys.platform == "linux2":
        return _get_steam_install_path_linux()
    elif sys.platform == "win32":
        return _get_steam_install_path_win32()
    else:
        raise FileNotFoundError("No known Steam locations for this platform.")


def _get_steam_install_path_linux() -> Path:
    # TODO: Test if this works properly.
    # Default path, there is a 99% chance a user has installed steam there.
    path = Path("~/.local/share/Steam").resolve()
    if not path.exists():
        raise FileNotFoundError("Steam not found in default location.")
    return path


def _get_steam_install_path_win32() -> Path:
    """Find Steam's installation path, using the Registry."""
    assert sys.platform == 'win32'
    from winreg import OpenKeyEx, QueryValueEx, HKEY_LOCAL_MACHINE
    try:
        with OpenKeyEx(HKEY_LOCAL_MACHINE, REG_STEAM) as key:
            installpath = QueryValueEx(key, "InstallPath")[0]
    except OSError:
        raise FileNotFoundError("Steam not present in registry.")

    return Path(installpath)


def get_libraries(steam_installpath: Path) -> Iterator[Path]:
    """Locate all Steam library folders."""
    # This file contains information on the directories where games are installed
    fpath = steam_installpath.joinpath("steamapps/libraryfolders.vdf")

    if not fpath.exists():
        return None # type: ignore

    with open(fpath, "r") as libraryfolders:
        lf = Keyvalues.parse(libraryfolders)
    
    lf = lf.find_key("libraryfolders")

    for block in lf:
        path = Path(block["path"], "steamapps")
        if path.exists():
            yield path


def build_installation_dictionary(library_folders: Iterable[Path], only_appid: int=-1) -> dict:
    """Builds and returns a dictionary, {int appid : (str app_name, Path installationPath)}"""

    games_dict = {}
    for libfolder in library_folders:
        local_path = libfolder.joinpath("common")  # Games are located in the common folder

        if only_appid >= 0:
            # Try to nail where the file could be
            acf_filename = libfolder.joinpath(f"appmanifest_{only_appid}.acf")
            try:
                appinfo = AppInfo.parse(local_path, acf_filename)
            except FileNotFoundError:
                continue
            games_dict[appinfo.id] = appinfo

        else:
            # Locate every ACF file.
            for acf_filename in libfolder.rglob("appmanifest_*.acf"):
                try:
                    appinfo = AppInfo.parse(local_path, acf_filename)
                except FileNotFoundError:
                    continue
                games_dict[appinfo.id] = appinfo
    
    return games_dict


def _ensure_dict_is_built(app_id: int=-1) -> None:
    """Ensures the game dictionary is ready to be delivered to the caller."""
    global GAMES_DICTIONARY

    if app_id <= 0:
        if GAMES_DICTIONARY:  # If not we need to build it
            return
    else:
        if app_id in GAMES_DICTIONARY:
            return
    
    installpath = get_steam_install_path()
    
    GAMES_DICTIONARY = build_installation_dictionary(get_libraries(installpath), only_appid=app_id)


def find_all_apps() -> Mapping[int, AppInfo]:
    """Parse all apps, then return a appid -> info mapping."""
    _ensure_dict_is_built()
    return GAMES_DICTIONARY


def find_app(app_id: int) -> AppInfo:
    """Locate information for the specified app ID.

    :raises KeyError: if the app could not be found.
    """
    _ensure_dict_is_built(app_id)
    return GAMES_DICTIONARY[app_id]  # Or keyError
