from pathlib import Path
from sys import platform
from typing import List

from srctools import Keyvalues

if platform == "win32":
    from winreg import OpenKeyEx, QueryValueEx, HKEY_LOCAL_MACHINE, CloseKey

REG_STEAM = "SOFTWARE\\WOW6432Node\\Valve\\Steam"

GAMES_DICTIONARY = {}


def get_steam_install_path() -> Path:
    """Retrieve the installation path of Steam."""
    if platform == "linux" or platform == "linux2":
        return _get_steam_install_path_linux()
    elif platform == "win32":
        return _get_steam_install_path_win32()
    else:
        return None # type: ignore



def _get_steam_install_path_linux() -> Path:
    #TODO: Test if this works properly.
    path = "~/.local/share/Steam" # Default path, there is a 99% chance a user has installed steam there.
    path = Path(path)
    if not path.exists():
        return None  # type: ignore
    
    path = path.resolve()
    return path


def _get_steam_install_path_win32() -> Path:
    """Find Steam's installation path, using the Registry."""
    try:
        key = OpenKeyEx(HKEY_LOCAL_MACHINE, REG_STEAM) # type: ignore
        installpath = QueryValueEx(key, "InstallPath")[0] # type: ignore
        if key:
            CloseKey(key) # type: ignore
    except:
        return None # type: ignore
    
    installpath = Path(installpath)
    return installpath


def get_libraries(steam_installpath: Path) -> List[Path]:
    """Locate all Steam library folders."""
    # This file contains information on the directories where games are installed
    fpath = steam_installpath.joinpath("steamapps/libraryfolders.vdf")

    if not fpath.exists():
        return None # type: ignore

    with open(fpath, "r") as libraryfolders:
        lf = Keyvalues.parse(libraryfolders)
    
    lf = lf.find_key("libraryfolders")

    dirs = []

    for block in lf:
        dirs.append(block["path"])

    dirs = [Path(x).joinpath("steamapps") for x in dirs]  # Already convert to also include steamapps
    dirs = [x for x in dirs if x.exists()]  # Make sure this path exists

    return dirs


def _parse_acf(filepath: Path) -> tuple:
    """Internal, processes an ACF file. Returns (appid, name, installation folder name). Ensure the file exists!"""
    if not filepath.exists():
        return None  # type: ignore

    with open(filepath, "r") as acf_f:
            acf = Keyvalues.parse(acf_f)
    acf = acf.find_key("AppState")
    appid = acf.find_key("appid").value
    appid = int(appid)
    name = acf.find_key("name").value
    installdir = acf.find_key("installdir").value

    return appid, name, installdir


def build_installation_dictionary(gameinstalldirs: list, only_appid = -1) -> dict:
    """Builds and returns a dictionary, {int appid : (str app_name, Path installationPath)}"""

    games_dict = {}
    for dir_ in gameinstalldirs:
        local_path = dir_.joinpath("common") # Games are located in the common folder

        if only_appid >= 0:
            # Try to nail where the file could be
            acf_file = dir_.joinpath(f"appmanifest_{only_appid}.acf")
            if not (acf_file := _parse_acf(acf_file)):
                continue
            #Ensure the file actually exists
            appid, name, installpath = acf_file # Unpack
            installpath = local_path.joinpath(installpath)
            games_dict[appid] = (name, installpath)



        else:
            acf_files = dir_.rglob("appmanifest_*.acf") # Locate every ACF file.
            for acf_p in acf_files:

                appid, name, installpath = _parse_acf(acf_p)

                installpath = local_path.joinpath(installpath)

                games_dict[appid] = (name, installpath)
    
    return games_dict


def _ensure_dict_is_built(app_id = -1) -> None:
    """Ensures the game dictionary is ready to be delivered to the caller."""
    global GAMES_DICTIONARY

    if not app_id >= 0:
        if GAMES_DICTIONARY: # If not we need to build it
            return
    else:
        try:
            GAMES_DICTIONARY[app_id] # Does it exist?
            return # It does, return
        except:
            pass # It doesn't, find it
    
    installpath = get_steam_install_path()
    if not installpath:
        return None
    
    gameinstalldirs = get_libraries(installpath)
    if not gameinstalldirs:
        return None
    
    GAMES_DICTIONARY = build_installation_dictionary(gameinstalldirs, only_appid=app_id)


def get_apps_dictionary() -> dict:
    """Retrieve the whole dictionary of installation paths."""

    _ensure_dict_is_built()

    return GAMES_DICTIONARY


def get_app_install_path(app_id: int) -> Path:
    """Get only the installation path of a specific app-id. Returns None if app was not found."""
    _ensure_dict_is_built(app_id)
    try:
        return GAMES_DICTIONARY[app_id][1] # type: ignore
    except KeyError:
        return None # type: ignore


def get_app_name(app_id: int) -> str:
    """Get only the app name of a specific app-id. Returns None if app was not found."""
    _ensure_dict_is_built(app_id)
    try:
        return GAMES_DICTIONARY[app_id][0]
    except KeyError:
        return None  # type: ignore


def get_app(app_id: int) -> tuple:
    """Get a tuple (name, installpath) a specific app-id. Returns None if app was not found."""
    _ensure_dict_is_built(app_id)
    try:
        return GAMES_DICTIONARY[app_id]
    except KeyError:
        return None  # type: ignore
