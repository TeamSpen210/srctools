from pathlib import Path
from sys import platform
from srctools import Keyvalues

if platform == "win32":
    from winreg import OpenKeyEx, QueryValueEx, HKEY_LOCAL_MACHINE, CloseKey

REG_STEAM = "SOFTWARE\\WOW6432Node\\Valve\\Steam"

GAMES_DICTIONARY = {}


def GetSteamInstallPath() -> Path:
    """Retrieve the installation path of Steam."""
    if platform == "linux" or platform == "linux2":
        return GetSteamInstallPathLinux()
    elif platform == "win32":
        return GetSteamInstallPathWin32()
    else:
        return None # type: ignore



def GetSteamInstallPathLinux() -> Path:
    #TODO: Test if this works properly.
    path = "~/.local/share/Steam" # Default path, there is a 99% chance a user has installed steam there.
    path = Path(path)
    if not path.exists():
        return None # type: ignore
    
    path = path.resolve()
    return path



def GetSteamInstallPathWin32() -> Path:
    try:
        key = OpenKeyEx(HKEY_LOCAL_MACHINE, REG_STEAM) # type: ignore
        installpath = QueryValueEx(key, "InstallPath")[0] # type: ignore

        if key:
            CloseKey(key) # type: ignore
    except:
        return None # type: ignore
    
    installpath = Path(installpath)
    return installpath



def GetInstallationDirs(steam_installpath: Path) -> list:
    """Retrieve information about where installed Steam games are stored in."""
    fpath = steam_installpath.joinpath("steamapps/libraryfolders.vdf") # This file contains information on the directories where games are installed

    if not fpath.exists():
        return None # type: ignore

    with open(fpath, "r") as libraryfolders:
        lf = Keyvalues.parse(libraryfolders) # type: ignore
    
    lf = lf.find_key("libraryfolders")  # type: ignore
    lf: dict = lf.as_dict() # type: ignore

    dirs = []

    for _, block in lf.items():
        dirs.append(block["path"])

    dirs = [Path(x).joinpath("steamapps") for x in dirs] # Already convert to also include steamapps
    dirs = [x for x in dirs if x.exists()] # Make sure this path exists

    return dirs



def _ProcesACF(filepath: Path) -> tuple:
    """Internal, processes an ACF file. Returns (appid, name, installation folder name). Ensure the file exists!"""
    if not filepath.exists():
        return None # type: ignore
    

    with open(filepath, "r") as acf_f:
            acf = Keyvalues.parse(acf_f)
    acf = acf.find_key("AppState")
    appid = acf.find_key("appid").value
    appid = int(appid)
    name = acf.find_key("name").value
    installdir = acf.find_key("installdir").value

    return appid, name, installdir


def BuildInstallationDictionary(gameinstalldirs: list, only_appid = -1) -> dict: 
    """Builds and returns a dictionary, {int appid : (str app_name, Path installationPath)}"""


    games_dict = {}
    for dir_ in gameinstalldirs:
        local_path = dir_.joinpath("common") # Games are located in the common folder

        if only_appid >= 0:
            # Try to nail where the file could be
            acf_file = dir_.joinpath(f"appmanifest_{only_appid}.acf")
            if not (acf_file := _ProcesACF(acf_file)):
                continue
            #Ensure the file actually exists
            appid, name, installpath = acf_file # Unpack
            installpath = local_path.joinpath(installpath)
            games_dict[appid] = (name, installpath)



        else:
            acf_files = dir_.rglob("appmanifest_*.acf") # Locate every ACF file.
            for acf_p in acf_files:

                appid, name, installpath = _ProcesACF(acf_p)

                installpath = local_path.joinpath(installpath)

                games_dict[appid] = (name, installpath)
    
    return games_dict


def _EnsureDictIsBuilt(app_id = -1) -> None:
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
    
    installpath = GetSteamInstallPath()
    if not installpath:
        return None
    
    gameinstalldirs = GetInstallationDirs(installpath)
    if not gameinstalldirs:
        return None
    
    GAMES_DICTIONARY = BuildInstallationDictionary(gameinstalldirs, only_appid=app_id)




def GetAppsDictionary() -> dict:
    """Retrieve the whole dictionary of installation paths."""

    _EnsureDictIsBuilt()

    return GAMES_DICTIONARY

def GetAppInstallPath(app_id: int) -> Path:
    """Get only the installation path of a specific app-id. Returns None if app was not found."""
    _EnsureDictIsBuilt(app_id)
    try:
        return GAMES_DICTIONARY[app_id][1] # type: ignore
    except KeyError:
        return None # type: ignore


def GetAppName(app_id: int) -> str:
    """Get only the app name of a specific app-id. Returns None if app was not found."""
    _EnsureDictIsBuilt(app_id)
    try:
        return GAMES_DICTIONARY[app_id][0]
    except KeyError:
        return None # type: ignore
    
def GetApp(app_id: int) -> tuple:
    """Get a tuple (name, installpath) a specific app-id. Returns None if app was not found."""
    _EnsureDictIsBuilt(app_id)
    try:
        return GAMES_DICTIONARY[app_id]
    except KeyError:
        return None # type: ignore
