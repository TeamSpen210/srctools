"""Reads the GameInfo file to determine where Source game data is stored."""
from pathlib import Path
import os
import sys

from srctools import Property

GINFO = 'gameinfo.txt'

class Game:
    def __init__(self, path: str):
        """Parse a game from a folder."""
        self.path = Path(path)
        with open(path / GINFO) as f:
            gameinfo = Property.parse(f).find_key('FileSystem')
        self.app_id = gameinfo['SteamAppId']
        self.tools_id = gameinfo['ToolsAppId', None]
        self.additional_content = gameinfo['AdditionalContentId', None]
        self.search_paths = []
        for path in gameinfo.find_children('SearchPaths'):
            self.search_paths.append(self.parse_search_path(path))
            
    def parse_search_path(self, path: Property) -> Path:
        """Evaluate options like |gameinfo_path|."""
        if '|' not in path:
            return Path(path).absolute()
        if path.startswith('|gameinfo_path|'):
            path = self.path / path[15:]
        if path.startswith('|all_source_engine_paths|'):
            # We have to figure out which of the possible paths this is.
            rel_path = Path(path[25:])
            
def find_gameinfo() -> Game:
    """Locate the game we're in, if launched as a a compiler.
    
    This checks the following:
    * -vproject
    * -game
    * the VPROJECT evnironment variable
    * the current folder.
    """
    path = None
    for i, value in enumerate(sys.argv):
        if value.casefold() in ('-vproject', '-game'):
            try:
                path = sys.argv[i+1]
            except IndexError:
                raise ValueError(
                    '"{}" argument 
                    'has no value!'.format(value)
                ) from None
            if Path(path, GINFO).exists():
                return Game(path)
    else:
        # Check VPROJECT
        if 'VPROJECT' in os.environ:
            path = os.environ['VPOROJECT']
            if Path(path, GINFO).exists():
                return Game(path)
        else:
            path = os.getcwd()
            if Path(path, GINFO).exists():
                return Game(path)
            else:
                raise ValueError("Couldn't find gameinfo.txt!"")
