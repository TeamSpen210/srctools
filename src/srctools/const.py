"""Various useful constants and enums."""
from typing import Any, MutableMapping
from enum import Enum, Flag
import functools
import operator
import sys


__all__ = [
    'add_unknown',
    'GameID', 'FileType', 'SurfFlags', 'BSPContents',
]


def add_unknown(ns: MutableMapping[str, Any], long: bool = False) -> None:
    """Add dummy members for :external:class:`enum.Flag` to allow all bits to be set.

    It should be called at the end of the class body. This is useful to allow for some
    compatibility with unhandled games, ensuring extra bits are preserved when resaving.
    All existing bits will be skipped.

    :param ns: The class namespace to add members to. This should be set to \
        :external:func:`locals()` or :external:func:`vars()`.
    :param long: If set, extend to 64 bits, not 32 bits.
    """

    # Don't alias bits we already have.
    used_bits = functools.reduce(
        operator.or_,
        # Skip dunder names etc added to the namespace.
        filter(lambda n: isinstance(n, int), ns.values()),
    )
    for i in range(64 if long else 32):
        bit = 1 << i
        if not bit & used_bits:
            # We don't have to stick to var naming rules, so just name it
            # after the number. Intern so repeated calls share strings.
            ns[sys.intern(str(i))] = bit


class FileType(Enum):
    """Different kinds of files for Source, mainly to indicate resources to pack.

    If this represents a specific file type, the value is the extension. Otherwise, it's an
    arbitrary integer.
    """
    GENERIC = 0  #: Arbitrary file type.
    SOUNDSCRIPT = 1  #: Script file containing soundscripts.

    GAME_SOUND = 2  #: ``world.blah`` sound - lookup the soundscript, and raw files.
    PARTICLE = PARTICLE_SYSTEM = 3  #: The name of a particle system.

    PARTICLE_FILE = 'pcf'  #: A particle collection file.

    VSCRIPT_SQUIRREL = 'nut'  #: Squirrel VScript file.

    #: Classname of another entity that this entity includes.
    #: This is only permitted in the FGD file.
    ENTITY = 4
    #: Name of a function to call defined inside the packlist module.
    ENTCLASS_FUNC = 5

    #: Randomised generic chunk gibs.
    #: This corresponds to ``g_PropDataSystem.GetRandomChunkModel()``.
    BREAKABLE_CHUNK = 6

    #: Material file.
    MATERIAL = 'vmt'

    TEXTURE = 'vtf'  #: VTF texture, implicitly looks for ``.hdr.vtf`` too.

    CHOREO = 'vcd'  #: Choreographed scenes.

    #: Source model, implies the ``vtx``, ``vvd``, ``phy`` files too.
    MODEL = 'mdl'


class GameID(Enum):
    """Steam appIDs for Source Engine games."""
    BASE_SOURCE_ENGINE_2 = '212'
    SOURCE_BASE_2007 = SRC_2007 = '216'
    ORANGE_BOX_MULTIPLAYER = OBOX_MP = '217'
    HALF_LIFE_2 = HL2 = '220'
    COUNTER_STRIKE_SOURCE = CS = '240'
    HALF_LIFE_SOURCE = HL1 = '280'
    DAY_OF_DEFEAT_SOURCE = DOD = '300'
    ALL_SOURCE_ENGINE_PATHS_HL2 = '312'
    HALF_LIFE_2_DEATHMATCH = HL2_DM = '320'
    HALF_LIFE_2_LOST_COAST = HL2_LC = '340'
    HALF_LIFE_DEATHMATCH_SOURCE = HL_DM = '360'
    HALF_LIFE_2_EPISODE_1 = HL2_EP1 = '380'
    PORTAL = P1 = '400'
    HALF_LIFE_2_EPISODE_TWO = HL2_EP2 = '420'
    TEAM_FORTRESS_2 = TF2 = '440'
    LEFT_4_DEAD = L4D = '500'
    DOTA_2 = DOTA2 = '570'

    PORTAL_2 = P2 = '620'
    APERTURE_TAG = TAG = '280740'
    THINKING_WITH_TIME_MACHINE = TWTM = '286080'
    PORTAL_STORIES_MEL = MEL = '317400'
    REXAURA = '317790'

    ALIEN_SWARM = ASW = '630'
    COUNTER_STRIKE_GLOBAL_OFFENSIVE = CSGO = '730'
    SIN_EPISODES_EMERGENCE = '1300'
    SIN_EPISODES_ARENA = '1308'
    SIN_EPISODES_UNABRIDGED = '1316'
    DARK_MESSIAH_OF_MIGHT_AND_MAGIC = DM_MAM = '2100'
    DARK_MESSIAH_MIGHT_AND_MAGIC_MULTIPLAYER = DM_MAM_MP = '2130'
    THE_SHIP = SHIP = '2400'
    BLOODY_GOOD_TIME = BGT = '2450'
    VAMPIRE_THE_MASQUERADE_BLOODLINES = VTMB = '2600'
    GARRYS_MOD = GARRY = '4000'
    ZOMBIE_PANIC_SOURCE = ZP = '17500'
    AGE_OF_CHIVALRY = AOC = '17510'
    SYNERGY = '17520'
    DIPRIP = '17530'
    ETERNAL_SILENCE = '17550'
    PIRATES_VIKINGS_KNIGHTS_II = PVKII = '17570'
    DYSTOPIA = '17580'
    INSURGENCY = '17700'
    NUCLEAR_DAWN = '17710'
    SMASHBALL = '17730'
    INSURGENCY_2 = '222880'
    CONTAGION = '238430'


class SurfFlags(Flag):
    """The various `SURF_*`_ flags, indicating different attributes for faces.

    .. _SURF_*: https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/public/bspflags.h#L80-L97
    """
    NONE = 0
    LIGHT = 0x1
    """The face has lighting info."""
    SKYBOX_2D = 0x2
    """Nodraw, but when visible 2D skybox should be rendered."""
    SKYBOX_3D = 0x4
    """Nodraw, but when visible 2D and 3D skybox should be rendered."""
    WATER_WARP = 0x8
    """According to the header, "turbulent water warp"."""
    TRANSLUCENT = 0x10
    """Translucent material."""
    NOPORTAL = 0x20
    """Portalgun blocking material."""
    TRIGGER = 0x40
    """XBox only - is a trigger surface."""
    NODRAW = 0x80
    """Texture isn't used, it's invisible."""
    HINT = 0x100
    """A hint brush."""
    SKIP = 0x200
    """Skip brush, removed from map."""
    NOLIGHT = 0x400
    """No light needs to be calculated for the surface."""
    BUMPLIGHT = 0x800
    """Needs three lightmaps for bumpmapping."""
    NO_SHADOWS = 0x1000
    """Doesn't receive shadows."""
    NO_DECALS = 0x2000
    """Rejects decals."""
    NO_SUBDIVIDE = 0x4000
    """Not allowed to split up the brush face."""
    HITBOX = 0x8000
    """According to the header, "Part of a hitbox"."""

    add_unknown(locals())


class BSPContents(Flag):
    """The various `CONTENTS_*`_ flags, indicating different collision types.

    This is normally for brushes, but is also used on other things like models.

    .. _CONTENTS_*: https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/public/bspflags.h#L24-L76
    """
    EMPTY = 0
    SOLID = 0x1
    """Regular solid brush. Player camera is not valid inside here."""
    WINDOW = 0x2
    """Translucent glass."""
    AUX = 0x4
    GRATE = 0x8
    """Grating, bullets/LOS pass, objects do not."""
    SLIME = 0x10
    """Slime-style liquid, opaque."""
    WATER = 0x20
    """Is a water brush"""
    MIST = 0x40
    OPAQUE = 0x80
    """Blocks LOS"""
    TEST_FOG_VOLUME = 0x100
    """May be non-solid, but cannot be seen through."""
    TEAM1 = 0x800
    """Special team-only clips."""
    TEAM2 = 0x1000
    """Special team-only clips."""
    IGNORE_NODRAW_OPAQUE = 0x2000
    """"Ignore ``CONTENTS_OPAQUE`` on surfaces that have ``SURF_NODRAW``."""
    MOVABLE = 0x4000

    AREAPORTAL = 0x8000
    """Is an areaportal brush."""
    PLAYER_CLIP = 0x10000
    """Functions like ``tools/toolsplayerclip``."""
    NPC_CLIP = 0x20000
    """Functions like ``tools/toolsclip``."""

    CURRENT_0 = 0x40000
    """Specifies water currents, can be mixed."""
    CURRENT_90 = 0x80000
    CURRENT_180 = 0x100000
    CURRENT_270 = 0x200000
    CURRENT_UP = 0x400000
    CURRENT_DOWN = 0x800000

    ORIGIN = 0x1000000
    """Behaves like a ``tools/toolsorigin`` brush, used to set origin. 
    This is Goldsource era and is very deprecated.
    """
    NPC = 0x2000000
    """This shouldn't be on brushes, it's used for NPCs."""
    DEBRIS = 0x4000000
    DETAIL = 0x8000000
    """The brush is marked as ``func_detail``."""
    TRANSLUCENT = 0x10000000
    """A face uses ``$translucent``, ``$alphatest``, ``$alpha``, etc."""
    LADDER = 0x20000000
    HITBOX = 0x40000000

    add_unknown(locals())
