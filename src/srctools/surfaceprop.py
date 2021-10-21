"""Parse surfaceproperties files, to determine surface physics.
"""
from srctools.property_parser import Property
from srctools.filesys import FileSystem, File

from typing import Optional, Dict, TypeVar
from enum import Enum


__all__ = ['SurfChar', 'SurfaceProp']


class SurfChar(Enum):
    """Code classification for this material.

    This is a single ASCII character.
    """
    ANTLION = 'A'
    BLOODYFLESH = 'B'
    CONCRETE = 'C'
    DIRT = 'D'
    EGGSHELL = 'E'  # The egg sacs in the tunnels in EP2.
    FLESH = 'F'
    GRATE = 'G'
    ALIENFLESH = 'H'
    CLIP = 'I'
    GRASS = 'J'  # L4D addition
    # In ASW, this is mud. By CSGO it's snow.
    MUD_ASW = SNOW = 'K'
    PLASTIC = 'L'
    METAL = 'M'
    SAND = 'N'
    FOLIAGE = 'O'
    COMPUTER = 'P'
    ASPHALT = 'Q'  # L4D addition
    # 2013 and P2 assigns this to reflective, brick in L4D+
    REFLECTIVE = BRICK = 'R'
    SLOSH = 'S'
    TILE = 'T'
    CARDBOARD = 'U'  # L4D addition
    VENT = 'V'
    WOOD = 'W'
    NOFX = 'X'  # "fake" materials use this (ladders, wading, clips, etc)
    GLASS = 'Y'
    WARPSHIELD = 'Z'  # Weird-looking jello effect for advisor shield.

    # L4D adds these:
    CLAY = '1'
    PLASTER = '2'
    ROCK = '3'
    RUBBER = '4'
    SHEETROCK = '5'
    CLOTH = '6'
    CARPET = '7'
    PAPER = '8'
    UPHOLSTERY = '9'
    PUDDLE = '10'
    MUD_L4D = '11'

    # ASW, shoots out steam
    STEAM_PIPE = '11'

    # CSGO
    SANDBARREL = '12'

_InitArgT = TypeVar('_InitArgT', float, str, bool, SurfChar)


def _attr_value(
    parent: Optional['SurfaceProp'],
    name: str,
    arg: Optional[_InitArgT],
    default: _InitArgT,
) -> _InitArgT:
    """Internal function for initialising."""
    if arg is not None:
        return arg
    if parent:
        return getattr(parent, name)
    else:
        return default


class SurfaceProp:
    """A material surface type."""

    __slots__ = [
        'name',
        'density',
        'elasticity',
        'friction',
        'dampening',
        'thickness',
        'snd_stepleft',
        'snd_stepright',
        'snd_bulletimpact',
        'snd_scraperough',
        'snd_scrapesmooth',
        'snd_impacthard',
        'snd_impactsoft',
        'snd_break',
        'snd_strain',
        'snd_roll',
        'snd_shake',
        'audio_reflectivity',
        'audio_hardness_factor',
        'audio_roughness_factor',
        'scrape_rough_threshold',
        'impact_hard_threshold',
        'hard_velocity_threshold',
        'gamematerial',
        'jump_factor',
        'max_speed_factor',
        'climbable',
    ]

    def __init__(
        self,
        name: str,
        parent: 'SurfaceProp'=None,
        *,
        density: float=None,
        elasticity: float=None,
        friction: float=None,
        dampening: float=None,
        thickness: float=None,

        snd_stepleft: str=None,
        snd_stepright: str=None,
        snd_bulletimpact: str=None,
        snd_scraperough: str=None,
        snd_scrapesmooth: str=None,
        snd_impacthard: str=None,
        snd_impactsoft: str=None,
        snd_strain: str=None,
        snd_break: str=None,
        snd_roll: str=None,
        snd_shake: str=None,

        audio_reflectivity: float=None,
        audio_hardness_factor: float=None,
        audio_roughness_factor: float=None,
        scrape_rough_thres: float=None,
        impact_hard_thres: float=None,
        hard_velocity_thres: float=None,

        gamemat: SurfChar=None,
        jump_factor: float=None,
        max_speed_factor: float=None,
        climbable: bool=False,
    ) -> None:
        """Create a surfaceprop definition.

        If parent is passed, it will be used for any unset values.
        """
        self.name = name

        self.density = _attr_value(parent, 'density', density, 2000)
        self.elasticity = _attr_value(parent, 'elasticity', elasticity, 0.25)
        self.friction = _attr_value(parent, 'friction', friction, 0.8)
        self.dampening = _attr_value(parent, 'dampening', dampening, 0.0)
        self.thickness = _attr_value(parent, 'thickness', thickness, -1.0)

        self.snd_stepleft = _attr_value(parent, 'snd_stepleft', snd_stepleft, "Default.StepLeft")
        self.snd_stepright = _attr_value(parent, 'snd_stepright', snd_stepright, "Default.StepRight")
        self.snd_bulletimpact = _attr_value(parent, 'snd_bulletimpact', snd_bulletimpact, "Default.BulletImpact")
        self.snd_scraperough = _attr_value(parent, 'snd_scraperough', snd_scraperough, "Default.ScrapeRough")
        self.snd_scrapesmooth = _attr_value(parent, 'snd_scrapesmooth', snd_scrapesmooth, "Default.ScrapeSmooth")
        self.snd_impacthard = _attr_value(parent, 'snd_impacthard', snd_impacthard, "Default.ImpactHard")
        self.snd_impactsoft = _attr_value(parent, 'snd_impactsoft', snd_impactsoft, "Default.ImpactSoft")
        self.snd_break = _attr_value(parent, 'snd_break', snd_break, "")
        self.snd_strain = _attr_value(parent, 'snd_strain', snd_strain, "")
        self.snd_shake = _attr_value(parent, 'snd_shake', snd_shake, "")
        self.snd_roll = _attr_value(parent, 'snd_roll', snd_roll, "")

        self.audio_reflectivity = _attr_value(parent, 'audio_reflectivity', audio_reflectivity, 0.66)
        self.audio_hardness_factor = _attr_value(parent, 'audio_hardness_factor', audio_hardness_factor, 1.0)
        self.audio_roughness_factor = _attr_value(parent, 'audio_roughness_factor', audio_roughness_factor, 1.0)

        self.scrape_rough_threshold = _attr_value(parent, 'scrape_rough_threshold', scrape_rough_thres, 0.)
        self.impact_hard_threshold = _attr_value(parent, 'impact_hard_threshold', impact_hard_thres, 0.5)
        self.hard_velocity_threshold = _attr_value(parent, 'hard_velocity_threshold', hard_velocity_thres, 0)

        self.gamematerial = _attr_value(parent, 'gamematerial', gamemat, SurfChar.CONCRETE)
        self.jump_factor = _attr_value(parent, 'jump_factor', jump_factor, 1.0)
        self.max_speed_factor = _attr_value(parent, 'max_speed_factor', max_speed_factor, 1.0)
        self.climbable = _attr_value(parent, 'climbable', climbable, False)

    def __repr__(self) -> str:
        return '<SurfaceProp "{}", char={}>'.format(self.name, self.gamematerial.value)

    def copy(self) -> 'SurfaceProp':
        """Duplicate this surfaceprop."""
        return SurfaceProp(
            self.name,

            density=self.density,
            elasticity=self.elasticity,
            friction=self.friction,
            dampening=self.dampening,
            thickness=self.thickness,

            snd_stepleft=self.snd_stepleft,
            snd_stepright=self.snd_stepright,
            snd_bulletimpact=self.snd_bulletimpact,
            snd_scraperough=self.snd_scraperough,
            snd_scrapesmooth=self.snd_scrapesmooth,
            snd_impacthard=self.snd_impacthard,
            snd_impactsoft=self.snd_impactsoft,
            snd_break=self.snd_break,
            snd_strain=self.snd_strain,
            snd_roll=self.snd_roll,
            snd_shake=self.snd_shake,

            audio_reflectivity=self.audio_reflectivity,
            audio_hardness_factor=self.audio_hardness_factor,
            audio_roughness_factor=self.audio_roughness_factor,
            scrape_rough_thres=self.scrape_rough_threshold,
            impact_hard_thres=self.impact_hard_threshold,
            hard_velocity_thres=self.hard_velocity_threshold,

            gamemat=self.gamematerial,
            jump_factor=self.jump_factor,
            max_speed_factor=self.max_speed_factor,
            climbable=self.climbable,
        )

    __copy__ = copy

    @staticmethod
    def parse_file(
        props: Property,
        prev: Dict[str, 'SurfaceProp']=None,
    ) -> Dict[str, 'SurfaceProp']:
        """Parse surfaceproperties from a file.

        prev if passed is used to read parent properties from.

        DEFAULT will be inserted as the "default" surfaceprop if not provided
        in prev.
        """
        if prev is None:
            prev = {}

        try:
            default = prev['default']
        except KeyError:
            default = prev['default'] = SurfaceProp('default')

        for prop in props:
            try:
                base = prev[prop['base'].casefold()]
            except KeyError:
                raise ValueError(
                    'Missing base surface "{}"'.format(prop['base'])
                )
            except IndexError:
                base = default

            try:
                game_mat = SurfChar(prop['gamematerial'])  # type: Optional[SurfChar]
            except (LookupError, ValueError):
                game_mat = None

            prev[prop.name] = SurfaceProp(
                prop.real_name or '',
                base,
                density=prop.float('density', None),
                elasticity=prop.float('elasticity', None),
                friction=prop.float('friction', None),
                dampening=prop.float('dampening', None),

                snd_stepleft=prop['stepleft', None],
                snd_stepright=prop['stepright', None],
                snd_bulletimpact=prop['bulletimpact', None],
                snd_scraperough=prop['scraperough', None],
                snd_scrapesmooth=prop['scrapesmooth', None],
                snd_impacthard=prop['impacthard', None],
                snd_impactsoft=prop['impactsoft', None],
                snd_strain=prop['strain', None],
                snd_break=prop['break', None],
                snd_roll=prop['roll', None],

                audio_reflectivity=prop.float('audioreflectivity', None),
                audio_hardness_factor=prop.float('audiohardnessfactor', None),
                audio_roughness_factor=prop.float('audioroughnessfactor', None),
                scrape_rough_thres=prop.float('scrapeRoughThreshold', None),
                impact_hard_thres=prop.float('impactHardThreshold', None),
                hard_velocity_thres=prop.float('audioHardMinVelocity', None),

                gamemat=game_mat,
                jump_factor=prop.float('jump_factor', None),
                climbable=prop.bool('climbable', None),
            )

        return prev

    @staticmethod
    def parse_manifest(fsys: FileSystem, file: File=None) -> Dict[str, 'SurfaceProp']:
        """Load surfaceproperties from a manifest.

        "scripts/surfaceproperties_manifest.txt" will be used if a file is
        not specified.
        """
        if not file:
            file = fsys['scripts/surfaceproperties_manifest.txt']

        with file.open_str() as f:
            manifest = Property.parse(f, file.path)

        surf: dict[str, SurfaceProp] = {}

        for prop in manifest.find_all('surfaceproperties_manifest', 'file'):
            surf = SurfaceProp.parse_file(fsys.read_prop(prop.value), surf)

        return surf
