"""Parse surfaceproperties files, to determine surface physics.
"""
from typing import TYPE_CHECKING, Dict, Optional, TypeVar
from enum import Enum

import attrs

from srctools.filesys import File, FileSystem
from srctools.keyvalues import Keyvalues


__all__ = ['SurfChar', 'SurfaceProp']


class SurfChar(Enum):
    """Code classification for this material.

    This is a single ASCII character.
    """
    ANTLION = 'A'
    BLOODYFLESH = 'B'
    CONCRETE = 'C'
    DIRT = 'D'
    EGGSHELL = 'E'  #: The egg sacs in the tunnels in EP2.
    FLESH = 'F'
    GRATE = 'G'
    ALIENFLESH = 'H'
    CLIP = 'I'
    GRASS = 'J'  #: L4D addition
    #: In ASW, this is mud. In CSGO it's snow.
    MUD_ASW = SNOW = 'K'
    PLASTIC = 'L'
    METAL = 'M'
    SAND = 'N'
    FOLIAGE = 'O'
    COMPUTER = 'P'
    ASPHALT = 'Q'  #: L4D addition
    #: 2013 and P2 assigns this to reflective, brick in L4D+
    REFLECTIVE = BRICK = 'R'
    SLOSH = 'S'
    TILE = 'T'
    CARDBOARD = 'U'  #: L4D addition
    VENT = 'V'
    WOOD = 'W'
    NOFX = 'X'  #: "fake" materials use this (ladders, wading, clips, etc)
    GLASS = 'Y'
    WARPSHIELD = 'Z'  #: Weird-looking jello effect for advisor shield.

    #: L4D adds these:
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

    #: ASW, shoots out steam
    STEAM_PIPE = '11'

    #: CSGO
    SANDBARREL = '12'


T = TypeVar('T')


@attrs.define(init=False)
class SurfaceProp:
    """A material surface type."""
    name: str
    density: float = 2000
    elasticity: float = 0.25
    friction: float = 0.8
    dampening: float = 0.0
    thickness: float = -1.0

    snd_stepleft: str = "Default.StepLeft"
    snd_stepright: str = "Default.StepRight"
    snd_bulletimpact: str = "Default.BulletImpact"
    snd_scraperough: str = "Default.ScrapeRough"
    snd_scrapesmooth: str = "Default.ScrapeSmooth"
    snd_impacthard: str = "Default.ImpactHard"
    snd_impactsoft: str = "Default.ImpactSoft"
    snd_break: str = ""
    snd_strain: str = ""
    snd_shake: str = ""
    snd_roll: str = ""

    audio_reflectivity: float = 0.66
    audio_hardness_factor: float = 1.0
    audio_roughness_factor: float = 1.0

    scrape_rough_threshold: float = 0.0
    impact_hard_threshold: float = 0.5
    hard_velocity_threshold: float = 0.0

    gamematerial: SurfChar = SurfChar.CONCRETE
    jump_factor: float = 1.0
    max_speed_factor: float = 1.0
    climbable: bool = False

    if TYPE_CHECKING:
        def __init__(
            self,
            name: str,
            parent: Optional['SurfaceProp'] = None,
            *,
            density: Optional[float] = None,
            elasticity: Optional[float] = None,
            friction: Optional[float] = None,
            dampening: Optional[float] = None,
            thickness: Optional[float] = None,

            snd_stepleft: Optional[str] = None,
            snd_stepright: Optional[str] = None,
            snd_bulletimpact: Optional[str] = None,
            snd_scraperough: Optional[str] = None,
            snd_scrapesmooth: Optional[str] = None,
            snd_impacthard: Optional[str] = None,
            snd_impactsoft: Optional[str] = None,
            snd_strain: Optional[str] = None,
            snd_break: Optional[str] = None,
            snd_roll: Optional[str] = None,
            snd_shake: Optional[str] = None,

            audio_reflectivity: Optional[float] = None,
            audio_hardness_factor: Optional[float] = None,
            audio_roughness_factor: Optional[float] = None,
            scrape_rough_threshold: Optional[float] = None,
            impact_hard_threshold: Optional[float] = None,
            hard_velocity_threshold: Optional[float] = None,

            gamematerial: Optional[SurfChar] = None,
            jump_factor: Optional[float] = None,
            max_speed_factor: Optional[float] = None,
            climbable: bool = False,
        ) -> None:
            """Create a surfaceprop definition.

            If parent is passed, it will be used for any unset values.
            """
    else:
        def __init__(self, name: str, parent: Optional['SurfaceProp'] = None, **kwargs: object) -> None:
            """Create a surfaceprop definition.

            If parent is passed, it will be used for any unset values.
            """
            self.name = name
            for field in attrs.fields(type(self)):
                field_name = field.name
                if field_name == 'name':
                    continue
                value = kwargs.get(field_name)
                if value is not None:
                    if not isinstance(value, field.type):
                        # Special case, int->float is fine, everything else is not.
                        if field.type is float and isinstance(value, int):
                            value = float(value)
                        else:
                            raise TypeError(f'{field_name} must be of type {field.type.__name__}, not {value!r}')
                    setattr(self, field_name, value)
                elif parent is not None:
                    setattr(self, field_name, getattr(parent, field_name))
                else:
                    setattr(self, field_name, field.default)

    def copy(self) -> 'SurfaceProp':
        """Duplicate this surfaceprop."""
        return attrs.evolve(self)

    __copy__ = copy

    @staticmethod
    def parse_file(
        props: Keyvalues,
        prev: Optional[Dict[str, 'SurfaceProp']] = None,
    ) -> Dict[str, 'SurfaceProp']:
        """Parse surfaceproperties from a file.

        :param props: The keyvalues block to parse.
        :param prev: If passed, this is used to read parent properties from.

        A blank "default" surfaceprop  will be generated if not already present.
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
                raise ValueError(f'Missing base surface "{prop["base"]}"') from None
            except IndexError:  # Not in keyvalues.
                base = default

            game_mat: Optional[SurfChar]
            try:
                game_mat = SurfChar(prop['gamematerial'])
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
                scrape_rough_threshold=prop.float('scrapeRoughThreshold', None),
                impact_hard_threshold=prop.float('impactHardThreshold', None),
                hard_velocity_threshold=prop.float('audioHardMinVelocity', None),

                gamematerial=game_mat,
                jump_factor=prop.float('jump_factor', None),
                climbable=prop.bool('climbable', False),
            )

        return prev

    @staticmethod
    def parse_manifest(fsys: FileSystem[T], file: Optional[File[FileSystem[T]]] = None) -> Dict[str, 'SurfaceProp']:
        """Load surfaceproperties from a manifest.

        :file:`scripts/surfaceproperties_manifest.txt` will be used if a file is
        not specified.
        """
        if not file:
            file = fsys['scripts/surfaceproperties_manifest.txt']

        with file.open_str() as f:
            manifest = Keyvalues.parse(f, file.path)

        surf: Dict[str, SurfaceProp] = {}

        for prop in manifest.find_all('surfaceproperties_manifest', 'file'):
            surf = SurfaceProp.parse_file(fsys.read_kv1(prop.value), surf)

        return surf
