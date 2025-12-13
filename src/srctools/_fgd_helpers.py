"""Implementations of specific code for each FGD helper type."""
from typing import ClassVar, Optional, Union, Literal
from typing_extensions import Self, deprecated
from collections.abc import Collection, Iterable, Iterator, Mapping

import attrs

from srctools import conv_float
from srctools.fgd import EntityDef, Helper, HelperTypes, TagsSet
from srctools.math import Vec, format_float


__all__ = [
    'HelperAxis', 'HelperBBox', 'HelperBoundingBox', 'HelperBreakableSurf',
    'HelperBrushSides', 'HelperCylinder', 'HelperDecal',
    'HelperEnvSprite', 'HelperFrustum', 'HelperHalfGridSnap',
    'HelperInherit', 'HelperInstance', 'HelperLight', 'HelperLightSpot',
    'HelperLine', 'HelperModel', 'HelperModelLight', 'HelperModelProp',
    'HelperOrientedBBox', 'HelperOrigin', 'HelperOverlay', 
    'HelperOverlayTransition', 'HelperRenderColor', 'HelperRope',
    'HelperSize', 'HelperSphere', 'HelperSprite',
    'HelperSweptPlayerHull', 'HelperTrack', 'HelperTypes',
    'HelperVecLine', 'HelperWorldText',

    'HelperExtAppliesTo', 'HelperExtAutoVisgroups', 'HelperExtOrderBy',
    'HelperExtNoInherit',
    'HelperExtCatapult', 'HelperExtStrataFlatOBB', 'HelperExtStrataHalfFlatOBB',
    'HelperExtStrataClusteredLight',
]


def parse_byte(text: Union[int, str]) -> int:
    """Parse an single byte."""
    try:
        return max(0, min(255, int(text)))
    except (TypeError, ValueError):
        return 255


def repr_float(value: object) -> str:
    """Strip .0 from floats, but keep everything else."""
    return format_float(value) if isinstance(value, float) else repr(value)


@attrs.define
class _HelperOneOptional(Helper):
    """Utility base class for a helper with one optional parameter."""
    _DEFAULT: ClassVar[str] = ''
    key: str

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse a single optional keyl."""
        if len(args) > 1:
            raise ValueError(
                'Expected up to 1 argument, got ({})!'.format(', '.join(args))
            )
        elif len(args) == 1:
            key = args[0].strip('"')
        else:
            key = cls._DEFAULT
        return cls(key, tags=tags)

    def export(self) -> list[str]:
        """Export the helper.

        If the key is the default it is omitted.
        """
        if self.key == self._DEFAULT:
            return []
        else:
            return [self.key]


class HelperInherit(Helper):
    """Helper implementation for base().

    These specify the base classes for an entity def.
    This implementation isn't used, the EntityDef special-cases it.
    """
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.INHERIT


@attrs.define
class HelperHalfGridSnap(Helper):
    """Helper implementation for halfgridsnap().

    This causes the entity to snap to half a grid.
    This argument doesn't use () in Valve's files.
    """
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.HALF_GRID_SNAP


@attrs.define(init=False)
class HelperSize(Helper):
    """Helper implementation for size().

    This sets the bbox for the entity.
    """
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.CUBE
    bbox_min: Vec
    bbox_max: Vec

    def __init__(self, point1: Vec, point2: Vec, tags: TagsSet = frozenset()) -> None:
        super().__init__(tags=tags)
        self.bbox_min, self.bbox_max = Vec.bbox(point1, point2)

    def overrides(self) -> Collection[HelperTypes]:
        """Additional versions of this are not available."""
        return [HelperTypes.CUBE]

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse size(x1 y1 z1, x2 y2 z2)."""
        if len(args) not in (3, 6):
            raise ValueError(
                'Expected 3 or 6 arguments, got ({})!'.format(', '.join(args))
            )
        size_min = Vec(conv_float(args[0]), conv_float(args[1]), conv_float(args[2]))
        if len(args) == 6:
            size_max = Vec(conv_float(args[3]), conv_float(args[4]), conv_float(args[5]))
        else:
            # "min" is actually the dimensions.
            size_max = size_min / 2
            size_min = -size_max

        return cls(size_min, size_max, tags)

    def export(self) -> list[str]:
        """Produce (x1 y1 z1, x2 y2 z2)."""
        return [
            str(self.bbox_min),
            str(self.bbox_max),
        ]


@deprecated("Does not exist, use HelperSize = size() instead.")
class HelperBBox(HelperSize):
    """Deprecated, implementation for the non-existent bbox() type. Use size() instead."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.BBOX


@attrs.define
class HelperCatapult(Helper):
    """Helper implementation for catapult(), specific to Portal: Revolution."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_CATAPULT


@attrs.define
class HelperRenderColor(Helper):
    """Helper implementation for color()."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.TINT

    r: int
    g: int
    b: int

    def overrides(self) -> Collection[HelperTypes]:
        """Previous ones of these are overridden by this."""
        return [HelperTypes.TINT]

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse color(R G B)."""
        try:
            [r, g, b] = args
        except ValueError:
            raise ValueError(
                f'Expected 3 arguments, got ({", ".join(args)})!'
            ) from None

        return cls(parse_byte(r), parse_byte(g), parse_byte(b), tags=tags)

    def export(self) -> list[str]:
        """Produce color(R G B)."""
        return [f'{self.r:g} {self.g:g} {self.b:g}']


@attrs.define
class HelperSphere(Helper):
    """Helper implementation for sphere()."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.SPHERE
    r: int
    g: int
    b: int
    size_key: str

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse sphere(radius, r g b)."""
        arg_count = len(args)
        if arg_count not in (0, 1, 4):
            raise ValueError(f'Expected 0, 1 or 4 arguments, got {args})!')
        r = g = b = 255

        if arg_count > 0:
            size_key = args[0]
            if arg_count == 4:
                r = parse_byte(args[1])
                g = parse_byte(args[2])
                b = parse_byte(args[3])
        else:
            size_key = 'radius'

        return cls(r, g, b, size_key, tags=tags)

    def export(self) -> list[str]:
        """Export the helper."""
        if self.r != 255 or self.g != 255 or self.b != 255:
            return [self.size_key, f'{self.r:g} {self.g:g} {self.b:g}']
        # Always explicitly pass radius. If we use the default value,
        # Hammer doesn't display the "camera" button in options to set
        # the value to the distance to the entity.
        return [self.size_key]


@attrs.define
class HelperLine(Helper):
    """Helper implementation for line().

    Line has the arguments line(r g b, start_key, start_value, end_key, end_value)
    It searches for the first entity where ent[start_key] == self[start_value].
    If the second pair are present it does the same for those for the other
    line end.
    """
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.LINE

    r: int
    g: int
    b: int
    start_key: str
    start_value: str
    end_key: Optional[str] = None
    end_value: Optional[str] = None

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse line(r g b, start_key, start_value, end_key, end_value)."""
        arg_count = len(args)
        if arg_count not in (5, 7):
            raise ValueError(f'Expected 5 or 7 arguments, got {args!r}!')

        r = parse_byte(args[0])
        g = parse_byte(args[1])
        b = parse_byte(args[2])
        start_key = args[3]
        start_value = args[4]

        end_key: Optional[str]
        end_value: Optional[str]

        if arg_count == 7:
            end_key = args[5]
            end_value = args[6]
        else:
            end_key = end_value = None

        return cls(r, g, b, start_key, start_value, end_key, end_value, tags=tags)

    def export(self) -> list[str]:
        """Produce the correct line() arguments."""
        args = [
            f'{self.r:g} {self.g:g} {self.b:g}',
            self.start_key,
            self.start_value,
        ]
        if self.end_key is not None and self.end_value is not None:
            args += [self.end_key, self.end_value]
        return args


@attrs.define
class HelperFrustum(Helper):
    """Helper for env_projectedtexture visuals.

    As an extension, values can be literals as well as key names. This is only natively supported
    by Vitamin Source, but HammerAddons will generate keyvalues to make this work.
    """
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.FRUSTUM

    fov: Union[str, float] = attrs.field(repr=repr_float)
    near_z: Union[str, float] = attrs.field(repr=repr_float)
    far_z: Union[str, float] = attrs.field(repr=repr_float)
    color: Union[str, tuple[int, int, int]]
    pitch_scale: float = 1.0

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse frustum(fov, near, far, color, pitch_scale)."""
        # These are the default values if not provided.
        fov: Union[str, float] = '_fov'
        nearz: Union[str, float] = '_nearplane'
        farz: Union[str, float] = '_farplane'
        color: Union[str, tuple[int, int, int]] = '_light'
        pitch = 1.0

        # All values are optional.
        # Colour can be 1 key or 3 literal, Fortunately this is unambiguous.
        # (fov, near, far, color, pitch) == 1 - 5
        # (fov, near, far, r, g, b) == 6
        # (fov, near, far, r, g, b, pitch) == 7
        try:
            fov = args[0]
            nearz = args[1]
            farz = args[2]
        except IndexError:
            pass  # Stop once out of args.
        else:
            # Handle colour/pitch.
            count = len(args)
            if count in (4, 5):
                color = args[3]
            if count in (6, 7):  # Literal colour.
                color = parse_byte(args[3]), parse_byte(args[4]), parse_byte(args[5])
            if count in (5, 7):  # We have pitch.
                pitch = conv_float(args[-1], 1.0)
            if count > 7:
                raise ValueError(f'Expected 1-7 arguments, got {args!r}!')

        # Try and parse everything else, but if it fails ignore since they could
        # be keyvalue names.
        try:
            fov = float(fov)
        except ValueError:
            pass
        try:
            nearz = float(nearz)
        except ValueError:
            pass
        try:
            farz = float(farz)
        except ValueError:
            pass

        return cls(fov, nearz, farz, color, pitch, tags=tags)

    def export(self) -> list[str]:
        """Export back out frustrum() arguments."""
        if isinstance(self.color, tuple):
            color = '{:g} {:g} {:g}'.format(*self.color)
        else:
            color = self.color

        def conv(x: 'Union[str, float]') -> str:
            """Ensure the .0 is removed from the float forms. """
            return x if isinstance(x, str) else format_float(x)

        result = [
            conv(self.fov),
            conv(self.near_z),
            conv(self.far_z),
            color,
            format_float(self.pitch_scale)
        ]
        # Try and remove redundant defaults.
        for default in ['1', '_light', '_farplane', '_nearplane', '_fov']:
            if result[-1] == default:
                result.pop()
            else:
                return result
        return result


@attrs.define
class HelperCylinder(HelperLine):
    """Helper implementation for cylinder().

    Cylinder has the same sort of arguments as line(), plus radii for both positions.
    """
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.CYLINDER

    start_radius: Optional[str] = None
    end_radius: Optional[str] = None

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse cylinder(r g b, start key/value/radius, end key/value/radius)."""
        arg_count = len(args)
        # Don't allow no keys (3), start key but no value (4), or end key but no value (7)
        if arg_count not in (5, 6, 8, 9):
            raise ValueError(f'Expected 5, 6, 8 or 9 arguments, got {args!r}!')

        r = parse_byte(args[0])
        g = parse_byte(args[1])
        b = parse_byte(args[2])
        start_key = args[3]
        start_value = args[4]

        start_radius = end_key = end_value = end_radius = None
        try:
            start_radius = args[5]
            end_key = args[6]
            end_value = args[7]
            end_radius = args[8]
        except IndexError:
            pass  # Use defaults.

        return cls(
            r, g, b,
            start_key, start_value,
            end_key, end_value,
            start_radius, end_radius,
            tags=tags,
        )

    def export(self) -> list[str]:
        """Produce the correct line() arguments."""
        args = [
            f'{self.r:g} {self.g:g} {self.b:g}',
            self.start_key,
            self.start_value,
        ]
        if self.start_radius is not None:
            args.append(self.start_radius)
            if self.end_key is not None and self.end_value is not None:
                args += [self.end_key, self.end_value]
                if self.end_radius is not None:
                    args.append(self.end_radius)
        return args


class HelperOrigin(_HelperOneOptional):
    """Parse the origin() helper."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ORIGIN
    _DEFAULT = 'origin'


class HelperVecLine(_HelperOneOptional):
    """A variant of line() which draws a line to the entity."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.VECLINE
    _DEFAULT = 'origin'


class HelperAxis(_HelperOneOptional):
    """A helper  which draws a line between two points."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.AXIS
    _DEFAULT = 'axis'


class HelperBrushSides(_HelperOneOptional):
    """Highlights brush faces in a space-sepearated keyvalue."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.BRUSH_SIDES
    _DEFAULT = 'sides'


@attrs.define
class HelperBoundingBox(Helper):
    """Displays bounding box between two keyvalues."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.BOUNDING_BOX_HELPER

    min: str
    max: str

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse wirebox(min, max)"""
        try:
            [key_min, key_max] = args
        except ValueError:
            raise ValueError(f'Expected 2 arguments, got {args!r}!') from None

        return cls(key_min, key_max, tags=tags)

    def export(self) -> list[str]:
        """Produce the wirebox(min, max) arguments."""
        return [self.min, self.max]


class HelperSweptPlayerHull(Helper):
    """Draws the movement of a player-sized bounding box from A to B."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.SWEPT_HULL


class HelperOrientedBBox(HelperBoundingBox):
    """A bounding box oriented to angles."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ORIENTED_BBOX


@attrs.define
class HelperSprite(Helper):
    """The sprite helper, for editor icons.

    If the material is not provided, the 'model' key is used.
    """
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.SPRITE

    mat: Optional[str]

    def overrides(self) -> Collection[HelperTypes]:
        """Specifies helpers which this makes irrelevant."""
        if self.mat is None:
            return ()  # When not set, this doesn't affect anything.
        else:
            # This doesn't override either of these,
            # but if you have two sprites it's pointless.
            # And so is a box + sprite.
            return [HelperTypes.CUBE, HelperTypes.SPRITE, HelperTypes.ENT_SPRITE]

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse iconsprite(mat)."""
        if len(args) > 1:
            raise ValueError(f'Expected up to 1 argument, got {args!r}!')
        elif len(args) == 1:
            return cls(args[0], tags=tags)
        else:
            return cls(None, tags=tags)

    def export(self) -> list[str]:
        """Produce the arguments for iconsprite()."""
        if self.mat is not None:
            # / characters etc require quotes.
            return [f'"{self.mat}"']
        else:
            return []

    def get_resources(self, entity: EntityDef) -> Iterator[str]:
        """iconsprite() uses a single material."""
        materials: Iterable[str]
        if self.mat is None:
            try:
                materials = entity.kv['model'].known_options()
            except KeyError:
                return
        else:
            materials = [self.mat]

        for material in materials:
            material = material.replace('\\', '/')

            # Strip leading and trailing quotation marks if both present
            if material.startswith('"') and material.endswith('"'):
                material = material[1:-1]

            # Don't append if the string is just empty!
            if len(material) == 0:
                continue

            if not material.casefold().endswith('.vmt'):
                material += '.vmt'
            if not material.casefold().startswith('materials/'):
                material = 'materials/' + material

            yield material


class HelperEnvSprite(HelperSprite):
    """Variant of iconsprite() specifically for env_sprite."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_SPRITE


@attrs.define
class HelperModel(Helper):
    """Helper which displays models.

    If the model is not provided, the 'model' key is used.
    """
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.MODEL
    model: Optional[str]

    def overrides(self) -> Collection[HelperTypes]:
        """Avoid some issues where other helpers break this one."""
        # If multiple model helpers are provided, rotating the entity in the views causes Hammer
        # to apply the transform multiple times to `angles` which is very undesirable.
        if self.model is None:
            # If no model is provided, line() and similar helpers make
            # the default cube size break.
            return [HelperTypes.LINE, HelperTypes.MODEL, HelperTypes.MODEL_NEG_PITCH, HelperTypes.MODEL_PROP]
        else:
            # If we provide a model, CUBE is useless.
            return [HelperTypes.CUBE, HelperTypes.MODEL, HelperTypes.MODEL_NEG_PITCH, HelperTypes.MODEL_PROP]

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse iconsprite(mat)."""
        if len(args) > 1:
            raise ValueError(f'Expected up to 1 argument, got {args!r}!')
        elif len(args) == 1:
            return cls(args[0], tags=tags)
        else:
            return cls(None, tags=tags)

    def export(self) -> list[str]:
        """Produce the arguments for iconsprite()."""
        if self.model is not None:
            return [f'"{self.model}"']
        else:
            return []

    def get_resources(self, entity: EntityDef) -> Iterable[str]:
        """studio() uses a single model."""
        models: Iterable[str]
        if self.model is None:
            try:
                models = entity.kv['model'].known_options()
            except KeyError:
                return
        else:
            models = [self.model]

        for mdl in models:
            mdl = mdl.replace('\\', '/')

            # Strip leading and trailing quotation marks if both present
            if mdl.startswith('"') and mdl.endswith('"'):
                mdl = mdl[1:-1]

            # Don't append if the string is just empty!
            if len(mdl) == 0:
                continue

            if not mdl.casefold().endswith('.mdl'):
                mdl += '.mdl'
            if not mdl.casefold().startswith('models/'):
                mdl = 'models/' + mdl

            yield mdl


class HelperModelProp(HelperModel):
    """Model helper which does not affect the bounding box."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.MODEL_PROP


# These are all specialised for a particular entity.
# There's rarely options available.


class HelperModelLight(HelperModel):
    """Special model helper, with inverted pitch."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.MODEL_NEG_PITCH


class HelperInstance(Helper):
    """Specialized helper for func_instance."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_INSTANCE


class HelperDecal(Helper):
    """Specialized helper for infodecal."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_DECAL


class HelperOverlay(Helper):
    """Specialized helper for env_overlay."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_OVERLAY


class HelperOverlayTransition(Helper):
    """Specialized helper for env_overlay_transition."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_OVERLAY_WATER


class HelperLight(Helper):
    """Specialized helper for omnidirectional lights."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_LIGHT


@attrs.define
class HelperLightSpot(Helper):
    """Specialized helper for displaying spotlight previews."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_LIGHT_CONE

    inner: str
    outer: str
    color: str
    pitch_scale: float

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse lightcone(inner, outer, color, pitch_scale)."""
        inner_cone = '_inner_cone'
        outer_cone = '_cone'
        color = '_light'
        pitch_scale = 1.0
        try:
            inner_cone = args[0]
            outer_cone = args[1]
            color = args[2]
            pitch_scale = float(args[3])
        except IndexError:
            pass
        else:
            if len(args) > 4:
                raise ValueError(f'Expected 0-4 arguments, got {args!r}!')

        return cls(inner_cone, outer_cone, color, pitch_scale, tags=tags)

    def export(self) -> list[str]:
        """Produce the arguments for lightcone()."""
        result = [
            self.inner, self.outer, self.color,
            format_float(self.pitch_scale),
        ]
        # Try and remove redundant defaults.
        for default in ['1', '_light', '_cone', '_inner_cone']:
            if result[-1] == default:
                result.pop()
            else:
                return result
        return result


@attrs.define
class HelperLightSpotBlackMesa(Helper):
    """A new helper for Black Mesa's new spot entity."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_LIGHT_CONE_BLACK_MESA

    theta_kv: str
    phi_kv: str
    color_kv: str

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse newlightcone(theta, phi, lightcolor)."""
        if len(args) != 3:
            raise ValueError(f'Expected 3 arguments, got {args!r}!')
        return cls(args[0], args[1], args[2], tags=tags)

    def export(self) -> list[str]:
        """Produce the arguments for iconsprite()."""
        return [self.theta_kv, self.phi_kv, self.color_kv]


@attrs.define
class HelperRope(Helper):
    """Specialized helper for displaying move_rope and keyframe_rope."""

    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_ROPE

    name_kv: Optional[str] = None  # Extension in Portal: Revolution

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse keyframe(name)."""
        if len(args) > 1:
            raise ValueError(f'Expected up to one argument, got {args!r}!')
        if len(args) == 0:
            return cls(None, tags=tags)
        return cls(args[0], tags=tags)

    def export(self) -> list[str]:
        """Produce the arguments for keyframe()."""
        if not self.name_kv:
            return []
        return [self.name_kv]


class HelperTrack(Helper):
    """Specialized helper for path_track-style entities.

    This no longer does anything.
    """
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_TRACK


class HelperBreakableSurf(Helper):
    """Specialized helper for func_breakable_surf."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_BREAKABLE_SURF


class HelperWorldText(Helper):
    """Specialized helper for point_worldtext."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_WORLDTEXT


# Extensions to the FGD format.

@attrs.define
class HelperExtAppliesTo(Helper):
    """Allows specifying "tags" to indicate an entity is only used in certain games."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.EXT_APPLIES_TO
    IS_EXTENSION: ClassVar[bool] = True

    #: The tags this entity requires.
    applies: list[str] = attrs.Factory(list)

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse appliesto(tag1, tag2)."""
        if tags:
            raise ValueError('Cannot have tags on appliesto() helper.')
        return cls(args)

    def export(self) -> list[str]:
        """Produce the arguments for appliesto()."""
        return self.applies


@attrs.define
class HelperExtOrderBy(Helper):
    """Reorder keyvalues. Args = names in order."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.EXT_ORDERBY
    IS_EXTENSION: ClassVar[bool] = True

    order: list[str] = attrs.Factory(list)

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse orderby(key1, key2)."""
        if tags:
            raise ValueError('Cannot have tags on orderby() helper.')
        return cls(args, tags=frozenset())

    def export(self) -> list[str]:
        """Produce the arguments for orderby()."""
        return self.order


@attrs.define
class HelperExtAutoVisgroups(Helper):
    """Convenience for parsing, adds @AutoVisgroups to entities.

    'Auto' is implied at the start."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.EXT_AUTO_VISGROUP
    IS_EXTENSION: ClassVar[bool] = True

    path: list[str] = attrs.Factory(list)

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse autovis(group1, group2)."""
        if tags:
            raise ValueError('Cannot have tags on autovis() helper.')

        if len(args) > 0 and args[0].casefold() != 'auto':
            args.insert(0, 'Auto')
        if len(args) < 2:
            raise ValueError(f'Expected 2 or more arguments, got {args!r}!')
        return cls(args, tags=frozenset())

    def export(self) -> list[str]:
        """Produce the arguments for autovis()."""
        return self.path


@attrs.define
class HelperExtNoInherit(Helper):
    """Specifies keyvalues/inputs/outputs which are not inherited from the base classes."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.EXT_NO_INHERIT

    kind: Literal['keyvalue', 'input', 'output']
    names: set[str] = attrs.Factory(set)

    # noinspection PyClassVar
    _KINDS: ClassVar[Mapping[str, Literal['keyvalue', 'input', 'output']]] = {
        'keyvalue': 'keyvalue',
        'k': 'keyvalue',
        'key': 'keyvalue',
        'kv': 'keyvalue',
        'i': 'input',
        'in': 'input',
        'inp': 'input',
        'o': 'output',
        'out': 'output',
        'output': 'output',
    }

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse noinherit(keyvalue, key1, key2, key3)."""
        # 1 arg is useless but technically valid.
        if len(args) == 0:
            raise ValueError(f'Expected 1 or more arguments, got {args!r}!')
        kind: Literal['keyvalue', 'input', 'output']
        try:
            kind = cls._KINDS[args[0].casefold()]
        except KeyError:
            # Allow shorthands, but use the full form in the error.
            raise ValueError(
                f'Unknown type "{args[0]}", expected "keyvalue", "input" or "output"'
            ) from None
        return cls(kind, set(args[1:]), tags=tags)

    def export(self) -> list[str]:
        """Produce the arguments for orderby()."""
        result = sorted(self.names)
        result.insert(0, self.kind)
        return result


@attrs.define
class HelperExtCatapult(Helper):
    """Specialized helper for trigger_catapult, in Hammer++ and Strata Source."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_CATAPULT


@attrs.define
class HelperExtStrataFlatOBB(Helper):
    """A 2D oriented bounding box, specified by width/height. Used to visualise VGUI entities."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.STRATA_FLAT_OBB

    width_kv: str
    height_kv: str

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse orientedwidthheight(width, height)."""
        if len(args) != 2:
            raise ValueError(f'Expected 2 arguments, got {args!r}!')
        return cls(args[0], args[1], tags=tags)

    def export(self) -> list[str]:
        """Produce the arguments for orientedwidthheight()."""
        return [self.width_kv, self.height_kv]


@attrs.define
class HelperExtStrataHalfFlatOBB(HelperExtStrataFlatOBB):
    """A 2D oriented bounding box, specified by a half-width/height. Used to visualise portal entities."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.STRATA_HALF_FLAT_OBB


@attrs.define
class HelperExtStrataClusteredLight(Helper):
    """Visualises parameters for clustered lights."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_STRATA_CLUSTERED_LIGHT

    #: Type of clustered light.
    kind: Literal['point', 'spot']

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse clusteredlight(width, height)."""
        if len(args) != 1:
            raise ValueError(f'Expected 1 argument, got {args!r}!')
        kind = args[0].casefold()
        # Feels redundant, but technically kind could be a subclass, therefore not a literal.
        if kind == 'point':
            return cls('point', tags=tags)
        elif kind == 'spot':
            return cls('spot', tags=tags)
        else:
            raise ValueError(f'Unknown light kind "{kind}"!')

    def export(self) -> list[str]:
        """Produce the arguments for clusteredlight()."""
        return [self.kind]
