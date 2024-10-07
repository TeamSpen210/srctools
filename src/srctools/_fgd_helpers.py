"""Implemenations of specific code for each FGD helper type."""
from typing import ClassVar, Collection, Iterable, Iterator, List, Optional, Tuple, Union
from typing_extensions import Self

import attrs

from srctools.fgd import EntityDef, Helper, HelperTypes
from srctools.math import Vec, format_float, parse_vec_str


__all__ = [
    'HelperBBox', 'HelperBoundingBox', 'HelperBreakableSurf',
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
]


@attrs.define
class _HelperOneOptional(Helper):
    """Utility base class for a helper with one optional parameter."""
    _DEFAULT: ClassVar[str] = ''
    key: str

    @classmethod
    def parse(cls, args: List[str]) -> Self:
        """Parse a single optional keyl."""
        if len(args) > 1:
            raise ValueError(
                'Expected up to 1 argument, got ({})!'.format(', '.join(args))
            )
        elif len(args) == 1:
            key = args[0]
        else:
            key = cls._DEFAULT
        return cls(key)

    def export(self) -> List[str]:
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

    def __init__(self, point1: Vec, point2: Vec) -> None:
        self.bbox_min, self.bbox_max = Vec.bbox(point1, point2)

    def overrides(self) -> Collection[HelperTypes]:
        """Additional versions of this are not available."""
        return [HelperTypes.CUBE]

    @classmethod
    def parse(cls, args: List[str]) -> Self:
        """Parse size(x1 y1 z1, x2 y2 z2)."""
        if len(args) > 2:
            raise ValueError(
                'Expected 1 or 2 arguments, got ({})!'.format(', '.join(args))
            )
        size_min = Vec.from_str(args[0])
        if len(args) == 2:
            size_max = Vec.from_str(args[1])
        else:
            # "min" is actually the dimensions.
            size_max = size_min / 2
            size_min = -size_max

        return cls(size_min, size_max)

    def export(self) -> List[str]:
        """Produce (x1 y1 z1, x2 y2 z2)."""
        return [
            str(self.bbox_min),
            str(self.bbox_max),
        ]


@attrs.define
class HelperBBox(HelperSize):
    """Helper implementation for bbox()."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.BBOX


@attrs.define
class HelperCatapult(Helper):
    """Helper implementation for catapult(), specific to Portal: Revolution."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_CATAPULT


@attrs.define
class HelperRenderColor(Helper):
    """Helper implementation for color()."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.TINT

    r: float
    g: float
    b: float

    def overrides(self) -> Collection[HelperTypes]:
        """Previous ones of these are overridden by this."""
        return [HelperTypes.TINT]

    @classmethod
    def parse(cls, args: List[str]) -> Self:
        """Parse color(R G B)."""
        try:
            [tint] = args
        except ValueError:
            raise ValueError(
                f'Expected 1 argument, got ({", ".join(args)})!'
            ) from None

        r, g, b = parse_vec_str(tint)

        return cls(r, g, b)

    def export(self) -> List[str]:
        """Produce color(R G B)."""
        return [f'{self.r:g} {self.g:g} {self.b:g}']


@attrs.define
class HelperSphere(Helper):
    """Helper implementation for sphere()."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.SPHERE
    r: float
    g: float
    b: float
    size_key: str

    @classmethod
    def parse(cls, args: List[str]) -> Self:
        """Parse sphere(radius, r g b)."""
        arg_count = len(args)
        if arg_count > 2:
            raise ValueError(f'Expected 1 or 2 arguments, got ({", ".join(args)})!')
        r = g = b = 255.0

        if arg_count > 0:
            size_key = args[0]
            if arg_count == 2:
                r, g, b = parse_vec_str(args[1])
        else:
            size_key = 'radius'

        return cls(r, g, b, size_key)

    def export(self) -> List[str]:
        """Export the helper."""
        if self.r != 255.0 or self.g != 255.0 or self.b != 255.0:
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

    r: float
    g: float
    b: float
    start_key: str
    start_value: str
    end_key: Optional[str] = None
    end_value: Optional[str] = None

    @classmethod
    def parse(cls, args: List[str]) -> Self:
        """Parse line(r g b, start_key, start_value, end_key, end_value)."""
        arg_count = len(args)
        if arg_count not in (3, 5):
            raise ValueError(f'Expected 3 or 5 arguments, got {args!r}!')

        r, g, b = parse_vec_str(args[0])
        start_key = args[1]
        start_value = args[2]

        end_key: Optional[str]
        end_value: Optional[str]

        if arg_count == 5:
            end_key = args[3]
            end_value = args[4]
        else:
            end_key = end_value = None

        return cls(r, g, b, start_key, start_value, end_key, end_value)

    def export(self) -> List[str]:
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
    """Helper for env_projectedtexture visuals."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.FRUSTUM

    fov: Union[str, float]
    near_z: Union[str, float]
    far_z: Union[str, float]
    color: Union[str, Tuple[float, float, float]]
    pitch_scale: Union[str, float]

    @classmethod
    def parse(cls, args: List[str]) -> Self:
        """Parse frustum(fov, near, far, color, pitch_scale)."""
        # These are the default values if not provided.
        fov: Union[str, float] = '_fov'
        nearz: Union[str, float] = '_nearplane'
        farz: Union[str, float] = '_farz'
        color: Union[str, Tuple[float, float, float]] = '_light'
        pitch: Union[str, float] = -1.0

        try:
            fov = args[0]
            nearz = args[1]
            farz = args[2]
            color = args[3]
            pitch = args[4]
        except IndexError:
            pass  # Stop once out of args.
        else:
            if len(args) > 5:
                raise ValueError(f'Expected at most 5 arguments, got {args!r}!')

        # Try and parse everything, but if it fails ignore since they could
        # be property names.
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
        try:
            pitch = float(pitch)
        except ValueError:
            pass

        if isinstance(color, str):
            try:
                r, g, b = color.split()
                color = (float(r), float(g), float(b))
            except ValueError:
                pass

        return cls(fov, nearz, farz, color, pitch)

    def export(self) -> List[str]:
        """Export back out frustrum() arguments."""

        if isinstance(self.color, tuple):
            color = '{:g} {:g} {:g}'.format(*self.color)
        else:
            color = self.color

        def conv(x: 'Union[str, float]') -> str:
            """Ensure the .0 is removed from the float forms. """
            return x if isinstance(x, str) else format_float(x)

        return [
            conv(self.fov),
            conv(self.near_z),
            conv(self.far_z),
            color,
            conv(self.pitch_scale),
        ]


@attrs.define
class HelperCylinder(HelperLine):
    """Helper implementation for cylinder().

    Cylinder has the same sort of arguments as line(), plus radii for both positions.
    """
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.CYLINDER

    start_radius: Optional[str] = None
    end_radius: Optional[str] = None

    @classmethod
    def parse(cls, args: List[str]) -> Self:
        """Parse cylinder(r g b, start key/value/radius, end key/value/radius)."""
        arg_count = len(args)
        if arg_count not in (3, 4, 6, 7):
            raise ValueError(f'Expected 3, 4, 6 or 7 arguments, got {args!r}!')

        r, g, b = parse_vec_str(args[0])
        start_key = args[1]
        start_value = args[2]

        start_radius = end_key = end_value = end_radius = None
        if arg_count > 3:
            start_radius = args[3]
            if arg_count >= 6:
                end_key = args[4]
                end_value = args[5]
                if arg_count == 7:
                    end_radius = args[6]

        return cls(
            r, g, b,
            start_key, start_value,
            end_key, end_value,
            start_radius, end_radius,
        )

    def export(self) -> List[str]:
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
    def parse(cls, args: List[str]) -> Self:
        """Parse wirebox(min, max)"""
        try:
            [key_min, key_max] = args
        except ValueError:
            raise ValueError(f'Expected 2 arguments, got {args!r}!') from None

        return cls(key_min, key_max)

    def export(self) -> List[str]:
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
    def parse(cls, args: List[str]) -> Self:
        """Parse iconsprite(mat)."""
        if len(args) > 1:
            raise ValueError(f'Expected up to 1 argument, got {args!r}!')
        elif len(args) == 1:
            return cls(args[0].strip('"'))
        else:
            return cls(None)

    def export(self) -> List[str]:
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
    def parse(cls, args: List[str]) -> Self:
        """Parse iconsprite(mat)."""
        if len(args) > 1:
            raise ValueError(f'Expected up to 1 argument, got {args!r}!')
        elif len(args) == 1:
            return cls(args[0])
        else:
            return cls(None)

    def export(self) -> List[str]:
        """Produce the arguments for iconsprite()."""
        if self.model is not None:
            return [self.model]
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
    def parse(cls, args: List[str]) -> Self:
        """Parse lightcone(inner, outer, color, pitch_scale)."""
        if len(args) >= 1:
            inner_cone = args[0]
        else:
            inner_cone = '_inner_cone'
        if len(args) >= 2:
            outer_cone = args[1]
        else:
            outer_cone = '_cone'
        if len(args) >= 3:
            color = args[2]
        else:
            color = '_light'
        if len(args) >= 4:
            pitch_scale = float(args[3])
        else:
            pitch_scale = 1.0

        return cls(inner_cone, outer_cone, color, pitch_scale)

    def export(self) -> List[str]:
        """Produce the arguments for lightcone()."""
        # If any parameter is different, all previous must be provided.
        if self.pitch_scale != 1.0:
            return [
                self.inner, self.outer, self.color,
                format(self.pitch_scale, 'g'),
            ]
        if self.color != '_light':
            return [self.inner, self.outer, self.color]
        if self.outer != '_cone':
            return [self.inner, self.outer]
        if self.inner != '_inner_cone':
            return [self.inner]
        return []


@attrs.define
class HelperLightSpotBlackMesa(Helper):
    """A new helper for Black Mesa's new spot entity."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_LIGHT_CONE_BLACK_MESA

    theta_kv: str
    phi_kv: str
    color_kv: str

    @classmethod
    def parse(cls, args: List[str]) -> Self:
        """Parse newlightcone(theta, phi, lightcolor)."""
        if len(args) != 3:
            raise ValueError(f'Expected 3 arguments, got {args!r}!')
        return cls(args[0], args[1], args[2])

    def export(self) -> List[str]:
        """Produce the arguments for iconsprite()."""
        return [self.theta_kv, self.phi_kv, self.color_kv]


@attrs.define
class HelperRope(Helper):
    """Specialized helper for displaying move_rope and keyframe_rope."""

    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.ENT_ROPE

    name_kv: Optional[str]  # Extension in Portal: Revolution

    @classmethod
    def parse(cls, args: List[str]) -> Self:
        """Parse keyframe(name)."""
        if len(args) > 1:
            raise ValueError(f'Expected up to one argument, got {args!r}!')
        if len(args) == 0:
            return cls(None)
        return cls(args[0])

    def export(self) -> List[str]:
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

    tags: List[str] = attrs.Factory(list)

    @classmethod
    def parse(cls, args: List[str]) -> Self:
        """Parse appliesto(tag1, tag2)."""
        return cls(args)

    def export(self) -> List[str]:
        """Produce the arguments for appliesto()."""
        return self.tags


@attrs.define
class HelperExtOrderBy(Helper):
    """Reorder keyvalues. Args = names in order."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.EXT_ORDERBY
    IS_EXTENSION: ClassVar[bool] = True

    order: List[str] = attrs.Factory(list)

    @classmethod
    def parse(cls, args: List[str]) -> Self:
        """Parse orderby(key1, key2)."""
        return cls(args)

    def export(self) -> List[str]:
        """Produce the arguments for orderby()."""
        return self.order


@attrs.define
class HelperExtAutoVisgroups(Helper):
    """Convenience for parsing, adds @AutoVisgroups to entities.

    'Auto' is implied at the start."""
    TYPE: ClassVar[Optional[HelperTypes]] = HelperTypes.EXT_AUTO_VISGROUP
    IS_EXTENSION: ClassVar[bool] = True

    path: List[str] = attrs.Factory(list)

    @classmethod
    def parse(cls, args: List[str]) -> Self:
        """Parse autovis(group1, group2)."""
        if len(args) > 0 and args[0].casefold() != 'auto':
            args.insert(0, 'Auto')
        if len(args) < 2:
            raise ValueError(f'Expected 2 or more arguments, got {args!r}!')
        return cls(args)

    def export(self) -> List[str]:
        """Produce the arguments for autovis()."""
        return self.path
