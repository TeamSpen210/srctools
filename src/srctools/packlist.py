"""Handles the list of files which are desired to be packed into the BSP."""
from typing import Generic, Optional, Union
from typing_extensions import TypeVar, deprecated
from collections import OrderedDict
from collections.abc import Callable, Collection, Iterable, Iterator
from enum import Enum
from pathlib import Path
from zipfile import ZipFile
import io
import os
import re
import shutil

import attrs

from srctools import conv_bool, choreo as choreo_scenes
from srctools.bsp import BSP, DetailPropModel, DetailPropSprite
from srctools.const import FileType
from srctools.dmx import Attribute, Element, ValueType
from srctools.fgd import FGD, EntityDef, ResourceCtx, ValueTypes as KVTypes
from srctools.filesys import (
    CACHE_KEY_INVALID, File, FileSystem, FileSystemChain, VirtualFileSystem,
    VPKFileSystem,
)
from srctools.keyvalues import KeyValError, Keyvalues, NoKeyError
from srctools.mdl import MDL_EXTS_EXTRA, AnimEvents, Model
from srctools.particles import FORMAT_NAME as PARTICLE_FORMAT_NAME, Particle
from srctools.sndscape import Soundscape
from srctools.sndscript import SND_CHARS, Sound
from srctools.tokenizer import TokenSyntaxError, Tokenizer
from srctools.vmf import VMF, Entity
from srctools.vmt import Material, VarType
import srctools.logger


LOGGER = srctools.logger.get_logger(__name__)
SOUND_CACHE_VERSION = '2'  # Used to allow ignoring incompatible versions.
ParsedT = TypeVar('ParsedT')

__all__ = [
    'FileType', 'FileMode', 'SoundScriptMode',
    'PackFile', 'PackList',
    'unify_path', 'strip_extension',
    'entclass_canonicalize', 'entclass_canonicalise', 'entclass_packfunc',
    'entclass_resources', 'entclass_iter',
]


class FileMode(Enum):
    """Mode for files we may want to pack like soundscripts or particles."""
    UNKNOWN = 'unknown'  # Normal, we know about this file but it's not used.
    INCLUDE = 'include'  # Something uses this file.
    PRELOAD = 'preload'  # This file is used, and we want to preload it.
    EXCLUDE = 'exclude'  # Ordered to NOT include this file.

    @property
    def is_used(self) -> bool:
        """Return if this file should be packed into the map."""
        return self.value in ['include', 'preload']


SoundScriptMode = FileMode  # Old name, deprecated.
# Binary keyvalues only stores 32-bit integers, ensure file mod times are within that
# range.
FILE_CACHE_TRUNC = 0x7FFF_FFFF

EXT_TYPE = {
    '.' + filetype.value: filetype
    for filetype in FileType
    if isinstance(filetype.value, str)
}

# VScript function names that imply resources. This assumes it's the first
# argument.
SCRIPT_FUNC_TYPES: dict[bytes, tuple[str, FileType]] = {
    b'IncludeScript': ('scripts/vscripts/', FileType.VSCRIPT_SQUIRREL),
    b'DoIncludeScript': ('scripts/vscripts/', FileType.VSCRIPT_SQUIRREL),
    b'PrecacheScriptSound': ('', FileType.GAME_SOUND),
    b'PrecacheSoundScript': ('', FileType.GAME_SOUND),
    b'PrecacheModel': ('', FileType.MODEL),
}

# Events which play sounds
ANIM_EVENT_SOUND = {
    AnimEvents.AE_CL_PLAYSOUND,
    AnimEvents.AE_SV_PLAYSOUND,
    AnimEvents.SCRIPT_EVENT_SOUND,
    AnimEvents.SCRIPT_EVENT_SOUND_VOICE,
}
# Events which play a hardcoded footstep.
ANIM_EVENT_FOOTSTEP = {
    AnimEvents.AE_NPC_LEFTFOOT,
    AnimEvents.AE_NPC_RIGHTFOOT,
}
ANIM_EVENT_PARTICLE = AnimEvents.AE_CL_CREATE_PARTICLE_EFFECT
ANIM_EVENT_PARTICLE_SCRIPT = AnimEvents.CL_EVENT_SPRITEGROUP_CREATE

_FGD_TO_FILE = {
    # Appears differently in Hammer etc, but all load mats.
    KVTypes.STR_MATERIAL: FileType.MATERIAL,
    KVTypes.STR_SPRITE: FileType.MATERIAL,
    KVTypes.STR_DECAL: FileType.MATERIAL,

    KVTypes.STR_MODEL: FileType.MODEL,
    KVTypes.EXT_STR_TEXTURE: FileType.TEXTURE,
    KVTypes.STR_SCENE: FileType.CHOREO,
    KVTypes.STR_SOUND: FileType.GAME_SOUND,
    KVTypes.EXT_SOUNDSCAPE: FileType.SOUNDSCAPE_NAME,

    # These don't do anything, avoid checking the rest.
    KVTypes.VOID: None,
    KVTypes.CHOICES: None,
    KVTypes.SPAWNFLAGS: None,
    KVTypes.STRING: None,
    KVTypes.BOOL: None,
    KVTypes.INT: None,
    KVTypes.FLOAT: None,
    KVTypes.VEC: None,
    KVTypes.ANGLES: None,
    KVTypes.TARG_DEST: None,
    KVTypes.TARG_DEST_CLASS: None,
    KVTypes.TARG_SOURCE: None,
    KVTypes.TARG_NPC_CLASS: None,
    KVTypes.TARG_POINT_CLASS: None,
    KVTypes.TARG_FILTER_NAME: None,
    KVTypes.TARG_NODE_DEST: None,
    KVTypes.TARG_NODE_SOURCE: None,
    KVTypes.ANGLE_NEG_PITCH: None,
    KVTypes.VEC_LINE: None,
    KVTypes.VEC_ORIGIN: None,
    KVTypes.VEC_AXIS: None,
    KVTypes.COLOR_1: None,
    KVTypes.COLOR_255: None,
    KVTypes.SIDE_LIST: None,
    KVTypes.INST_FILE: None,  # Don't need to pack instances.
    KVTypes.INST_VAR_DEF: None,
    KVTypes.INST_VAR_REP: None,
    KVTypes.EXT_VEC_DIRECTION: None,
    KVTypes.EXT_VEC_LOCAL: None,
    KVTypes.EXT_ANGLE_PITCH: None,
    KVTypes.EXT_ANGLES_LOCAL: None,
}
# The set of types with packing behaviour.
_USEFUL_KV_TYPES = {
    kv_type for kv_type, result in _FGD_TO_FILE.items()
    if result is not None
}
_USEFUL_KV_TYPES |= {KVTypes.STR_VSCRIPT, KVTypes.STR_VSCRIPT_SINGLE, KVTypes.STR_PARTICLE}


@deprecated(
    'Use EntityDef.engine_def() instead if possible, or FGD.engine_dbase() '
    'if all entities are required.'
)
def load_fgd() -> FGD:
    """Extract the local copy of FGD data.

    This allows the analysis to not depend on local files.
    """
    return FGD.engine_dbase()


def strip_extension(filename: str) -> str:
    """Strip extensions from a filename, like Q_StripExtension()."""
    try:
        dot_pos = filename.rindex('.')
    except ValueError:
        return filename  # No extension.
    # If there's a slash after here, it's probably a ../, dotted folder, etc.
    if '/' in filename[dot_pos:]:
        return filename
    return filename[:dot_pos]


def default_extension(filename: str, ext: str) -> str:
    """If an extension isn't specified, use the provided one."""
    try:
        dot_pos = filename.rindex('.')
    except ValueError:
        return filename + ext  # No existing one.
    if '/' in filename[dot_pos:]:
        return filename + ext
    return filename  # We do have one.


@attrs.define(eq=False, repr=False)
class PackFile:
    """Represents a single file we are packing.

    data is raw data to pack directly, instead of from the filesystem.
    """
    type: FileType
    filename: str
    data: Optional[bytes] = None
    optional: bool = False
    # Human-readable indication as to the immediate cause of packing. Only the first is remembered.
    source: str = ''
    # If we've checked for dependencies of this yet.
    _analysed: bool = attrs.field(init=False, default=False)

    @property
    def virtual(self) -> bool:
        """Virtual files do not exist on the file system."""
        return self.data is not None

    def __repr__(self) -> str:
        text = f'<{"virtual " if self.virtual else ""}{self.type.name} Packfile "{self.filename}"'
        if self.data is not None:
            text += f' with {len(self.data)} bytes data'
        if self.source:
            text += f', source="{self.source}"'
        return text + '>'


def unify_path(path: str) -> str:
    """Convert paths to a unique form."""
    path = os.path.normpath(path).casefold().replace('\\', '/')
    if '../' in path:
        raise ValueError('Path tried to escape root!')
    return path.lstrip('/')


@attrs.define
class ManifestedFiles(Generic[ParsedT]):
    """Handles a file type which contains a bunch of named objects.

    We parse those to load the names, then when the names are referenced we pack the files they're
    defined in.
    """
    name: str  # Our file type.
    # When packing the file, use this filetype.
    pack_type: FileType
    # A function which parses the data, given the filename and contents.
    parse_func: Callable[[File], dict[str, ParsedT]]
    # For each identifier, the filename it's in and whatever data this was parsed into.
    # Do not display in the repr, there's thousands of these.
    name_to_parsed: dict[str, tuple[str, Optional[ParsedT]]] = attrs.field(factory=dict, repr=False)
    # Identifiers which have been packed
    packed_ids: set[str] = attrs.field(factory=set, repr=False)
    # All the filenames we know about, in order. The value is then
    # whether they should be packed.
    _files: dict[str, FileMode] = attrs.Factory(OrderedDict)
    _unparsed_file: dict[str, File] = attrs.Factory(dict)
    # Records the contents of the cache file.
    # filename -> (cache_key, identifier_list)
    _cache: dict[str, tuple[int, list[str]]] = attrs.Factory(dict)
    _cache_changed: bool = True

    def force_exclude(self, filename: str) -> None:
        """Mark this soundscript file as excluded."""
        self._files[filename] = FileMode.EXCLUDE

    def __len__(self) -> int:
        """Return the number of items we know about."""
        return len(self.name_to_parsed)

    def load_cache(self, filename: Union[str, 'os.PathLike[str]']) -> None:
        """Load the cache data. If the file is invalid, this does nothing."""
        try:
            with open(filename, 'rb') as f:
                root, file_type, file_version = Element.parse(f)
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            LOGGER.warning('Could not parse cache file "{}"!', filename)
            return
        if file_type != 'SrcPacklistCache' or file_version != 1:
            LOGGER.warning(
                'Unknown cache file "{}" with type {} v{}!',
                filename, file_type, file_version,
            )
            return
        for file_elem in root['files'].iter_elem():
            self._cache[file_elem.name] = (
                file_elem['key'].val_int & FILE_CACHE_TRUNC,
                list(file_elem['files'].iter_string()),
            )
        self._cache_changed = False

    def save_cache(self, filename: Union[str, 'os.PathLike[str]'], force: bool = False) -> None:
        """Write back new cache data."""
        if not force and not self._cache_changed:
            LOGGER.debug('Skipping resaving {}', filename)
            return
        root = Element('FileList', 'SrcFileList')
        file_arr = Attribute.array('files', ValueType.ELEMENT)
        root['files'] = file_arr
        for cached_file, (cache_key, files) in self._cache.items():
            if cache_key == CACHE_KEY_INVALID:
                continue
            elem = Element(cached_file, 'SrcCacheFile')
            # Truncate to avoid overflow errors.
            elem['key'] = cache_key & FILE_CACHE_TRUNC
            elem['files'] = Attribute('files', ValueType.STR, files)
            file_arr.append(elem)
        # No need to be atomic, if we corrupt this it'll just be rebuilt.
        with open(filename, mode='wb') as f:
            root.export_binary(f, fmt_name='SrcPacklistCache', fmt_ver=1, unicode='format')

    def add_cached_file(
        self, filename: str, file: File,
        mode: FileMode = FileMode.UNKNOWN,
    ) -> None:
        """Load a file which may have been cached.

        If the file is new we parse immediately, otherwise defer until actually required.
        """
        # Don't override exclude specifications.
        if self._files.get(filename, None) is not FileMode.EXCLUDE:
            self._files[filename] = mode
        key = file.cache_key()
        if key != CACHE_KEY_INVALID:
            key &= FILE_CACHE_TRUNC

        identifiers: list[str]
        try:
            cached_key, identifiers = self._cache[filename]
        except KeyError:
            pass
        else:
            if cached_key == key and key != CACHE_KEY_INVALID:
                LOGGER.debug('Loading {} from cache', filename)
                # Apply cache.
                self._unparsed_file[filename] = file
                for identifier in identifiers:
                    identifier = identifier.casefold()
                    if identifier not in self.name_to_parsed:
                        self.name_to_parsed[identifier] = (filename, None)
                return
        LOGGER.debug('Loading {}: not in cache', filename)
        # Otherwise, parse and add to the cache.
        identifiers = []
        self._cache_changed = True
        self._cache[filename] = key, identifiers
        for identifier, data in self.parse_func(file).items():
            identifiers.append(identifier)
            identifier = identifier.casefold()
            if identifier not in self.name_to_parsed:
                self.name_to_parsed[identifier] = (filename, data)

    def add_file(
        self, filename: str,
        items: Iterable[tuple[str, ParsedT]],
        mode: FileMode = FileMode.UNKNOWN,
    ) -> None:
        """Add a file with its parsed items."""
        # Don't override exclude specifications.
        if self._files.get(filename, None) is not FileMode.EXCLUDE:
            self._files[filename] = mode
        for identifier, data in items:
            identifier = identifier.casefold()
            if identifier not in self.name_to_parsed:
                self.name_to_parsed[identifier] = (filename, data)

    def fetch_data(self, identifier: str) -> tuple[str, ParsedT]:
        """Fetch the parsed form of this data and the file it's in, without packing."""
        [filename, data] = self.name_to_parsed[identifier.casefold()]
        if data is None:
            # Parse right now.
            LOGGER.debug('Parsing {}', filename)
            for ident, data in self.parse_func(self._unparsed_file.pop(filename)).items():
                ident = ident.casefold()
                if ident not in self.name_to_parsed or self.name_to_parsed[ident][1] is None:
                    self.name_to_parsed[ident] = (filename, data)
            [filename, data] = self.name_to_parsed[identifier.casefold()]
            if data is None:
                raise ValueError(f'Parsed "{filename}", but identifier "{identifier}" was not present!')
        return filename, data

    def pack_and_get(
        self, lst: 'PackList',
        identifier: str,
        preload: bool = False, source: str = '',
    ) -> ParsedT:
        """Pack the associated filename, then return the data."""
        self.packed_ids.add(identifier)
        filename, data = self.fetch_data(identifier)

        old = self._files[filename]
        if old is not FileMode.EXCLUDE:
            self._files[filename] = FileMode.PRELOAD if preload else FileMode.INCLUDE
            lst.pack_file(filename, self.pack_type, source=source)
        return data

    def packed_files(self) -> Iterator[tuple[str, FileMode]]:
        """Yield the used files in order."""
        for file, mode in self._files.items():
            if mode.is_used:
                yield file, mode


def _make_load_sound_file(
    kind: str, parse: Callable[[Keyvalues], dict[str, ParsedT]],
) -> Callable[[File], dict[str, ParsedT]]:
    """Make a function that parses a soundscript or soundscape file, logging errors that occur."""
    def func(file: File) -> dict[str, ParsedT]:
        try:
            with file.open_str(encoding='cp1252') as f:
                kv = Keyvalues.parse(f, file.path, allow_escapes=False)
            return parse(kv)
        except FileNotFoundError:
            # It doesn't exist, complain and pretend it's empty.
            LOGGER.warning('{} "{}" does not exist!', kind, file.path)
            return {}
        except (KeyValError, ValueError):
            LOGGER.warning('{} "{}" could not be parsed:', kind, file.path, exc_info=True)
            return {}
    return func


_load_soundscript = _make_load_sound_file('Soundscript', Sound.parse)
_load_soundscape = _make_load_sound_file('Soundscape', Soundscape.parse)


def _load_particle_system(file: File) -> dict[str, Particle]:
    """Parse a particle system file, logging errors that occur."""
    try:
        with file.open_bin() as f:
            dmx, fmt_name, fmt_version = Element.parse(f)
        if fmt_name != PARTICLE_FORMAT_NAME:
            raise ValueError(f'"{file.path}" is not a particle file!')
        return Particle.parse(dmx, fmt_version)
    except FileNotFoundError:
        # It doesn't exist, complain and pretend it's empty.
        LOGGER.warning('Particle system "{}" does not exist!', file.path)
        return {}
    except ValueError:
        LOGGER.warning('Particle system "{}" could not be parsed:', file.path, exc_info=True)
        return {}


class PackList:
    """Represents a list of resources for a map."""
    fsys: FileSystemChain

    soundscript: ManifestedFiles[Sound]
    soundscapes: ManifestedFiles[Soundscape]
    particles: ManifestedFiles[Particle]

    _packed_particles: set[str]
    _packed_soundscapes: set[str]
    _files: dict[str, PackFile]
    # folder, ext, data -> filename used
    _inject_files: dict[tuple[str, str, bytes], str]
    # Cache of the models used for breakable chunks.
    _break_chunks: Optional[dict[str, list[str]]]
    choreo: dict[choreo_scenes.CRC, choreo_scenes.Entry]
    # For each model, defines the skins the model uses. None means at least
    # one use is unknown, so all skins could potentially be used.
    skinsets: dict[str, Optional[set[int]]]

    def __init__(self, fsys: FileSystemChain) -> None:
        self.fsys = fsys
        self.soundscript = ManifestedFiles('soundscript', FileType.SOUNDSCRIPT, _load_soundscript)
        self.soundscapes = ManifestedFiles('soundscapes', FileType.SOUNDSCAPE_FILE, _load_soundscape)
        self.particles = ManifestedFiles('particle', FileType.PARTICLE_FILE, _load_particle_system)
        self._packed_particles = set()
        self._packed_soundscapes = set()
        self._files = {}
        self._inject_files = {}
        self._break_chunks = {}
        self.choreo = {}
        self.skinsets = {}

    def __getitem__(self, path: str) -> PackFile:
        """Look up a packfile by filename."""
        return self._files[unify_path(path)]

    def __len__(self) -> int:
        return len(self._files)

    def __iter__(self) -> Iterator[PackFile]:
        return iter(self._files.values())

    def filenames(self) -> Iterator[str]:
        """The filenames of all packed files."""
        return iter(self._files.keys())

    def __contains__(self, path: str) -> bool:
        return unify_path(path) in self._files

    def pack_file(
        self,
        filename: 'str | os.PathLike[str]',
        data_type: FileType = FileType.GENERIC,
        data: Optional[bytes] = None,
        *,
        skinset: Optional[set[int]] = None,
        optional: bool = False,
        source: str = '',
    ) -> None:
        """Queue the given file to be packed.

        :param filename: The name of the file to pack. This could be symbolic
            if certain file types are specified.
        :param data_type: Indicates definiteively what type of file this is.
            If generic, this is inferred from the extension.
        :param data: If provided, this file will use the given data instead of any
            on-disk data.
        :param skinset: If the file is a model, skinset allows restricting which skins are used.
            If None (default), all skins may be used. Otherwise, it is a set of
            skins. If all uses of a model restrict the skin, only those skins need
            to be packed.
        :param optional: If set, no errors will occur if it isn't in the filesystem.
        :param source: Human-readable string indicating the reason for packing.
        """
        filename = os.fspath(filename)

        # Assume an empty filename is an optional value.
        if not filename:
            if data is not None:
                raise ValueError('Data provided with no filename!')
            return

        if data_type is FileType.ENTCLASS_FUNC or data_type is FileType.ENTITY:
            raise ValueError(f'File type "{data_type.name}" must not be packed directly!')

        # Try to promote generic to other types if known.
        if data_type is FileType.GENERIC:
            try:
                data_type = EXT_TYPE[os.path.splitext(filename)[1].casefold()]
            except KeyError:
                pass

        # These are entry names, not files - the function packs any actual resources.
        # If this matches any of these, we don't want to pack a 'file', they'll do that.
        if data_type is FileType.GAME_SOUND:
            self.pack_soundscript(filename, source=source)
            return
        if data_type is FileType.PARTICLE:
            self.pack_particle(filename, source=source)
            return
        if data_type is FileType.BREAKABLE_CHUNK:
            self.pack_breakable_chunk(filename)
            return
        if data_type is FileType.SOUNDSCAPE_NAME:
            self.pack_soundscape(filename, source=source)
            return

        # If soundscript data is provided, load and force-include it.
        if data_type is FileType.SOUNDSCRIPT and data:
            try:
                sounds = Sound.parse(Keyvalues.parse(data.decode('cp1252'), filename))
            except (KeyValError, ValueError):
                LOGGER.warning('Soundscript "{}" could not be parsed:', filename, exc_info=True)
            else:
                self.soundscript.add_file(filename, sounds.items(), FileMode.INCLUDE)

        # Disallow tabs, to guard against cases where we incorrectly parse \t in file paths.
        if '\t' in filename:
            raise ValueError(f'No tabs are allowed in filenames ({filename!r})')

        filename = unify_path(filename)

        if data_type is FileType.MATERIAL:
            if filename.endswith('.spr'):
                # Legacy handling, sprite-type materials are found in materials/sprites.
                filename = f'sprites/{filename[:-4]}.vmt'
            if not filename.startswith('materials/'):
                filename = f'materials/{filename}'
            # This will replace .spr materials, which don't exist any more.
            if not filename.endswith(('.vmt', '.zmat')):
                filename = strip_extension(filename) + '.vmt'
        elif data_type is FileType.TEXTURE:
            if not filename.startswith('materials/'):
                filename = 'materials/' + filename
            if not filename.endswith('.hdr'):
                # Strip all other extensions, then add vtf unconditionally.
                filename = strip_extension(filename)
            filename = filename + '.vtf'
        elif data_type is FileType.VSCRIPT_SQUIRREL:
            filename = strip_extension(filename) + '.nut'
        elif data_type is FileType.RAW_SOUND:
            filename = filename.lstrip(SND_CHARS)
            if not filename.startswith('sound/'):
                filename = 'sound/' + filename
        elif data_type is FileType.PARTICLE_FILE:
            if data:
                virtual_fsys = VirtualFileSystem({filename: data})
                self.particles.add_cached_file(filename, virtual_fsys[filename], FileMode.INCLUDE)
            else:
                try:
                    fsys_file = self.fsys[filename]
                except FileNotFoundError:
                    pass
                else:
                    self.particles.add_cached_file(filename, fsys_file, FileMode.INCLUDE)

        if data_type is FileType.MODEL:
            if not filename.startswith('models/'):
                filename = 'models/' + filename
            if filename.endswith(MDL_EXTS_EXTRA):
                # It's a .vvd, .vtx etc file. Treat as generic, don't check for skinsets
                # or force the extension to be .mdl!!
                data_type = FileType.GENERIC
            else:
                # Allow passing skinsets via filename. This isn't useful if read from entities,
                # but is good for filenames in resource lists.
                if '.mdl#' in filename:
                    filename, skinset_int = filename.rsplit('.mdl#', 1)
                    try:
                        skinset = set(map(int, skinset_int.split(',')))
                    except (TypeError, ValueError):
                        LOGGER.warning(
                            'Invalid skinset for "{}.mdl": {} should be comma-separated skins!',
                            filename, skinset_int,
                        )
                if not filename.endswith(('.mdl', '.glb', '.gltf')):
                    filename = strip_extension(filename) + '.mdl'
                if skinset is None:
                    # It's dynamic, this overrides any previous specific skins.
                    self.skinsets[filename] = None
                else:
                    try:
                        existing_skins = self.skinsets[filename]
                    except KeyError:
                        self.skinsets[filename] = skinset.copy()
                    else:
                        # Merge the two.
                        if existing_skins is not None:
                            self.skinsets[filename] = existing_skins | skinset

        if not data and filename.endswith(('.wav', '.mp3', '.ogg')) and '$gender' in filename:
            # Special case for raw sounds, they can have gendered files.
            # Just include both variants.
            self.pack_file(
                filename.replace('$gender', 'female'),
                data_type, optional=optional, source=source,
            )
            filename = filename.replace('$gender', 'male')

        try:
            file = self._files[filename]
        except KeyError:
            pass  # We need to make it.
        else:
            # It's already here, is that OK?

            # Allow overriding data on disk with ours..
            if file.data is None:
                if data is not None:
                    file.data = data
                # else: no data on either, that's fine.
            elif data == file.data:
                pass  # Overrode with the same data, that's fine
            elif data:
                raise ValueError(f'"{filename}": two different data streams!')
            # else: we had an override, but asked to just pack now. That's fine.

            if not file.source:
                file.source = source

            # Override optional packing with required packing.
            if not optional:
                file.optional = False

            if file.type is data_type:
                # Same, no problems - just packing on top.
                return

            if file.type is FileType.GENERIC:
                file.type = data_type  # This is fine, we now know it has behaviour...
            elif data_type is FileType.GENERIC:
                pass  # If we know it has behaviour already, that trumps generic.
            else:
                raise ValueError(
                    f'"{filename}": {file.type.name} '
                    f"can't become a {data_type.name}!"
                )
            return  # Don't re-add this.

        start, ext = os.path.splitext(filename)

        if data_type is FileType.SOUNDSCRIPT:
            if ext != '.txt':
                raise ValueError(f'"{filename}" cannot be a soundscript!')

        self._files[filename] = PackFile(data_type, filename, data, optional, source)

    def inject_file(
        self, data: bytes, folder: str, ext: str, *,
        prefix: str = 'INJECT',
        source: str = 'inject',
    ) -> str:
        """Inject a generated file into the map and return the full name.

        The file will be named using the format :file:`{folder}/{prefix}_{hex}.{ext}`.
        If the same file is requested twice (same folder,
        extension and data), only one will be included.
        """
        folder = folder.rstrip('\\/').replace('\\', '/')
        ext = ext.lstrip('.')
        try:
            return self._inject_files[folder, ext, data]
        except KeyError:
            pass
        # Repeatedly hashing permutes the data, until we stop colliding.
        # Also abs() to remove ugly minus signs.
        name_hash = format(abs(hash(data)), 'x')
        while True:
            full_name = f"{folder}/{prefix}_{name_hash}.{ext}"
            if full_name not in self._files:
                break
            name_hash = format(abs(hash(name_hash)), 'x')
        self.pack_file(full_name, data=data, source=source)
        self._inject_files[folder, ext, data] = full_name
        return full_name

    def inject_vscript(self, code: str, folder: str = 'inject', prefix: str = 'INJECT') -> str:
        """Specialised variant of inject_file() for VScript code specifically.

        This returns the script name suitable for passing to Entity Scripts.
        """
        return self.inject_file(
            code.encode('ascii'),
            os.path.join('scripts/vscripts', folder), '.nut',
            source='inject_vscript',
            prefix=prefix,
            # Strip off the scripts/vscripts/ folder since it's implied.
        )[17:]

    def pack_soundscript(self, sound_name: str, *, source: str='') -> None:
        """Pack a soundscript or raw sound file."""
        # Blank means no sound is used.
        if not sound_name:
            return

        sound_name = sound_name.casefold().replace('\\', '/')
        # Check for raw sounds first.
        if sound_name.endswith(('.wav', '.mp3', '.ogg')):
            self.pack_file(sound_name, FileType.RAW_SOUND, source=source)
            return

        try:
            soundscript = self.soundscript.pack_and_get(self, sound_name, source=source)
        except KeyError:
            LOGGER.warning('Unknown sound "{}"!', sound_name)
            return

        for sound in soundscript.sounds:
            # The soundscript is the source, not what packed the soundscript.
            self.pack_file(sound, FileType.RAW_SOUND, source=sound_name)

    def pack_soundscape(self, soundscape_name: str, *, source: str = '') -> None:
        """Pack a soundscape, and all the sounds it needs."""
        # Blank produces a dev message, also skip already-packed names.
        if not soundscape_name or (folded := soundscape_name.casefold()) in self._packed_soundscapes:
            return
        self._packed_soundscapes.add(folded)
        try:
            soundscape = self.soundscapes.pack_and_get(self, soundscape_name, source=source)
        except KeyError:
            LOGGER.warning('Unknown soundscape "{}"!', soundscape_name)
            return
        for loop in soundscape.loop_sounds:
            # The soundscape is the source, not what packed the soundscape.
            self.pack_file(loop.sound, FileType.RAW_SOUND, source=soundscape_name)
        for rand in soundscape.rand_sounds:
            for sound in rand.sounds:
                self.pack_file(sound, FileType.RAW_SOUND, source=soundscape_name)
        # This is recursive, but the packed-soundscapes set means all levels need to be unique to
        # have any effect. Game only allows 8 levels, so whatever we hit is already far exceeding
        # that.
        for child in soundscape.children:
            self.pack_soundscape(child.name, source=soundscape_name)

    def pack_particle(self, particle_name: str, preload: bool = False, *, source: str = '') -> None:
        """Pack a particle system and the raw PCFs."""
        # Blank means no particle is used, also skip if we already packed.
        if not particle_name or particle_name in self._packed_particles:
            return
        self._packed_particles.add(particle_name)
        try:
            particle = self.particles.pack_and_get(self, particle_name, preload=preload, source=source)
        except KeyError:
            LOGGER.warning('Unknown particle "{}"!', particle_name)
            return
        # Pack the sprites the particle system uses.
        try:
            mat = particle.options['material'].val_str
        except KeyError:
            pass
        else:
            self.pack_file(mat, FileType.MATERIAL, source=particle_name)
        for rend in particle.renderers:
            if rend.function.casefold() == 'render models':
                try:
                    mdl = rend.options['sequence 0 model'].val_str
                except KeyError:
                    LOGGER.warning('Particle {} has model render with no model?', particle_name)
                else:
                    self.pack_file(mdl, FileType.MODEL, source=particle_name)
        for child in particle.children:
            self._packed_particles.add(child.particle)
            self.pack_particle(child.particle, preload, source=particle_name)

    def pack_breakable_chunk(self, chunkname: str) -> None:
        """Pack the generic gib model for the given chunk name."""
        if self._break_chunks is None:
            # Need to load the file.
            self.pack_file('scripts/propdata.txt', source='BreakableModels')
            try:
                propdata = self.fsys['scripts/propdata.txt']
            except FileNotFoundError:
                LOGGER.warning('No scripts/propdata.txt for breakable chunks!')
                return
            with propdata.open_str(encoding='cp1252') as f:
                kv = Keyvalues.parse(f, 'scripts/propdata.txt', allow_escapes=False)
            self._break_chunks = {}
            for chunk_prop in kv.find_children('BreakableModels'):
                self._break_chunks[chunk_prop.name] = [
                    prop.real_name for prop in chunk_prop
                ]
        try:
            mdl_list = self._break_chunks[chunkname.casefold()]
        except KeyError:
            LOGGER.warning('Unknown gib chunks type "{}"!', chunkname)
            return
        source = f'scripts/propdata.txt:{chunkname}'
        for mdl in mdl_list:
            self.pack_file(mdl, FileType.MODEL, source=source)

    def load_soundscript(
        self,
        file: File,
        *,
        always_include: bool = False,
    ) -> Iterable[Sound]:
        """Read in a soundscript and record which files use it.

        If always_include is True, it will be included in the manifests even
        if it isn't used.

        The sounds registered by this soundscript are returned.
        """
        scripts = _load_soundscript(file)
        self.soundscript.add_file(
            file.path, scripts.items(),
            FileMode.INCLUDE if always_include else FileMode.UNKNOWN
        )
        return scripts.values()

    def load_particle_system(self, filename: str, mode: FileMode = FileMode.UNKNOWN) -> Iterable[Particle]:
        """Read in the specified particle system and record the particles for usage checking."""
        try:
            particles = _load_particle_system(self.fsys[filename])
        except FileNotFoundError:
            # It doesn't exist, complain and pretend it's empty.
            LOGGER.warning('Particle system "{}" does not exist!', filename)
            return ()

        self.particles.add_file(filename, particles.items(), mode)
        return particles.values()

    def load_manifests(self, cache_folder: Union[Path, str, None] = None) -> None:
        """Parse the manifests and script files for things like soundscripts or particles.

        This is necessary to perform lookups by name.
        If the cache prefix is provided, this is used as a path and prefix for files writted
        to cache results to speed up later executions.
        """
        if cache_folder is not None:
            cache = Path(cache_folder)
            prefix = cache.stem
            self.load_soundscript_manifest(cache.with_name(f'{prefix}_soundscript.dmx'))
            self.load_soundscape_manifest(cache.with_name(f'{prefix}_soundscapes.dmx'))
            self.load_particle_manifest(cache.with_name(f'{prefix}_particles.dmx'))
        else:
            self.load_soundscript_manifest()
            self.load_soundscape_manifest()
            self.load_particle_manifest()

        self.load_choreo_scenes()  # Image is already a compact cache.

    def load_soundscript_manifest(self, cache_file: Union[Path, str, None] = None) -> None:
        """Read the soundscript manifest, and read all mentioned scripts.

        If cache_file is provided, it should be a path to a file used to
        cache the file reading for later use.
        """
        try:
            man = self.fsys.read_kv1('scripts/game_sounds_manifest.txt', encoding='cp1252')
        except FileNotFoundError:
            LOGGER.warning('No soundscripts manifest.')
            return

        if cache_file is not None:
            self.soundscript.load_cache(cache_file)

        for prop in man.find_children('game_sounds_manifest'):
            if not prop.name.endswith('_file'):
                continue
            try:
                file = self.fsys[prop.value]
            except FileNotFoundError:
                LOGGER.warning('Soundscript "{}" does not exist!', prop.value)
                # Don't write anything into the cache, so we check this
                # every time.
                continue
            # The soundscripts in the manifests are always included,
            # since many would be part of the core code (physics, weapons,
            # ui, etc). Just keep those loaded, no harm since the game does.
            self.soundscript.add_cached_file(prop.value, file, FileMode.INCLUDE)

        if cache_file is not None:
            self.soundscript.save_cache(cache_file)

    def load_soundscape_manifest(self, cache_file: Union[Path, str, None] = None) -> None:
        """Read the soundscape manifest, and read all mentioned scripts.

        If cache_file is provided, it should be a path to a file used to
        cache the file reading for later use.
        """
        try:
            man = self.fsys.read_kv1('scripts/soundscapes_manifest.txt', encoding='cp1252')
        except FileNotFoundError:
            LOGGER.warning('No soundscapes manifest.')
            return

        if cache_file is not None:
            self.soundscapes.load_cache(cache_file)

        for kv in man.find_children('soundscapes_manifest'):
            if kv.name != 'file':
                continue
            try:
                file = self.fsys[kv.value]
            except FileNotFoundError:
                LOGGER.warning('Soundscape "{}" does not exist!', kv.value)
                # Don't write anything into the cache, so we check this
                # every time.
                continue
            self.soundscapes.add_cached_file(kv.value, file)

        if cache_file is not None:
            self.soundscapes.save_cache(cache_file)

    def load_particle_manifest(self, cache_file: Union[Path, str, None] = None) -> None:
        """Read the particle manifest, and read all mentioned scripts.

        If cache_file is provided, it should be a path to a file used to
        cache the file reading for later use.
        """
        try:
            man = self.fsys.read_kv1('particles/particles_manifest.txt')
        except FileNotFoundError:
            LOGGER.warning('No particles manifest.')
            man = Keyvalues.root()

        if cache_file is not None:
            self.particles.load_cache(cache_file)

        in_manifest: set[str] = set()

        for prop in man.find_children('particles_manifest'):
            if prop.value.startswith('!'):
                file_mode = FileMode.PRELOAD
                fname = prop.value[1:]
            else:
                file_mode = FileMode.INCLUDE
                fname = prop.value
            in_manifest.add(fname)
            try:
                file = self.fsys[fname]
            except FileNotFoundError:
                # It doesn't exist, complain and pretend it's empty.
                LOGGER.warning('Particle system "{}" does not exist!', fname)
            else:
                self.particles.add_cached_file(fname, file, file_mode)

        # Now, manually look for any particles not in the manifest, those are added if referenced.
        for part_file in self.fsys.walk_folder('particles/'):
            if not part_file.path[-4:].casefold() == '.pcf':
                continue
            if part_file.path not in in_manifest:
                self.particles.add_cached_file(part_file.path, part_file)

        if cache_file is not None:
            self.particles.save_cache(cache_file)

    def load_choreo_scenes(self) -> None:
        """Load the scenes manifest."""
        try:
            image = self.fsys['scenes/scenes.image']
        except FileNotFoundError:
            LOGGER.warning('No scenes.image!')
        else:
            with image.open_bin() as file:
                self.choreo |= choreo_scenes.parse_scenes_image(file)

    @deprecated('Renamed to write_soundscript_manifest()')
    def write_manifest(self) -> None:
        """Deprecated, call write_soundscript_manifest()."""
        self.write_soundscript_manifest()

    def write_soundscript_manifest(self) -> None:
        """Produce and pack a soundscript manifest file for this map.

        It will be packed such that it can override the master manifest with
        sv_soundemitter_flush.
        """
        manifest = Keyvalues('game_sounds_manifest', [
            Keyvalues('precache_file', snd)
            for snd, _ in self.soundscript.packed_files()
        ])

        buf = io.BytesIO()
        wrapper = io.TextIOWrapper(buf, encoding='cp1252')
        manifest.serialise(wrapper)
        wrapper.detach()
        self.pack_file(
            'scripts/game_sounds_manifest.txt', FileType.GENERIC, buf.getvalue(),
            source='<generated>',
        )

    def write_particles_manifest(self, manifest_name: str) -> None:
        """Write a particles manifest, so that used particles can be loaded."""
        manifest = Keyvalues('particles_manifest', [])
        for filename, mode in self.particles.packed_files():
            if mode is FileMode.PRELOAD:
                filename = '!' + filename
            manifest.append(Keyvalues('file', filename))

        buf = io.BytesIO()
        wrapper = io.TextIOWrapper(buf, encoding='cp1252')
        manifest.serialise(wrapper)
        wrapper.detach()
        self.pack_file(manifest_name, FileType.GENERIC, buf.getvalue(), source='<generated>')

    def pack_from_bsp(self, bsp: BSP) -> None:
        """Pack files found in BSP data (excluding entities)."""
        for prop in bsp.props:
            # Static props obviously only use one skin.
            self.pack_file(prop.model, FileType.MODEL, skinset={prop.skin}, source='prop_static')

        # These are all the materials the BSP references, including brushes and overlays.
        for mat in bsp.textures:
            self.pack_file(f'materials/{mat.lower()}.vmt', FileType.MATERIAL, source='brush/overlay')

        # detail.vbsp is only used by VBSP itself, so we don't need to pack.
        has_sprite = False  # All sprites use a single texture sheet.
        for detail in bsp.detail_props:
            if isinstance(detail, DetailPropModel):
                # Always skin 0.
                self.pack_file(detail.model, FileType.MODEL, skinset={0}, source='prop_detail')
            elif isinstance(detail, DetailPropSprite):
                has_sprite = True
        if has_sprite:
            self.pack_file(
                # Unfortunate, pack_from_ents() deals with everything else ent-wise, but we should
                # only pack this if detail props exist.
                bsp.ents.spawn['detailmaterial'], FileType.MATERIAL,
                source='prop_detail_sprite',
            )

    @deprecated("The provided FGD is no longer necessary, call pack_with_ents instead.",)
    def pack_fgd(self, vmf: VMF, fgd: FGD, mapname: str = '', tags: Iterable[str] = ()) -> None:
        """Deprecated version of pack_from_ents(). The FGD parameter is no longer necessary."""
        self.pack_from_ents(vmf, mapname, tags)

    def pack_from_ents(
        self,
        vmf: VMF,
        mapname: str = '',
        tags: Iterable[str] = (),
    ) -> None:
        """Analyse the map to pack files, using an internal database of keyvalues.

        'detailmaterial' is handled in `pack_from_bsp()`, we only need to include it if
        a detail sprite is actually present in the BSP.
        """
        # Don't show the same keyvalue warning twice, it's just noise.
        unknown_keys: set[tuple[str, str]] = set()

        # Definitions for the common keyvalues on all entities.
        base_entity = EntityDef.engine_def('_CBaseEntity_')

        ent_cache: dict[str, EntityDef] = {}
        kv_cache: dict[EntityDef, set[str]] = {}

        def get_ent(classname: str) -> EntityDef:
            """Look up the FGD for an entity."""
            try:
                return ent_cache[classname]
            except KeyError:
                pass
            try:
                ent_class = ent_cache[classname] = EntityDef.engine_def(classname)
                return ent_class
            except KeyError:
                if (classname, '') not in unknown_keys:
                    LOGGER.warning('Unknown class "{}"!', classname)
                    unknown_keys.add((classname, ''))
                raise

        res_ctx = ResourceCtx(
            fgd=get_ent,
            fsys=self.fsys,
            mapname=mapname,
            tags=tags,
        )

        for ent in vmf.entities:
            # Allow opting out packing specific entities.
            if conv_bool(ent.pop('srctools_nopack', '')):
                continue

            classname = ent['classname'].casefold()
            if classname == 'worldspawn':
                continue  # Don't try packing things like detail prop materials.
            source = f'{classname}:{source}' if (source := ent['targetname']) else classname

            try:
                ent_class = get_ent(classname)
            except KeyError:
                # Fall back to generic keyvalues.
                ent_class = base_entity
            try:
                relevant_keys = kv_cache[ent_class]
            except KeyError:
                # Filter keyvalues down to those which could result in packing.
                # There's a lot of random keys in CBaseEntity etc which are never going
                # to produce files.
                relevant_keys = kv_cache[ent_class] = {
                    name
                    for name in ent_class.kv
                    if name.casefold() == 'model' or ent_class.kv[name].type in _USEFUL_KV_TYPES
                }

            skinset: Optional[set[int]]
            if 'skinset' in ent:
                # Special key for us - if set this is a list of skins this
                # entity is pledging it will restrict itself to.
                skinset = {
                    int(x)
                    for x in ent['skinset'].split()
                }
            else:
                skinset = None

            value: str
            key: str
            # Check both keys set on the ent (for unknown-kv warnings), and potentially packable
            # ones in the FGD, so we can check the defaults.
            for key in set(ent) | relevant_keys:
                key = key.casefold()
                # These are always present on entities, and we don't have to do
                # any packing for them.
                # Origin/angles might be set (brushes, instances) even for ents
                # that don't use them.
                if key in (
                    'classname', 'hammerid',
                    'origin', 'angles',
                    'skin',
                    'pitch',
                    'skinset',
                ):
                    continue
                elif key == 'model':
                    # Models are set on all brush entities, and are either a '*37' brush ref,
                    # a model, or a sprite.
                    # But look up the KV anyway - if it's explicitly not one of those, don't pack
                    # this. That indicates it's just there to let you swap the model for in Hammer.
                    value = ent[key]
                    if not value or value.startswith('*'):
                        continue  # Do not "pack" brush references, or blank ones.

                    try:
                        val_type = ent_class.kv[key].type
                    except KeyError:
                        val_type = KVTypes.STR_MODEL  # Try to pack this anyway.
                    if val_type is KVTypes.STR_MODEL or val_type is KVTypes.STR_SPRITE:
                        self.pack_file(value, skinset=skinset, source=source)
                        continue
                    # Else, it's another type. Do the generic handling below.
                try:
                    kv = ent_class.kv[key]
                    val_type = kv.type
                    default = kv.default
                except KeyError:
                    # If ent_class is base_entity, this is an unknown class.
                    # Suppress the error, we already showed a warning above.
                    if ent_class is not base_entity and (classname, key) not in unknown_keys:
                        unknown_keys.add((classname, key))
                        LOGGER.warning(
                            'Unknown keyvalue "{}" for ent of type "{}"!',
                            key, classname,
                        )
                    continue

                value = ent[key, default]

                # Ignore blank values, they're not useful.
                if not value:
                    continue

                try:
                    file_type = _FGD_TO_FILE[val_type]
                except KeyError:
                    # Handle some file types with unique behaviour.
                    # Note, if adding more, add to _USEFUL_KV_TYPES too!

                    # Either a space-separated list of scripts, or a single one. Valve only used
                    # Squirrel (.nut) for VScript, but it's possible mods could have added others.
                    # The extension is optional, so assume nut if no other is provided.
                    if val_type is KVTypes.STR_VSCRIPT:
                        for script in value.split():
                            self.pack_file(
                                default_extension(f'scripts/vscripts/{script}', '.nut'),
                                source=source,
                            )
                    elif val_type is KVTypes.STR_VSCRIPT_SINGLE:
                        self.pack_file(
                            default_extension(f'scripts/vscripts/{value}', '.nut'),
                            source=source,
                        )
                    elif val_type is KVTypes.STR_PARTICLE:
                        self.pack_particle(value, source=source)
                else:
                    if file_type is not None:
                        self.pack_file(value, file_type, source=source)

            # Handle resources that's coded into different entities with our internal database.
            for file_type, filename in ent_class.get_resources(res_ctx, ent=ent, on_error=LOGGER.warning):
                self.pack_file(filename, file_type, source=source)

        # Handle worldspawn here - this is fairly special.
        sky_name = vmf.spawn['skyname']
        if sky_name:
            for suffix in ['bk', 'dn', 'ft', 'lf', 'rt', 'up']:
                self.pack_file(
                    f'materials/skybox/{sky_name}{suffix}.vmt',
                    FileType.MATERIAL,
                    source='2D Skybox',
                )

    def pack_into_zip(
        self,
        bsp: BSP,
        *,
        whitelist: Iterable[FileSystem] = (),
        blacklist: Iterable[FileSystem] = (),
        callback: Callable[[str], Optional[bool]] = lambda f: None,
        dump_loc: Optional[Path] = None,
        only_dump: bool = False,
        ignore_vpk: bool = True,
    ) -> None:
        """Pack all our files into the packfile in the BSP.

        The filesystem is used to find files to pack.
        First it is passed to the callback (if provided), which should return True/False to
        determine if the file should be packed. If it returns None, then the whitelist/blacklist
        is checked.
        Filesystems must be in the whitelist and not in the blacklist, if provided.
        If ignore_vpk is True, files in VPK won't be packed unless that system
        is in allow_filesys.
        If dump_loc is set, files will be copied there as well. If only_dump is
        set, they won't be packed at all.
        """
        # We need to rebuild the zipfile from scratch, so we can overwrite
        # old data if required.

        # First retrieve existing files.
        # The packed_files dict is a casefolded name -> (orig name, bytes) tuple.
        packed_files: dict[str, tuple[str, bytes]] = {}

        all_systems: set[FileSystem] = {
            sys for sys, _ in
            self.fsys.systems
        }

        allowed = set(all_systems)

        if ignore_vpk:
            for fsys in all_systems:
                if isinstance(fsys, VPKFileSystem):
                    allowed.discard(fsys)

        # Add these on top, so this overrides ignore_vpk.
        allowed.update(whitelist)
        # Then remove blacklisted systems.
        allowed.difference_update(blacklist)

        LOGGER.debug('Allowed filesystems:\n{}', '\n'.join([
            ('+ ' if sys in allowed else '- ') + repr(sys) for sys, _ in
            self.fsys.systems
        ]))

        if dump_loc is not None:
            # Always write to a subfolder named after the map.
            # This ensures we're unlikely to overwrite important folders.
            dump_loc /= Path(bsp.filename).stem
            LOGGER.info('Dumping pakfile to "{}"..', dump_loc)
            shutil.rmtree(dump_loc, ignore_errors=True)
        else:
            only_dump = False  # Pointless to not dump to pakfile or folder.

        # Test existing files against the callback, so we can remove them.
        for info in bsp.pakfile.infolist():
            # Explicitly compare because None = no opinion, so we want to keep then.
            if callback(info.filename.replace('\\', '/')) is False:
                LOGGER.debug('REMOVE: {}', info.filename)
            else:
                packed_files[info.filename.casefold()] = (info.filename, bsp.pakfile.read(info))

        for file in self._files.values():
            # Need to ensure / separators.
            fname = file.filename.replace('\\', '/')

            fname_source = f'{fname}\t\t(Source={file.source})' if file.source else fname

            if file.data is not None:
                # Always pack, we've got custom data.
                LOGGER.debug('CUSTOM DATA: {}', fname_source)
                if not only_dump:
                    packed_files[fname.casefold()] = (fname, file.data)
                if dump_loc is not None:
                    path = dump_loc / fname
                    path.parent.mkdir(exist_ok=True, parents=True)
                    path.write_bytes(file.data)
                continue

            try:
                sys_file = self.fsys[file.filename]
            except FileNotFoundError:
                if not file.optional and fname.casefold() not in packed_files:
                    LOGGER.warning('WARNING: "{}" not packed! Source="{}"', file.filename, file.source)
                continue

            if fname.casefold().endswith('.bik'):
                # BINK cannot be packed, always skip. TODO: Still true?
                LOGGER.debug('EXT:  {}', fname_source)
                continue

            should_pack = callback(sys_file.path)
            if should_pack is None:
                should_pack = self.fsys.get_system(sys_file) in allowed

            if should_pack:
                LOGGER.debug('ADD:  {}', fname_source)
                with sys_file.open_bin() as f:
                    data = f.read()
                if not only_dump:
                    packed_files[fname.casefold()] = (fname, data)
                if dump_loc is not None:
                    path = dump_loc / fname
                    path.parent.mkdir(exist_ok=True, parents=True)
                    path.write_bytes(data)
            else:
                LOGGER.debug('SKIP: {}', fname_source)

        LOGGER.info('Compressing packfile...')
        # Note no with statement, the BSP takes ownership and needs it open.
        new_zip = ZipFile(io.BytesIO(), 'w')
        for fname, data in packed_files.values():
            new_zip.writestr(fname, data)
        bsp.pakfile = new_zip

    def eval_dependencies(self) -> None:
        """Add files to the list which need to also be packed.

        This requires parsing through many files.
        """
        # Run through repeatedly, until all are analysed.
        todo = True
        while todo:
            todo = False

            for file in list(self._files.values()):
                # noinspection PyProtectedMember
                if file._analysed:
                    continue
                file._analysed = True
                todo = True

                try:
                    if file.type is FileType.MATERIAL:
                        self._get_material_files(file)
                    elif file.type is FileType.MODEL:
                        self._get_model_files(file)
                    elif file.type is FileType.TEXTURE:
                        # Try packing the '.hdr.vtf' file as well if present.
                        # But don't recurse!
                        if not file.filename.endswith('.hdr.vtf'):
                            hdr_tex = file.filename[:-3] + 'hdr.vtf'
                            if hdr_tex in self.fsys:
                                self.pack_file(hdr_tex, optional=True, source=file.filename)
                    elif file.type is FileType.VSCRIPT_SQUIRREL:
                        self._get_vscript_files(file)
                    elif file.type is FileType.CHOREO:
                        self._get_choreo_files(file)
                    elif file.type is FileType.WEAPON_SCRIPT:
                        # Black Mesa Source uses DMX files instead of KV1.
                        if file.filename.endswith('.dmx'):
                            self._get_weaponscript_dmx_files(file)
                        else:
                            self._get_weaponscript_kv1_files(file)
                except Exception as exc:
                    # Skip errors in the file format - means we can't find the dependencies.
                    LOGGER.warning(
                        'Could not evaluate dependencies for file "{}"!',
                        file.filename, exc_info=exc,
                    )

    def _get_model_files(self, file: PackFile) -> None:
        """Find any needed files for a model."""
        filename = file.filename
        file_stem, ext = os.path.splitext(filename)

        if ext in ('.glb', '.gltf'):
            return

        # Some of these are optional, so just skip. Do not re-pack the MDL itself, that's
        # pointless and will also erase the skinset!
        for ext in MDL_EXTS_EXTRA:
            component = file_stem + ext
            if component in self.fsys:
                # The sub-files are really part of the main one, they should be packed with
                # the same source.
                self.pack_file(component, source=file.source)

        if file.data is not None:
            # We need to add that file onto the system, so it's loaded.
            self.fsys.add_sys(VirtualFileSystem({
                filename: file.data,
            }), priority=True)
            try:
                mdl = Model(self.fsys, self.fsys[filename])
            finally:  # Remove this system.
                self.fsys.systems.pop(0)
        else:
            try:
                mdl = Model(self.fsys, self.fsys[filename])
            except FileNotFoundError:
                if not file.optional:
                    LOGGER.warning('Can\'t find model "{}"!', filename)
                return

        skinset = self.skinsets.get(filename, None)
        for tex in mdl.iter_textures(skinset):
            self.pack_file(tex, FileType.MATERIAL, optional=file.optional, source=filename)

        for mdl_file in mdl.included_models:
            self.pack_file(
                mdl_file.filename, FileType.MODEL,
                optional=file.optional, source=filename,
            )

        for seq in mdl.sequences:
            for event in seq.events:
                if event.type in ANIM_EVENT_SOUND:
                    self.pack_soundscript(event.options)
                elif event.type in ANIM_EVENT_FOOTSTEP:
                    npc = event.options or "NPC_CombineS"
                    self.pack_soundscript(npc + ".RunFootstepLeft", source=filename)
                    self.pack_soundscript(npc + ".RunFootstepRight", source=filename)
                    self.pack_soundscript(npc + ".FootstepLeft", source=filename)
                    self.pack_soundscript(npc + ".FootstepRight", source=filename)
                elif event.type is ANIM_EVENT_PARTICLE:
                    try:
                        part_name, attach_type, attach_name = event.options.split()
                    except ValueError:
                        LOGGER.warning(
                            'Invalid particle anim event params "{}" in "{}" sequence on "{}"!',
                            event.options, seq.label, filename,
                        )
                    else:
                        self.pack_particle(part_name, source=filename)
                elif event.type is ANIM_EVENT_PARTICLE_SCRIPT:
                    # This is used for env_particlescript only.
                    try:
                        attach_name, sprite_name = event.options.split()
                    except ValueError:
                        LOGGER.warning(
                            'Invalid env_particlescript sprite event params "{}" in "{}" sequence on "{}"!',
                            event.options, seq.label, filename,
                        )
                    else:
                        self.pack_file(sprite_name, FileType.MATERIAL, source=filename)

        for break_mdl in mdl.phys_keyvalues.find_all('break', 'model'):
            # Breakable gibs inherit the base prop skin. If the gib prop doesn't have multiple
            # skins, it'll correctly just include skin 0.
            self.pack_file(
                break_mdl.value,
                FileType.MODEL,
                optional=file.optional,
                skinset=skinset,
                source=filename,
            )

    def _get_material_files(self, file: PackFile) -> None:
        """Find any needed files for a material."""
        parents: list[str] = []
        try:
            if file.data is not None:
                # Read directly from the data we have.
                mat = Material.parse(
                    io.TextIOWrapper(io.BytesIO(file.data), encoding='utf8'),
                    file.filename,
                )
            else:
                with self.fsys.open_str(file.filename) as f:
                    mat = Material.parse(f, file.filename)
        except FileNotFoundError:
            if not file.optional:
                LOGGER.warning('File "{}" does not exist!', file.filename)
            return
        except TokenSyntaxError as exc:
            LOGGER.warning(
                'File "{}" cannot be parsed:\n{}',
                file.filename,
                exc,
            )
            return

        try:
            # For 'patch' shaders, apply the originals.
            mat = mat.apply_patches(self.fsys, parent_func=parents.append)
        except ValueError:
            LOGGER.warning(
                'Error parsing Patch shader in "{}":',
                file.filename,
                exc_info=True,
            )
            return

        for vmt in parents:
            self.pack_file(vmt, FileType.MATERIAL, optional=file.optional, source=file.filename)

        for param_name, param_value in mat.items():
            param_value = param_value.casefold()
            param_type = VarType.from_name(param_name)
            if param_type is VarType.TEXTURE:
                # Skip over reference to cubemaps, or realtime buffers.
                if param_value == 'env_cubemap' or param_value.startswith('_rt_'):
                    continue
                self.pack_file(param_value, FileType.TEXTURE, optional=file.optional, source=file.filename)
            # $bottommaterial for water brushes mainly.
            if param_type is VarType.MATERIAL:
                self.pack_file(param_value, FileType.MATERIAL, optional=file.optional, source=file.filename)

    def _get_vscript_files(self, file: PackFile) -> None:
        """Find dependencies in VScripts.

        Since it's very dynamic, this only looks for obvious calls
        to PrecacheSoundScript, IncludeScript, DoIncludeScript, etc.
        """
        # Be fairly sloppy, just match func("param"
        # Also do in the binary level, so we're tolerant of any ASCII-compatible
        # encodings. Squirrel itself doesn't have complex Unicode support.
        func_pattern = re.compile(rb'([a-zA-Z]+)\s*\(\s*"([^"]+)"')
        if file.data:
            data = file.data
        else:
            try:
                with self.fsys.open_bin(file.filename) as f:
                    data = f.read()
            except FileNotFoundError:
                if not file.optional:
                    LOGGER.warning('File "{}" does not exist!', file.filename)
                return

        func: bytes
        arg: bytes
        for func, arg in func_pattern.findall(data):
            try:
                prefix, param_type = SCRIPT_FUNC_TYPES[func]
            except KeyError:
                continue
            try:
                filename = prefix + arg.decode('utf8')
            except UnicodeDecodeError:
                LOGGER.warning("Can't read filename in VScript:", exc_info=True)
                continue
            self.pack_file(filename, param_type, optional=file.optional, source=file.filename)

    def _get_choreo_files(self, file: PackFile) -> None:
        """Find sound dependencies for choreo scenes."""
        crc = choreo_scenes.checksum_filename(file.filename)
        if file.data:
            try:
                scene = choreo_scenes.Scene.parse_text(Tokenizer(file.data.decode('utf8')))
            except (TokenSyntaxError, UnicodeDecodeError) as exc:
                LOGGER.warning(
                    'Choreo scene "{}" cannot be parsed:\n{}',
                    file.filename, exc,
                )
                return
        else:
            try:
                entry = self.choreo[crc]
            except KeyError:
                try:
                    with self.fsys[file.filename].open_str('utf8') as f:
                        scene = choreo_scenes.Scene.parse_text(Tokenizer(f))
                except (TokenSyntaxError, UnicodeDecodeError) as exc:
                    LOGGER.warning(
                        'Choreo scene "{}" cannot be parsed:\n{}',
                        file.filename, exc,
                    )
                    return
            else:
                # Directly stored in the entry.
                # This doesn't include sub-scenes, TODO handle those?
                # If in the image, packing probably isn't too necessary,
                # and would require wasting time uncompressing/parsing.
                for sound in entry.sounds:
                    self.pack_soundscript(sound)
                return

        for sound in scene.used_sounds():
            self.pack_soundscript(sound, source=file.filename)
        for event in scene.iter_events(choreo_scenes.EventType.SubScene):
            self.pack_file(event.parameters[0], FileType.CHOREO, source=file.filename)

    def _get_weaponscript_kv1_files(self, file: PackFile) -> None:
        """Find any dependencies in a Keyvalues1 weapon script."""
        try:
            data = self.fsys.read_kv1(file.filename).find_key('WeaponData')
        except FileNotFoundError:
            extensionless = strip_extension(file.filename)
            try:
                encrypted_file = self.fsys[extensionless + '.ctx']
            except FileNotFoundError:
                LOGGER.warning(
                    'Weapon script {}.txt/.ctx does not exist!',
                    extensionless,
                )
            else:
                # TODO: Implement parsing ICE scripts, or provide an alt location?
                LOGGER.error(
                    'Cannot parse ICE-encrypted weapon script "{}"',
                    encrypted_file.path,
                )
            return
        except (TokenSyntaxError, NoKeyError) as exc:
            LOGGER.warning(
                'Weapon script "{}" cannot be parsed:\n{}',
                file.filename,
                exc,
            )
            return
        for kv in data.find_children('SoundData'):
            self.pack_file(kv.value, FileType.GAME_SOUND, source=file.filename)
        for mdl_name in [
            'viewmodel', 'playermodel',
            'viewmodel_dual', 'playermodel_dual',  # L4D2
            'vrmodel', 'vrmodel_l',  # HL2 VR Mod
            'worldmodel', 'addonmodel',  # L4D2
            'displaymodel',  # ASW
        ]:
            try:
                mdl_value = data[mdl_name]
            except LookupError:
                pass
            else:
                self.pack_file(mdl_value, FileType.MODEL, source=file.filename)
        for part_name in [
            'MuzzleFlashEffect_1stPerson',
            'MuzzleFlashEffect_3rdPerson',
            'EjectBrassEffect',
        ]:
            try:
                particle = data[part_name]
            except LookupError:
                pass
            else:
                self.pack_file(particle, FileType.PARTICLE_SYSTEM, source=file.filename)
        # L4D viewmodel arms
        for viewmodel in data.find_children('CharacterViewmodelAddon'):
            self.pack_file(viewmodel.value, FileType.MODEL, source=file.filename)
        for sprite_block in data.find_children('texturedata'):
            if 'file' in sprite_block:
                self.pack_file(sprite_block['file'], FileType.TEXTURE, source=file.filename)

    def _get_weaponscript_dmx_files(self, file: PackFile) -> None:
        """Find any dependencies in a DMX weapon script.

        This is used in Black Mesa Source.
        """
        try:
            with self.fsys[file.filename].open_bin() as f:
                data, fmt_name, fmt_ver = Element.parse(f)
        except FileNotFoundError:
            LOGGER.warning(
                'Weapon script "{}" does not exist!',
                file.filename,
            )
            return
        except (OSError, ValueError) as exc:
            LOGGER.warning(
                'Weapon script "{}" cannot be parsed:\n{}',
                file.filename,
                exc,
            )
            return

        try:
            sound_elem = data['sounds'].val_elem
        except KeyError:
            pass
        else:
            for attr in sound_elem.values():
                if attr.type is ValueType.STRING:
                    self.pack_file(attr.val_str, FileType.GAME_SOUND, source=file.filename)
        for mdl_name in ['viewmodel', 'playermodel', 'playermodel_multiplayer']:
            try:
                mdl_attr = data[mdl_name]
            except LookupError:
                pass
            else:
                self.pack_file(mdl_attr.val_string, FileType.MODEL, source=file.filename)


@deprecated(
    'Using entclass_resources() is deprecated, access EntityDef.engine_def() and '
    'then EntityDef.get_resources() instead.',
)
def entclass_resources(classname: str) -> Iterable[tuple[str, FileType]]:
    """Fetch a list of resources this entity class is known to use in code.

    :deprecated: Use :py:meth:`EntityDef.engine_def() <srctools.fgd.EntityDef.engine_def>` \
    then :py:meth:`EntityDef.get_resources() <srctools.fgd.EntityDef.get_resources>`.
    """
    ent_def = EntityDef.engine_def(classname)
    for filetype, filename in ent_def.get_resources(ResourceCtx(), ent=None):
        yield (filename, filetype)


@deprecated('Using entclass_canonicalise() is deprecated, access EntityDef.engine_def() instead.')
def entclass_canonicalise(classname: str) -> str:
    """Canonicalise classnames - some ents have old classnames for compatibility and the like.

    For example ``func_movelinear`` was originally ``momentary_door``. This doesn't include names
    which have observably different behaviour, like ``prop_physics_override``.

    :deprecated: Use :py:meth:`EntityDef.engine_def() <srctools.fgd.EntityDef.engine_def>`, then \
    check if the entity is an alias.
    """
    ent_def = EntityDef.engine_def(classname)
    if ent_def.is_alias:
        try:
            [base] = ent_def.bases
            return base.classname if isinstance(base, EntityDef) else base
        except ValueError:
            pass
    return classname


entclass_canonicalize = entclass_canonicalise  # noqa  # America.


@deprecated(
    'Using entclass_packfunc() is deprecated, access EntityDef.engine_def() and '
    'then EntityDef.get_resources() instead.'
)
def entclass_packfunc(classname: str) -> Callable[[PackList, Entity], object]:
    """For some entities, they have unique packing behaviour depending on keyvalues.

    If the specified classname is one, return a callable that packs it.

    :deprecated: Use :py:meth:`EntityDef.engine_def() <srctools.fgd.EntityDef.engine_def>` \
    then :py:meth:`EntityDef.get_resources() <srctools.fgd.EntityDef.get_resources>`.
    """
    ent_def = EntityDef.engine_def(classname)

    def pack_shim(packlist: PackList, ent: Entity) -> None:
        """Translate to old parameters."""
        for filetype, filename in ent_def.get_resources(ResourceCtx(
            fsys=packlist.fsys,
        ), ent=ent):
            packlist.pack_file(filename, filetype, source=classname)
    return pack_shim


@deprecated('Using entclass_iter() is deprecated, access EntityDef.engine_classes() instead.')
def entclass_iter() -> Collection[str]:
    """Yield all classnames with known behaviour.

    :deprecated: Use :py:meth:`EntityDef.engine_classes() <srctools.fgd.EntityDef.engine_classes>` instead.
    """
    return EntityDef.engine_classes()
