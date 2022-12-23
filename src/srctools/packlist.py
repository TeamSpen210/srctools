"""Handles the list of files which are desired to be packed into the BSP."""
from typing import (
    Any, Callable, Collection, Dict, Generic, Iterable, Iterator, List, Optional, Set, Tuple,
    TypeVar, Union,
)
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from zipfile import ZipFile
import io
import os
import re
import shutil
import warnings

import attrs

from srctools import conv_bool
from srctools.bsp import BSP
from srctools.const import FileType
from srctools.dmx import Attribute, Element, ValueType
from srctools.fgd import FGD, EntityDef, EntityTypes, ResourceCtx, ValueTypes as KVTypes
from srctools.filesys import (
    CACHE_KEY_INVALID, File, FileSystem, FileSystemChain, VirtualFileSystem, VPKFileSystem,
)
from srctools.keyvalues import KeyValError, Keyvalues
from srctools.mdl import MDL_EXTS, AnimEvents, Model
from srctools.particles import FORMAT_NAME as PARTICLE_FORMAT_NAME, Particle
from srctools.sndscript import SND_CHARS, Sound
from srctools.tokenizer import TokenSyntaxError
from srctools.vmf import VMF, Entity
from srctools.vmt import Material, VarType
import srctools.logger


LOGGER = srctools.logger.get_logger(__name__)
SOUND_CACHE_VERSION = '2'  # Used to allow ignoring incompatible versions.
ParsedT = TypeVar('ParsedT')

__all__ = [
    'FileType', 'FileMode', 'SoundScriptMode',
    'PackFile', 'PackList',
    'unify_path',
    'entclass_canonicalize', 'entclass_canonicalise', 'entclass_packfunc', 'entclass_resources', 'entclass_iter',
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
SCRIPT_FUNC_TYPES: Dict[bytes, Tuple[str, FileType]] = {
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


def load_fgd() -> FGD:
    """Extract the local copy of FGD data.

    This allows the analysis to not depend on local files.
    """
    import warnings
    warnings.warn(
        'Use FGD.engine_dbase() instead, '
        'this has been moved there.',
        DeprecationWarning,
        stacklevel=2,
    )
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


@attrs.define(eq=False, repr=False)
class PackFile:
    """Represents a single file we are packing.

    data is raw data to pack directly, instead of from the filesystem.
    """
    type: FileType
    filename: str
    data: Optional[bytes] = None
    optional: bool = False
    # If we've checked for dependencies of this yet.
    _analysed: bool = attrs.field(init=False, default=False)

    @property
    def virtual(self) -> bool:
        """Virtual files do not exist on the file system."""
        return self.data is not None

    def __repr__(self) -> str:
        text = f'<{"virtual " if self.virtual else ""}{self.type.name} Packfile "{self.filename}"'
        if self.data is not None:
            text += f' with {len(self.data)} bytes data>'
        else:
            text += '>'
        return text


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
    parse_func: Callable[[File[Any]], Dict[str, ParsedT]]
    # For each identifier, the filename it's in and whatever data this was parsed into.
    # Do not display in the repr, there's thousands of these.
    name_to_parsed: Dict[str, Tuple[str, Optional[ParsedT]]] = attrs.field(factory=dict, repr=False)
    # All the filenames we know about, in order. The value is then
    # whether they should be packed.
    _files: Dict[str, FileMode] = attrs.Factory(OrderedDict)
    _unparsed_file: Dict[str, File[Any]] = attrs.Factory(dict)
    # Records the contents of the cache file.
    # filename -> (cache_key, identifier_list)
    _cache: Dict[str, Tuple[int, List[str]]] = attrs.Factory(dict)

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
        except (FileNotFoundError, IOError, ValueError):
            return
        if file_type != 'SrcPacklistCache':
            LOGGER.warning('Unknown cache file "{}"!', file_type)
            return
        if file_version != 1:
            LOGGER.warning('Unrecognised cache file version {}', file_version)
            return
        for file_elem in root['files'].iter_elem():
            self._cache[file_elem.name] = (
                file_elem['key'].val_int & FILE_CACHE_TRUNC,
                list(file_elem['files'].iter_string()),
            )

    def save_cache(self, filename: Union[str, 'os.PathLike[str]']) -> None:
        """Write back new cache data."""
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
        self, filename: str, file: File[Any],
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

        identifiers: List[str]
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
        self._cache[filename] = key, identifiers
        for identifier, data in self.parse_func(file).items():
            identifiers.append(identifier)
            identifier = identifier.casefold()
            if identifier not in self.name_to_parsed:
                self.name_to_parsed[identifier] = (filename, data)

    def add_file(
        self, filename: str,
        items: Iterable[Tuple[str, ParsedT]],
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

    def fetch_data(self, identifier: str) -> Tuple[str, ParsedT]:
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

    def pack_and_get(self, lst: 'PackList', identifier: str, preload: bool=False) -> ParsedT:
        """Pack the associated filename, then return the data."""
        filename, data = self.fetch_data(identifier)

        old = self._files[filename]
        if old is not FileMode.EXCLUDE:
            self._files[filename] = FileMode.PRELOAD if preload else FileMode.INCLUDE
            lst.pack_file(filename, self.pack_type)
        return data

    def packed_files(self) -> Iterator[Tuple[str, FileMode]]:
        """Yield the used files in order."""
        for file, mode in self._files.items():
            if mode.is_used:
                yield file, mode


def _load_soundscript(file: File[Any]) -> Dict[str, Sound]:
    """Parse a soundscript file, logging errors that occur."""
    try:
        with file.open_str(encoding='cp1252') as f:
            kv = Keyvalues.parse(f, file.path, allow_escapes=False)
        return Sound.parse(kv)
    except FileNotFoundError:
        # It doesn't exist, complain and pretend it's empty.
        LOGGER.warning('Soundscript "{}" does not exist!', file.path)
        return {}
    except (KeyValError, ValueError):
        LOGGER.warning('Soundscript "{}" could not be parsed:', file.path, exc_info=True)
        return {}


def _load_particle_system(file: File[Any]) -> Dict[str, Particle]:
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
    particles: ManifestedFiles[Particle]

    _packed_particles: Set[str]
    _files: Dict[str, PackFile]
    # folder, ext, data -> filename used
    _inject_files: Dict[Tuple[str, str, bytes], str]
    # Cache of the models used for breakable chunks.
    _break_chunks: Dict[str, List[str]]
    # For each model, defines the skins the model uses. None means at least
    # one use is unknown, so all skins could potentially be used.
    skinsets: Dict[str, Optional[Set[int]]]

    def __init__(self, fsys: FileSystemChain) -> None:
        self.fsys = fsys
        self.soundscript = ManifestedFiles('soundscript', FileType.SOUNDSCRIPT, _load_soundscript)
        self.particles = ManifestedFiles('particle', FileType.PARTICLE_FILE, _load_particle_system)
        self._packed_particles = set()
        self._files = {}
        self._inject_files = {}
        self._break_chunks = {}
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
        filename: str,
        data_type: FileType=FileType.GENERIC,
        data: Optional[bytes] = None,
        skinset: Optional[Set[int]] = None,
        optional: bool = False,
    ) -> None:
        """Queue the given file to be packed.

        If data is set, this file will use the given data instead of any
        on-disk data. The data_type parameter allows specifying the kind of
        file, which ensures it can be treated appropriately.

        If the file is a model, skinset allows restricting which skins are used.
        If None (default), all skins may be used. Otherwise it is a set of
        skins. If all uses of a model restrict the skin, only those skins need
        to be packed.
        If optional is set, this will be marked as optional so no errors occur
        if it isn't in the filesystem.
        """
        filename = os.fspath(filename)

        # Assume an empty filename is an optional value.
        if not filename:
            if data is not None:
                raise ValueError('Data provided with no filename!')
            return

        # Disallow tabs, to guard against cases where we incorrectly parse \t in file paths.
        if '\t' in filename:
            raise ValueError(f'No tabs are allowed in filenames ({filename!r})')

        if data_type is FileType.ENTCLASS_FUNC or data_type is FileType.ENTITY:
            raise ValueError(f'File type "{data_type.name}" must not be packed directly!')

        if data_type is FileType.GAME_SOUND:
            self.pack_soundscript(filename)
            return  # This packs the soundscript and wav for us.
        if data_type is FileType.PARTICLE:
            self.pack_particle(filename)
            return  # This packs the PCF and material if required.
        if data_type is FileType.CHOREO:
            # self.pack_choreo(filename)  # TODO: Choreo scene parsing
            return
        if data_type is FileType.BREAKABLE_CHUNK:
            self.pack_breakable_chunk(filename)
            return  # Packs additional models.

        # If soundscript data is provided, load it and force-include it.
        elif data_type is FileType.SOUNDSCRIPT and data:
            try:
                sounds = Sound.parse(Keyvalues.parse(data.decode('cp1252'), filename))
            except (KeyValError, ValueError):
                LOGGER.warning('Soundscript "{}" could not be parsed:', filename, exc_info=True)
            else:
                self.soundscript.add_file(filename, sounds.items(), FileMode.INCLUDE)

        filename = unify_path(filename)

        if data_type is FileType.MATERIAL or (
            data_type is FileType.GENERIC and filename.endswith('.vmt')
        ):
            data_type = FileType.MATERIAL
            if not filename.startswith('materials/'):
                filename = 'materials/' + filename
            # This will replace .spr materials, which don't exist any more.
            filename = strip_extension(filename) + '.vmt'
        elif data_type is FileType.TEXTURE or (
            data_type is FileType.GENERIC and filename.endswith('.vtf')
        ):
            data_type = FileType.TEXTURE
            if not filename.startswith('materials/'):
                filename = 'materials/' + filename
            if not filename.endswith('.hdr'):
                # Strip all other extensions, then add vtf unconditionally.
                filename = strip_extension(filename)
            filename = filename + '.vtf'
        elif data_type is FileType.VSCRIPT_SQUIRREL or (
            data_type is FileType.GENERIC and filename.endswith('.nut')
        ):
            data_type = FileType.VSCRIPT_SQUIRREL
            filename = strip_extension(filename) + '.nut'

        if data_type is FileType.MODEL or filename.endswith('.mdl'):
            data_type = FileType.MODEL
            if not filename.startswith('models/'):
                filename = 'models/' + filename
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

        if filename.endswith('.nut'):
            data_type = FileType.VSCRIPT_SQUIRREL

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

        # Try to promote generic to other types if known.
        if data_type is FileType.GENERIC:
            try:
                data_type = EXT_TYPE[ext]
            except KeyError:
                pass
        elif data_type is FileType.SOUNDSCRIPT:
            if ext != '.txt':
                raise ValueError(f'"{filename}" cannot be a soundscript!')

        self._files[filename] = PackFile(data_type, filename, data, optional)

    def inject_file(self, data: bytes, folder: str, ext: str) -> str:
        """Inject a generated file into the map and return the full name.

        The file will be named using the format "INJECT_<hex>".
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
            full_name = f"{folder}/INJECT_{name_hash}.{ext}"
            if full_name not in self._files:
                break
            name_hash = format(abs(hash(name_hash)), 'x')
        self.pack_file(full_name, data=data)
        self._inject_files[folder, ext, data] = full_name
        return full_name

    def inject_vscript(self, code: str, folder: str='inject') -> str:
        """Specialised variant of inject_file() for VScript code specifically.

        This returns the script name suitable for passing to Entity Scripts.
        """
        return self.inject_file(
            code.encode('ascii'),
            os.path.join('scripts/vscripts', folder), '.nut'
            # Strip off the scripts/vscripts/ folder since it's implied.
        )[17:]

    def pack_soundscript(self, sound_name: str) -> None:
        """Pack a soundscript or raw sound file."""
        # Blank means no sound is used.
        if not sound_name:
            return

        sound_name = sound_name.casefold().replace('\\', '/')
        # Check for raw sounds first.
        if sound_name.endswith(('.wav', '.mp3')):
            sound_name = sound_name.lstrip(SND_CHARS)
            if not sound_name.startswith('sound/'):
                sound_name = 'sound/' + sound_name
            self.pack_file(sound_name)
            return

        try:
            soundscript = self.soundscript.pack_and_get(self, sound_name)
        except KeyError:
            LOGGER.warning('Unknown sound "{}"!', sound_name)
            return

        for sound in soundscript.sounds:
            self.pack_file('sound/' + sound.lstrip(SND_CHARS).replace('\\', '/'))

    def pack_particle(self, particle_name: str, preload: bool = False) -> None:
        """Pack a particle system and the raw PCFs."""
        # Blank means no particle is used, also skip if we already packed.
        if not particle_name or particle_name in self._packed_particles:
            return
        try:
            particle = self.particles.pack_and_get(self, particle_name, preload)
        except KeyError:
            LOGGER.warning('Unknown particle "{}"!', particle_name)
            self._packed_particles.add(particle_name)
            return
        # Pack the sprites the particle system uses.
        try:
            mat = particle.options['material'].val_str
        except KeyError:
            pass
        else:
            self.pack_file(mat, FileType.MATERIAL)
        for rend in particle.renderers:
            if rend.function.casefold() == 'render models':
                try:
                    mdl = rend.options['sequence 0 model'].val_str
                except KeyError:
                    LOGGER.warning('Particle {} has model render with no model?', particle_name)
                else:
                    self.pack_file(mdl, FileType.MODEL)
        for child in particle.children:
            self._packed_particles.add(child.particle)
            self.pack_particle(child.particle, preload)

    def pack_breakable_chunk(self, chunkname: str) -> None:
        """Pack the generic gib model for the given chunk name."""
        if self._break_chunks is None:
            # Need to load the file.
            self.pack_file('scripts/propdata.txt')
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
        for mdl in mdl_list:
            self.pack_file(mdl, FileType.MODEL)

    def load_soundscript(
        self,
        file: File[Any],
        *,
        always_include: bool=False,
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

    def load_particle_system(self, filename: str, mode: FileMode=FileMode.UNKNOWN) -> Iterable[Particle]:
        """Read in the specified particle system and record the particles for usage checking."""
        try:
            particles = _load_particle_system(self.fsys[filename])
        except FileNotFoundError:
            # It doesn't exist, complain and pretend it's empty.
            LOGGER.warning('Particle system "{}" does not exist!', filename)
            return ()

        self.particles.add_file(filename, particles.items(), mode)
        return particles.values()

    def load_soundscript_manifest(self, cache_file: Union[Path, str, None]=None) -> None:
        """Read the soundscript manifest, and read all mentioned scripts.

        If cache_file is provided, it should be a path to a file used to
        cache the file reading for later use.
        """
        try:
            man = self.fsys.read_kv1('scripts/game_sounds_manifest.txt', encoding='cp1252')
        except FileNotFoundError:
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

    def load_particle_manifest(self, cache_file: Union[Path, str, None]=None) -> None:
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

        in_manifest: Set[str] = set()

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

    def write_manifest(self) -> None:
        """Deprecated, call write_soundscript_manifest()."""
        warnings.warn('Renamed to write_soundscript_manifest()', DeprecationWarning)
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

        buf = bytearray()
        for line in manifest.export():
            buf.extend(line.encode('utf8'))

        self.pack_file('scripts/game_sounds_manifest.txt', FileType.GENERIC, bytes(buf))

    def write_particles_manifest(self, manifest_name: str) -> None:
        """Write a particles manifest, so that used particles can be loaded."""
        manifest = Keyvalues('particles_manifest', [])
        for filename, mode in self.particles.packed_files():
            if mode is FileMode.PRELOAD:
                filename = '!' + filename
            manifest.append(Keyvalues('file', filename))

        buf = bytearray()
        for line in manifest.export():
            buf.extend(line.encode('utf8'))

        self.pack_file(manifest_name, FileType.GENERIC, bytes(buf))

    def pack_from_bsp(self, bsp: BSP) -> None:
        """Pack files found in BSP data (excluding entities)."""
        for prop in bsp.props:
            # Static props obviously only use one skin.
            self.pack_file(prop.model, FileType.MODEL, skinset={prop.skin})

        # These are all the materials the BSP references, including brushes and overlays.
        for mat in bsp.textures:
            self.pack_file('materials/{}.vmt'.format(mat.lower()), FileType.MATERIAL)

    def pack_fgd(
        self,
        vmf: VMF,
        fgd: FGD,
        mapname: str='',
        tags: Iterable[str]=(),
    ) -> None:
        """Analyse the map to pack files. We use the FGD to easily handle this."""
        # Don't show the same keyvalue warning twice, it's just noise.
        unknown_keys: Set[Tuple[str, str]] = set()

        # Definitions for the common keyvalues on all entities.
        try:
            base_entity = fgd['_CBaseEntity_']
        except KeyError:
            LOGGER.warning('No CBaseEntity definition!')
            base_entity = EntityDef(EntityTypes.BASE)

        res_ctx = ResourceCtx(
            fgd=fgd,
            fsys=self.fsys,
            mapname=mapname,
            tags=tags,
        )

        for ent in vmf.entities:
            # Allow opting out packing specific entities.
            if conv_bool(ent.pop('srctools_nopack', '')):
                continue

            classname = ent['classname']
            try:
                ent_class = fgd[classname]
            except KeyError:
                if (classname, '') not in unknown_keys:
                    LOGGER.warning('Unknown class "{}"!', classname)
                    unknown_keys.add((classname, ''))
                # Fall back to generic keyvalues.
                ent_class = base_entity

            skinset: Optional[Set[int]]
            if ent['skinset'] != '':
                # Special key for us - if set this is a list of skins this
                # entity is pledging it will restrict itself to.
                skinset = {
                    int(x)
                    for x in ent.pop('skinset').split()
                }
            else:
                skinset = None

            value: str
            key: str
            for key in set(ent) | set(ent_class.kv):
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
                    # Models are set on all brush entities, and are always either
                    # a '*37' brush ref, a model, or a sprite.
                    value = ent[key]
                    if value and value[:1] != '*':
                        self.pack_file(value, skinset=skinset)
                    continue
                try:
                    kv = ent_class.kv[key]
                    val_type = kv.type
                    default = kv.default
                except KeyError:
                    # Suppress this error for unknown classes, we already
                    # showed a warning above.
                    if ent_class is not base_entity and (classname, key) not in unknown_keys:
                        unknown_keys.add((ent_class.classname, key))
                        LOGGER.warning('Unknown keyvalue "{}" for ent of type "{}"!',
                                       key, ent['classname'])
                    continue

                value = ent[key, default]

                # Ignore blank values, they're not useful.
                if not value:
                    continue

                if val_type is KVTypes.STR_MATERIAL:
                    self.pack_file(value, FileType.MATERIAL)
                elif val_type is KVTypes.STR_MODEL:
                    self.pack_file(value, FileType.MODEL)
                elif val_type is KVTypes.EXT_STR_TEXTURE:
                    self.pack_file(value, FileType.TEXTURE)
                elif val_type is KVTypes.STR_VSCRIPT:
                    for script in value.split():
                        self.pack_file('scripts/vscripts/' + script)
                elif val_type is KVTypes.STR_VSCRIPT_SINGLE:
                    self.pack_file('scripts/vscripts/' + value)
                elif val_type is KVTypes.STR_SPRITE:
                    if not value.casefold().startswith('sprites/'):
                        value = 'sprites/' + value
                    if not value.casefold().startswith('materials/'):
                        value = 'materials/' + value

                    self.pack_file(value, FileType.MATERIAL)
                elif val_type is KVTypes.STR_SOUND:
                    self.pack_soundscript(value)
                elif val_type is KVTypes.STR_PARTICLE:
                    self.pack_particle(value)

            # Handle resources that's coded into different entities with our internal database.
            for file_type, filename in ent_class.get_resources(res_ctx, ent=ent):
                self.pack_file(filename, file_type)

        # Handle worldspawn here - this is fairly special.
        sky_name = vmf.spawn['skyname']
        for suffix in ['bk', 'dn', 'ft', 'lf', 'rt', 'up']:
            self.pack_file(
                'materials/skybox/{}{}.vmt'.format(sky_name, suffix),
                FileType.MATERIAL,
            )
        self.pack_file(vmf.spawn['detailmaterial'], FileType.MATERIAL)

        detail_script = vmf.spawn['detailvbsp']
        if detail_script:
            self.pack_file(detail_script, FileType.GENERIC)
            try:
                detail_props = self.fsys.read_kv1(detail_script, 'ansi')
            except FileNotFoundError:
                LOGGER.warning('detail.vbsp file does not exist: "{}"', detail_script)
            except Exception:
                LOGGER.warning(
                    'Could not parse detail.vbsp file: ',
                    exc_info=True
                )
            else:
                # We only need to worry about models, the sprites are a single
                # sheet packed above.
                for prop in detail_props.iter_tree():
                    if prop.name == 'model':
                        self.pack_file(prop.value, FileType.MODEL)

    def pack_into_zip(
        self,
        bsp: BSP,
        *,
        whitelist: Iterable[FileSystem[Any]]=(),
        blacklist: Iterable[FileSystem[Any]]=(),
        callback: Callable[[str], Optional[bool]]=lambda f: None,
        dump_loc: Optional[Path]=None,
        only_dump: bool=False,
        ignore_vpk: bool=True,
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
        packed_files: Dict[str, Tuple[str, bytes]] = {}

        all_systems: Set[FileSystem[Any]] = {
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

            if file.data is not None:
                # Always pack, we've got custom data.
                LOGGER.debug('CUSTOM DATA: {}', fname)
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
                    LOGGER.warning('WARNING: "{}" not packed!', file.filename)
                continue

            if fname.casefold().endswith('.bik'):
                # BINK cannot be packed, always skip.
                LOGGER.debug('EXT:  {}', fname)
                continue

            should_pack = callback(sys_file.path)
            if should_pack is None:
                should_pack = self.fsys.get_system(sys_file) in allowed

            if should_pack:
                LOGGER.debug('ADD:  {}', fname)
                with sys_file.open_bin() as f:
                    data = f.read()
                if not only_dump:
                    packed_files[fname.casefold()] = (fname, data)
                if dump_loc is not None:
                    path = dump_loc / fname
                    path.parent.mkdir(exist_ok=True, parents=True)
                    path.write_bytes(data)
            else:
                LOGGER.debug('SKIP: {}', fname)

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
                                self.pack_file(hdr_tex, optional=True)
                    elif file.type is FileType.VSCRIPT_SQUIRREL:
                        self._get_vscript_files(file)
                except Exception as exc:
                    # Skip errors in the file format - means we can't find the dependencies.
                    LOGGER.warning('Bad file "{}"!', file.filename, exc_info=exc)

    def _get_model_files(self, file: PackFile) -> None:
        """Find any needed files for a model."""
        filename, ext = os.path.splitext(file.filename)

        # Some of these are optional, so just skip. Do not re-pack the MDL itself, that's
        # pointless and will also erase the skinset!
        for ext in MDL_EXTS:
            if ext != '.mdl':
                component = filename + ext
                if component in self.fsys:
                    self.pack_file(component)

        if file.data is not None:
            # We need to add that file onto the system, so it's loaded.
            self.fsys.add_sys(VirtualFileSystem({
                file.filename: file.data,
            }), priority=True)
            try:
                mdl = Model(self.fsys, self.fsys[file.filename])
            finally:  # Remove this system.
                self.fsys.systems.pop(0)
        else:
            try:
                mdl = Model(self.fsys, self.fsys[file.filename])
            except FileNotFoundError:
                if not file.optional:
                    LOGGER.warning('Can\'t find model "{}"!', file.filename)
                return

        for tex in mdl.iter_textures(self.skinsets.get(file.filename, None)):
            self.pack_file(tex, FileType.MATERIAL, optional=file.optional)

        for mdl_file in mdl.included_models:
            self.pack_file(mdl_file.filename, FileType.MODEL, optional=file.optional)

        for seq in mdl.sequences:
            for event in seq.events:
                if event.type in ANIM_EVENT_SOUND:
                    self.pack_soundscript(event.options)
                elif event.type in ANIM_EVENT_FOOTSTEP:
                    npc = event.options or "NPC_CombineS"
                    self.pack_soundscript(npc + ".RunFootstepLeft")
                    self.pack_soundscript(npc + ".RunFootstepRight")
                    self.pack_soundscript(npc + ".FootstepLeft")
                    self.pack_soundscript(npc + ".FootstepRight")
                elif event.type is ANIM_EVENT_PARTICLE:
                    try:
                        part_name, attach_type, attach_name = event.options.split()
                    except ValueError:
                        LOGGER.warning(
                            'Invalid particle anim event params "{}" in "{}" sequence on "{}"!',
                            event.options, seq.label, file.filename,
                        )
                    else:
                        self.pack_particle(part_name)

        for break_mdl in mdl.phys_keyvalues.find_all('break', 'model'):
            self.pack_file(break_mdl.value, FileType.MODEL, optional=file.optional)

    def _get_material_files(self, file: PackFile) -> None:
        """Find any needed files for a material."""

        parents = []  # type: List[str]
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
            self.pack_file(vmt, FileType.MATERIAL, optional=file.optional)

        for param_name, param_value in mat.items():
            param_value = param_value.casefold()
            param_type = VarType.from_name(param_name)
            if param_type is VarType.TEXTURE:
                # Skip over reference to cubemaps, or realtime buffers.
                if param_value == 'env_cubemap' or param_value.startswith('_rt_'):
                    continue
                self.pack_file(param_value, FileType.TEXTURE, optional=file.optional)
            # $bottommaterial for water brushes mainly.
            if param_type is VarType.MATERIAL:
                self.pack_file(param_value, FileType.MATERIAL, optional=file.optional)

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
            self.pack_file(filename, param_type, optional=file.optional)


_engine_fgd: Optional[FGD] = None


def _get_engine_fgd() -> FGD:
    """Fetch and cache the engine FGD database."""
    global _engine_fgd
    if _engine_fgd is None:
        _engine_fgd = FGD.engine_dbase()
    return _engine_fgd


def entclass_resources(classname: str) -> Iterable[Tuple[str, FileType]]:
    """Fetch a list of resources this entity class is known to use in code.

    :deprecated: Use :py:func:`EntityDef.engine_cls()` then :py:func:`EntityDef.get_resources()`.
    """
    warnings.warn(
        'Using entclass_resources() is deprecated, access FGD.engine_db() and then '
        'EntityDef.get_resources() instead.',
        DeprecationWarning, stacklevel=2,
    )
    fgd = _get_engine_fgd()
    try:
        ent_def = fgd[classname]
    except KeyError:
        raise KeyError(classname) from None

    for filetype, filename in ent_def.get_resources(ResourceCtx(fgd=fgd), ent=None):
        yield (filename, filetype)


def entclass_canonicalise(classname: str) -> str:
    """Canonicalise classnames - some ents have old classnames for compatibility and the like.

    For example func_movelinear was originally momentary_door. This doesn't include names which
    have observably different behaviour, like prop_physics_override.

    :deprecated: Use :py:func:`FGD.engine_db()`, then check if the entity is an alias.
    """
    warnings.warn(
        f'Using entclass_packfun() is deprecated, access FGD.engine_db() and then '
        f'EntityDef.get_resources() instead.',
        DeprecationWarning, stacklevel=2,
    )
    fgd = _get_engine_fgd()
    try:
        ent_def = fgd[classname]
    except KeyError:
        return classname
    if ent_def.is_alias:
        try:
            [base] = ent_def.bases
            return base.classname if isinstance(base, EntityDef) else base
        except ValueError:
            pass
    return classname


entclass_canonicalize = entclass_canonicalise  # noqa  # America.


def entclass_packfunc(classname: str) -> Callable[[PackList, Entity], object]:
    """For some entities, they have unique packing behaviour depending on keyvalues.

    If the specified classname is one, return a callable that packs it.

    :deprecated: Use :py:func:`EntityDef.engine_cls()` then :py:func:`EntityDef.get_resources()`.
    """
    warnings.warn(
        'Using entclass_packfunc() is deprecated, access FGD.engine_db() and then '
        'EntityDef.get_resources() instead.',
        DeprecationWarning, stacklevel=2,
    )
    fgd = _get_engine_fgd()
    try:
        ent_def = fgd[classname]
    except KeyError:
        raise KeyError(classname) from None

    def pack_shim(packlist: PackList, ent: Entity) -> None:
        """Translate to old parameters."""
        for filetype, filename in ent_def.get_resources(ResourceCtx(
            fgd=fgd,
            fsys=packlist.fsys,
        ), ent=ent):
            packlist.pack_file(filename, filetype)
    return pack_shim


def entclass_iter() -> Collection[str]:
    """Yield all classnames with known behaviour.

    :deprecated: Use :py:func:`FGD.engine_db()` instead.
    """
    warnings.warn(
        'Using entclass_iter() is deprecated, access FGD.engine_db() instead.',
        DeprecationWarning, stacklevel=2,
    )
    return [
        ent.classname
        for ent in _get_engine_fgd().entities.values()
        if not isinstance(ent.resources, tuple)
    ]
