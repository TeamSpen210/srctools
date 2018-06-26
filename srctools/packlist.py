"""Handles the list of files which are desired to be packed into the BSP."""
from collections import OrderedDict
from typing import Iterable, Dict, Tuple, List
from enum import Enum
from zipfile import ZipFile
import os

from srctools.property_parser import Property
from srctools.vmf import VMF
from srctools.fgd import FGD, ValueTypes as KVTypes, KeyValues
from srctools.bsp import BSP
from srctools.filesys import FileSystem, VPKFileSystem, FileSystemChain, File
from srctools.mdl import Model
from srctools.vmt import Material, VarType
from srctools.sndscript import Sound
from srctools.logger import get_logger

LOGGER = get_logger(__name__)


class FileType(Enum):
    """Types of files we might pack."""
    GENERIC = 0  # Other file types.
    SOUNDSCRIPT = 1  # Should be added to the manifest

    GAME_SOUND = 2  # 'world.blah' sound - lookup the soundscript, and raw files.

    # Assume these files are present even
    # if we can't find them. (rt_ textures for example.)
    # also don't bother looking for dependencies.
    WHITELIST = 3
    
    # This file might be present - if it is pack it.
    # If not it's not an error.
    OPTIONAL = 4
    
    PARTICLE_FILE = 'pcf'  # Should be added to the manifest
    
    VSCRIPT_SQUIRREL = 'nut'
    
    # Implies packing referenced materials and textures.
    MATERIAL = 'vmt'
    
    TEXTURE = 'vtf'  # May want .hdr.vtf too.
    
    # Requires lookup of vtx, vvd, phy files too - in the model data.
    # also any skins used.
    MODEL = 'mdl'


EXT_TYPE = {
    '.' + filetype.value: filetype
    for filetype in FileType
    if isinstance(filetype.value, str)
}

INJECT_FORMAT = "{}/INJECT_{:X}.{ext}"

# noinspection PyProtectedMember
from srctools._class_resources import CLASS_RESOURCES


def load_fgd() -> FGD:
    """Extract the local copy of FGD data.

    This allows the analysis to not depend on local files.
    """
    # Import these here, so the overall package doesn't require them.
    from pkg_resources import resource_stream
    from lzma import LZMAFile
    with LZMAFile(resource_stream('srctools', 'fgd.lzma')) as f:
        return FGD.unserialise(f)


class PackFile:
    """Represents a single file we are packing.
    
    data is raw data to pack directly, instead of from the filesystem.
    """
    __slots__ = ['type', 'filename', 'data', '_analysed']
    def __init__(
        self, 
        type: FileType,
        filename: str, 
        data: bytes=None,
    ):
        self.type = type
        self.filename = filename
        self.data = data
        # If we've checked for dependencies
        self._analysed = False

    @property
    def virtual(self) -> bool:
        return self.data is not None

    def __repr__(self):
        text = '<{}{} Packfile "{}"'.format(
            'virtual ' if self.virtual else '',
            self.type.name,
            self.filename,
        )
        if self.data is not None:
            text += ' with {} bytes data>'.format(len(self.data))
        else:
            text += '>'
        return text


def unify_path(path: str):
    """Convert paths to a unique form."""
    path = os.path.normpath(path).casefold().replace('\\', '/')
    if '../' in path:
        raise ValueError('Path tried to escape root!')
    return path.lstrip('/')


class PackList:
    """Represents a list of resources for a map."""
    def __init__(self, fsys: FileSystemChain):
        self._files = {}  # type: Dict[str, PackFile]
        self.fsys = fsys
        # Soundscript name -> soundscript path, raw WAVs
        self.soundscripts = {}  # type: Dict[str, Tuple[str, List[str]]]
        # Filenames of soundscripts - used to generate the manifest.
        # Ordered dictionary to keep the order intact, with
        # filename keys mapping to a bool value indicating if it's included
        # in the map.
        self.soundscript_files = OrderedDict()

        # folder, ext, data -> filename used
        self._inject_files = {}  # type: Dict[Tuple[str, str, bytes], str]

    def __getitem__(self, path: str):
        """Look up a packfile by filename."""
        return self._files[unify_path(path)]

    def __len__(self):
        return len(self._files)

    def __iter__(self):
        return iter(self._files.values())

    def __contains__(self, path):
        return unify_path(path) in self._files

    def pack_file(
        self,
        filename: str,
        data_type: FileType=FileType.GENERIC,
        data: bytes=None,
    ) -> None:
        """Queue the given file to be packed.

        If data is set, this file will use the given data instead of any
        on-disk data. The data_type parameter allows specifying the kind of
        file, which ensures it can be treated appropriately.
        """
        filename = os.fspath(filename)

        if '\t' in filename:
            raise ValueError(
                'No tabs are allowed in filenames ({!r})'.format(filename)
            )

        if data_type is FileType.GAME_SOUND:
            self.pack_soundscript(filename)
            return  # This packs the soundscript and wav for us.

        # If soundscript data is provided, load it and force-include it.
        elif data_type is FileType.SOUNDSCRIPT and data:
            self._parse_soundscript(
                Property.parse(data.decode('utf8'), filename),
                filename,
                always_include=True,
            )

        if data_type is FileType.MATERIAL or (
            data_type is FileType.GENERIC and filename.endswith('.vmt')
        ):
            data_type = FileType.MATERIAL
            if not filename.startswith('materials/'):
                filename = 'materials/' + filename
            if not filename.endswith(('.vmt', '.spr')):
                filename = filename + '.vmt'
        elif data_type is FileType.TEXTURE or (
            data_type is FileType.GENERIC and filename.endswith('.vtf')
        ):
            data_type = FileType.TEXTURE
            if not filename.startswith('materials/'):
                filename = 'materials/' + filename
            if not filename.endswith('.vtf'):
                filename = filename + '.vtf'

        path = unify_path(filename)

        try:
            file = self._files[path]
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
                raise ValueError('"{}": two different data streams!'.format(filename))
            # else: we had an override, but asked to just pack now. That's fine.

            if file.type is data_type:
                # Same, no problems - just packing on top.
                return

            if file.type is FileType.GENERIC:
                file.type = data_type  # This is fine, we now know it has behaviour...
            elif data_type is FileType.GENERIC:
                pass  # If we know it has behaviour already, that trumps generic.
            elif data_type is FileType.WHITELIST:
                file.type = data_type  # Blindly believe this.
            else:
                raise ValueError('"{}": {} can\'t become a {}!'.format(
                    filename,
                    file.type.name,
                    data_type.name,
                ))
            return  # Don't re-add this.

        start, ext = os.path.splitext(path)

        # Try to promote generic to other types if known.
        if data_type is FileType.GENERIC:
            try:
                data_type = EXT_TYPE[ext]
            except KeyError:
                pass
        elif data_type is FileType.SOUNDSCRIPT:
            if ext != '.txt':
                raise ValueError('"{}" cannot be a soundscript!'.format(filename))

        self._files[path] = PackFile(
            data_type,
            filename,
            data,
        )

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
        name_hash = data
        while True:
            # Repeatedly hashing permutes the data, until we stop colliding.
            name_hash = format(hash(name_hash), 'x')
            full_name = INJECT_FORMAT.format(folder, name_hash, ext)
            if full_name not in self._files:
                break
        self.pack_file(full_name, data=data)
        self._inject_files[folder, ext, data] = full_name
        return full_name

    def pack_soundscript(self, sound_name: str):
        """Pack a soundscript or raw sound file."""
        sound_name = sound_name.casefold()
        # Check for raw sounds first.
        if sound_name.endswith(('.wav', '.mp3')):
            self.pack_file('sound/' + sound_name)
            return

        try:
            script_path, sound = self.soundscripts[sound_name]
        except KeyError:
            LOGGER.warning('Unknown sound "{}"!', sound_name)
            return

        # Mark the soundscript as something we need to add to the manifest.
        self.soundscript_files[script_path] = True
        self.pack_file(script_path, FileType.SOUNDSCRIPT)

        for raw_file in sound:
            self.pack_file('sound/' + raw_file)

    def load_soundscript(
        self,
        file: File,
        *,
        always_include: bool=False,
    ) -> Iterable[str]:
        """Read in a soundscript and record which files use it.

        If always_include is True, it will be included in the manifests even
        if it isn't used.

        The sounds registered by this soundscript are returned.
        """
        with file.sys, file.open_str() as f:
            props = Property.parse(f, file.path)
        return self._parse_soundscript(props, file.path, always_include)

    def _parse_soundscript(
        self,
        props: Property,
        path: str,
        always_include: bool = False,
    ) -> Iterable[str]:
        """Read in a soundscript and record which files use it.

        If always_include is True, it will be included in the manifests even
        if it isn't used.
        """
        if always_include or path not in self.soundscript_files:
            self.soundscript_files[path] = always_include

        scripts = Sound.parse(props)

        for name, sound in scripts.items():
            self.soundscripts[name] = path, [
                snd.lstrip('*@#<>^)}$!?').replace('\\', '/')
                for snd in sound.sounds
            ]

        return list(scripts.keys())

    def load_soundscript_manifest(self, cache_file: str=None):
        """Read the soundscript manifest, and read all mentioned scripts.

        If cache_file is provided, it should be a path to a file used to
        cache the file reading for later use.
        """
        try:
            man = self.fsys.read_prop('scripts/game_sounds_manifest.txt')
        except FileNotFoundError:
            return

        cache_data = {}  # type: Dict[str, Tuple[int, Property]]
        if cache_file is not None:
            try:
                f = open(cache_file)
            except FileNotFoundError:
                pass
            else:
                with f:
                    old_cache = Property.parse(f, cache_file)
                for cache_prop in old_cache:
                    cache_data[cache_prop.name] = (
                        cache_prop.int('cache_key'),
                        cache_prop.find_key('files')
                    )

            # Regenerate from scratch each time - that way we remove old files
            # from the list.
            new_cache_data = Property(None, [])
        else:
            new_cache_data = None

        with self.fsys:
            for prop in man.find_children('game_sounds_manifest'):
                if not prop.name.endswith('_file'):
                    continue
                try:
                    cache_key, cache_files = cache_data[prop.value.casefold()]
                except KeyError:
                    cache_key = -1
                    cache_files = None

                file = self.fsys[prop.value]
                cur_key = file.cache_key()

                if cache_key != cur_key or cache_key == -1:
                    sounds = self.load_soundscript(file, always_include=True)
                else:
                    # Read from cache.
                    sounds = []
                    for cache_prop in cache_files:
                        sounds.append(cache_prop.real_name)
                        self.soundscripts[cache_prop.real_name] = (prop.value, [
                            snd.value
                            for snd in cache_prop
                        ])

                # The soundscripts in the manifests are always included,
                # since many would be part of the core code (physics, weapons,
                # ui, etc). Just keep those loaded, no harm since vanilla does.
                self.soundscript_files[file.path] = True

                if new_cache_data is not None:
                    new_cache_data.append(Property(prop.value, [
                        Property('cache_key', str(cur_key)),
                        Property('Files', [
                            Property(snd, [
                                Property('snd', raw)
                                for raw in self.soundscripts[snd][1]
                            ])
                            for snd in sounds
                        ])
                    ]))

        if cache_file is not None:
            # Write back out our new cache with updated data.
            with open(cache_file, 'w') as f:
                for line in new_cache_data.export():
                    f.write(line)

    def write_manifest(self, map_name: str=None):
        """Produce and pack a manifest file for this map.

        If map_name is provided, the script in the custom content position
        to be automatically loaded for that name. Otherwise, it will be packed
        such that it can override the master manifest with
        sv_soundemitter_flush.
        """
        manifest = Property('game_sounds_manifest', [])
        for snd, is_enabled in self.soundscript_files.items():
            if not is_enabled:
                continue
            manifest.append(Property('precache_file', snd))

        buf = bytearray()
        for line in manifest.export():
            buf.extend(line.encode('utf8'))

        self.pack_file(
            'map/{}_level_sounds.txt'.format(map_name)
            if map_name else
            'scripts/game_sounds_manifest.txt',
            FileType.SOUNDSCRIPT,
            bytes(buf),
        )

    def pack_from_bsp(self, bsp: BSP):
        """Pack files found in BSP data (excluding entities)."""
        for static_prop in bsp.static_prop_models():
            self.pack_file(static_prop, FileType.MODEL)

        for mat in bsp.read_texture_names():
            self.pack_file('materials/{}.vmt'.format(mat.lower()), FileType.MATERIAL)

    def pack_fgd(self, vmf: VMF, fgd: FGD):
        """Analyse the map to pack files. We use the FGD to easily handle this."""
        for ent in vmf.entities:
            classname = ent['classname']
            try:
                ent_class = fgd[classname]
            except KeyError:
                LOGGER.warning('Unknown class "{}"!', classname)
                continue
            for key in set(ent.keys) | set(ent_class.kv):
                # These are always present on entities, and we don't have to do
                # any packing for them.
                # Origin/angles might be set (brushes, instances) even for ents
                # that don't use them.
                if key in ('classname', 'hammerid', 'origin', 'angles', 'skin', 'pitch'):
                    continue
                elif key == 'model':
                    # Models are set on all brush entities, and are always either
                    # a '*37' brush ref, a model, or a sprite.
                    value = ent[key]
                    if value and value[:1] != '*':
                        self.pack_file(value)
                    continue
                try:
                    kv = ent_class.kv[key]  # type: KeyValues
                    val_type = kv.type
                    default = kv.default
                except KeyError:
                    LOGGER.warning('Unknown keyvalue "{}" for ent of type "{}"!',
                                   key, ent['classname'])
                    val_type = None  # Doesn't match any enum.
                    default = ''

                value = ent[key, default]

                # Ignore blank values, they're not useful.
                if not value:
                    continue

                if classname == 'env_projectedtexture' and key == 'texturename':
                    # Special case - this is a VTF, not a material.
                    self.pack_file(value, FileType.TEXTURE)
                    continue

                if val_type is KVTypes.STR_MATERIAL:
                    self.pack_file(value, FileType.MATERIAL)
                elif val_type is KVTypes.STR_MODEL:
                    self.pack_file(value, FileType.MODEL)
                elif val_type is KVTypes.STR_VSCRIPT:
                    for script in value.split():
                        self.pack_file('scripts/vscripts/' + script)
                elif val_type is KVTypes.STR_SPRITE:
                    self.pack_file('materials/sprites/' + value, FileType.MATERIAL)
                elif val_type is KVTypes.STR_SOUND:
                    self.pack_soundscript(value)

        for classname in vmf.by_class.keys():
            try:
                res = CLASS_RESOURCES[classname]
            except KeyError:
                continue
            if callable(res):
                for ent in vmf.by_class[classname]:
                    for file in res(ent):
                        self.pack_file(file)
            else:
                for file in res:
                    self.pack_file(file)

    def pack_into_zip(
        self,
        zip_file: ZipFile,
        *,
        whitelist: Iterable[FileSystem]=(),
        blacklist: Iterable[FileSystem]=(),
        ignore_vpk=True,
    ):
        """Pack all our files into the given zipfile.

        The filesys is used to find files to pack.
        Filesystems must be in the whitelist and not in the blacklist, if provided.
        If ignore_vpk is True, files in VPK won't be packed unless that system
        is in allow_filesys.
        """
        existing_names = {
            name.replace('\\', '/')
            for name in zip_file.namelist()
        }

        all_systems = {
            sys for sys, prefix in
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

        with self.fsys:
            for file in self._files.values():
                # Need to ensure / separators.
                fname = file.filename.replace('\\', '/')

                if file.virtual:
                    # Always pack.
                    zip_file.writestr(fname, file.data)
                    continue

                if file.filename in existing_names:
                    # Already in the zip - cubemap patch files, or something
                    # else has already added it. Ignore.
                    continue

                try:
                    sys_file = self.fsys[file.filename]
                except FileNotFoundError:
                    if file.type is not FileType.OPTIONAL:
                        print('WARNING: "{}" not packed!'.format(file.filename))
                    continue

                if self.fsys.get_system(sys_file) in allowed:
                    with sys_file.open_bin() as f:
                        zip_file.writestr(fname, f.read())

    def eval_dependencies(self) -> None:
        """Add files to the list which need to also be packed.

        This requires parsing through many files.
        """
        # Run though repeatedly, until all are analysed.
        todo = True
        with self.fsys:
            while todo:
                todo = False

                for file in list(self._files.values()):
                    if file._analysed:
                        continue
                    file._analysed = True

                    if file.type is FileType.MATERIAL:
                        if self._get_material_files(file):
                            todo = True
                    elif file.type is FileType.MODEL:
                        self._get_model_files(file)
                        todo = True
                    elif file.type is FileType.TEXTURE:
                        # Try packing the '.hdr.vtf' file as well if present.
                        # But don't recurse!
                        if not file.filename.endswith('.hdr.vtf'):
                            self.pack_file(
                                file.filename[:-3] + 'hdr.vtf',
                                FileType.OPTIONAL
                            )
                            todo = True

    def _get_model_files(self, file: PackFile):
        """Find any needed files for a model."""
        filename, ext = os.path.splitext(file.filename)
        self.pack_file(filename + '.vvd')  # Must be present.

        # Some of these are optional.
        for ext in ['.phy', '.vtx', '.sw.vtx', '.dx80.vtx', '.dx90.vtx']:
            component = filename + ext
            if component in self.fsys:
                self.pack_file(component)

        try:
            mdl = Model(self.fsys, self.fsys[file.filename])
        except FileNotFoundError:
            LOGGER.warning('Can\'t find model "{}"!', file.filename)
            return

        for tex in mdl.iter_textures():
            self.pack_file(tex, FileType.MATERIAL)

        for file in mdl.included_models:
            self.pack_file(file.filename, FileType.MODEL)

        for snd in mdl.find_sounds():
            self.pack_file(snd, FileType.GAME_SOUND)

    def _get_material_files(self, file: PackFile):
        """Find any needed files for a material."""

        parents = []
        try:
            with self.fsys, self.fsys.open_str(file.filename) as f:
                mat = Material.parse(f, file.filename)
        except FileNotFoundError:
            print('WARNING: File "{}" does not exist!'.format(file.filename))
            return

        # For 'patch' shaders, apply the originals.
        mat = mat.apply_patches(self.fsys, parent_func=parents.append)

        for vmt in parents:
            self.pack_file(vmt, FileType.MATERIAL)

        has_deps = bool(parents)

        for param_name, param_type, param_value in mat:
            param_value = param_value.casefold()
            if param_type is VarType.TEXTURE:
                # Skip over reference to cubemaps, or realtime buffers.
                if param_value == 'env_cubemap' or param_value.startswith('_rt_'):
                    continue
                self.pack_file(
                    'materials/' + param_value + '.vtf',
                    FileType.TEXTURE,
                )
                has_deps = True
            # Bottommaterial for water brushes mainly.
            if param_type is VarType.MATERIAL:
                self.pack_file(
                    'materials/' + param_value + '.vmt',
                    FileType.MATERIAL,
                )
                has_deps = True

        return has_deps
