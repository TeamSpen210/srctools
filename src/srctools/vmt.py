"""Parses material files."""
from typing import (
    Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, TextIO,
    Tuple, TypeVar, Union, overload,
)
from enum import Enum
import sys

import attrs

from srctools import EmptyMapping
from srctools.filesys import FileSystem
from srctools.keyvalues import Keyvalues
from srctools.tokenizer import BARE_DISALLOWED, Token as Tok, Tokenizer as Tokenizer


class VarType(Enum):
    """The different types shader variables can be.

    The value is the name used in the game code.
    """
    # Vars like $selfillum which set a bitmask - these cannot be altered later.
    FLAG = 0

    MATERIAL = 'SHADER_PARAM_TYPE_MATERIAL'  # models/blah.vmt
    STR = 'SHADER_PARAM_TYPE_STRING'  # Basic string, nothing special.
    TEXTURE = 'SHADER_PARAM_TYPE_TEXTURE'

    INT = 'SHADER_PARAM_TYPE_INTEGER'
    FLOAT = 'SHADER_PARAM_TYPE_FLOAT'  # 4.0, .3, 2.3f
    BOOL = 'SHADER_PARAM_TYPE_BOOL'  # 0, 1

    COLOR = 'SHADER_PARAM_TYPE_COLOR'  # RGBA, optional A
    VEC2 = 'SHADER_PARAM_TYPE_VEC2'  # [0 0]
    VEC3 = 'SHADER_PARAM_TYPE_VEC3'  # [0 0 0]
    VEC4 = 'SHADER_PARAM_TYPE_VEC4'  # [0 0 0 0]

    # Transformation matrix in the following forms:
    # 'center .5 .5 scale 1 1 rotate 0 translate 0 0'
    # '[1 0 0 0 1 0 0 0 1]'
    MATRIX = 'SHADER_PARAM_TYPE_MATRIX'  # 9-matrix, or center scale rotate etc
    MATRIX_4X2 = 'SHADER_PARAM_TYPE_MATRIX4X2'  # Partially implemented, only a 2x4=8 matrix?

    # ENVMAP = 'SHADER_PARAM_TYPE_ENVMAP'  # Obsolete apparently
    # Special case - pointer to arbitrary shader-specific data.
    FOUR_CC = 'SHADER_PARAM_TYPE_FOURCC'

    @classmethod
    def from_name(cls, name: str) -> 'VarType':
        """Given a shader parameter name, return the type this value is.

        If not known, return STR.
        """
        return get_parm_type(name, cls.STR)


@attrs.define
class Variable:
    """Allow storing the original case of the name."""
    name: str  # With correct case
    value: str


__all__ = ['VarType', 'Material', 'get_parm_type']
ArgT = TypeVar('ArgT')
_SHADER_PARAM_TYPES: Dict[str, VarType] = {}


@overload
def get_parm_type(name: str) -> Optional[VarType]: ...
@overload
def get_parm_type(name: str, default: ArgT) -> Union[VarType, ArgT]: ...


def get_parm_type(name: str, default: Optional[ArgT] = None) -> Union[VarType, ArgT, None]:
    """Retrieve the type a parameter has, or return the default."""
    # Import and load the parameters.
    # noinspection PyProtectedMember
    from srctools._shaderdb import _shader_db

    _shader_db(VarType, _SHADER_PARAM_TYPES)

    # Delete the module - that way it'll be garbage
    # collected - no need to keep it around.
    del sys.modules['srctools._shaderdb']

    # Redirect this to always call the normal function.
    get_parm_type.__code__ = _get_parm_type_real.__code__
    return _get_parm_type_real(name, default)


def _get_parm_type_real(name: str, default: Optional[ArgT] = None) -> Union[VarType, ArgT, None]:
    """Retrieve the type a parameter has, or return the default."""
    if '?' in name:
        flag, name = name.split('?', 1)
    try:
        return _SHADER_PARAM_TYPES[name.lstrip('$').casefold()]
    except KeyError:
        return default


class Material(MutableMapping[str, str]):
    """Represents a material.

    This behaves as a mapping, storing the shader parameters.
    """
    shader: str
    """The name of the shader."""
    proxies: List[Keyvalues]
    """
    List of Material Proxies defined for the material.
    Each is a tuple of the string name and a dict of keys-> values.
    "Empty" proxies are removed.
    """
    blocks: List[Keyvalues]
    """Other sub-blocks inside the material definition. These are usually fallbacks or other similar definitions."""

    def __init__(
        self,
        shader: str,
        params: Mapping[str, str] = EmptyMapping,
        blocks: Iterable[Keyvalues] = (),
        proxies: Iterable[Keyvalues] = (),
    ) -> None:
        """Create a material."""
        self.shader = shader
        self._params: Dict[str, Variable] = {}
        self.blocks = list(blocks)
        self.proxies = list(proxies)

        for key, value in params.items():
            self[key] = value

    @classmethod
    def parse(cls, data: Iterable[str], filename: str = '') -> 'Material':
        """Parse a VMT from the file."""
        # Block escapes, so "files\test\tex" doesn't have a tab in it.
        tok = Tokenizer(
            data, filename,
            string_bracket=True,
            allow_escapes=False,
            allow_star_comments=True,
        )

        # First look for the shader name -
        # which must be the first string
        # in the file.
        shader_name = None
        for token, shader_name in tok:
            if token is Tok.NEWLINE:
                continue
            elif token is Tok.STRING:
                break
            else:
                raise tok.error(token)

        if not shader_name:
            raise tok.error("No shader name!")

        # Open the parameters body.
        tok.expect(Tok.BRACE_OPEN)

        mat = cls(shader_name)

        # Look for parameter names
        for token, param_name in tok:
            if token is Tok.NEWLINE:
                continue
            # End of body.
            elif token is Tok.BRACE_CLOSE:
                break
            elif token is Tok.PROP_FLAG:
                tok.expect(Tok.NEWLINE)
                continue
            elif token is not Tok.STRING:
                raise tok.error(token)
            token, param_value = tok()

            if token is Tok.STRING:
                # We have the value.
                pass
            elif token is Tok.NEWLINE or token is Tok.BRACE_OPEN:
                # Name by itself: '%compilenodraw' etc if newline,
                # or 'Proxies {' etc.
                param_value = ''
                # Skip newlines, figure out what this is.
                while token is Tok.NEWLINE:
                    token, ignored = tok()

                if token is Tok.BRACE_OPEN:
                    if param_name.casefold() == "proxies":
                        mat.proxies.extend(cls._parse_block(tok, 'Proxy'))
                    else:
                        mat.blocks.append(cls._parse_block(tok, param_name))

                    continue  # Don't replace with None.
                elif token is Tok.BRACE_CLOSE:
                    # End of us after single name.
                    mat[param_name] = param_value
                    break
                else:
                    raise tok.error(token)
            else:
                raise tok.error(token)

            mat[param_name] = param_value

        # We expect nothing else now.
        tok.expect(Tok.EOF)

        return mat

    @staticmethod
    def _parse_proxies(tok: Tokenizer) -> Iterator[Tuple[str, Dict[str, str]]]:
        # Parse the proxy block.

        for token, proxy_name in tok.skipping_newlines():
            if token is Tok.BRACE_CLOSE:
                return
            elif token is not Tok.STRING:
                raise tok.error(token)

            opts: Dict[str, str] = {}

            tok.expect(Tok.BRACE_OPEN)  # Start of proxy values.

            # Looking for key-value parameters for a proxy
            for par_tok, param_name in tok.skipping_newlines():
                if par_tok is Tok.BRACE_CLOSE:
                    break
                elif par_tok is not Tok.STRING:
                    raise tok.error(par_tok)

                par_tok, param_value = tok()

                if par_tok is Tok.STRING:
                    opts[param_name.casefold()] = param_value
                else:
                    raise tok.error(par_tok)
            else:
                raise tok.error('EOF while reading options for "{}" proxy', proxy_name)
            yield proxy_name, opts
        # We hit EOF, still expecting the }.
        raise tok.error('Proxy block not closed!')

    @staticmethod
    def _parse_block(tok: Tokenizer, name: str) -> Keyvalues:
        """Parse a block into a block of properties."""
        prop = Keyvalues(name, [])

        for token, param_name in tok:
            # End of our block
            if token is Tok.BRACE_CLOSE:
                return prop
            elif token is Tok.NEWLINE:
                continue
            elif token is not Tok.STRING:
                raise tok.error(token)

            token, param_value = tok()

            if token is Tok.STRING:
                # We have the value.
                pass
            elif token is Tok.NEWLINE or token is Tok.BRACE_OPEN:
                # Name by itself: '%compilenodraw' etc if newline,
                # or 'Proxies {' etc. Skip newlines, figure out what this is.
                while token is Tok.NEWLINE:
                    token, ignored = tok()

                if token is Tok.BRACE_OPEN:
                    prop.append(Material._parse_block(tok, param_name))
                    continue
                elif token is Tok.BRACE_CLOSE:
                    # End of us after single name.
                    prop.append(Keyvalues(param_name, ''))
                    break
                else:
                    raise tok.error(token)
            else:
                raise tok.error(token)
            prop.append(Keyvalues(param_name, param_value))

        raise tok.error('EOF without closed block!')

    def export(self, f: TextIO) -> None:
        """Write the material back to a file."""
        f.write(self.shader + '\n\t{\n')
        for param in self._params.values():
            name = param.name
            value = param.value
            if any(c in BARE_DISALLOWED for c in name):
                name = f'"{name}"'
            if not value or any(c in BARE_DISALLOWED for c in value):
                value = f'"{value}"'
            f.write(f'\t{name} {value}\n')
        for block in self.blocks:
            block.serialise(f, start_indent='\t')
        if self.proxies:
            f.write('\n\tProxies\n\t\t{\n')
            for block in self.proxies:
                block.serialise(f, start_indent='\t\t')
            f.write('\t\t}\n')
        f.write('\t}\n')

    def apply_patches(
        self,
        fsys: FileSystem,
        *,
        limit: int = 100,
        parent_func: Optional[Callable[[str], None]] = None,
    ) -> 'Material':
        """If the material is a `Patch <https://developer.valvesoftware.com/wiki/Patch>` shader expand to the full material.

        :param fsys: This reads from the supplied filesystem as required.
        :param limit: If more than this many files are parsed, a RecursionError is raised.
        :param parent_func: If this is provided, it will be called with the filenames of the \
        VMTs which are looked up. This allows tracking which are used.
        """
        return self._apply_patch(fsys, 1, limit, parent_func)

    def _apply_patch(
        self,
        fsys: FileSystem,
        count: int,
        limit: int,
        parent_func: Optional[Callable[[str], None]],
    ) -> 'Material':
        """Do apply_patches()."""
        if count > limit:
            raise RecursionError('Parsed too deep a Patch tree!')

        if self.shader.casefold() != 'patch':
            return self

        try:
            filename = self._params['include'].value
        except KeyError:
            raise ValueError('No "include" key for Patch shader!') from None
        try:
            parent_file = fsys[filename]
        except FileNotFoundError:
            raise ValueError(
                f'Material file "{filename}" does not exist when patching!'
            ) from None

        if parent_func is not None:
            parent_func(parent_file.path)

        with parent_file.open_str() as f:
            parent = Material.parse(f, filename)

        parent = parent._apply_patch(fsys, count + 1, limit, parent_func)

        copy = Material(parent.shader)
        # Transfer them all over, the parent is being deleted so we don't need to bother copying.
        copy._params.update(parent._params)
        copy.proxies.extend(parent.proxies)
        copy.blocks.extend(parent.blocks)

        # Empty strings in these delete the value.
        # If replace is used, the value must exist, otherwise it's skipped.
        for block in self.blocks:
            if block.name == 'insert':
                always_add = True
            elif block.name == 'replace':
                always_add = False
            else:
                raise ValueError(f'Unknown patch command "{block.real_name}"!')
            if not block.has_children():
                raise ValueError(f'"{block.real_name}" must be a block, not a single value.')
            for prop in block:
                if prop.name == 'proxies':
                    if not prop.has_children():
                        raise ValueError('Proxies must be a block, not a string!')
                    for prox_block in prop:
                        copy.proxies.append(prox_block.copy())
                elif prop.has_children():
                    # For replace keyvalues, they recursively append.
                    if not always_add:  # Todo: recursively merge.
                        copy.blocks.append(prop.copy())
                elif prop.value == '':
                    try:
                        del copy._params[prop.name]
                    except KeyError:
                        pass
                elif always_add or prop.name in copy._params:
                    copy[prop.real_name] = prop.value
                else:
                    pass
        return copy

    def __iter__(self) -> Iterator[str]:
        for var in self._params.values():
            yield var.name

    def __len__(self) -> int:
        return len(self._params)

    def __contains__(self, item: object) -> bool:
        """Check if the given property is present."""
        if isinstance(item, str):
            return item.casefold() in self._params
        return False

    def __getitem__(self, key: str) -> str:
        """Get the value of the specified property."""
        return self._params[key.casefold()].value

    def __setitem__(self, key: str, value: str) -> None:
        """Set the specified property."""
        folded = key.casefold()
        try:
            self._params[folded].value = value
        except KeyError:
            self._params[folded] = Variable(key, value)

    def __delitem__(self, key: str) -> None:
        """Remove the specified property."""
        del self._params[key.casefold()]
