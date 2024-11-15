"""Reads and parses Valve's KeyValues files.

These files follow the following general format::

    "Name"
        {
        "Name" "Value" // Comment
        "Name"
            {
            "Name" "Value"
            }
        "Name" "Value"
        "SecondName" "ps3-only value" [ps3]
        "Third_Name" "never on Linux" [!linux]
        "Name" "multi-line values
    are supported like this.
    They end with a quote."
        }

The names are usually case-insensitive.
Call ``Keyvalues(name, value)`` to get a keyvalues object, or ``Keyvalues.parse(file, 'filename')``
to parse a file.

This will perform a round-trip file read::

    >>> with open('filename.txt', 'r') as read_file:  # doctest: +SKIP
    ...     kv = Keyvalues.parse(read_file, 'filename.txt')
    ... with open('filename_2.txt', 'w') as write_file:
    ...     kv.serialise(write_file)

Keyvalue ``values`` should be either a string, or a list of children Keyvalues.
Names will be converted to lowercase automatically; use kv.real_name to
obtain the original spelling. To allow multiple root blocks in a file, the
returned keyvalues from Keyvalues.parse() is special and will export with
un-indented children.

Keyvalues with children can be indexed by their names, or by a
('name', default) tuple::

    >>> kv = Keyvalues('Top', [
    ...     Keyvalues('child1', '1'),
    ...     Keyvalues('child2', '0'),
    ... ])
    >>> kv['child1']
    '1'
    >>> kv['child3']
    Traceback (most recent call last):
        ...
    IndexError: No key child3!
    >>> kv['child3', 'default']
    'default'
    >>> kv['child4', object()]
    <object object at 0x...>
    >>> del kv['child2']
    >>> kv['child3'] = 'new value'
    >>> kv
    Keyvalues('Top', [Keyvalues('child1', '1'), Keyvalues('child3', 'new value')])

Handling ``\\n``, ``\\t``, ``\\"``, and ``\\\\`` escape characters can be enabled.
"""
from typing import (
    Any, Callable, ClassVar, Dict, Final, Iterable, Iterator, List, Mapping, Optional,
    Protocol, Tuple, Type, TypeVar, Union, cast,
)
from typing_extensions import Literal, TypeAlias, deprecated, overload, ContextManager
import builtins  # Keyvalues.bool etc shadows these.
import io
import keyword
import os
import sys
import types
import warnings

from exceptiongroup import BaseExceptionGroup

from srctools import BOOL_LOOKUP, EmptyMapping, StringPath
from srctools.math import Vec as _Vec
from srctools.tokenizer import (
    BaseTokenizer, Token, Tokenizer, TokenSyntaxError, escape_text, format_exc_fileinfo,
)


__all__ = ['KeyValError', 'NoKeyError', 'LeafKeyvalueError', 'Keyvalues', 'escape_text']

# Sentinel value to indicate that no default was given to find_key()
_NO_KEY_FOUND: str = cast(str, object())

_KV_Value = Union[List['Keyvalues'], str, Any]
_As_Dict_Ret: TypeAlias = Union[str, Dict[str, '_As_Dict_Ret']]

T = TypeVar('T')

# Various [flags] used after keyvalues names in some Valve files.
# See https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/tier1/KeyValues.cpp#L2055
FLAGS_DEFAULT = {
    # We know we're not on a console...
    'x360': False,
    'ps3': False,
    'gameconsole': False,

    'win32': True,  # Not actually windows, it actually means 'PC'
    'osx': sys.platform.startswith('darwin'),
    'linux': sys.platform.startswith('linux'),
    # Not in the SDK, but shows up in compatible games.
    'deck': 'steamdeck' in os.environ,
}


class KeyValError(TokenSyntaxError):
    """An error that occurred when parsing a Valve KeyValues file.

    See the base class :py:class:`TokenSyntaxError` for available attributes.
    """


class NoKeyError(LookupError):
    """Raised if a key is not found when searching with
    :py:meth:`~Keyvalues.find_key()`, :py:meth:`~Keyvalues.find_block()`, etc."""
    key: str  #: The key that was missing.
    line_num: Optional[int]  #: The line number where the block is defined, if known.
    filename: Optional[str]  #: The filename where the block is defined, if known.
    def __init__(
        self, key: str,
        line_num: Optional[int] = None,
        filename: Optional[str] = None,
    ) -> None:
        super().__init__(key)
        self.key = key
        self.line_num = line_num
        self.filename = filename

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.key!r})'

    def __str__(self) -> str:
        return format_exc_fileinfo(f"No key {self.key}!", self.filename, self.line_num)

    @staticmethod
    def apply_filename(filename: str) -> ContextManager[str, None]:
        """Applies a filename to any :py:class:`NoKeyError` raised inside this scope.

        This will cause the errors to display that filename. Has no effect if the filename
        was already set, potentially by another :py:func:`apply_filename` call closer to the
        exception source.
        This recurses into any raised :py:class:`ExceptionGroup`.
        """
        return _FilenameSetter(filename)


class _FilenameSetter(ContextManager[str, None]):
    """Applies a filename to all :py:class:`NoKeyError`s raised inside this scope."""
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def __enter__(self) -> str:
        return self.filename

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[types.TracebackType],
    ) -> None:
        if exc_val is None:
            return
        if isinstance(exc_val, NoKeyError):
            if exc_val.filename is None:
                exc_val.filename = self.filename
        elif isinstance(exc_val, BaseExceptionGroup):
            self._modify_group(exc_val)

    def _modify_group(self, group: BaseExceptionGroup[BaseException]) -> None:
        """Recursively modify ExceptionGroups."""
        for exc in group.exceptions:
            if isinstance(exc, BaseExceptionGroup):
                self._modify_group(exc)
            elif isinstance(exc, NoKeyError) and  exc.filename is None:
                exc.filename = self.filename


class LeafKeyvalueError(ValueError):
    """Raised when a method requiring a keyvalue block is run on a leaf keyvalue.

    Leaf keyvalues only have a string value, so this operation is not valid.
    """
    leaf: 'Keyvalues'  #: The keyvalue being used.
    operation: str  #: Name of the operation being performed.
    line_num: Optional[int]  #: The line number where the leaf is defined, if known.
    filename: Optional[str]  #: The filename where the leaf is defined, if known.
    def __init__(self, leaf: 'Keyvalues', operation: str, filename: Optional[str] = None) -> None:
        super().__init__(operation)
        self.leaf = leaf
        self.operation = operation
        self.line_num = leaf.line_num
        self.filename = filename

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.leaf!r}, {self.operation!r})'

    def __str__(self) -> str:
        return format_exc_fileinfo(
            f'Cannot {self.operation} a leaf {self.leaf!r}, since it has no children!',
            self.filename, self.line_num,
        )


class _SupportsWrite(Protocol):
    """We accept any file object with a ``write()`` method."""
    def write(self, data: str, /) -> object: ...


def _read_flag(flags: Mapping[str, bool], flag_val: str) -> bool:
    """Check whether a flag is True or False."""
    flag_inv = flag_val[:1] == '!'
    if flag_inv:
        flag_val = flag_val[1:]
    flag_val = flag_val.casefold()
    try:
        flag_result = bool(flags[flag_val])
    except KeyError:
        flag_result = FLAGS_DEFAULT.get(flag_val, False)
    # If flag succeeds
    return flag_inv is not flag_result


class Keyvalues:
    """Represents Valve's Keyvalues 1 file format.

    Value should be a string (for leaf Keyvalues), or a list of children Keyvalues objects.
    The name should be a string, or None for a root object.
    Root objects export each child at the topmost indent level. This is produced from :py:meth:`Keyvalues.parse()` calls.
    """
    # Helps decrease memory footprint with lots of Keyvalues values.
    __slots__ = ('_folded_name', '_real_name', '_value', 'line_num')
    _folded_name: Optional[str]
    _real_name: Optional[str]
    _value: _KV_Value
    #: If parsed from a file, the line number this keyvalue starts on. This allows providing a
    #: source location for missing-value exceptions.
    line_num: Optional[int]

    @overload
    def __init__(
        self,
        name: str,
        value: Union[List['Keyvalues'], str],
        line_num: Optional[int] = None,
    ) -> None: ...
    @overload
    @deprecated("Use Keyvalues.root() to construct.", category=None)
    def __init__(
        self,
        name: None,
        value: Union[List['Keyvalues'], str],
        line_num: Optional[int] = None,
    ) -> None: ...

    def __init__(
        self,
        name: Optional[str],
        value: Union[List['Keyvalues'], str],
        line_num: Optional[int] = None,
    ) -> None:
        """Create a new keyvalues instance."""
        if name is None:
            warnings.warn("Root keyvalues will change to a new class.", DeprecationWarning, 2)
            self._folded_name = self._real_name = None
        else:
            self._real_name = sys.intern(name)
            self._folded_name = sys.intern(name.casefold())

        self._value = value
        self.line_num = line_num

    @property
    def value(self) -> str:
        """Return the value of a leaf keyvalue.

        :deprecated: Accessing the internal list for keyvalue blocks.
        """
        if isinstance(self._value, list):
            warnings.warn("Accessing internal keyvalues block list is deprecated", DeprecationWarning, 2)
        return cast(str, self._value)

    @value.setter
    def value(self, value: str) -> None:
        """Set the value of a leaf keyvalue."""
        if not isinstance(value, str):
            warnings.warn("Setting keyvalues to non-string values will be fixed soon.", DeprecationWarning, 2)
        self._value = value

    @property
    def name(self) -> str:
        """Produces a :py:meth:`str.casefold`-ed version of the keyvalue's name. Usually names are case-insensitive.

        Assigning to this automatically updates both this and :py:attr:`real_name`.

        :deprecated: 'Root' keyvalues have a name of :py:const:`None`. This will be replaced by a blank string, use :py:meth:`is_root()` instead.
        """
        if self._folded_name is None:
            warnings.warn("The name of root keyvalues will change to a blank string in the future.", DeprecationWarning, 2)
        return self._folded_name  # type: ignore

    @name.setter
    def name(self, new_name: str) -> None:
        if new_name is None:
            warnings.warn("Root keyvalues will change to a new class.", DeprecationWarning, 2)
            self._folded_name = self._real_name = None
        else:
            # Intern names to help reduce duplicates in memory.
            self._real_name = sys.intern(new_name)
            self._folded_name = sys.intern(new_name.casefold())

    @property
    def real_name(self) -> str:
        """The original, case-sensitive version of this name.

        Assigning to this automatically updates both this and :py:attr:`name`.

        :deprecated: 'Root' keyvalues have a name of :py:const:`None`. This will be replaced by a blank string, use :py:meth:`is_root()` instead.
        """
        if self._real_name is None:
            warnings.warn("The name of root keyvalues will change to a blank string in the future.", DeprecationWarning, 2)
        return self._real_name  # type: ignore

    @real_name.setter
    def real_name(self, new_name: str) -> None:
        if new_name is None:
            warnings.warn("Root keyvalues will change to a new class.", DeprecationWarning, 2)
            self._folded_name = self._real_name = None
        else:
            # Intern names to help reduce duplicates in memory.
            self._real_name = sys.intern(new_name)
            self._folded_name = sys.intern(new_name.casefold())

    def edit(self, name: Optional[str] = None, value: Optional[str] = None) -> 'Keyvalues':
        """Simultaneously modify the name and value."""
        if name is not None:
            self._real_name = name
            self._folded_name = name.casefold()
        if value is not None:
            self._value = value
        return self

    @classmethod
    def root(cls, *children: 'Keyvalues') -> 'Keyvalues':
        """Return a new 'root' keyvalue. These have no name, and are returned from :py:meth:`parse()`.

        When serialised, their children are directly written with no indents, allowing multiple keyvalues
        blocks to exist on the topmost level.
        """
        kv = cls.__new__(cls)
        kv._folded_name = kv._real_name = None
        for child in children:
            if not isinstance(child, Keyvalues):
                raise TypeError(f'{type(kv).__name__} is not a Keyvalues!')
        kv._value = list(children)
        kv.line_num = None
        return kv

    @staticmethod
    def parse(
        file_contents: Union[str, BaseTokenizer, Iterator[str]],
        filename: StringPath = '', *,
        flags: Mapping[str, bool] = EmptyMapping,
        newline_keys: bool = False,
        newline_values: bool = True,
        allow_escapes: bool = True,
        single_line: bool = False,
        single_block: bool = False,
    ) -> "Keyvalues":
        """Returns a Keyvalues tree parsed from given text.

        :param file_contents: should be an iterable of strings or a single string. Alternatively,
          file_contents may be an already created tokenizer. In this case ``allow_escapes`` is ignored.
        :param filename: If set this should be the source of the text for debug purposes. If not
          supplied, ``file_contents.name`` will be used if present.
        :param single_block: If set, parse a single keyvalues block, instead of a file with multiple
          roots. Importantly this will exit after hitting this brace, allowing it to be called in
          the middle of parsing a larger document.
        :param flags: This should be a mapping for additional ``[flag]`` suffixes to accept.
        :param allow_escapes: This allows choosing if ``\\t`` or similar escapes are parsed.
        :param single_line: If this is set, allow multiple keyvalues to be on the same line.
          This means unterminated strings will be caught late (if at all), but it allows parsing
          some generated data blocks inside things like `.PHY` files.
        :param newline_keys: This specifies if newline characters are allowed in keys.
          Keys are prohibited by default, since this is fairly useless, but if quote characters are
          mismatched it'll catch the mistake early.
        :param newline_values: This specifies if newline characters are allowed in string values.
        """
        # The block we are currently adding to.

        # The special name 'None' marks it as the root keyvalue, which
        # just outputs its children when exported. This way we can handle
        # multiple root blocks in the file, while still returning a single
        # Keyvalues object which has all the methods.
        # Skip calling __init__ for speed.
        cur_block = root = Keyvalues.__new__(Keyvalues)
        cur_block._folded_name = cur_block._real_name = None
        cur_block.line_num = 1

        # Cache off the value list.
        cur_block_contents: List[Keyvalues]
        cur_block_contents = cur_block._value = []
        # A queue of the keyvalues we are currently in (outside to inside).
        # And the line numbers of each of these, for error reporting.
        open_keyvalues: List[Keyvalues] = [cur_block]

        # Grab a reference to the token values, so we avoid global lookups.
        STRING: Final = Token.STRING
        PROP_FLAG: Final = Token.PROP_FLAG
        NEWLINE: Final = Token.NEWLINE
        BRACE_OPEN: Final = Token.BRACE_OPEN
        BRACE_CLOSE: Final = Token.BRACE_CLOSE

        if isinstance(file_contents, BaseTokenizer):
            tokenizer = file_contents
            tokenizer.filename = os.fspath(filename)
            tokenizer.error_type = KeyValError
        else:
            tokenizer = Tokenizer(
                file_contents,
                filename,
                KeyValError,
                string_bracket=True,
                allow_escapes=allow_escapes,
            )

        # A pseudo-enum to track whether we expect a block next.
        BLOCK_LINE_EXPECT: Literal[2] = 2  # We require a block to open next ("name"\n must have { next.)
        BLOCK_LINE_NONE: Literal[0] = 0  # there's no block.
        BLOCK_LINE_SKIP: Literal[1] = 1  # the block is disabled, so we need to skip it.
        block_line: Literal[0, 1, 2] = BLOCK_LINE_NONE
        # Are we permitted to replace the last keyvalue with a flagged version of the same?
        can_flag_replace = False

        for token_type, token_value in tokenizer:
            if token_type is BRACE_OPEN:  # {
                # Open a new block - make sure the last token was a name.
                if block_line is BLOCK_LINE_NONE:
                    raise tokenizer.error(
                        'Keyvalues cannot have sub-section if it already '
                        'has an in-line value.\n\n'
                        'A "name" "value" line cannot then open a block.',
                    )
                can_flag_replace = False
                if block_line is BLOCK_LINE_SKIP:
                    # It failed the flag check. Use a dummy keyvalue object.
                    # This isn't put into the tree, so after we parse the block it's popped
                    # and discarded.
                    cur_block = Keyvalues.__new__(Keyvalues)
                    cur_block._folded_name = cur_block.real_name = '<skipped>'
                    cur_block.line_num = None  # Not used, but make sure to keep it valid.
                else:
                    cur_block = cur_block_contents[-1]
                cur_block_contents = cur_block._value = []
                open_keyvalues.append(cur_block)
                block_line = BLOCK_LINE_NONE
                continue
            # Something else, but followed by '{'
            elif block_line is not BLOCK_LINE_NONE and token_type is not NEWLINE:
                raise tokenizer.error(
                    'Block opening ("{{") required!\n\n'
                    'A single "name" on a line should next have a open brace '
                    'to begin a block.',
                )

            if token_type is NEWLINE:
                continue
            if token_type is STRING:   # "string"
                if not newline_keys and ('\n' in token_value or '\r' in token_value):
                    raise tokenizer.error('Illegal newline found in key "{}"!', token_value)
                # Skip calling __init__ for speed. Value needs to be set
                # before using this, since it's unset here.
                keyvalue = Keyvalues.__new__(Keyvalues)
                keyvalue._folded_name = sys.intern(token_value.casefold())
                keyvalue.real_name = sys.intern(token_value)
                keyvalue.line_num = tokenizer.line_num

                # We need to check the next token to figure out what kind of
                # prop it is.
                prop_type, prop_value = tokenizer()

                # It's a block followed by flag. ("name" [stuff])
                if prop_type is PROP_FLAG:
                    # That must be the end of the line...
                    tokenizer.expect(NEWLINE)
                    if _read_flag(flags, prop_value):
                        block_line = BLOCK_LINE_EXPECT
                        keyvalue._value = []

                        # Special function - if the last prop was a
                        # keyvalue with this name, replace it instead.
                        if (
                            can_flag_replace and
                            cur_block_contents[-1]._real_name == token_value and
                            cur_block_contents[-1].has_children()
                        ):
                            cur_block_contents[-1] = keyvalue
                        else:
                            cur_block_contents.append(keyvalue)
                        # Can't do twice in a row
                        can_flag_replace = False
                    else:
                        # Signal that the block needs to be discarded.
                        block_line = BLOCK_LINE_SKIP

                elif prop_type is STRING:
                    # A value.. ("name" "value")
                    if block_line is not BLOCK_LINE_NONE:
                        raise tokenizer.error(
                            'Keyvalue split across lines!\n\n'
                            'A value like "name" "value" must be on the same '
                            'line.'
                        )
                    if not newline_values and ('\n' in prop_value or '\r' in prop_value):
                        raise tokenizer.error('Illegal newline found in value "{}"!', prop_value)

                    keyvalue._value = prop_value

                    # Check for flags.
                    flag_token, flag_val = tokenizer()
                    if flag_token is PROP_FLAG:
                        # Should be the end of the line here.
                        tokenizer.expect(NEWLINE)
                        if _read_flag(flags, flag_val):
                            # Special function - if the last prop was a
                            # keyvalue with this name, replace it instead.
                            if (
                                can_flag_replace and
                                cur_block_contents[-1]._real_name == token_value and
                                isinstance(cur_block_contents[-1].value, str)
                            ):
                                cur_block_contents[-1] = keyvalue
                            else:
                                cur_block_contents.append(keyvalue)
                            # Can't do twice in a row
                            can_flag_replace = False
                        else:
                            # Flag was disabled, so ignore this block.
                            continue  # Skip the single_block check below.
                    elif flag_token is STRING:
                        # Specifically disallow multiple text on the same line
                        # normally.
                        # ("name" "value" "name2" "value2")
                        if single_line:
                            cur_block_contents.append(keyvalue)
                            tokenizer.push_back(flag_token, flag_val)
                            continue
                        else:
                            raise tokenizer.error(
                                "Cannot have multiple names on the same line!"
                            )
                    else:
                        # Otherwise, it's got nothing after.
                        # So insert the keyvalue, and check the token
                        # in the next loop. This allows braces to be
                        # on the same line.
                        cur_block_contents.append(keyvalue)
                        can_flag_replace = True
                        tokenizer.push_back(flag_token, flag_val)
                    if single_block and cur_block is root:
                        # The 'single block' turned out to be just a value, return it.
                        return keyvalue
                    continue
                else:
                    # Something else - treat this as a block, and
                    # then re-evaluate the token in the next loop.
                    keyvalue._value = []

                    block_line = BLOCK_LINE_EXPECT
                    can_flag_replace = False
                    cur_block_contents.append(keyvalue)
                    tokenizer.push_back(prop_type, prop_value)
                    continue

            elif token_type is BRACE_CLOSE:  # }
                # Move back a block
                open_keyvalues.pop()
                try:
                    cur_block = open_keyvalues[-1]
                except IndexError:
                    # It's empty, we've closed one too many properties.
                    raise tokenizer.error(
                        'Too many closing brackets.\n\n'
                        'An extra closing bracket was added which would '
                        'close the outermost level.',
                    ) from None
                if single_block and cur_block is root:
                    # Single-block mode - we just exited out of the main block.
                    # Return our child.
                    return root[0]
                # We know this isn't a leaf KV, we made it earlier.
                assert not isinstance(cur_block._value, str)
                cur_block_contents = cur_block._value
                # For replacing the block.
                can_flag_replace = True
            else:
                raise tokenizer.error(token_type, token_value)

        # We're out of data, do some final sanity checks.

        # We last had a ("name"\n), so we were expecting a block
        # next.
        if block_line is not BLOCK_LINE_NONE:
            raise KeyValError(
                'Block opening ("{") required, but hit EOF!\n'
                'A "name" line was located at the end of the file, which needs'
                ' a {} block to follow.',
                tokenizer.filename,
                line=None,
            )

        # All the keyvalues in the file should be closed,
        # so the only thing in open_keyvalues should be the
        # root one we added.

        if len(open_keyvalues) > 1:
            raise KeyValError(
                'End of text reached with remaining open sections.\n\n'
                "File ended with at least one keyvalue that didn't "
                'have an ending "}".\n'
                'Open keyvalues: \n- Root at line 1\n' + '\n'.join([
                    f'- "{kv.real_name}" on line {kv.line_num}'
                    for kv in open_keyvalues[1:]
                ]),
                tokenizer.filename,
                line=None,
            )
        # Return that root keyvalue.
        return root

    def find_all(self, *keys: str) -> Iterator['Keyvalues']:
        """Search through the tree, yielding all keyvalues that match a particular path."""
        depth = len(keys)
        if depth == 0:
            raise ValueError("Cannot find_all without commands!")

        targ_key = keys[0].casefold()
        for kv in self:
            if not isinstance(kv, Keyvalues):
                raise LeafKeyvalueError(kv, 'find children of')
            if kv._folded_name == targ_key is not None:
                if depth > 1:
                    if kv.has_children():
                        yield from Keyvalues.find_all(kv, *keys[1:])
                else:
                    yield kv

    def find_children(self, *keys: str) -> Iterator['Keyvalues']:
        """Search through the tree, yielding children of keyvalues in a path."""
        for block in self.find_all(*keys):
            yield from block

    @overload
    def find_key(self, key: str, *, or_blank: bool) -> 'Keyvalues': ...
    @overload
    def find_key(self, key: str, def_: str = ...) -> 'Keyvalues': ...

    def find_key(self, key: str, def_: str = _NO_KEY_FOUND, *, or_blank: bool = False) -> 'Keyvalues':
        """Obtain the child Keyvalue with a given name.

        - If no child is found with the given name, this will return the
          default value wrapped in a new :py:class:`Keyvalue <Keyvalues>`. If or_blank is set,
          it will be a blank block instead. If neither default is provided
          this will raise :py:class:`NoKeyError`.
        - This prefers keys located closer to the end of the value list.
        """
        if not isinstance(self._value, list):
            raise LeafKeyvalueError(self, 'find a key in')
        folded = key.casefold()
        kv: Keyvalues
        for kv in reversed(self._value):
            if kv._folded_name == folded:
                return kv
        if or_blank:
            return Keyvalues(key, [])
        elif def_ is _NO_KEY_FOUND:
            raise NoKeyError(key, self.line_num)
        else:
            # We were given a default, return it wrapped in a Keyvalue.
            return Keyvalues(key, def_)

    def find_block(self, key: str, or_blank: bool = False) -> 'Keyvalues':
        """Obtain the child Keyvalue block with a given name.

        - If no child is found with the given name and ``or_blank`` is true, a
          blank :py:class:`Keyvalues` block will be returned. Otherwise, :py:class:`NoKeyError` will
          be raised.
        - This prefers keys located closer to the end of the value list.
        """
        if not isinstance(self._value, list):
            raise LeafKeyvalueError(self, 'find a sub-block in')
        folded = key.casefold()
        kv: Keyvalues
        for kv in reversed(self._value):
            if kv._folded_name == folded and kv.has_children():
                return kv
        if or_blank:
            return Keyvalues(key, [])
        else:
            raise NoKeyError(key, self.line_num)

    @overload
    def _get_value(self, key: str, /) -> builtins.str: ...
    @overload
    def _get_value(self, key: str, default: T, /) -> Union[builtins.str, T]: ...
    def _get_value(
        self,
        key: str,
        default: Union[builtins.str, T] = _NO_KEY_FOUND,
        /,
    ) -> Union[builtins.str, T]:
        """Obtain the value of the child Keyvalue with a given name.

        Effectively :py:meth:`find_key()` but doesn't make a new keyvalue.

        - If no child is found with the given name, this will return the
          default value, or raise :py:class:`NoKeyError` if none is provided.
        - This prefers keys located closer to the end of the value list.
        """
        if not isinstance(self._value, list):
            raise LeafKeyvalueError(self, 'find a key in')
        folded = key.casefold()
        kv: Keyvalues
        block_kv = False
        for kv in reversed(self._value):
            if kv._folded_name == folded:
                if kv.has_children():
                    block_kv = True
                else:
                    return cast(str, kv._value)
        if block_kv:
            warnings.warn('This will ignore block keyvalues!', DeprecationWarning, stacklevel=3)
        if default is _NO_KEY_FOUND:
            raise NoKeyError(key, self.line_num)
        else:
            return default

    @overload
    def int(self, key: str) -> builtins.int: ...
    @overload
    def int(self, key: str, def_: T) -> Union[builtins.int, T]: ...

    def int(self, key: str, def_: Union[builtins.int, T] = 0) -> Union[builtins.int, T]:
        """Return the value of an integer key.

        Equivalent to ``int(kv[key])``, but with a default value if missing or
        invalid.
        If multiple keys with the same name are present, this will use the
        last only.
        """
        try:
            return int(self._get_value(key))
        except LeafKeyvalueError:
            raise  # Incorrect usage, not missing.
        except (NoKeyError, ValueError, TypeError):
            return def_

    @overload
    def float(self, key: str) -> builtins.float: ...
    @overload
    def float(self, key: str, def_: T) -> Union[builtins.float, T]: ...

    def float(self, key: str, def_: Union[builtins.float, T] = 0.0) -> Union[builtins.float, T]:
        """Return the value of an integer key.

        Equivalent to ``float(kv[key])``, but with a default value if missing or
        invalid.
        If multiple keys with the same name are present, this will use the
        last only.
        """
        try:
            return float(self._get_value(key))
        except LeafKeyvalueError:
            raise  # Incorrect usage, not missing.
        except (NoKeyError, ValueError, TypeError):
            return def_

    @overload
    def bool(self, key: str) -> builtins.bool: ...
    @overload
    def bool(self, key: str, def_: T) -> Union[builtins.bool, T]: ...

    def bool(self, key: str, def_: Union[builtins.bool, T] = False) -> Union[builtins.bool, T]:
        """Return the value of a boolean key.

        The value may be case-insensitively 'true', 'false', '1', '0', 'T',
        'F', 'y', 'n', 'yes', or 'no'.
        If multiple keys with the same name are present, this will use the
        last only.
        """
        try:
            return BOOL_LOOKUP[self._get_value(key).casefold()]
        except LeafKeyvalueError:
            raise  # Incorrect usage, not missing.
        except LookupError:  # base for NoKeyError and KeyError
            return def_

    def vec(
        self, key: str,
        x: builtins.float = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
    ) -> _Vec:
        """Return the given keyvalue, converted to a vector.

        If multiple keys with the same name are present, this will use the
        last only.
        """
        try:
            return _Vec.from_str(self._get_value(key), x, y, z)
        except LeafKeyvalueError:
            raise  # Incorrect usage, not missing.
        except NoKeyError:  # key not present, defaults.
            return _Vec(x, y, z)

    def set_key(self, path: Union[Tuple[str, ...], str], value: str) -> None:
        """Set the value of a key deep in the tree hierarchy.

        - If any of the hierarchy do not exist (or do not have children), blank keyvalues will be added automatically.
        - path should be a tuple of names, or a single string.
        """
        if not isinstance(self._value, list):
            raise LeafKeyvalueError(self, 'set a child key of')

        current_kv = self
        if isinstance(path, tuple):
            # Search through each item in the tree!
            for key in path[:-1]:
                folded_key = key.casefold()
                # We can't use find_key() here because we also need to check that the keyvalue
                # has children to search through.
                for kv in reversed(self._value):
                    if (
                        kv.name is not None and
                        kv.name == folded_key and
                        kv.has_children()
                    ):
                        current_kv = kv
                        break
                else:
                    # No matching keyvalue found
                    new_kv = Keyvalues(key, [])
                    current_kv.append(new_kv)
                    current_kv = new_kv
            path = path[-1]
        try:
            current_kv.find_key(path).value = value
        except NoKeyError:
            current_kv.append(Keyvalues(path, value))

    def copy(self) -> 'Keyvalues':
        """Deep copy this Keyvalue tree and return it."""
        # Bypass __init__() and name=None warnings.
        result = Keyvalues.__new__(Keyvalues)
        result._real_name = self._real_name
        result._folded_name = self._folded_name
        result.line_num = self.line_num
        if isinstance(self._value, list):
            # This recurses if needed
            result._value = [child.copy() for child in self._value]
        else:
            result._value = self._value
        return result

    def as_dict(self) -> _As_Dict_Ret:
        """Convert this keyvalue tree into a tree of dictionaries.

        This keeps only the last if multiple items have the same name.
        """
        if isinstance(self._value, list):
            return {item.name: item.as_dict() for item in self}
        else:
            return self._value

    @overload
    def as_array(self) -> List[str]: ...
    @overload
    def as_array(self, *, conv: Callable[[str], T]) -> List[T]: ...

    def as_array(self, *, conv: Callable[[str], T] = cast(Any, str)) -> Union[List[T], List[str]]:
        """Convert a keyvalue block into a list of values.

        If the keyvalue is a single keyvalue, the single value will be
        yielded. Otherwise, each child must be a single value and each
        of those will be yielded. The name is ignored.
        """
        if isinstance(self._value, list):
            arr: List[T] = []
            child: Keyvalues
            for child in self._value:
                if not isinstance(child._value, str):
                    raise ValueError(
                        'Cannot have sub-children in a '
                        f'"{self.real_name}" array of values!'
                    )
                arr.append(conv(child._value))
            return arr
        else:
            return [conv(self._value)]

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> builtins.bool:
        """Compare two items and determine if they are equal.

        This ignores names.
        """
        if isinstance(other, Keyvalues):
            return self._value == other._value
        return NotImplemented

    def __ne__(self, other: object) -> builtins.bool:
        """Not-Equal To comparison. This ignores names.
        """
        if isinstance(other, Keyvalues):
            return self._value != other._value
        return NotImplemented

    def __len__(self) -> builtins.int:
        """Determine the number of child keyvalues."""
        if isinstance(self._value, list):
            return len(self._value)
        raise LeafKeyvalueError(self, 'compute the length of')

    def __bool__(self) -> builtins.bool:
        """Keyvalues are true if we have children, or have a value."""
        if isinstance(self._value, list):
            return len(self._value) > 0
        else:
            return bool(self._value)

    def __iter__(self) -> Iterator['Keyvalues']:
        """Iterate through the value list.

        """
        if isinstance(self._value, list):
            return iter(self._value)
        else:
            raise LeafKeyvalueError(self, 'iterate')

    def iter_tree(self, blocks: builtins.bool = False) -> Iterator['Keyvalues']:
        """Iterate through all keyvalues in this tree.

        This goes through keyvalues in the same order that they will serialise
        into.
        If blocks is True, the keyvalue blocks will be returned as well as
        keyvalues. If false, only keyvalues will be yielded.
        """
        if isinstance(self._value, list):
            return self._iter_tree(blocks)
        else:
            raise LeafKeyvalueError(self, 'recursively iterate')

    def _iter_tree(self, blocks: builtins.bool) -> Iterator['Keyvalues']:
        """Implementation of iter_tree(). This assumes self has children."""
        assert isinstance(self._value, list)
        kv: Keyvalues
        for kv in self._value:
            if kv.has_children():
                if blocks:
                    yield kv
                yield from kv._iter_tree(blocks)
            else:
                yield kv

    def __contains__(self, key: str) -> builtins.bool:
        """Check to see if a name is present in the children."""
        key = key.casefold()
        if isinstance(self._value, list):
            kv: Keyvalues
            for kv in self._value:
                if kv._folded_name == key:
                    return True
            return False

        raise LeafKeyvalueError(self, 'search through')

    @overload
    def __getitem__(self, index: builtins.int) -> 'Keyvalues': ...
    @overload
    def __getitem__(self, index: slice) -> List['Keyvalues']: ...
    @overload
    def __getitem__(self, index: str) -> str: ...
    @overload
    def __getitem__(self, index: Tuple[str, T]) -> Union[str, T]: ...

    def __getitem__(
        self,
        index: Union[
            str,
            builtins.int,
            slice,
            Tuple[str, Union[str, T]],
        ],
    ) -> Union['Keyvalues', List['Keyvalues'], str, T]:
        """Allow indexing the children directly.

        - If given an index, it will return the keyvalues in that position.
        - If given a string, it will find the last Keyvalue with that name.
          (Default can be chosen by passing a 2-tuple like kv[key, default])
        - If none are found, it raises IndexError.
        """
        if isinstance(self._value, list):
            if isinstance(index, (int, slice)):
                return self._value[index]
            elif isinstance(index, tuple):
                # With default value
                return self._get_value(index[0], index[1])
            elif isinstance(index, str):
                try:
                    return self._get_value(index)
                except NoKeyError as no_key:
                    raise IndexError(no_key) from no_key
            else:
                raise TypeError(f'Unknown key type: {index!r}')
        else:
            raise LeafKeyvalueError(self, 'find a child in')

    @overload
    def __setitem__(self, index: slice, value: Iterable['Keyvalues']) -> None: ...
    @overload
    def __setitem__(self, index: builtins.int, value: 'Keyvalues') -> None: ...
    @overload
    def __setitem__(self, index: str, value: str) -> None: ...

    def __setitem__(
        self,
        index: Union[builtins.int, slice, str],
        value: Union['Keyvalues', Iterable['Keyvalues'], str],
    ) -> None:
        """Allow setting the values of the children directly.

        - If given an index or slice, it will add these keyvalues in these positions.
        - If given a string, it will set the last Keyvalue with that name.
        - If none are found, it appends the value to the tree.
        """
        if not isinstance(self._value, list):
            raise LeafKeyvalueError(self, 'set a child in')
        if isinstance(index, int):
            if isinstance(value, Keyvalues):
                self._value[index] = value
            else:
                raise TypeError(f"Cannot assign non-Keyvalue to position {index}: {value!r}")
        elif isinstance(index, slice):
            kv_list = []
            for kv in value:
                if isinstance(kv, Keyvalues):
                    kv_list.append(kv)
                else:
                    raise TypeError(f'Must assign Keyvalues to positions, not {type(kv).__name__}!')
            self._value[index] = kv_list
        elif isinstance(index, str):
            if isinstance(value, Keyvalues):
                # We don't want to assign keyvalues, we want to add them under
                # this name!,
                value.name = index
                try:
                    # Replace at the same location.
                    pos = self._value.index(self.find_key(index))
                except NoKeyError:
                    self._value.append(value)
                else:
                    self._value[pos] = value
            else:
                try:
                    self.find_key(index)._value = value
                except NoKeyError:
                    if isinstance(value, str):
                        self._value.append(Keyvalues(index, value))
                    else:
                        self._value.append(Keyvalues(index, list(value)))
        else:
            raise TypeError(f'Unknown key type: {index!r}')

    def __delitem__(self, index: Union[builtins.int, slice, str]) -> None:
        """Delete the given keyvalues index.

        - If given an integer, it will delete by position.
        - If given a string, it will delete the last Keyvalue with that name.
        """
        if not isinstance(self._value, list):
            raise LeafKeyvalueError(self, 'delete a child from')
        if isinstance(index, (int, slice)):
            del self._value[index]
        else:
            try:
                self._value.remove(self.find_key(index))
            except NoKeyError as no_key:
                raise IndexError(no_key) from no_key

    def clear(self) -> None:
        """Delete the contents of a block."""
        if isinstance(self._value, list):
            self._value.clear()
        else:
            raise LeafKeyvalueError(self, 'clear')

    def __add__(self, other: Iterable['Keyvalues']) -> 'Keyvalues':
        """Extend this keyvalue with the contents of another, or an iterable.

        This deep-copies the Keyvalue tree first.
        Deprecated behaviour: This also accepts a non-root keyvalue, which will be appended
        instead.
        """
        if isinstance(self._value, list):
            copy = self.copy()
            assert isinstance(copy._value, list)
            if isinstance(other, Keyvalues) and other._folded_name is not None:
                # Deprecated behaviour, add the other keyvalue to ourselves,
                # not its values.
                warnings.warn(
                    "Using + to add a single Keyvalue is confusing, use append() instead.",
                    DeprecationWarning, 2,
                )
                copy._value.append(other.copy())
            else:  # Assume a sequence.
                for kv in other:
                    if not isinstance(kv, Keyvalues):
                        raise TypeError(f'{type(kv).__name__} is not a Keyvalue!')
                    self._value.append(kv.copy())
            return copy
        else:
            return NotImplemented

    def __iadd__(self, other: Iterable['Keyvalues']) -> 'Keyvalues':
        """Extend this keyvalue with the contents of another, or an iterable.

        Deprecated behaviour: This also accepts a non-root keyvalue, which will be appended
        instead.
        """
        if isinstance(self._value, list):
            if isinstance(other, Keyvalues) and other._folded_name is not None:
                # Deprecated behaviour, add the other keyvalue to ourselves,
                # not its values.
                warnings.warn(
                    "Using += to add a single Keyvalue is confusing, use append() instead.",
                    DeprecationWarning, 2,
                )
                self._value.append(other.copy())
            else:
                for kv in other:
                    if not isinstance(kv, Keyvalues):
                        raise TypeError(f'{type(kv).__name__} is not a Keyvalue!')
                    self._value.append(kv.copy())
            return self
        else:
            raise LeafKeyvalueError(self, 'extend')

    def append(self, other: Union[Iterable['Keyvalues'], 'Keyvalues']) -> None:
        """Append another keyvalue to this one.

        Deprecated behaviour: Accept an iterable of keyvalues or a root keyvalue
        which are merged into this one.
        """
        if isinstance(self._value, list):
            if isinstance(other, Keyvalues):
                if other._folded_name is None:
                    warnings.warn(
                        "Append()ing a root Keyvalue is confusing, use extend() instead.",
                        DeprecationWarning, 2,
                    )
                    if isinstance(other._value, str):
                        raise ValueError('A leaf root Keyvalue should not exist!')
                    self._value.extend(other._value)
                else:
                    self._value.append(other)
            else:
                warnings.warn(
                    "Use extend() for appending iterables of Keyvalues, not append().",
                    DeprecationWarning, 2,
                )
                for kv in other:
                    if not isinstance(kv, Keyvalues):
                        raise TypeError(f'{type(kv).__name__} is not a Keyvalue!')
                    self._value.append(kv)
        else:
            raise LeafKeyvalueError(self, 'append to')

    def extend(self, other: Iterable['Keyvalues']) -> None:
        """Extend this keyvalue with the contents of another, or an iterable."""
        if not isinstance(self._value, list):
            raise LeafKeyvalueError(self, 'extend')

        for kv in other:
            if not isinstance(kv, Keyvalues):
                raise TypeError(f'{type(kv)} is not a Keyvalue!')
            self._value.append(kv.copy())

    def merge_children(self, *names: str) -> None:
        """Merge together any children of ours with the given names.

        After execution, this tree will have only one sub-Keyvalue for each of the given names.
        Direct leaf keyvalues will be left unchanged, even if they happen to match the given names.
        """
        if not isinstance(self._value, list):
            raise LeafKeyvalueError(self, 'merge children in')
        folded_names = [name.casefold() for name in names]
        new_list = []
        # Optional only here because _folded_name may be None - it'll never actually be there.
        merge: Dict[Optional[str], Keyvalues] = {
            name: Keyvalues(name, [])
            for name in folded_names
        }
        assert None not in merge, 'Names cannot be none!'

        item: Keyvalues
        for item in self._value[:]:
            if isinstance(item._value, list) and item._folded_name in folded_names:
                kv = merge[item._folded_name]
                assert isinstance(kv._value, list)
                kv._value.extend(item)
            else:
                new_list.append(item)

        for kv_name in names:
            kv = merge[kv_name.casefold()]
            if len(kv._value) > 0:
                new_list.append(kv)

        self._value = new_list

    def ensure_exists(self, key: str) -> 'Keyvalues':
        """Ensure a Keyvalue block exists with this name, and return it."""
        if not isinstance(self._value, list):
            raise LeafKeyvalueError(self, 'add children to')
        try:
            return self.find_key(key)
        except NoKeyError:
            kv = Keyvalues(key, [])
            self._value.append(kv)
            return kv

    def has_children(self) -> builtins.bool:
        """Does this have child keyvalues?"""
        return isinstance(self._value, list)

    def is_root(self) -> builtins.bool:
        """Check if the keyvalue is a root, returned from the :py:meth:`parse()` or :py:meth:`root()` methods.

        See :py:meth:`root()` for more information.
        """
        return self._real_name is None

    def __repr__(self) -> str:
        return f'Keyvalues({self._real_name!r}, {self._value!r})'

    def __str__(self) -> str:
        return self.serialise()

    @overload
    def serialise(
        self, file: _SupportsWrite,
        /, *, indent: str = '\t', indent_braces: builtins.bool = True,
        start_indent: str = '',
    ) -> None: ...
    @overload
    def serialise(
        self,
        /, *, indent: str = '\t', indent_braces: builtins.bool = True,
        start_indent: str = '',
    ) -> str: ...

    def serialise(
        self,
        file: Optional[_SupportsWrite] = None,
        /, *,
        indent: str = '\t',
        indent_braces: builtins.bool = True,
        start_indent: str = '',
    ) -> Optional[str]:
        """Serialise the keyvalues data to a file, or return as a string.

        Recursive trees are not permitted.

        :param file: The file to write to. If omitted, the data is returned instead.
        :param indent: The characters to use for each indentation level.
        :param indent_braces: If enabled, indent the braces to match the block contents, instead of the name.
        :param start_indent: The initial starting indentation. Useful to allow serialising inside an existing file.
        """
        buffer: Optional[io.StringIO] = None
        if file is None:
            file = buffer = io.StringIO()

        if indent_braces:
            open_brace = f'{indent}{{\n'
            close_brace = f'{indent}}}\n'
        else:
            open_brace, close_brace = '{\n', '}\n'

        self._serialise(file, indent, open_brace, close_brace, start_indent)

        if buffer is not None:
            return buffer.getvalue()
        return None

    def _serialise(
        self,
        file: _SupportsWrite,
        indent: str, open_brace: str, close_brace: str,
        cur_indent: str,
    ) -> None:
        """Implements write(), after values are prepared."""
        child: Keyvalues
        if isinstance(self._value, list):
            if self._real_name is None:
                # If the name is None, we just output the children
                # without a "Name" { } surround. These Keyvalue objects represent the root.
                for child in self._value:
                    child._serialise(file, indent, open_brace, close_brace, '')
            else:
                file.write(f'{cur_indent}"{self._real_name}"\n')
                file.write(f'{cur_indent}{open_brace}')
                child_indent = f"{cur_indent}{indent}"
                for child in self._value:
                    child._serialise(file, indent, open_brace, close_brace, child_indent)
                file.write(f'{cur_indent}{close_brace}')
        else:
            # We need to escape quotes and backslashes, so they don't get detected.
            assert self._real_name is not None, repr(self)
            file.write(f'{cur_indent}"{escape_text(self._real_name)}" "{escape_text(self._value)}"\n')

    serialize = serialise  # Alias

    @deprecated('Use serialise() instead.')
    def export(self) -> Iterator[str]:
        """Generate the set of strings for a keyvalues file.

        Recursively calls itself for all child keyvalues.

        :deprecated: Use :py:meth:`serialise()` instead.
        """
        if isinstance(self._value, list):
            if self._real_name is None:
                # If the name is None, we just output the children
                # without a "Name" { } surround. These Keyvalue objects represent the root.
                for kv in self._value:
                    yield from kv.export()
            else:
                assert self._real_name is not None, repr(self)
                yield f'"{self._real_name}"\n'
                yield '\t{\n'
                yield from (
                    '\t' + line
                    for kv in self._value
                    for line in kv.export()
                )
                yield '\t}\n'
        else:
            # We need to escape quotes and backslashes, so they don't get detected.
            assert self._real_name is not None, repr(self)
            yield f'"{escape_text(self._real_name)}" "{escape_text(self._value)}"\n'

    def build(self) -> '_Builder':
        """Allows appending a tree to this keyvalue in a convenient way.

        Use as follows::

            # doctest: +NORMALIZE_WHITESPACE
            >>> kv = Keyvalues('name', [])
            >>> with kv.build() as builder:
            ...     builder.root1('blah')
            ...     builder.root2('blah')
            ...     with builder.subprop:
            ...         subprop = builder.config('value')
            ...         builder['unusual name']('value')
            Keyvalues('root1', 'blah')
            Keyvalues('root2', 'blah')
            Keyvalues('unusual name', 'value')
            >>> print(subprop) # doctest: +NORMALIZE_WHITESPACE
            "config" "value"
            >>> print(kv.serialise()) # doctest: +NORMALIZE_WHITESPACE
            "name"
                {
                "root1" "blah"
                "root2" "blah"
                "subprop"
                    {
                    "config" "value"
                    "unusual name" "value"
                    }
                }

        The return values/results of the context manager are the keyvalues.
        Set names by ``builder.name``, ``builder['name']``. For keywords append ``_``.

        Alternatively::

            >>> with Keyvalues('name', []).build() as builder:
            ...     builder.root1('blah')
            ...     builder.root2('blah')
            Keyvalues('root1', 'blah')
            Keyvalues('root2', 'blah')
            >>> kv = builder()
            >>> print(repr(kv))
            Keyvalues('name', [Keyvalues('root1', 'blah'), Keyvalues('root2', 'blah')])
        """
        if not isinstance(self._value, list):
            raise LeafKeyvalueError(self, 'add children to')
        return _Builder(self)


class _Builder:
    """Allows constructing keyvalues trees using with: chains.

    This is the builder you get directly, which is interacted with for
    all the hierachies.  Calling it then returns the original tree.
    """
    def __init__(self, parent: Keyvalues) -> None:
        self._parents = [parent]

    def __getattr__(self, name: str) -> '_BuilderElem':
        """Accesses by attribute produce a keyvalue with that name.

        For all Python keywords a trailing underscore is ignored.
        """
        return _BuilderElem(self, self._keywords.get(name, name))

    def __getitem__(self, name: str) -> '_BuilderElem':
        """This allows accessing names dynamically easily, or names which aren't identifiers."""
        return _BuilderElem(self, name)

    def __enter__(self) -> '_Builder':
        """Start a keyvalue block."""
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException], exc_val: BaseException, exc_tb: types.TracebackType,
    ) -> None:
        """Ends the keyvalue block."""
        pass

    def __call__(self) -> Keyvalues:
        """Return the tree root."""
        return self._parents[0]

    _keywords: ClassVar[Mapping[str, str]] = {kw + '_': kw for kw in keyword.kwlist}


# noinspection PyProtectedMember
class _BuilderElem:
    """Allows constructing keyvalue trees using with: chains.

    This is produced when indexing or accessing attributes on the builder,
    and is then called or used as a context manager to enter the builder.
    """
    def __init__(self, builder: _Builder, name: str) -> None:
        self._builder = builder
        self._name = name

    def __call__(self, value: str) -> Keyvalues:
        """Add a key-value pair."""
        kv = Keyvalues(self._name, value)
        self._builder._parents[-1].append(kv)
        return kv

    def __enter__(self) -> Keyvalues:
        """Start a keyvalue block."""
        kv = Keyvalues(self._name, [])
        self._builder._parents[-1].append(kv)
        self._builder._parents.append(kv)
        return kv

    def __exit__(
        self,
        exc_type: Type[BaseException], exc_val: BaseException, exc_tb: types.TracebackType,
    ) -> None:
        """End a keyvalue block."""
        self._builder._parents.pop()
