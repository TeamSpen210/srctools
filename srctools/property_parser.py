"""Reads and parses Valve's KeyValues files.

These files follow the following general format:
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
    Call 'Property(name, value) to get a property object, or
    Property.parse(file, 'name') to parse a file.

    This will perform a round-trip file read:
    >>> with open('filename.txt', 'r') as f:
    ...     props = Property.parse(f, 'filename.txt')
    >>> with open('filename_2.txt', 'w') as f:
    ...     for line in props.export():
                f.write(line)

    Property values should be either a string, or a list of children Properties.
    Names will be converted to lowercase automatically; use Prop.real_name to
    obtain the original spelling. To allow multiple root blocks in a file, the
    returned property from Property.parse() has a name of None - this property
    type will export with un-indented children.

    Properties with children can be indexed by their names, or by a
    ('name', default) tuple:

    >>> props = Property('Top', [
    ...     Property('child1', '1'),
    ...     Property('child2', '0'),
    ... ])
    ... props['child1']
    '1'
    >>> props['child3']
    Traceback (most recent call last):
        ...
    IndexError: No key child3!
    >>> props['child3', 'default']
    'default'
    >>> props['child4', object()]
    <object object ax 0x...>
    >>> del props['child2']
    >>> props['child3'] = 'new value'
    >>> props
    Property('Top', [Property('child1', '1'), Property('child3', 'new value')])

    \n, \t, and \\ will be converted in Property values.
"""
import re
import sys
import srctools

from srctools import BOOL_LOOKUP, Vec as _Vec

from typing import (
    Optional, Union, Any,
    Dict, List, Tuple, Iterator,
)

__all__ = ['KeyValError', 'NoKeyError', 'Property']

# various escape sequences that we allow
REPLACE_CHARS = [
    (r'\n',  '\n'),
    (r'\t',  '\t'),
    (r'\/',  '/'),
    ('\\\\', '\\'),
]

# Sentinel value to indicate that no default was given to find_key()
_NO_KEY_FOUND = object()

# We allow bare identifiers on lines, but they can't contain quotes or brackets.
_RE_IDENTIFIER = re.compile('^[^"\'{}<>();:\[\]]+\w*$')

_Prop_Value = Union[List['Property'], str]

# Various [flags] used after property names in some Valve files.
# See https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/tier1/KeyValues.cpp#L2055
PROP_FLAGS = {
    # We know we're not on a console...
    'x360': False,
    'ps3': False,
    'gameconsole': False,

    'win32': True,  # Not actually windows, it actually means 'PC'
    'osx': sys.platform.startswith('darwin'),
    'linux': sys.platform.startswith('linux'),
}


class KeyValError(Exception):
    """An error that occured when parsing a Valve KeyValues file.

    mess = The error message that occured.
    file = The filename passed to Property.parse(), if it exists
    line_num = The line where the error occured.
    """
    def __init__(
            self,
            message: str,
            file: Optional[str],
            line: Optional[int]
            ) -> None:
        super().__init__()
        self.mess = message
        self.file = file
        self.line_num = line

    def __repr__(self):
        return 'KeyValError({!r}, {!r}, {!r})'.format(
            self.mess,
            self.file,
            self.line_num,
            )

    def __str__(self):
        """Generate the complete error message.

        This includes the line number and file, if avalible.
        """
        mess = self.mess
        if self.line_num:
            mess += '\nError occured on line ' + str(self.line_num)
            if self.file:
                mess += ', with file'
        if self.file:
            if not self.line_num:
                mess += '\nError occured with file'
            mess += ' "' + self.file + '"'
        return mess


class NoKeyError(LookupError):
    """Raised if a key is not found when searching from find_key().

    key = The missing key that was asked for.
    """
    def __init__(self, key):
        super().__init__()
        self.key = key

    def __repr__(self):
        return 'NoKeyError({!r})'.format(self.key)

    def __str__(self):
        return "No key " + self.key + "!"


def read_multiline_value(file, line_num, filename):
    """Pull lines out until a quote character is reached."""
    lines = ['']  # We return with a beginning newline
    # Re-looping over the same iterator means we don't repeat lines
    for line_num, line in file:
        if isinstance(line, bytes):
            # Decode bytes using utf-8
            line = line.decode('utf-8')
        if '"' in line:
            # Split the line, handling \" inside
            line_parts = srctools.escape_quote_split(line)
            # A correct line will result in ['blah', '  ']
            if len(line_parts) > 1:
                comment = line_parts[1].lstrip()
                # Non-whitespace after the quote, or multiple quotes.
                if comment and not comment.startswith('//') or len(line_parts) > 2:
                    raise KeyValError(
                        'Extra characters after string end! ' + repr(line_parts),
                        filename,
                        line_num,
                    )
                lines.append(line_parts[0])
                return '\n'.join(lines)
            elif len(line_parts) == 1:
                # No actual quote here, continue.
                lines.append(line_parts[0])
                continue  # Don't write the original
            else:
                assert 0, 'escape_quote_split() returned nothing!'
        lines.append(line)
    else:
        # We hit EOF!
        raise KeyValError(
            "Reached EOF without ending quote!",
            filename,
            None,
        )


def read_flag(line_end, filename, line_num):
    """Read a potential [] flag."""
    flag = line_end.lstrip()
    if flag[:1] == '[':
        flag = flag[1:]
        if ']' not in flag:
            raise KeyValError(
                'Unterminated [flag] on '
                'line: "{}"'.format(line_end),
                filename,
                line_num,
            )
        flag, comment = flag.split(']', 1)
        # Parse the flag
        if flag[:1] == '!':
            inv = True
            flag = flag[1:]
        else:
            inv = False
    else:
        comment = flag
        flag = inv = None

    # Check for unexpected text at the end of a line..
    comment = comment.lstrip()
    if comment and not comment.startswith('//'):
        raise KeyValError(
            'Extra text after '
            'line: "{}"'.format(line_end),
            filename,
            line_num,
        )

    if flag:
        # If inv is False, we need True flags.
        # If inv is True, we need False flags.
        return inv is not PROP_FLAGS.get(flag.casefold(), True)
    return True


class Property:
    """Represents Property found in property files, like those used by Valve.

    Value should be a string (for leaf properties), or a list of children
    Property objects.
    The name should be a string, or None for a root object.
    Root objects export each child at the topmost indent level.
        This is produced from Property.parse() calls.
    """
    # Helps decrease memory footprint with lots of Property values.
    __slots__ = ('_folded_name', 'real_name', 'value')

    def __init__(
            self: 'Property',
            name: Optional[str],
            value: _Prop_Value,
            ):
        """Create a new property instance.

        """
        self.real_name = name  # type: Optional[str]
        self.value = value  # type: _Prop_Value
        self._folded_name = (
            None if name is None
            else name.casefold()
        )  # type: Optional[str]

    @property
    def name(self) -> Optional[str]:
        """Name automatically casefolds() any given names.

        This ensures comparisons are always case-sensitive.
        Read .real_name to get the original value.
        """
        return self._folded_name

    @name.setter
    def name(self, new_name):
        self.real_name = new_name
        if new_name is None:
            self._folded_name = None
        else:
            self._folded_name = new_name.casefold()

    def edit(self, name=None, value=None):
        """Simultaneously modify the name and value."""
        if name is not None:
            self.real_name = name
            self._folded_name = name.casefold()
        if value is not None:
            self.value = value
        return self

    @staticmethod
    def parse(file_contents, filename='') -> "Property":
        """Returns a Property tree parsed from given text.

        filename, if set should be the source of the text for debug purposes.
        file_contents should be an iterable of strings
        """
        if not filename:
            # Try to pull the name off the file, if it's a file object.
            filename = getattr(file_contents, 'name', '')

        file_iter = enumerate(file_contents, start=1)

        # The block we are currently adding to.

        # The special name 'None' marks it as the root property, which
        # just outputs its children when exported. This way we can handle
        # multiple root blocks in the file, while still returning a single
        # Property object which has all the methods.
        cur_block = Property(None, [])

        # A queue of the properties we are currently in (outside to inside).
        open_properties = [cur_block]

        # Do we require a block to be opened next? ("name"\n must have { next.)
        requires_block = False

        is_identifier = _RE_IDENTIFIER.match

        for line_num, line in file_iter:
            if isinstance(line, bytes):
                # Decode bytes using utf-8
                line = line.decode('utf-8')
            freshline = line.strip()

            if not freshline or freshline[:2] == '//':
                # Skip blank lines and comments!
                continue

            if freshline[0] == '{':
                # Open a new block - make sure the last token was a name..
                if not requires_block:
                    raise KeyValError(
                        'Property cannot have sub-section if it already '
                        'has an in-line value.',
                        filename,
                        line_num,
                    )
                requires_block = False
                cur_block = cur_block[-1]
                cur_block.value = []
                open_properties.append(cur_block)
                continue
            else:
                # A "name" line was found, but it wasn't followed by '{'!
                if requires_block:
                    raise KeyValError(
                        "Block opening ('{') required!",
                        filename,
                        line_num,
                    )

            if freshline[0] == '"':   # data string
                if '\\"' in freshline:
                    # There's escaped double-quotes in here - handle that
                    # split slowly but properly.
                    line_contents = srctools.escape_quote_split(freshline)
                else:
                    # Just call split normally.
                    line_contents = freshline.split('"')
                # Line_contents = [indent, name, space, value, flags/comments]

                name = line_contents[1]
                try:
                    value = line_contents[3]
                except IndexError:  # It doesn't have a value - it's a block.
                    cur_block.append(Property(name, ''))
                    requires_block = True  # Ensure the next token must be a '{'.
                    continue  # Skip to next line

                # Special case - comment between name/value sections -
                # it's a name block then.
                if line_contents[2].lstrip().startswith('//'):
                    cur_block.append(Property(name, ''))
                    requires_block = True
                    continue
                # Check there isn't text between name and value!
                elif line_contents[2].strip():
                    raise KeyValError(
                        "Extra text (" + line_contents[2] + ") in line!",
                        filename,
                        line_num
                    )

                if len(line_contents) < 5:
                    # It's a multiline value - no ending quote!
                    value += read_multiline_value(
                        file_iter,
                        line_num,
                        filename,
                    )
                if value and '\\' in value:
                    for orig, new in REPLACE_CHARS:
                        value = value.replace(orig, new)

                # Line_contents[4] is the start of the comment, check for [] flags.
                if len(line_contents) >= 5:
                    # Pass along all the parts after, so this can validate
                    # them (for extra quotes.)
                    if read_flag('"'.join(line_contents[4:]), filename, line_num):
                        cur_block.append(Property(name, value))
                else:
                    # No flag, add unconditionally
                    cur_block.append(Property(name, value))
            elif freshline[0] == '}':
                # Move back a block
                open_properties.pop()
                try:
                    cur_block = open_properties[-1].value
                except IndexError:
                    # No open blocks!
                    raise KeyValError(
                        'Too many closing brackets.',
                        filename,
                        line_num,
                    )

            # Handle name bare on one line - it's a name block. This is used
            # in VMF files...
            elif is_identifier(freshline):
                cur_block.append(Property(freshline, ''))
                requires_block = True  # Ensure the next token must be a '{'.
                continue
            else:
                raise KeyValError(
                    "Unexpected beginning character '"
                    + freshline[0]
                    + '"!',
                    filename,
                    line_num,
                )

        if requires_block:
            raise KeyValError(
                "Block opening ('{') required, but hit EOF!",
                filename,
                line=None,
            )
        
        if len(open_properties) > 1:
            raise KeyValError(
                'End of text reached with remaining open sections.',
                filename,
                line=None,
            )
        return open_properties[0]

    def find_all(self, *keys) -> Iterator['Property']:
        """Search through the tree, yielding all properties that match a particular path.

        """
        depth = len(keys)
        if depth == 0:
            raise ValueError("Cannot find_all without commands!")

        targ_key = keys[0].casefold()
        for prop in self:
            if not isinstance(prop, Property):
                raise ValueError(
                    'Cannot find_all on a value that is not a Property!'
                )
            if prop._folded_name == targ_key is not None:
                if depth > 1:
                    if prop.has_children():
                        yield from Property.find_all(prop, *keys[1:])
                else:
                    yield prop

    def find_children(self, *keys) -> Iterator['Property']:
        """Search through the tree, yielding children of properties in a path.

        """
        for block in self.find_all(*keys):
            yield from block

    def find_key(self, key, def_: _Prop_Value=_NO_KEY_FOUND):
        """Obtain the child Property with a given name.

        - If no child is found with the given name, this will return the
          default value wrapped in a Property, or raise NoKeyError if
          none is provided.
        - This prefers keys located closer to the end of the value list.
        """
        key = key.casefold()
        for prop in reversed(self.value):  # type: Property
            if prop._folded_name == key:
                return prop
        if def_ is _NO_KEY_FOUND:
            raise NoKeyError(key)
        else:
            return Property(key, def_)
            # We were given a default, return it wrapped in a Property.

    def _get_value(self, key, def_=_NO_KEY_FOUND):
        """Obtain the value of the child Property with a given name.

        - If no child is found with the given name, this will return the
          default value, or raise NoKeyError if none is provided.
        - This prefers keys located closer to the end of the value list.
        """
        key = key.casefold()
        for prop in reversed(self.value):  # type: Property
            if prop._folded_name == key:
                return prop.value
        if def_ is _NO_KEY_FOUND:
            raise NoKeyError(key)
        else:
            return def_

    def int(self, key: str, def_: Any=0) -> int:
        """Return the value of an integer key.

        Equivalent to int(prop[key]), but with a default value if missing or
        invalid.
        If multiple keys with the same name are present, this will use the
        last only.
        """
        try:
            return int(self._get_value(key))
        except (NoKeyError, ValueError, TypeError):
            return def_

    def float(self, key: str, def_: Any=0.0) -> float:
        """Return the value of an integer key.

        Equivalent to float(prop[key]), but with a default value if missing or
        invalid.
        If multiple keys with the same name are present, this will use the
        last only.
        """
        try:
            return float(self._get_value(key))
        except (NoKeyError, ValueError, TypeError):
            return def_

    def bool(self, key: str, def_: Any=False) -> bool:
        """Return the value of an boolean key.

        The value may be case-insensitively 'true', 'false', '1', '0', 'T',
        'F', 'y', 'n', 'yes', or 'no'.
        If multiple keys with the same name are present, this will use the
        last only.
        """
        try:
            return BOOL_LOOKUP[self._get_value(key).casefold()]
        except LookupError:  # base for NoKeyError, KeyError
            return def_

    def vec(self, key, x=0.0, y=0.0, z=0.0) -> _Vec:
        """Return the given property, converted to a vector.

        If multiple keys with the same name are present, this will use the
        last only.
        """
        try:
            return _Vec.from_str(self._get_value(key), x, y, z)
        except LookupError:  # base for NoKeyError, KeyError
            return _Vec(x, y, z)

    def set_key(self, path, value):
        """Set the value of a key deep in the tree hierarchy.

        -If any of the hierarchy do not exist (or do not have children),
          blank properties will be added automatically
        - path should be a tuple of names, or a single string.
        """
        current_prop = self
        if isinstance(path, tuple):
            # Search through each item in the tree!
            for key in path[:-1]:
                folded_key = key.casefold()
                # We can't use find_key() here because we also
                # need to check that the property has chilren to search
                # through
                for prop in reversed(self.value):
                    if (prop.name is not None and
                            prop.name == folded_key and
                            prop.has_children()):
                        current_prop = prop
                        break
                else:
                    # No matching property found
                    new_prop = Property(key, [])
                    current_prop.append(new_prop)
                    current_prop = new_prop
            path = path[-1]
        try:
            current_prop.find_key(path).value = value
        except NoKeyError:
            current_prop.value.append(Property(path, value))

    def copy(self):
        """Deep copy this Property tree and return it."""
        if self.has_children():
            # This recurses if needed
            return Property(
                self.real_name,
                [
                    child.copy()
                    for child in
                    self.value
                ]
            )
        else:
            return Property(self.real_name, self.value)

    def as_dict(self):
        """Convert this property tree into a tree of dictionaries.

        This keeps only the last if multiple items have the same name.
        """
        if self.has_children():
            return {item._folded_name: item.as_dict() for item in self}
        else:
            return self.value

    def __eq__(self, other):
        """Compare two items and determine if they are equal.

        This ignores names.
        """
        if isinstance(other, Property):
            return self.value == other.value
        else:
            return self.value == other  # Just compare values

    def __ne__(self, other):
        """Not-Equal To comparison. This ignores names.
        """
        if isinstance(other, Property):
            return self.value != other.value
        else:
            return self.value != other # Just compare values

    def __lt__(self, other):
        """Less-Than comparison. This ignores names.
        """
        if isinstance(other, Property):
            return self.value < other.value
        else:
            return self.value < other

    def __gt__(self, other):
        "Greater-Than comparison. This ignores names."
        if isinstance(other, Property):
            return self.value > other.value
        else:
            return self.value > other

    def __le__(self, other):
        "Less-Than or Equal To comparison. This ignores names."
        if isinstance(other, Property):
            return self.value <= other.value
        else:
            return self.value <= other

    def __ge__(self, other):
        "Greater-Than or Equal To comparison. This ignores names."
        if isinstance(other, Property):
            return self.value >= other.value
        else:
            return self.value >= other

    def __len__(self):
        """Determine the number of child properties."""
        if self.has_children():
            return len(self.value)
        raise ValueError("{!r} has no children!".format(self))

    def __bool__(self):
        """Properties are true if we have children, or have a value."""
        if self.has_children():
            return len(self.value) > 0
        else:
            return bool(self.value)

    def __iter__(self) -> Iterator['Property']:
        """Iterate through the value list.

        """
        if self.has_children():
            return iter(self.value)
        else:
            raise ValueError(
                "Can't iterate through {!r} without children!".format(self)
            )

    def iter_tree(self, blocks=False) -> Iterator['Property']:
        """Iterate through all properties in this tree.

        This goes through properties in the same order that they will serialise
        into.
        If blocks is True, the property blocks will be returned as well as
        keyvalues. If false, only keyvalues will be yielded.
        """
        if self.has_children():
            return self._iter_tree(blocks)
        else:
            raise ValueError(
                "Can't iterate through {!r} without children!".format(self)
            )

    def _iter_tree(self, blocks):
        """Implementation of iter_tree(). This assumes self has children."""
        for prop in self.value:  # type: Property
            if prop.has_children():
                if blocks:
                    yield prop
                yield from prop._iter_tree(blocks)
            else:
                yield prop

    def __contains__(self, key):
        """Check to see if a name is present in the children."""
        key = key.casefold()
        if self.has_children():
            for prop in self.value:  # type: Property
                if prop._folded_name == key:
                    return True
            return False

        raise ValueError("Can't search through properties without children!")

    def __getitem__(
            self,
            index: Union[
                str,
                int,
                slice,
                Tuple[Union[str, int, slice], Union[_Prop_Value, Any]],
            ],
            ) -> str:
        """Allow indexing the children directly.

        - If given an index, it will search by position.
        - If given a string, it will find the last Property with that name.
          (Default can be chosen by passing a 2-tuple like Prop[key, default])
        - If none are found, it raises IndexError.
        """
        if self.has_children():
            if isinstance(index, int):
                return self.value[index]
            else:
                if isinstance(index, tuple):
                    # With default value
                    return self._get_value(index[0], def_=index[1])
                else:
                    try:
                        return self._get_value(index)
                    except NoKeyError as no_key:
                        raise IndexError(no_key) from no_key
        else:
            raise ValueError("Can't index a Property without children!")

    def __setitem__(
            self,
            index: Union[int, slice, str],
            value: _Prop_Value
            ):
        """Allow setting the values of the children directly.

        If the value is a Property, this will be inserted under the given
        name or index.
        - If given an index, it will search by position.
        - If given a string, it will set the last Property with that name.
        - If none are found, it appends the value to the tree.
        - If given a tuple of strings, it will search through that path,
          and set the value of the last matching Property.
        """
        if self.has_children():
            if isinstance(index, int):
                self.value[index] = value
            else:
                if isinstance(value, Property):
                    # We don't want to assign properties, we want to add them under
                    # this name!
                    value.name = index
                    try:
                        # Replace at the same location..
                        index = self.value.index(self.find_key(index))
                    except NoKeyError:
                        self.value.append(value)
                    else:
                        self.value[index] = value
                else:
                    try:
                        self.find_key(index).value = value
                    except NoKeyError:
                        self.value.append(Property(index, value))
        else:
            raise ValueError("Can't index a Property without children!")

    def __delitem__(self, index):
        """Delete the given property index.

        - If given an integer, it will delete by position.
        - If given a string, it will delete the last Property with that name.
        """
        if self.has_children():
            if isinstance(index, int):
                del self.value[index]
            else:
                try:
                    self.value.remove(self.find_key(index))
                except NoKeyError as no_key:
                    raise IndexError(no_key) from no_key
        else:
            raise IndexError("Can't index a Property without children!")

    def clear(self):
        """Delete the contents of a block."""
        if self.has_children():
            self.value.clear()
        else:
            raise ValueError("Can't clear a Property without children!")

    def __add__(self, other):
        """Allow appending other properties to this one.

        This deep-copies the Property tree first.
        Works with either a sequence of Properties or a single Property.
        """
        if self.has_children():
            copy = self.copy()
            if isinstance(other, Property):
                if other._folded_name is None:
                    copy.value.extend(other.value)
                else:
                    # We want to add the other property tree to our
                    # own, not its values.
                    copy.value.append(other)
            else: # Assume a sequence.
                copy.value += other # Add the values to ours.
            return copy
        else:
            return NotImplemented

    def __iadd__(self, other):
        """Allow appending other properties to this one.

        This is the += op, where it does not copy the object.
        """
        if self.has_children():
            if isinstance(other, Property):
                if other._folded_name is None:
                    self.value.extend(other.value)
                else:
                    self.value.append(other)
            else:
                self.value += other
            return self
        else:
            return NotImplemented

    append = __iadd__

    def merge_children(self, *names: str):
        """Merge together any children of ours with the given names.

        After execution, this tree will have only one sub-Property for
        each of the given names. This ignores leaf Properties.
        """
        folded_names = [name.casefold() for name in names]
        new_list = []
        merge = {
            name.casefold(): Property(name, [])
            for name in
            names
        }
        if self.has_children():
            for item in self.value[:]:  # type: Property
                if item._folded_name in folded_names:
                    merge[item._folded_name].value.extend(item.value)
                else:
                    new_list.append(item)
        for prop_name in names:
            prop = merge[prop_name.casefold()]
            if len(prop.value) > 0:
                new_list.append(prop)

        self.value = new_list

    def ensure_exists(self, key) -> 'Property':
        """Ensure a Property group exists with this name, and return it."""
        try:
            return self.find_key(key)
        except NoKeyError:
            prop = Property(key, [])
            self.value.append(prop)
            return prop

    def has_children(self):
        """Does this have child properties?"""
        return isinstance(self.value, list)

    def __repr__(self):
        return 'Property(' + repr(self.real_name) + ', ' + repr(self.value) + ')'

    def __str__(self):
        return ''.join(self.export())

    def export(self):
        """Generate the set of strings for a property file.

        Recursively calls itself for all child properties.
        """
        if isinstance(self.value, list):
            if self.name is None:
                # If the name is None, we just output the chilren
                # without a "Name" { } surround. These Property
                # objects represent the root.
                for prop in self.value:
                    yield from prop.export()
            else:
                yield '"' + self.real_name + '"\n'
                yield '\t{\n'
                yield from (
                    '\t' + line
                    for prop in self.value
                    for line in prop.export()
                )
                yield '\t}\n'
        else:
            yield '"{}" "{}"\n'.format(
                self.real_name,
                self.value.replace('"', '\\"')
            )
