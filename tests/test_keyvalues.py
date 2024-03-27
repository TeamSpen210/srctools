"""Test the keyvalues module."""
from typing import Any, Generator, List, Type, Union
import itertools
import io

import pytest

from srctools import keyvalues as kv_mod
from srctools.keyvalues import KeyValError, Keyvalues, LeafKeyvalueError, NoKeyError
# noinspection PyProtectedMember
from srctools.tokenizer import Cy_Tokenizer, Py_Tokenizer, Tokenizer


if Cy_Tokenizer is not None:
    parms = [Cy_Tokenizer, Py_Tokenizer]
    ids = ['Cython tokenizer', 'Python tokenizer']
else:
    pytest.fail('No _tokenizer')
    parms = [Py_Tokenizer]
    ids = ['Python tokenizer']


@pytest.fixture(params=parms, ids=ids)
def py_c_token(request: Any) -> Generator[None, None, None]:
    """Run the test twice, for the Python and C versions of Tokenizer."""
    orig_tok = kv_mod.Tokenizer
    try:
        kv_mod.Tokenizer = request.param
        yield None
    finally:
        kv_mod.Tokenizer = orig_tok


def assert_tree(first: Keyvalues, second: Keyvalues, path: str = '') -> None:
    """Check that two keyvalues trees match exactly (including case)."""
    if first.is_root():
        path = (path + '.<root>') if path else '<root>'
        assert second.is_root(), (first, second)
    else:
        assert not second.is_root(), (first, second)
        path = f'{path}.{first.real_name}' if path else first.real_name
        assert first.name == second.name, (first, second)
        assert first.real_name == second.real_name, (first, second)

    assert first.has_children() == second.has_children(), (first, second)
    if first.has_children():
        for child1, child2 in itertools.zip_longest(first, second):
            assert child1 is not None, f'None != {path}.{child2.name}'
            assert child2 is not None, f'{path}.{child1.name} != None'
            assert_tree(child1, child2, path)
    else:
        assert first.value == second.value, (first, second)


def test_docstring() -> None:
    """Run doctests on the module."""
    import doctest
    assert doctest.testmod(kv_mod, optionflags=doctest.ELLIPSIS).failed == 0


def test_constructor() -> None:
    """Test the constructor for Keyvalues objects."""
    with pytest.deprecated_call(match='Root properties'):
        Keyvalues(None, [])
    Keyvalues('Test', 'value with spaces and ""')
    block = Keyvalues('Test_block', [
        Keyvalues('Test', 'value\0'),
        Keyvalues('Test', [
            Keyvalues('leaf', 'data'),
        ]),
        Keyvalues('Test2', 'other'),
        Keyvalues('Block', []),
    ])
    assert block.real_name == 'Test_block'
    children = list(block)
    assert children[0].real_name == 'Test'
    assert children[1].real_name == 'Test'
    assert children[2].real_name == 'Test2'
    assert children[3].real_name == 'Block'

    assert children[0].value == 'value\0'
    assert children[2].value, 'other'
    assert list(children[3]) == []

    sub_children = list(children[1])
    assert sub_children[0].real_name == 'leaf'
    assert sub_children[0].value == 'data'
    assert len(sub_children) == 1


def test_names() -> None:
    """Test the behaviour of Keyvalues.name."""
    prop = Keyvalues('Test1', 'value')

    # Property.name casefolds the argument.
    assert prop.name == 'test1'
    assert prop.real_name == 'Test1'

    # Editing name modifies both values
    prop.name = 'SECOND_test'
    assert prop.name == 'second_test'
    assert prop.real_name == 'SECOND_test'

    # It can also be set to None - deprecated
    with pytest.deprecated_call(match='[r|R]oot [p|P]ropert[y|ies]'):
        prop.name = None  # type: ignore
    with pytest.deprecated_call(match='[r|R]oot [p|P]ropert[y|ies]'):
        assert prop.name is prop.real_name is None

# If edited, update test_parse() and tokeniser check too!
parse_test = '''

// """"" should be ignored

"Root1"
    {
        

    "Key" "Value"
        "Extra"        "Spaces"
    // "Commented" "out"
    "Block"  {
        "Empty"
             {
             } }
    "Block" // "with value"
  {
 bare
    {   "block" "he\\tre"
          }
            }
       }
    "Root2"
    {
    "Name with \\" in it" "Value with \\" inside"
    "multiline" "text
\tcan continue
for many \\"lines\\" of
  possibly indented

text"
    "Escapes" "\\t \\n \\d"
    "Oneliner" { "name" "value" }
    }
    "CommentChecks"
        {
        "after " "value" //comment [ ] ""
        "FlagBlocks" "This" [test_disabled]
        "Flag" "allowed" [!test_disabled]
        "FlagAllows" "This" [test_enabled]
        "Flag" "blocksthis" [!test_enabled]

        "Replaced" "shouldbe"
        "Replaced" "toreplace" [test_enabled]
        "Replaced" "alsothis"  [test_enabled]
        
        "Replaced" "shouldbe2"
        "Replaced" "toreplace2" [!test_disabled]
        "Replaced" "alsothis2"  [!test_disabled]

        "Replaced"
            {
            "shouldbe3" "replaced3"
            "prop2" "blah"
            }
        "Replaced" [test_enabled]
            {
            "lambda" "should"
            "replace" "above"
            }
        
        "Replaced"
            {
            "shouldbe4" "replaced4"
            "prop2" "blah"
            }
        "Replaced" [!test_disabled]
            {
            "lambda2" "should2"
            "replace2" "above2"
            }
        "otherval" "blah"
        "shouldnotreplace" [test_enabled]
            {
            "key" "value1"
            "key" "value2"
            }
        "skipped" [test_disabled]
            {
            "ignore" "value"
            }
        }
'''

P = Keyvalues
parse_result = Keyvalues.root(
    P('Root1', [
        P("Key", "Value"),
        P("Extra", "Spaces"),
        P("Block", [
            P('Empty', []),
        ]),
        P('Block', [
            P('bare', [
                P('block', 'he\tre'),
            ]),
        ]),
    ]),
    P('Root2', [
        P('Name with " in it', 'Value with \" inside'),
        P('multiline',
          'text\n\tcan continue\nfor many "lines" of\n  possibly indented\n\ntext'
          ),
        # Note, invalid = unchanged.
        P('Escapes', '\t \n \\d'),
        P('Oneliner', [P('name', 'value')]),
    ]),
    P('CommentChecks', [
        P('after ', 'value'),
        P('Flag', 'allowed'),
        P('FlagAllows', 'This'),
        P('Replaced', 'toreplace'),
        P('Replaced', 'alsothis'),
        P('Replaced', 'toreplace2'),
        P('Replaced', 'alsothis2'),
        P('Replaced', [
            P('lambda', 'should'),
            P('replace', 'above'),
        ]),
        P('Replaced', [
            P('lambda2', 'should2'),
            P('replace2', 'above2'),
        ]),
        P('otherval', 'blah'),
        P('shouldnotreplace', [
            P('key', 'value1'),
            P('key', 'value2'),
        ]),
    ]),
)
del P


def test_parse(py_c_token: Type[Tokenizer]) -> None:
    """Test parsing strings."""
    result = Keyvalues.parse(
        # iter() ensures sequence methods aren't used anywhere.
        iter(parse_test.splitlines(keepends=True)),
        # Check active and inactive flags are correctly treated.
        flags={
            'test_enabled': True,
            'test_disabled': False,
        }
    )
    assert_tree(parse_result, result)

    # Test the whole string can be passed too.
    result = Keyvalues.parse(
        parse_test,
        flags={
            'test_enabled': True,
            'test_disabled': False,
        },
    )
    assert_tree(parse_result, result)

    # Check export roundtrips.
    assert_tree(parse_result, Keyvalues.parse(parse_result.export()))


def test_build() -> None:
    """Test the .build() constructor."""
    prop = Keyvalues.root()

    with prop.build() as b:
        with b.Root1:
            b.Key("Value")
            b.Extra("Spaces")
            with b.Block:
                with b.Empty:
                    pass
            with b.Block:
                with b.bare:
                    b.block('he\tre')
        with b.Root2:
            b['Name with " in it']('Value with \" inside')
            b.multiline(
              'text\n\tcan continue\nfor many "lines" of\n  possibly '
              'indented\n\ntext'
            )
            # Note invalid = unchanged.
            b.Escapes('\t \n \\d')
            with b.Oneliner:
                b.name('value')

        with b.CommentChecks:
            b['after ']('value')
            b.Flag('allowed')
            b.FlagAllows('This')
            b.Replaced('toreplace')
            b.Replaced('alsothis')
            b.Replaced('toreplace2')
            b.Replaced('alsothis2')
            with b.Replaced:
                b.lambda_('should')
                b.replace('above')
            with b.Replaced:
                b['lambda2']('should2')
                b.replace2('above2')
            b.otherval('blah')
            with b.shouldnotreplace:
                b.key('value1')
                b.key('value2')

    assert_tree(parse_result, prop)


def test_build_exc() -> None:
    """Test the with statement handles exceptions correctly."""
    class Exc(Exception):
        pass

    prop = Keyvalues('Root', [])

    # Check this does not swallow exceptions.
    with pytest.raises(Exc):
        with prop.build() as build:
            build.prop('Hi')
            raise Exc
    # Exception doesn't rollback.
    assert_tree(Keyvalues('Root', [
        Keyvalues('prop', 'Hi'),
    ]), prop)

    prop.clear()

    with prop.build() as build:
        build.leaf('value')
        with pytest.raises(Exc):
            with build.subprop:
                raise Exc
    assert_tree(Keyvalues('Root', [
        Keyvalues('leaf', 'value'),
        Keyvalues('subprop', []),
    ]), prop)


def test_parse_fails(py_c_token: Type[Tokenizer]) -> None:
    """Test various forms of invalid syntax to ensure they indeed fail."""
    def t(text: str) -> None:
        """Test a string to ensure it fails parsing."""
        try:
            result = Keyvalues.parse(text)
        except KeyValError:
            pass
        else:
            pytest.fail(f"Successfully parsed bad text ({text!r}) to {result!r}")
    # Bare text at end of file
    t('''\
regular text. with sentences.
    ''')
    # Bare text in the middle
    t('''\
regular text. with sentences.
    "blah" "value"
    ''')
    t('''\
"Ok block"
    {
    "missing" //value
    }
''')

    # Test block without a block
    t('''\
"block1"
"no_block" 
''')

    # Test block expecting a {
    t('''\
"block"
    {
    "blsh" "Val"
    }
"block1"
''')

    # Test characters before a keyvalue
    t('''\
bbhf  "text before"
    "key" "value
''')
    t('''
  "text" bl "between"
    "key" "value
''')
    # Test text after the keyvalue
    t('''\
    "text" "value" blah
    "key" "value
    ''')
    # Test quotes after the keyvalue
    t('''
    "text" "with extra" "
''')

    t('''
    "multi" "line
text with
  multiple
  quotes" "
''')

    # Test a flag without ] at end
    t('''
    "Name" "value" [flag
    ''')

    # Test a flag with values after the bracket.
    t('''
    "Name" "value" [flag ] hi
    ''')

    # Test too many closing brackets
    t('''
    "Block"
        {
        "Opened"
            {
            "Closed" "value"
            }
            }
        }
    "More text" "value"
    ''')

    # Test property with a value and block
    t('''
    "Block" "value"
        {
        "Name" "value"
        }
    ''')

    # Test '/' in text by itself (not a comment!)
    t('''\
    "Block"
        {
        "Name" / "Value"
            {
            }
        }
    ''')

    # Test unterminated strings
    t('''\
    "Block"
        {
        "blah
        }
    ''')

    # Test unterminated string with '\' at the end
    t('''"Blah \\''')

    # Test too many open brackets
    t('''\
    "Block"
        {
        "Key" "Value"
        "Block"
            {
            {
            "Key" "value"
            }
        }
    ''')

    # Too many open blocks.
    t('''\
    "Block"
        {
        "Key" "Value"
        "Block2"
            {
            "Key" "Value"
            }
    ''')

    t('''\
    "Key" "value
    which is multi-line
    and no ending.
    ''')

    # Test a key and value split over a line.
    t('''\
    "block"
        {
        "key" "value"
        "key"
        "value"
        }
    ''')


def test_newline_strings(py_c_token: Type[Tokenizer]) -> None:
    """Test that newlines are correctly blocked if the parameter is set."""
    with pytest.raises(KeyValError):
        Keyvalues.parse('"key\nmultiline" "value"')
    with pytest.raises(KeyValError):
        Keyvalues.parse('"key" "value\nmultiline"', newline_values=False)
    root = Keyvalues.parse('"key" "value\rmulti"')
    assert_tree(root, Keyvalues.root(Keyvalues('key', 'value\nmulti')))

    root = Keyvalues.parse('"key\nmulti" "value"', newline_keys=True)
    assert_tree(root, Keyvalues.root(Keyvalues('key\nmulti', 'value')))


def test_serialise() -> None:
    """Test serialisation code."""

    basic_tree = Keyvalues.root(
        Keyvalues('Block1', [
            Keyvalues('kEy"', 'Value\t\n with "quotes" included.'),
            Keyvalues('regular', 'key'),
            Keyvalues('block2', [
                Keyvalues('lambda2', 'should2'),
                Keyvalues('Blank', []),
                Keyvalues('replace2', 'above2'),
            ]),
            Keyvalues('anotherKey', '1'),

        ]),
        Keyvalues('AnotherRoot', [
            Keyvalues('option', 'set'),
        ])
    )

    buf = []

    class File:
        """Implements only the write() method."""
        def write(self, the_line: str, /) -> float:
            buf.append(the_line)
            return 3.14

    basic_tree.serialise(File())
    buf2 = basic_tree.serialise()
    assert buf == [
        '"Block1"\n',
        '\t{\n',
        '\t"kEy\\"" "Value\\t\\n with \\"quotes\\" included."\n',
        '\t"regular" "key"\n',
        '\t"block2"\n',
        '\t\t{\n',
        '\t\t"lambda2" "should2"\n',
        '\t\t"Blank"\n',
        '\t\t\t{\n',
        '\t\t\t}\n',
        '\t\t"replace2" "above2"\n',
        '\t\t}\n',
        '\t"anotherKey" "1"\n',
        '\t}\n',
        '"AnotherRoot"\n',
        '\t{\n',
        '\t"option" "set"\n',
        '\t}\n',
    ]
    assert buf2 == ''.join(buf)

    buf.clear()
    basic_tree.serialise(File(), indent='>>|')
    assert buf == [
        '"Block1"\n',
        '>>|{\n',
        '>>|"kEy\\"" "Value\\t\\n with \\"quotes\\" included."\n',
        '>>|"regular" "key"\n',
        '>>|"block2"\n',
        '>>|>>|{\n',
        '>>|>>|"lambda2" "should2"\n',
        '>>|>>|"Blank"\n',
        '>>|>>|>>|{\n',
        '>>|>>|>>|}\n',
        '>>|>>|"replace2" "above2"\n',
        '>>|>>|}\n',
        '>>|"anotherKey" "1"\n',
        '>>|}\n',
        '"AnotherRoot"\n',
        '>>|{\n',
        '>>|"option" "set"\n',
        '>>|}\n',
    ]

    assert basic_tree.serialise(indent=' ', indent_braces=False) == '''\
"Block1"
{
 "kEy\\"" "Value\\t\\n with \\"quotes\\" included."
 "regular" "key"
 "block2"
 {
  "lambda2" "should2"
  "Blank"
  {
  }
  "replace2" "above2"
 }
 "anotherKey" "1"
}
"AnotherRoot"
{
 "option" "set"
}
'''


def test_edit() -> None:
    """Check functionality of Keyvalues.edit()"""
    test_prop = Keyvalues('Name', 'Value')

    def check(prop: Keyvalues, name: str, value: Union[str, List[str]]) -> None:
        """Check the property was edited, and has the given value."""
        nonlocal test_prop
        assert prop is test_prop
        assert prop.real_name == name
        assert prop.value == value
        test_prop = Keyvalues('Name', 'Value')

    check(test_prop.edit(), 'Name', 'Value')
    check(test_prop.edit(name='new_name',), 'new_name', 'Value')
    check(test_prop.edit(value='new_value'), 'Name', 'new_value')

    # Check converting a block into a keyvalue
    test_prop = Keyvalues('Name', [
        Keyvalues('Name', 'Value')
    ])
    check(test_prop.edit(value='Blah'), 'Name', 'Blah')

    # Check converting a keyvalue into a block.
    child_1 = Keyvalues('Key', 'Value')
    child_2 = Keyvalues('Key2', 'Value')
    new_prop = test_prop.edit(value=[child_1, child_2])
    assert test_prop is new_prop
    assert list(test_prop) == [child_1, child_2]


def test_bool() -> None:
    """Check bool(Property)."""
    assert bool(Keyvalues('Name', '')) is False
    assert bool(Keyvalues('Name', 'value')) is True
    assert bool(Keyvalues('Name', [])) is False
    assert bool(Keyvalues('Name', [
        Keyvalues('Key', 'Value')
    ])) is True


def test_blockfuncs_fail_on_leaf() -> None:
    """Check that methods requiring a block fail on a leaf key, and raise the specific exception."""
    leaf = Keyvalues('Name', 'blah')
    with pytest.raises(LeafKeyvalueError):
        next(leaf.find_all("blah"))
    with pytest.raises(LeafKeyvalueError):
        leaf.find_key("blah")
    with pytest.raises(LeafKeyvalueError):
        next(iter(leaf))
    with pytest.raises(LeafKeyvalueError):
        leaf['blah']
    with pytest.raises(LeafKeyvalueError):
        leaf[0]
    with pytest.raises(LeafKeyvalueError):
        leaf[1:2]
    with pytest.raises(LeafKeyvalueError):
        leaf['blah', '']

    with pytest.raises(LeafKeyvalueError):
        leaf['blah'] = 't'
    with pytest.raises(LeafKeyvalueError):
        leaf[0] = 't'  # type: ignore
    with pytest.raises(LeafKeyvalueError):
        leaf[1:2] = 't'  # type: ignore
    with pytest.raises(LeafKeyvalueError):
        leaf['blah', ''] = 't'  # type: ignore

    with pytest.raises(LeafKeyvalueError):
        leaf.int('blah')
    with pytest.raises(LeafKeyvalueError):
        leaf.bool('blah')
    with pytest.raises(LeafKeyvalueError):
        leaf.float('blah')
    with pytest.raises(LeafKeyvalueError):
        leaf.vec('blah')
    with pytest.raises(LeafKeyvalueError):
        len(leaf)
    with pytest.raises(LeafKeyvalueError):
        with leaf.build():
            pass
    with pytest.raises(LeafKeyvalueError):
        leaf.ensure_exists('blah')
    with pytest.raises(LeafKeyvalueError):
        leaf.set_key(("blah", "another"), 45)
    with pytest.raises(LeafKeyvalueError):
        leaf.merge_children()


def test_search() -> None:
    """Test various key search funcs."""
    key1 = Keyvalues('key1', '1')
    key2 = Keyvalues('key2', '2')
    kEy1 = Keyvalues('kEy1', '3')
    bLock1 = Keyvalues('bLock1', [Keyvalues('leaf', '45')])
    test1 = Keyvalues('Block', [key1, key2, bLock1, kEy1])

    # Search preferring later keys, case-insensitive.
    assert test1.find_key('keY1') is kEy1
    assert test1.find_key('key2') is key2
    # Default values.
    assert test1.find_key('missing', '45') == Keyvalues('missing', '45')
    assert test1.find_key('missing', or_blank=True) == Keyvalues('missing', [])
    assert test1.find_key('key1', '45') is kEy1
    assert test1.find_key('key1', or_blank=True) is kEy1

    assert test1['key1'] == '3'
    assert test1['key2'] == '2'
    assert test1['missing', 'default'] == 'default'

    assert list(test1.find_all('Key1')) == [key1, kEy1]

    with pytest.raises(IndexError):
        test1['notpresent']
    with pytest.raises(NoKeyError):
        test1.find_key('nonpresent')
    # Find_block ignores leaf keys.
    with pytest.raises(NoKeyError):
        test1.find_block('key1')
    with pytest.raises(NoKeyError):
        test1.find_block('Block2')
    assert test1.find_block('blOCk1') is bLock1
    assert test1.find_block('Block2', or_blank=True) == Keyvalues('Block2', [])


def test_getitem() -> None:
    """Test various modes of getitem functions correctly."""
    key1 = Keyvalues('key1', '1')
    key2 = Keyvalues('key2', '2')
    kEy1 = Keyvalues('kEy1', '3')
    bLock1 = Keyvalues('bLock1', [Keyvalues('leaf', '45')])
    root = Keyvalues('Block', [key1, key2, bLock1, kEy1])

    assert root[0] is key1
    assert root[2] is bLock1

    plist = root[1:4]
    assert isinstance(plist, list)
    assert len(plist) == 3
    assert plist[0] is key2
    assert plist[1] is bLock1
    assert plist[2] is kEy1

    plist = root[2::-2]
    assert isinstance(plist, list)
    assert len(plist) == 2
    assert plist[0] is bLock1
    assert plist[1] is key1

    assert root['key1'] == '3'
    assert root['key45', 'default'] == 'default'
    assert root['key2', any] == '2'
    assert root['key45', any] is any


def test_setitem() -> None:
    """Test various modes of setitem functions correctly."""
    key1 = Keyvalues('key1', '1')
    key2 = Keyvalues('key2', '2')
    kEy1 = Keyvalues('kEy1', '3')
    bLock1 = Keyvalues('bLock1', [Keyvalues('leaf', '45')])
    root = Keyvalues('Block', [key1, key2, Keyvalues('key3', '45'), bLock1, kEy1])

    root[0] = Keyvalues('kEy1', '2')
    assert root[0] == Keyvalues('kEy1', '2')
    assert root['key1'] == '3'  # It's not at the end.
    root[4] = Keyvalues('KEY1', '4')
    assert root['key1'] == '4'

    with pytest.raises(TypeError):
        root[0] = 'not_a_property'  # type: ignore

    root[1:3] = [Keyvalues('another', '12'), Keyvalues('keys', '12')]
    assert_tree(root, Keyvalues('Block', [
        Keyvalues('kEy1', '2'),
        Keyvalues('another', '12'),
        Keyvalues('keys', '12'),
        bLock1,
        Keyvalues('KEY1', '4'),
    ]))
    # Modifying an existing key assigns to it.
    assert len(root) == 5
    root['aNother'] = 'new_value'
    assert root['another'] == 'new_value'
    assert len(root) == 5

    # A new key is appended.
    root['mIssing'] = 'blah'
    assert root[-1] == Keyvalues('mIssing', 'blah')
    assert len(root) == 6

    # This also works when assigning property objects, with the name cleared.
    prop_1 = Keyvalues('ignored', 'second')
    root['anOther'] = prop_1
    assert root.find_key('another') is prop_1
    assert prop_1.real_name == 'anOther'
    assert len(root) == 6

    # And when appending.
    prop_2 = Keyvalues('new_prop', [Keyvalues('a', '1'), Keyvalues('b', '2')])
    root['added'] = prop_2
    assert root.find_key('added') is prop_2
    assert root[-1] is prop_2
    assert len(root) == 7
