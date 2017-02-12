import pytest
from srctools.property_parser import Property, KeyValError, NoKeyError, PROP_FLAGS


def assert_tree(first, second):
    """Check that two property trees match exactly (including case)."""
    assert first.name == second.name
    assert first.real_name == second.real_name
    assert first.has_children() == second.has_children()
    if first.has_children():
        for child1, child2 in zip(first, second):
            assert_tree(child1, child2)
    else:
        assert first.value == second.value


def test_constructor():
    """Test the constructor for Property objects."""
    Property(None, [])
    Property('Test', 'value with spaces and ""')
    block = Property('Test_block', [
        Property('Test', 'value\0'),
        Property('Test', [
            Property('leaf', 'data'),
        ]),
        Property('Test2', 'other'),
        Property('Block', []),
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


def test_names():
    """Test the behaviour of Property.name."""
    prop = Property('Test1', 'value')
    
    # Property.name casefolds the argument.
    assert prop.name == 'test1'
    assert prop.real_name == 'Test1'
    
    # Editing name modifies both values
    prop.name = 'SECOND_test'
    assert prop.name == 'second_test'
    assert prop.real_name == 'SECOND_test'
    
    # It can also be set to None.
    prop.name = None
    assert prop.name is prop.real_name is None
    
parse_test = '''

// """"" should be ignored

"Root1"
    {
        

    "Key" "Value"
        "Extra"        "Spaces"
    // "Commented" "out"
    "Block"
        {
        "Empty"
             {
             }
        }
    "Block" // "with value"
  {
 bare
    {
            "block" "he\\tre"
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
    }
    "CommentChecks"
        {
        "after " "value" //comment [ ] ""
        "FlagBlocks" "This" [test_disabled]
        "Flag" "allowed" [!test_disabled]
        "FlagAllows" "This" [test_enabled]
        "Flag" "blocksthis" [!test_enabled]

        }
'''


def test_parse():
    """Test parsing strings."""
    P = Property
    
    expected = P(None, [
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
        ]),
        P('CommentChecks', [
            P('after ', 'value'),
            P('Flag', 'allowed'),
            P('FlagAllows', 'This'),
        ])
    ])

    # Check active and inactive flags are correctly treated.
    PROP_FLAGS['test_enabled'] = True
    PROP_FLAGS['test_disabled'] = False

    # iter() ensures sequence methods aren't used anywhere.
    result = Property.parse(iter(parse_test.splitlines()))
    assert_tree(result, expected)

    del PROP_FLAGS['test_enabled']
    del PROP_FLAGS['test_disabled']


def test_parse_fails():
    """Test various forms of invalid syntax to ensure they indeed fail."""
    def t(text):
        """Test a string to ensure it fails parsing."""
        try:
            result = Property.parse(text)
        except KeyValError:
            pass
        else:
            pytest.fail("Successfully parsed bad text ({!r}) to {!r}".format(
                text,
                result,
            ))
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


def test_edit():
    """Check functionality of Property.edit()"""
    test_prop = Property('Name', 'Value')

    def check(prop: Property, name, value):
        """Check the property was edited, and has the given value."""
        nonlocal test_prop
        assert prop is test_prop
        assert prop.real_name == name
        assert prop.value == value
        test_prop = Property('Name', 'Value')

    check(test_prop.edit(), 'Name', 'Value')
    check(test_prop.edit(name='new_name',), 'new_name', 'Value')
    check(test_prop.edit(value='new_value'), 'Name', 'new_value')

    # Check converting a block into a keyvalue
    test_prop = Property('Name', [
        Property('Name', 'Value')
    ])
    check(test_prop.edit(value='Blah'), 'Name', 'Blah')

    # Check converting a keyvalue into a block.
    child_1 = Property('Key', 'Value')
    new_prop = test_prop.edit(value=[child_1, Property('Key2', 'Value')])
    assert test_prop is new_prop
    assert list(test_prop)[0] is child_1


def test_bool():
    """Check bool(Property)."""
    assert bool(Property('Name', '')) is False
    assert bool(Property('Name', 'value')) is True
    assert bool(Property('Name', [])) is False
    assert bool(Property('Name', [
        Property('Key', 'Value')
    ])) is True
