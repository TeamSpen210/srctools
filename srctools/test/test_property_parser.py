from nose.tools import *  # assert_equal, assert_is, etc

import srctools

from srctools.property_parser import Property, KeyValError, NoKeyError, PROP_FLAGS


def assert_tree(first, second):
    """Check that two property trees match exactly (including case)."""
    assert_equal(first.name, second.name)
    assert_equal(first.real_name, second.real_name)
    assert_equal(first.has_children(), second.has_children())
    if first.has_children():
        for child1, child2 in zip(first, second):
            assert_tree(child1, child2)
    else:
        if '"' in first.value:
            print(repr(first), '\n', repr(second))
        assert_equal(first.value, second.value)


def test_constructor():
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
    assert_equal(block.real_name, 'Test_block')
    children = list(block)
    assert_equal(children[0].real_name, 'Test')
    assert_equal(children[1].real_name, 'Test')
    assert_equal(children[2].real_name, 'Test2')
    assert_equal(children[3].real_name, 'Block')

    assert_equal(children[0].value, 'value\0')
    assert_equal(children[2].value, 'other')
    assert_equal(list(children[3].value), [])

    sub_children = list(children[1])
    assert_equal(sub_children[0].real_name, 'leaf')
    assert_equal(sub_children[0].value, 'data'),
    assert_equal(len(sub_children), 1)


def test_names():
    prop = Property('Test1', 'value')
    
    # Property.name casefolds the argument.
    assert_equal(prop.name, 'test1')
    assert_equal(prop.real_name, 'Test1')
    
    # Editing name modifies both values
    prop.name = 'SECOND_test'
    assert_equal(prop.name, 'second_test')
    assert_equal(prop.real_name, 'SECOND_test')
    
    # It can also be set to None.
    prop.name = None
    assert_is(prop.name, None)
    assert_is(prop.real_name, None)
    
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
    def t(line_num, text):
        """Test a string to ensure it fails parsing."""
        try:
            result = Property.parse(text.splitlines())
        except KeyValError as e:
            assert_equal(e.line_num, line_num)
        else:
            raise AssertionError("Successfully parsed bad text {!r}".format(
                result
            ))
    # Bare text at end of file
    t(None, '''\
regular text. with sentences.
    ''')
    # Bare text in the middle
    t(2, '''\
regular text. with sentences.
    "blah" "value"
    ''')
    t(4, '''\
"Ok block"
    {
    "missing" //value
    }
''')

    # Test block without a block
    t(2, '''\
"block1"
"no_block" 
''')

    # Test characters before a keyvalue
    t(1, '''\
bbhf  "text before"
    "key" "value
''')
    t(2, '''
  "text" bl "between"
    "key" "value
''')
    # Test text after the keyvalue
    t(1, '''\
    "text" "value" blah
    "key" "value
    ''')
    # Test quotes after the keyvalue
    t(2, '''
    "text" "with extra" "
''')

    t(5, '''
    "multi" "line
text with
  multiple
  quotes" "
''')

    # Test a flag without ] at end
    t(2, '''
    "Name" "value" [flag
    ''')

    # Test a flag with values after the bracket.
    t(2, '''
    "Name" "value" [flag ] hi
    ''')

    # Test too many closing brackets
    t(9, '''
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
    t(3, '''
    "Block" "value"
        {
        "Name" "value"
        }
    ''')

    # Test too many open brackets
    t(6, '''\
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
