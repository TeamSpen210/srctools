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
            "block" "here"
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
                    P('block', 'here'),
                ]),
            ]),
        ]),
        P('Root2', [
            P('Name with " in it', 'Value with \" inside'),
            P('multiline',
              'text\n\tcan continue\nfor many "lines" of\n  possibly indented\n\ntext'
              ),
        ]),
    ])
    
    # iter() ensures sequence methods aren't used anywhere.
    result = Property.parse(iter(parse_test.splitlines()))
    assert_tree(result, expected)


def test_parse_fails():
    def t(text):
        """Test a string to ensure it fails parsing."""
        try:
            result = Property.parse(text.splitlines())
        except KeyValError:
            pass
        else:
            raise AssertionError("Successfully parsed bad text {!r}".format(
                result
            ))
    
    t('''\
regular text. with sentences.
    ''')
    t('''\
"Ok block"
    {
    "missing" //value
    }
''')

    t('''\
"block1"
"no_block" 
''')

    t('''
bbhf  "text before"
''')
    t('''
  "text" bl "between"
''')

    t('''
    "text" "with extra" "
''')

    t('''
    "multi" "line
text with
  multiple
  quotes" "
''')
