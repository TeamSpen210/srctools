"""Test functionality in srctools.__init__."""
import unittest
import operator

from srctools import EmptyMapping
import srctools


class FalseObject:
    def __bool__(self):
        return False


class TrueObject:
    def __bool__(self):
        return True

true_vals = [1, 1.0, True, 'test', [2], (-1, ), TrueObject(), object()]
false_vals = [0, 0.0, False, '', [], (), FalseObject()]

ints = [
    ('0', 0),
    ('-0', -0),
    ('1', 1),
    ('12352343783', 12352343783),
    ('24', 24),
    ('-4784', -4784),
    ('28', 28),
    (1, 1),
    (-2, -2),
    (3783738378, 3783738378),
    (-23527, -23527),
]

floats = [
    ('0.0', 0.0),
    ('-0.0', -0.0),
    ('-4.5', -4.5),
    ('4.5', 4.5),
    ('1.2', 1.2),
    ('12352343783.189', 12352343783.189),
    ('24.278', 24.278),
    ('-4784.214', -4784.214),
    ('28.32', 28.32),
    (1.35, 1.35),
    (-2.26767, -2.26767),
    (338378.3246, 338378.234),
    (-23527.9573, -23527.9573),
]

false_strings = ['0', 'false', 'no', 'faLse', 'False', 'No', 'NO', 'nO']
true_strings = ['1', 'true', 'yes', 'True', 'trUe', 'Yes', 'yEs', 'yeS']

non_ints = ['-23894.0', '', 'hello', '5j', '6.2', '0.2', '6.9', None, object()]
non_floats = ['5j', '', 'hello', '6.2.5', '4F', '100-', None, object(), float]

def_vals = [
    1, 0, True, False, None, object(),
    TrueObject(), FalseObject(), 456.9,
    -4758.97
]


class TestConvFunc(unittest.TestCase):
    def test_bool_as_int(self):
        for val in true_vals:
            self.assertEqual(srctools.bool_as_int(val), '1', repr(val))
        for val in false_vals:
            self.assertEqual(srctools.bool_as_int(val), '0', repr(val))

    def test_conv_int(self):
        for string, result in ints:
            self.assertEqual(srctools.conv_int(string), result)

    def test_conv_int_fails_on_float(self):
        # Check that float values fail
        marker = object()
        for string, result in floats:
            if isinstance(string, str):  # We don't want to check float-rounding
                self.assertIs(
                    srctools.conv_int(string, marker),
                    marker,
                    msg=string,
                )

    def test_conv_int_fails_on_str(self):
        for string in non_ints:
            self.assertEqual(srctools.conv_int(string), 0)
            for default in def_vals:
                # Check all default values pass through unchanged
                self.assertIs(srctools.conv_int(string, default), default)

    def test_conv_bool(self):
        for val in true_strings:
            self.assertTrue(srctools.conv_bool(val))
        for val in false_strings:
            self.assertFalse(srctools.conv_bool(val))

        # Check that bools pass through
        self.assertTrue(srctools.conv_bool(True))
        self.assertFalse(srctools.conv_bool(False))

        # None passes through the default
        for val in def_vals:
            self.assertIs(srctools.conv_bool(None, val), val)

    def test_conv_float(self):
        # Float should convert integers too
        for string, result in ints:
            self.assertEqual(srctools.conv_float(string), float(result))
            self.assertEqual(srctools.conv_float(string), result)

    def test_conv_float_fails_on_str(self):
        for string in non_floats:
            self.assertEqual(srctools.conv_float(string), 0)
            for default in def_vals:
                # Check all default values pass through unchanged
                self.assertIs(srctools.conv_float(string, default), default)


class TestEmptyMapping(unittest.TestCase):
    """Test the EmptyMapping singleton."""

    def test_methods(self):
        # It should be possible to 'construct' an instance..
        self.assertIs(EmptyMapping(), EmptyMapping)

        # Must be passable to dict()
        self.assertEqual(dict(EmptyMapping), {})

        # EmptyMapping['x'] raises
        self.assertRaises(KeyError, operator.getitem, EmptyMapping, 'x')
        self.assertRaises(KeyError, operator.delitem, EmptyMapping, 'x')
        EmptyMapping['x'] = 4  # Shouldn't fail
        self.assertNotIn('x', EmptyMapping)

        self.check_empty_iterable(EmptyMapping, 'EmptyMapping')
        self.check_empty_iterable(EmptyMapping.keys(), 'keys()')
        self.check_empty_iterable(EmptyMapping.values(), 'values()')
        self.check_empty_iterable(EmptyMapping.items(), 'items()', item=('x', 'y'))

    def test_empty_dict_methods(self):

        marker = object()

        self.assertIs(EmptyMapping.get('x'), None)
        self.assertIs(EmptyMapping.setdefault('x'), None)
        self.assertIs(EmptyMapping.get('x', marker), marker)
        self.assertIs(EmptyMapping.setdefault('x', marker), marker)
        self.assertIs(EmptyMapping.pop('x', marker), marker)
        self.assertRaises(KeyError, EmptyMapping.popitem)
        self.assertRaises(KeyError, EmptyMapping.pop, 'x')
        self.assertFalse(EmptyMapping)
        self.assertEqual(len(EmptyMapping), 0)
        EmptyMapping.update({1: 23, 'test': 34,})
        EmptyMapping.update(other=5, a=1, b=3)
        # Can't give more than one item..
        self.assertRaises(TypeError, lambda: EmptyMapping.update({3: 4}, {1: 2}))

    def test_abc(self):
        from collections import abc
        self.assertIsInstance(EmptyMapping, abc.Container)
        self.assertIsInstance(EmptyMapping, abc.Sized)
        self.assertIsInstance(EmptyMapping, abc.Mapping)
        self.assertIsInstance(EmptyMapping, abc.MutableMapping)

    def check_empty_iterable(self, obj, name, item: object='x'):
        """Check the given object is iterable, and is empty."""
        try:
            iterator = iter(obj)
        except TypeError:
            self.fail(name + ' is not iterable!')
        else:
            self.assertNotIn(item, obj)
            self.assertRaises(StopIteration, next, iterator)
            self.assertRaises(StopIteration, next, iterator)

    def test_quote_escape(self):
        self.assertEqual(
            srctools.escape_quote_split('abcdef'),
            ['abcdef'],
        )
        # No escapes, equivalent to str.split
        self.assertEqual(
            srctools.escape_quote_split('"abcd"ef""  " test'),
            '"abcd"ef""  " test'.split('"'),
        )

        self.assertEqual(
            srctools.escape_quote_split(r'"abcd"ef\""  " test'),
            ['', 'abcd', 'ef"', '  ', ' test'],
        )
        self.assertEqual(
            srctools.escape_quote_split(r'"test\"\"" blah"'),
            ['', 'test""', ' blah', ''],
        )

if __name__ == '__main__':
    unittest.main()

