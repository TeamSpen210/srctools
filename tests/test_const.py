"""Test the consts module, specifically add_unknown.

Everything else is just enum data.
"""

from srctools.const import add_unknown


def test_add_unknown_32() -> None:
    ns = {
        'a': 0x1,
        'b': 0x4,
        'c': 0x10,
        'non_member': [1, 2, 3],
        # Python adds this, we need to ignore dunders.
        '__firstlineno__': 0x2938,
    }
    orig = ns.copy()
    add_unknown(ns, False)
    expect = {
        str(i): 1 << i
        for i in range(1, 32)
        if i not in [0, 2, 4]  # These are already defined.
    }
    assert ns == (orig | expect)


def test_add_unknown_64() -> None:
    ns = {
        'a': 0x1,
        'b': 0x4,
        'c': 0x48,
        'non_member': [1, 2, 3],
        # Python adds this, we need to ignore dunders.
        '__firstlineno__': 0x2938,
    }
    orig = ns.copy()
    add_unknown(ns, True)
    expect = {
        str(i): 1 << i
        for i in range(1, 64)
        if i not in [0, 2, 3, 6]  # These are already defined.
    }
    assert ns == (orig | expect)
