"""Test the PointsMap mapping."""
import copy
import pickle

import pytest
from srctools.math import FrozenVec, Vec
from srctools.points_map import PointsMap
from collections.abc import KeysView, ValuesView, ItemsView


def test_insertion() -> None:
    points = PointsMap(epsilon=0.1)
    assert len(points) == 0

    points[45, 30, 0] = 'a'
    points[45.4, 30.2, 0.1] = 'b'
    assert points[45, 30.0, 0] == 'a'
    assert points[45.4, 30.2, 0.1] == 'b'

    assert len(points) == 2
    assert sorted(points.keys()) == [Vec(45, 30, 0), Vec(45.4, 30.2, 0.1)]
    assert sorted(points.values()) == ['a', 'b']
    assert sorted(points.items()) == [
        (Vec(45, 30, 0), 'a'),
        (Vec(45.4, 30.2, 0.1), 'b'),
    ]

    assert points[45.01, 30.0, -0.01] == 'a'
    points[45.03, 29.95, 0.05] = 'c'
    assert points[45, 30.0, 0] == 'c'


def test_init() -> None:
    """Test initialisation with values."""
    points = PointsMap()
    assert len(points) == 0
    assert list(points) == list(points.keys()) == []
    assert list(points.values()) == list(points.items()) == []

    points = PointsMap([])
    assert len(points) == 0
    assert list(points) == list(points.keys()) == []
    assert list(points.values()) == list(points.items()) == []

    points = PointsMap([
        (Vec(1, 2, 3), 'a'),
        (Vec(4, 5, 6), 'b'),
    ])
    assert len(points) == 2
    assert points[1, 2, 3] == 'a'
    assert points[4, 5, 6] == 'b'

    points = PointsMap([
        (Vec(1, 2, 3), 'a'),
        (Vec(4, 5, 6), 'b'),
    ])
    assert len(points) == 2
    assert points[1, 2, 3] == 'a'
    assert points[4, 5, 6] == 'b'

    points = PointsMap([(Vec(), 'e')])
    assert len(points) == 1
    assert points[0, 0, 0] == 'e'

    points = PointsMap((
        (Vec(), 'a'),
        (FrozenVec(1, 1, 1), 'b')
    ))
    assert len(points) == 2
    assert points[0, 0, 0], 'a'
    assert points[Vec(1, 1, 1)] == 'b'
    assert points[FrozenVec(1, 1, 1)] == 'b'

    points = PointsMap((
        ((1.0, 1.5, 1.0), 'b'),
    ))
    assert len(points) == 1
    assert points[1.0, 1.5, 1.0] == 'b'

    with pytest.raises(TypeError):
        PointsMap([Vec(), 'a'])  # type: ignore
    with pytest.raises(TypeError):
        PointsMap((Vec(), 'a', 'c'))  # type: ignore
    with pytest.raises(TypeError):
        PointsMap(Vec())  # type: ignore
    with pytest.raises(TypeError):
        PointsMap('ab')  # type: ignore


def test_keys() -> None:
    """Test keys()."""
    points = PointsMap([
        (Vec(1, 2, 3), 'a'),
        (Vec(4, 5, 6), 'b'),
    ], epsilon=0.01)
    assert isinstance(points.keys(), KeysView)
    assert not isinstance(points.keys(), ValuesView)
    assert not isinstance(points.keys(), ItemsView)

    assert len(points.keys()) == 2
    assert Vec(1, 2, 3) in points.keys()
    assert (1.0, 2, 3.001) in points.keys()
    assert FrozenVec(1.0, 2, 3.001) in points.keys()
    assert (39, -20, 12) not in points.keys()
    assert sorted(points.keys()) == [Vec(1, 2, 3), Vec(4, 5, 6)]


def test_values() -> None:
    """Test values()."""
    points = PointsMap([
        (Vec(1, 2, 3), 'a'),
        (Vec(4, 5, 6), 'b'),
    ], epsilon=0.01)
    assert not isinstance(points.values(), KeysView)
    assert isinstance(points.values(), ValuesView)
    assert not isinstance(points.values(), ItemsView)

    assert len(points.values()) == 2
    assert 'a' in points.values()
    assert 'b' in points.values()
    assert Vec() not in points.values()
    assert FrozenVec(1, 2, 3) not in points.values()
    assert 'c' not in points.values()
    assert sorted(points.values()) == ['a', 'b']


def test_items() -> None:
    """Test items()."""
    points = PointsMap([
        (Vec(1, 2, 3), 'a'),
        (FrozenVec(4, 5, 6), 'b'),
    ], epsilon=0.01)
    assert not isinstance(points.items(), KeysView)
    assert not isinstance(points.items(), ValuesView)
    assert isinstance(points.items(), ItemsView)

    assert len(points.items()) == 2
    assert (Vec(1, 2, 3), 'a') in points.items()
    assert ((1, 2, 3.01), 'a') in points.items()
    assert (FrozenVec(1, 2, 3.01), 'a') in points.items()
    assert ((3.991, 5, 6), 'b') in points.items()
    assert ((45, 29,-20), 'c') not in points.items()
    assert sorted(points.items()) == [
        (Vec(1, 2, 3), 'a'),
        (Vec(4, 5, 6), 'b'),
    ]
    assert 'not_a_tuple' not in points.items()


def test_deletion() -> None:
    points = PointsMap([
        (Vec(), 'a'),
        (Vec(1, 1, 1), 'b')
    ])
    del points[1, 1, 1]

    with pytest.raises(KeyError):
        print(points[1, 1, 1])

    with pytest.raises(KeyError):
        del points[-4.0, 2.5, 4.9]


def test_copying() -> None:
    """Test copying pointsmaps."""
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    orig = PointsMap([
        (Vec(1, 0, 0), list1),
        (Vec(2, 0, 0), list2),
    ])
    cp = copy.copy(orig)
    assert len(cp) == 2
    assert cp[1, 0, 0] is list1
    assert cp[2, 0, 0] is list2

    dc = copy.deepcopy(orig)
    assert len(dc) == 2
    assert dc[1, 0, 0] == list1
    assert dc[2, 0, 0] == list2
    assert dc[1, 0, 0] is not list1
    assert dc[2, 0, 0] is not list2

    pick = pickle.dumps(orig)
    cp_pick = pickle.loads(pick)
    assert len(cp_pick) == 2
    assert cp_pick[1, 0, 0] == list1
    assert cp_pick[2, 0, 0] == list2
    assert cp_pick[1, 0, 0] is not list1
    assert cp_pick[2, 0, 0] is not list2
