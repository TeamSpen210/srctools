"""Test the BSP parser's functionality."""
import pytest

from srctools.bsp import _find_or_insert, _find_or_extend


@pytest.mark.parametrize('original, item, index', [
    ([1, 2, 3, 4], 3, 2),
    ([1, 2, 4, 38], 12, 4),
    ([3, 4, 5, 4], 4, 3),
])
def test_find_or_insert(original: list, item, index: int) -> None:
    """Test the find-or-insert helper function correctly inserts values."""
    array = original.copy()
    finder = _find_or_insert(array, lambda x: -x)
    assert finder(item) == index
    assert finder(item) == index  # Doesn't repeat.
    assert array[index] == item  # And put it in that spot.
    # But the original is the same.
    assert array[:len(original)] == original


@pytest.mark.parametrize('original, subset, start', [
    (['a', 'b', 'c', 'd', 'e'], ['c', 'd'], 2),
    (['a', 'b', 'c'], ['j', 'k'], 3),
    (['a', 'c', 'e', 'f', 'g', 'c', 'e', 'a', 'k'], ['c', 'e', 'a'], 5),
])
def test_find_or_extend(original: list, subset: list, start: int) -> None:
    """Test the find-or-extend helper function correctly inserts a subset."""
    array = original.copy()
    finder = _find_or_extend(array, str.swapcase)

    assert finder(subset) == start
    assert finder(subset) == start  # Doesn't repeat.
    assert array[start: start + len(subset)] == subset  # And put it in that spot.
    # But the original is the same.
    assert array[:len(original)] == original


def test_find_or_insert_repeat() -> None:
    """Test repeatedly inserting is still valid."""
    lst = [4, 5, 8]
    finder = _find_or_insert(lst, lambda x: x**2)

    assert finder(5) == 1
    assert finder(12) == 3
    assert lst == [4, 5, 8, 12]
    assert finder(38) == 4
    assert lst == [4, 5, 8, 12, 38]
    assert finder(8) == 2
    assert finder(200) == 5
    assert lst == [4, 5, 8, 12, 38, 200]


def test_find_or_extend_repeat() -> None:
    """asset repeatedly extending is still valid."""
    lst = [4, 8, 20, 12, -50, 3, 4, 12]
    finder = _find_or_extend(lst, lambda x: x-4)
    assert finder([20, 12]) == 2
    assert finder([12, -50, 3]) == 3
    assert finder([4, 12]) == 6

    assert finder([14, 12]) == 8
    assert finder([14, 12]) == 8
    assert lst == [4, 8, 20, 12, -50, 3, 4, 12, 14, 12]
    assert finder([3, 4]) == 5
    assert finder([12, 14, 12]) == 7
    assert finder([1, 2, 3, 4, 8, 63]) == 10
    assert finder([3, 4, 8]) == 12

    assert finder([-50, 3, 4, 12, 14]) == 4
    assert finder([1, 2]) == 10
    assert finder([4, 8]) == 0
    assert finder([20, 12]) == 2
    assert lst == [4, 8, 20, 12, -50, 3, 4, 12, 14, 12, 1, 2, 3, 4, 8, 63]
    assert finder([14, 12, 2]) == 16
