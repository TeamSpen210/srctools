"""Implements a mapping for positions to values.

This has a minimum distance, where points closer than that are treated equal.
"""
import itertools
import math
from copy import deepcopy
from typing import (
    Union, Tuple, overload, TypeVar, Generic,
    Iterator, Iterable, Mapping,
    MutableMapping, List,
)
from collections.abc import KeysView, ValuesView, ItemsView
from srctools.math import Vec, Vec_tuple


ValueT = TypeVar('ValueT')
AnyVec = Union[Vec, Vec_tuple, Tuple[float, float, float]]


class PointsMap(MutableMapping[Vec, ValueT], Generic[ValueT]):
    """A mutable mapping with vectors as keys.

    This is constructed with an epsilon distance, and lookups succeed if
    the distance is below this value.
    """
    @overload
    def __init__(self, __contents: Union[Mapping[AnyVec, ValueT], Iterable[Tuple[AnyVec, ValueT]]], *, epsilon: float = 1e-6) -> None: ...
    @overload
    def __init__(self, *contents: Tuple[AnyVec, ValueT], epsilon: float = 1e-6) -> None: ...
    def __init__(
        self,
        *contents: Union[
            Tuple[AnyVec, ValueT],
            Mapping[AnyVec, ValueT],
            Iterable[Tuple[AnyVec, ValueT]]
        ],
        epsilon: float = 1e-6,
    ) -> None:
        if not (0.0 < epsilon < 1.0):
            raise ValueError('Epsilon must be between 0 and 1.')
        self._map: dict[tuple[int, int, int], list[tuple[Vec, ValueT]]] = {}
        self._dist_sq = epsilon ** 2
        if len(contents) == 1:
            # Edge case - single tuple parameter, try as a pos-value pair first.
            if isinstance(contents[0], tuple):
                try:
                    [[key, value]] = contents
                    pos = Vec(key)
                except (TypeError, ValueError):
                    pass
                else:
                    self[pos] = value
                    return

            [contents] = contents
        if hasattr(contents, 'items'):
            contents = contents.items()
        for kv in contents:
            if not isinstance(kv, tuple):
                raise TypeError(
                    'PointsMap must be initialised with a mapping, '
                    'pairs of tuples or an iterable of tuple pairs, '
                    f'not {type(kv).__name__}!'
                )
            key, value = kv
            self[key] = value

    def _iter_cells(self, x: int, y: int, z: int) -> Iterator[
        tuple[int, int, int]]:
        """Iterate over the cells this index could match."""
        return itertools.product(
            [x - 1, x, x + 1],
            [y - 1, y, x + 1],
            [z - 1, z, x + 1],
        )

    def __repr__(self) -> str:
        if self._dist_sq != (1e-6 ** 2):
            return f'PointsMap({list(self.items())!r}, epsilon={math.sqrt(self._dist_sq)})'
        else:
            return f'PointsMap({list(self.items())!r})'

    def get_all(self, item: AnyVec) -> Iterator[ValueT]:
        """Find all items matching this position."""
        pos = Vec(item)
        x, y, z = round(pos.x), round(pos.y), round(pos.z)
        for key in self._iter_cells(x, y, z):
            try:
                lst = self._map[key]
            except KeyError:
                continue
            for map_pos, value in lst:
                if (pos - map_pos).mag_sq() < self._dist_sq:
                    yield value

    def __getitem__(self, item: AnyVec) -> ValueT:
        """Find the first item matching this position."""
        try:
            return next(self.get_all(item))
        except StopIteration:
            raise KeyError(item) from None

    def __setitem__(self, item: AnyVec, value: ValueT) -> None:
        """Set the first item matching this position, or add a new item."""
        pos = Vec(item)
        x, y, z = round(pos.x), round(pos.y), round(pos.z)
        for key in self._iter_cells(x, y, z):
            try:
                lst = self._map[key]
            except KeyError:
                continue
            for i, (map_pos, old_value) in enumerate(lst):
                if (pos - map_pos).mag_sq() < self._dist_sq:
                    lst[i] = (map_pos, value)
                    return
        self._map.setdefault((x, y, z), []).append((pos, value))

    def __delitem__(self, item: AnyVec) -> None:
        """Remove the first item matching this position."""
        pos = Vec(item)
        x, y, z = round(pos.x), round(pos.y), round(pos.z)
        for key in self._iter_cells(x, y, z):
            try:
                lst = self._map[key]
            except KeyError:
                continue
            for i, (map_pos, old_value) in enumerate(lst):
                if (pos - map_pos).mag_sq() < self._dist_sq:
                    del pos[i]
                    return
        raise KeyError(pos)

    def __len__(self) -> int:
        """Return the number of points set."""
        return sum(map(len, self._map.values()))

    def __iter__(self) -> Iterator[Vec]:
        for points in self._map.values():
            for pos, value in points:
                yield pos.copy()

    def values(self) -> 'PointValuesView[ValueT]':
        """Return a view over the values of this map."""
        return PointValuesView(self)

    def items(self) -> 'PointItemsView[ValueT]':
        """Iterate over point, value pairs."""
        return PointItemsView(self)

    def clear(self) -> None:
        """Remove all items."""
        self._map.clear()

    def __getstate__(self) -> Tuple[float, List[Tuple[float, float, float, ValueT]]]:
        """Allow pickling of a PointsMap."""
        return self._dist_sq, [
            (pos.x, pos.y, pos.z, value)
            for points in self._map.values()
            for pos, value in points
        ]

    def __setstate__(
        self,
        state: Tuple[float, List[Tuple[float, float, float, ValueT]]],
    ) -> None:
        """Apply the pickled state."""
        self._dist_sq, points = state
        if not isinstance(self._dist_sq, float):
            raise ValueError('Invalid epsilon distance.')
        for x, y, z, value in points:
            self[x, y, z] = value

    def __copy__(self) -> 'PointsMap[ValueT]':
        """Shallow-copy this PointsMap."""
        copy = PointsMap.__new__(PointsMap)
        copy._dist_sq = self._dist_sq
        copy._map = {
            key: [
                (pos.copy(), value)
                for pos, value in points
            ]
            for key, points in self._map.items()
        }
        return copy

    def __deepcopy__(self, memodict: dict) -> 'PointsMap[ValueT]':
        """Deep-copy this PointsMap."""
        copy = PointsMap.__new__(PointsMap)
        copy._dist_sq = self._dist_sq
        copy._map = {
            key: [
                (pos.copy(), deepcopy(copy, memodict))  # type: ignore  # Incorrect stub.
                for pos, value in points
            ]
            for key, points in self._map.items()
        }
        return copy


# noinspection PyProtectedMember
class PointValuesView(ValuesView[ValueT], Generic[ValueT]):
    """A view over the values in a PointsMap."""
    _mapping: PointsMap  # Superclass initialises.

    def __contains__(self, item: object) -> bool:
        """Check if this value is present in the PointsMap."""
        for points in self._mapping._map.values():
            for pos, value in points:
                if value is item or value == item:
                    return True
        return False

    def __iter__(self) -> ValueT:
        """Yield all values stored in the PointsMap."""
        for points in self._mapping._map.values():
            for pos, value in points:
                yield value


# noinspection PyProtectedMember
class PointItemsView(ItemsView[Vec, ValueT], Generic[ValueT]):
    """A view over the points and values in a PointsMap."""
    _mapping: PointsMap  # Superclass initialises.

    def __contains__(self, item: object) -> bool:
        """Check if this point and value is present in the PointsMap."""
        try:
            search_pos, search_value = item
            pos = Vec(search_pos)
        except (TypeError, ValueError):  # Can never be present.
            return False
        x, y, z = round(pos.x), round(pos.y), round(pos.z)
        for key in self._mapping._iter_cells(x, y, z):
            try:
                lst = self._mapping._map[key]
            except KeyError:
                continue
            for map_pos, map_value in lst:
                if (pos - map_pos).mag_sq() < self._mapping._dist_sq:
                    if map_value is search_value or map_value == search_value:
                        return True
                    # Else, continue, in case there's another matching point.
        return False

    def __iter__(self) -> ValueT:
        """Yield all values stored in the PointsMap."""
        for points in self._mapping._map.values():
            for pos, value in points:
                yield (pos.copy(), value)
