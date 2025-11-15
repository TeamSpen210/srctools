"""Implements a mapping for positions to values.

This has a minimum distance, where points closer than that are treated equal.
"""
from typing import Any, Union, TypeVar, Generic, cast
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, ValuesView, ItemsView
from copy import deepcopy
import itertools
import math

from srctools.math import AnyVec, FrozenVec, Vec


__all__ = ['PointsMap']
ValueT = TypeVar('ValueT')
_State = tuple[float, list[tuple[float, float, float, ValueT]]]


def _cell_neighbours(x: int, y: int, z: int) -> Iterator[tuple[int, int, int]]:
    """Iterate over the cells this index could match."""
    # Reorder such that (x, y, z) is the first result. Lookups are fairly likely
    # to be using the exact position, so ensure that is checked first.
    return itertools.product(
        (x, x - 1, x + 1),
        (y, y - 1, y + 1),
        (z, z - 1, z + 1),
    )


class PointsMap(MutableMapping[AnyVec, ValueT], Generic[ValueT]):
    """A :class:`~collections.abc.MutableMapping` with vectors as keys.

    This is constructed with an epsilon distance, and lookups succeed if
    the distance is below this value.

    When keys are set, any existing key is assigned to. However it is still possible to set
    keys such that a lookup of a particular location has multiple candidates - in this case,
    an arbitary value is returned. Use `get_all()` if you need all matches.

    Acess can be done with regular `Vecs<Vec>`, `FrozenVecs<FrozenVec>` and 3-tuples. In particular,
    ``pointmap[1, 2, 3]`` will lookup correctly. Iteration always produces a new `Vec` each time.

    .. warning::
        This stores keys partitioned by their integral position, then checks each position plus
        neighbours during lookups. If a large number of points are less than 1 away, performance
        will degrade into many linear comparisons.
    """
    _map: dict[tuple[int, int, int], list[tuple[FrozenVec, ValueT]]]
    _dist_sq: float
    def __init__(
        self,
        contents: Union[Mapping[AnyVec, ValueT], Iterable[tuple[AnyVec, ValueT]]] = (),
        epsilon: float = 1e-6,
    ) -> None:
        """Create a PointsMap.

        :param contents: If provided, used to set the initial contents. Note that it is not any
          more efficient to construct from an existing mapping.
        :param epsilon: If two positions are smaller than this distance, they will be treated as
          the same. Must be between 0 and 1
        """
        if not (0.0 < epsilon < 1.0):
            raise ValueError('Epsilon must be between 0 and 1.')
        self._map = {}
        self._dist_sq = epsilon ** 2
        cont_iter: Iterable[tuple[AnyVec, ValueT]]
        if isinstance(contents, Mapping):
            # Typing hole: contents could be a `Mapping[tuple[AnyVec, ValueT], ?]`, which is a
            # valid Iterable[...], but means key is a tuple now. Occurs with all mappings. Cast
            # away that invalid key type. If it happens in practice, the FrozenVec() conversion
            # will catch it.
            cont_iter = cast('Mapping[AnyVec, ValueT]', contents).items()
        else:
            cont_iter = contents
        for kv in cont_iter:
            if not isinstance(kv, tuple):
                raise TypeError(
                    'PointsMap must be initialised with a mapping, '
                    'pairs of tuples or an iterable of tuple pairs, '
                    f'not {type(kv).__name__}!'
                )
            key, value = kv
            self[key] = value

    def __repr__(self) -> str:
        if self._dist_sq != (1e-6 ** 2):
            return f'PointsMap({list(self.items())!r}, epsilon={math.sqrt(self._dist_sq)})'
        else:
            return f'PointsMap({list(self.items())!r})'

    def get_all(self, item: AnyVec, /) -> Iterator[ValueT]:
        """Iterate over all items matching this position."""
        pos = FrozenVec(item)
        x, y, z = round(pos.x), round(pos.y), round(pos.z)
        for key in _cell_neighbours(x, y, z):
            try:
                lst = self._map[key]
            except KeyError:
                continue
            for map_pos, value in lst:
                if (pos - map_pos).mag_sq() < self._dist_sq:
                    yield value

    def __getitem__(self, item: AnyVec, /) -> ValueT:
        """Find the first item matching this position."""
        try:
            return next(self.get_all(item))
        except StopIteration:
            raise KeyError(item) from None

    def __setitem__(self, item: AnyVec, value: ValueT, /) -> None:
        """Set the first item matching this position, or add a new item."""
        pos = FrozenVec(item)
        x, y, z = round(pos.x), round(pos.y), round(pos.z)
        for key in _cell_neighbours(x, y, z):
            try:
                lst = self._map[key]
            except KeyError:
                continue
            for i, (map_pos, old_value) in enumerate(lst):
                if (pos - map_pos).mag_sq() < self._dist_sq:
                    lst[i] = (map_pos, value)
                    return
        self._map.setdefault((x, y, z), []).append((pos, value))

    def __delitem__(self, item: AnyVec, /) -> None:
        """Remove the first item matching this position."""
        pos = FrozenVec(item)
        x, y, z = round(pos.x), round(pos.y), round(pos.z)
        for key in _cell_neighbours(x, y, z):
            try:
                lst = self._map[key]
            except KeyError:
                continue
            for i, (map_pos, old_value) in enumerate(lst):
                if (pos - map_pos).mag_sq() < self._dist_sq:
                    del lst[i]
                    return
        raise KeyError(pos)

    def __len__(self) -> int:
        """Return the number of points set."""
        return sum(map(len, self._map.values()))

    def __iter__(self) -> Iterator[Vec]:
        """Yield all points in the map."""
        for points in self._map.values():
            for pos, value in points:
                yield pos.thaw()

    def values(self) -> 'PointValuesView[ValueT]':
        """Return a view over the values of this map."""
        return PointValuesView(self)

    def items(self) -> 'PointItemsView[ValueT]':
        """Iterate over point, value pairs."""
        return PointItemsView(self)

    def clear(self) -> None:
        """Remove all items."""
        self._map.clear()

    def __getstate__(self) -> _State[ValueT]:
        """Allow pickling of a PointsMap."""
        return self._dist_sq, [
            (pos.x, pos.y, pos.z, value)
            for points in self._map.values()
            for pos, value in points
        ]

    def __setstate__(self, state: _State[ValueT], /) -> None:
        """Apply the pickled state."""
        self._dist_sq, points = state
        self._map = {}  # Pickle skips __init__!
        if not isinstance(self._dist_sq, float):
            raise ValueError('Invalid epsilon distance.')
        for x, y, z, value in points:
            self[x, y, z] = value

    def __copy__(self) -> 'PointsMap[ValueT]':
        """Shallow-copy this PointsMap."""
        copy = PointsMap.__new__(PointsMap)
        copy._dist_sq = self._dist_sq
        copy._map = {
            key: points.copy()
            for key, points in self._map.items()
        }
        return copy

    def __deepcopy__(self, memodict: dict[int, Any]) -> 'PointsMap[ValueT]':
        """Deep-copy this PointsMap."""
        copy = PointsMap.__new__(PointsMap)
        copy._dist_sq = self._dist_sq
        copy._map = {
            key: [
                (pos, deepcopy(value, memodict))
                for pos, value in points
            ]
            for key, points in self._map.items()
        }
        return copy


# noinspection PyProtectedMember
class PointValuesView(ValuesView[ValueT], Generic[ValueT]):
    """A view over the values in a PointsMap."""
    _mapping: PointsMap[ValueT]  # Superclass initialises.

    def __contains__(self, item: object) -> bool:
        """Check if this value is present in the PointsMap."""
        for points in self._mapping._map.values():
            for pos, value in points:
                if value is item or value == item:
                    return True
        return False

    def __iter__(self) -> Iterator[ValueT]:
        """Yield all values stored in the PointsMap."""
        for points in self._mapping._map.values():
            for pos, value in points:
                yield value


# noinspection PyProtectedMember
class PointItemsView(ItemsView[AnyVec, ValueT], Generic[ValueT]):
    """A view over the points and values in a PointsMap."""
    _mapping: PointsMap[ValueT]  # Superclass initialises.

    def __contains__(self, item: object) -> bool:
        """Check if this point and value is present in the PointsMap."""
        search_pos: AnyVec
        search_value: object
        try:
            search_pos, search_value = item  # type: ignore
            pos = FrozenVec(search_pos)
        except (TypeError, ValueError):  # Can never be present.
            return False
        x, y, z = round(pos.x), round(pos.y), round(pos.z)
        for key in _cell_neighbours(x, y, z):
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

    def __iter__(self) -> Iterator[tuple[Vec, ValueT]]:
        """Yield all values stored in the PointsMap."""
        for points in self._mapping._map.values():
            for pos, value in points:
                yield (pos.thaw(), value)
