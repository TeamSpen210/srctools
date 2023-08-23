from srctools.vmf import Side


def test_side_setters(side: Side) -> None:
    """Side has some properties that are setters only."""
    afloat: float = 12.0
    side.offset = 12
    side.offset = 3.14
    side.offset = 12
    side.scale = 3.14
    afloat = side.offset  # type: ignore
    afloat = side.scale  # type: ignore
