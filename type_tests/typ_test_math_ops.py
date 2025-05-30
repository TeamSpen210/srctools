"""Verify type definitions for math classes."""
from typing_extensions import assert_type

from srctools.math import Angle, FrozenAngle, FrozenMatrix, FrozenVec, Matrix, Vec
# + is not concatenation here, don't suggest (*Vec, x, y, z)
# ruff: noqa: RUF005


def test_addition() -> None:
    """Check addition operation types."""
    assert_type(Vec() + Vec(), Vec)
    assert_type(Vec() + FrozenVec(), Vec)
    Vec() + Angle()         # type: ignore
    Vec() + FrozenAngle()   # type: ignore
    Vec() + Matrix()        # type: ignore
    Vec() + FrozenMatrix()  # type: ignore
    assert_type(Vec() + (1.0, 2.0, 3.0), Vec)
    assert_type(Vec() + 5, Vec)
    assert_type(Vec() + 5.0, Vec)

    assert_type(FrozenVec() + Vec(), FrozenVec)
    assert_type(FrozenVec() + FrozenVec(), FrozenVec)
    FrozenVec() + Angle()         # type: ignore
    FrozenVec() + FrozenAngle()   # type: ignore
    FrozenVec() + Matrix()        # type: ignore
    FrozenVec() + FrozenMatrix()  # type: ignore
    assert_type(FrozenVec() + (1.0, 2.0, 3.0), FrozenVec)
    assert_type(FrozenVec() + 5, FrozenVec)
    assert_type(FrozenVec() + 5.0, FrozenVec)

    Angle() + Vec()            # type: ignore
    Angle() + FrozenVec()      # type: ignore
    Angle() + Angle()          # type: ignore
    Angle() + FrozenAngle()    # type: ignore
    Angle() + Matrix()         # type: ignore
    Angle() + FrozenMatrix()   # type: ignore
    Angle() + (1.0, 2.0, 3.0)  # type: ignore
    Angle() + 5                # type: ignore
    Angle() + 5.0              # type: ignore

    FrozenAngle() + Vec()            # type: ignore
    FrozenAngle() + FrozenVec()      # type: ignore
    FrozenAngle() + Angle()          # type: ignore
    FrozenAngle() + FrozenAngle()    # type: ignore
    FrozenAngle() + Matrix()         # type: ignore
    FrozenAngle() + FrozenMatrix()   # type: ignore
    FrozenAngle() + (1.0, 2.0, 3.0)  # type: ignore
    FrozenAngle() + 5                # type: ignore
    FrozenAngle() + 5.0              # type: ignore

    Matrix() + Vec()            # type: ignore
    Matrix() + FrozenVec()      # type: ignore
    Matrix() + Angle()          # type: ignore
    Matrix() + FrozenAngle()    # type: ignore
    Matrix() + Matrix()         # type: ignore
    Matrix() + FrozenMatrix()   # type: ignore
    Matrix() + (1.0, 2.0, 3.0)  # type: ignore
    Matrix() + 5                # type: ignore
    Matrix() + 5.0              # type: ignore

    FrozenMatrix() + Vec()            # type: ignore
    FrozenMatrix() + FrozenVec()      # type: ignore
    FrozenMatrix() + Angle()          # type: ignore
    FrozenMatrix() + FrozenAngle()    # type: ignore
    FrozenMatrix() + Matrix()         # type: ignore
    FrozenMatrix() + FrozenMatrix()   # type: ignore
    FrozenMatrix() + (1.0, 2.0, 3.0)  # type: ignore
    FrozenMatrix() + 5                # type: ignore
    FrozenMatrix() + 5.0              # type: ignore

    assert_type((1.0, 2.0, 3.0) + Vec(), Vec)
    assert_type((1.0, 2.0, 3.0) + FrozenVec(), FrozenVec)
    (1.0, 2.0, 3.0) + Angle()         # type: ignore
    (1.0, 2.0, 3.0) + FrozenAngle()   # type: ignore
    (1.0, 2.0, 3.0) + Matrix()        # type: ignore
    (1.0, 2.0, 3.0) + FrozenMatrix()  # type: ignore

    assert_type(5 + Vec(), Vec)
    assert_type(5 + FrozenVec(), FrozenVec)
    5 + Angle()         # type: ignore
    5 + FrozenAngle()   # type: ignore
    5 + Matrix()        # type: ignore
    5 + FrozenMatrix()  # type: ignore

    assert_type(5.0 + Vec(), Vec)
    assert_type(5.0 + FrozenVec(), FrozenVec)
    5.0 + Angle()         # type: ignore
    5.0 + FrozenAngle()   # type: ignore
    5.0 + Matrix()        # type: ignore
    5.0 + FrozenMatrix()  # type: ignore


def test_subtraction() -> None:
    """Check subtraction operation types."""
    assert_type(Vec() - Vec(), Vec)
    assert_type(Vec() - FrozenVec(), Vec)
    Vec() - Angle()         # type: ignore
    Vec() - FrozenAngle()   # type: ignore
    Vec() - Matrix()        # type: ignore
    Vec() - FrozenMatrix()  # type: ignore
    assert_type(Vec() - (1.0, 2.0, 3.0), Vec)
    assert_type(Vec() - 5, Vec)
    assert_type(Vec() - 5.0, Vec)

    assert_type(FrozenVec() - Vec(), FrozenVec)
    assert_type(FrozenVec() - FrozenVec(), FrozenVec)
    FrozenVec() - Angle()         # type: ignore
    FrozenVec() - FrozenAngle()   # type: ignore
    FrozenVec() - Matrix()        # type: ignore
    FrozenVec() - FrozenMatrix()  # type: ignore
    assert_type(FrozenVec() - (1.0, 2.0, 3.0), FrozenVec)
    assert_type(FrozenVec() - 5, FrozenVec)
    assert_type(FrozenVec() - 5.0, FrozenVec)

    Angle() - Vec()            # type: ignore
    Angle() - FrozenVec()      # type: ignore
    Angle() - Angle()          # type: ignore
    Angle() - FrozenAngle()    # type: ignore
    Angle() - Matrix()         # type: ignore
    Angle() - FrozenMatrix()   # type: ignore
    Angle() - (1.0, 2.0, 3.0)  # type: ignore
    Angle() - 5                # type: ignore
    Angle() - 5.0              # type: ignore

    FrozenAngle() - Vec()            # type: ignore
    FrozenAngle() - FrozenVec()      # type: ignore
    FrozenAngle() - Angle()          # type: ignore
    FrozenAngle() - FrozenAngle()    # type: ignore
    FrozenAngle() - Matrix()         # type: ignore
    FrozenAngle() - FrozenMatrix()   # type: ignore
    FrozenAngle() - (1.0, 2.0, 3.0)  # type: ignore
    FrozenAngle() - 5                # type: ignore
    FrozenAngle() - 5.0              # type: ignore

    Matrix() - Vec()            # type: ignore
    Matrix() - FrozenVec()      # type: ignore
    Matrix() - Angle()          # type: ignore
    Matrix() - FrozenAngle()    # type: ignore
    Matrix() - Matrix()         # type: ignore
    Matrix() - FrozenMatrix()   # type: ignore
    Matrix() - (1.0, 2.0, 3.0)  # type: ignore
    Matrix() - 5                # type: ignore
    Matrix() - 5.0              # type: ignore

    FrozenMatrix() - Vec()            # type: ignore
    FrozenMatrix() - FrozenVec()      # type: ignore
    FrozenMatrix() - Angle()          # type: ignore
    FrozenMatrix() - FrozenAngle()    # type: ignore
    FrozenMatrix() - Matrix()         # type: ignore
    FrozenMatrix() - FrozenMatrix()   # type: ignore
    FrozenMatrix() - (1.0, 2.0, 3.0)  # type: ignore
    FrozenMatrix() - 5                # type: ignore
    FrozenMatrix() - 5.0              # type: ignore

    assert_type((1.0, 2.0, 3.0) - Vec(), Vec)
    assert_type((1.0, 2.0, 3.0) - FrozenVec(), FrozenVec)
    (1.0, 2.0, 3.0) - Angle()         # type: ignore
    (1.0, 2.0, 3.0) - FrozenAngle()   # type: ignore
    (1.0, 2.0, 3.0) - Matrix()        # type: ignore
    (1.0, 2.0, 3.0) - FrozenMatrix()  # type: ignore

    assert_type(5 - Vec(), Vec)
    assert_type(5 - FrozenVec(), FrozenVec)
    5 - Angle()         # type: ignore
    5 - FrozenAngle()   # type: ignore
    5 - Matrix()        # type: ignore
    5 - FrozenMatrix()  # type: ignore

    assert_type(5.0 - Vec(), Vec)
    assert_type(5.0 - FrozenVec(), FrozenVec)
    5.0 - Angle()         # type: ignore
    5.0 - FrozenAngle()   # type: ignore
    5.0 - Matrix()        # type: ignore
    5.0 - FrozenMatrix()  # type: ignore


def test_multiplication() -> None:
    """Test multiplication operation types."""
    Vec() * Vec()            # type: ignore
    Vec() * FrozenVec()      # type: ignore
    Vec() * Angle()          # type: ignore
    Vec() * FrozenAngle()    # type: ignore
    Vec() * Matrix()         # type: ignore
    Vec() * FrozenMatrix()   # type: ignore
    Vec() * (1.0, 2.0, 3.0)  # type: ignore
    assert_type(Vec() * 5, Vec)
    assert_type(Vec() * 5.0, Vec)

    FrozenVec() * Vec()            # type: ignore
    FrozenVec() * FrozenVec()      # type: ignore
    FrozenVec() * Angle()          # type: ignore
    FrozenVec() * FrozenAngle()    # type: ignore
    FrozenVec() * Matrix()         # type: ignore
    FrozenVec() * FrozenMatrix()   # type: ignore
    FrozenVec() * (1.0, 2.0, 3.0)  # type: ignore
    assert_type(FrozenVec() * 5, FrozenVec)
    assert_type(FrozenVec() * 5.0, FrozenVec)

    Angle() * Vec()            # type: ignore
    Angle() * FrozenVec()      # type: ignore
    Angle() * Angle()          # type: ignore
    Angle() * FrozenAngle()    # type: ignore
    Angle() * Matrix()         # type: ignore
    Angle() * FrozenMatrix()   # type: ignore
    Angle() * (1.0, 2.0, 3.0)  # type: ignore
    assert_type(Angle() * 5, Angle)
    assert_type(Angle() * 5.0, Angle)

    FrozenAngle() * Vec()            # type: ignore
    FrozenAngle() * FrozenVec()      # type: ignore
    FrozenAngle() * Angle()          # type: ignore
    FrozenAngle() * FrozenAngle()    # type: ignore
    FrozenAngle() * Matrix()         # type: ignore
    FrozenAngle() * FrozenMatrix()   # type: ignore
    FrozenAngle() * (1.0, 2.0, 3.0)  # type: ignore
    assert_type(FrozenAngle() * 5, FrozenAngle)
    assert_type(FrozenAngle() * 5.0, FrozenAngle)

    Matrix() * Vec()             # type: ignore
    Matrix() * FrozenVec()       # type: ignore
    Matrix() * Angle()           # type: ignore
    Matrix() * FrozenAngle()     # type: ignore
    Matrix() * Matrix()          # type: ignore
    Matrix() * FrozenMatrix()    # type: ignore
    Matrix() * (1.0, 2.0, 3.0)   # type: ignore
    Matrix() * 5                 # type: ignore
    Matrix() * 5.0               # type: ignore

    FrozenMatrix() * Vec()             # type: ignore
    FrozenMatrix() * FrozenVec()       # type: ignore
    FrozenMatrix() * Angle()           # type: ignore
    FrozenMatrix() * FrozenAngle()     # type: ignore
    FrozenMatrix() * Matrix()          # type: ignore
    FrozenMatrix() * FrozenMatrix()    # type: ignore
    FrozenMatrix() * (1.0, 2.0, 3.0)   # type: ignore
    FrozenMatrix() * 5                 # type: ignore
    FrozenMatrix() * 5.0               # type: ignore

    (1.0, 2.0, 3.0) * Vec()           # type: ignore
    (1.0, 2.0, 3.0) * FrozenVec()     # type: ignore
    (1.0, 2.0, 3.0) * Angle()         # type: ignore
    (1.0, 2.0, 3.0) * FrozenAngle()   # type: ignore
    (1.0, 2.0, 3.0) * Matrix()        # type: ignore
    (1.0, 2.0, 3.0) * FrozenMatrix()  # type: ignore

    assert_type(5 * Vec(), Vec)
    assert_type(5 * FrozenVec(), FrozenVec)
    assert_type(5 * Angle(), Angle)
    assert_type(5 * FrozenAngle(), FrozenAngle)
    5 * Matrix()         # type: ignore
    5 * FrozenMatrix()   # type: ignore

    assert_type(5.0 * Vec(), Vec)
    assert_type(5.0 * FrozenVec(), FrozenVec)
    assert_type(5.0 * Angle(), Angle)
    assert_type(5.0 * FrozenAngle(), FrozenAngle)
    5.0 * Matrix()         # type: ignore
    5.0 * FrozenMatrix()   # type: ignore


def test_division() -> None:
    """Check division operation types."""
    Vec() / Vec()            # type: ignore
    Vec() / FrozenVec()      # type: ignore
    Vec() / Angle()          # type: ignore
    Vec() / FrozenAngle()    # type: ignore
    Vec() / Matrix()         # type: ignore
    Vec() / FrozenMatrix()   # type: ignore
    Vec() / (1.0, 2.0, 3.0)  # type: ignore
    assert_type(Vec() / 5, Vec)
    assert_type(Vec() / 5.0, Vec)

    FrozenVec() / Vec()            # type: ignore
    FrozenVec() / FrozenVec()      # type: ignore
    FrozenVec() / Angle()          # type: ignore
    FrozenVec() / FrozenAngle()    # type: ignore
    FrozenVec() / Matrix()         # type: ignore
    FrozenVec() / FrozenMatrix()   # type: ignore
    FrozenVec() / (1.0, 2.0, 3.0)  # type: ignore
    assert_type(FrozenVec() / 5, FrozenVec)
    assert_type(FrozenVec() / 5.0, FrozenVec)

    Angle() / Vec()            # type: ignore
    Angle() / FrozenVec()      # type: ignore
    Angle() / Angle()          # type: ignore
    Angle() / FrozenAngle()    # type: ignore
    Angle() / Matrix()         # type: ignore
    Angle() / FrozenMatrix()   # type: ignore
    Angle() / (1.0, 2.0, 3.0)  # type: ignore
    Angle() / 5,               # type: ignore
    Angle() / 5.0              # type: ignore

    FrozenAngle() / Vec()            # type: ignore
    FrozenAngle() / FrozenVec()      # type: ignore
    FrozenAngle() / Angle()          # type: ignore
    FrozenAngle() / FrozenAngle()    # type: ignore
    FrozenAngle() / Matrix()         # type: ignore
    FrozenAngle() / FrozenMatrix()   # type: ignore
    FrozenAngle() / (1.0, 2.0, 3.0)  # type: ignore
    FrozenAngle() / 5                # type: ignore
    FrozenAngle() / 5.0              # type: ignore

    Matrix() / Vec()             # type: ignore
    Matrix() / FrozenVec()       # type: ignore
    Matrix() / Angle()           # type: ignore
    Matrix() / FrozenAngle()     # type: ignore
    Matrix() / Matrix()          # type: ignore
    Matrix() / FrozenMatrix()    # type: ignore
    Matrix() / (1.0, 2.0, 3.0)   # type: ignore
    Matrix() / 5                 # type: ignore
    Matrix() / 5.0               # type: ignore

    FrozenMatrix() / Vec()             # type: ignore
    FrozenMatrix() / FrozenVec()       # type: ignore
    FrozenMatrix() / Angle()           # type: ignore
    FrozenMatrix() / FrozenAngle()     # type: ignore
    FrozenMatrix() / Matrix()          # type: ignore
    FrozenMatrix() / FrozenMatrix()    # type: ignore
    FrozenMatrix() / (1.0, 2.0, 3.0)   # type: ignore
    FrozenMatrix() / 5                 # type: ignore
    FrozenMatrix() / 5.0               # type: ignore

    (1.0, 2.0, 3.0) / Vec()           # type: ignore
    (1.0, 2.0, 3.0) / FrozenVec()     # type: ignore
    (1.0, 2.0, 3.0) / Angle()         # type: ignore
    (1.0, 2.0, 3.0) / FrozenAngle()   # type: ignore
    (1.0, 2.0, 3.0) / Matrix()        # type: ignore
    (1.0, 2.0, 3.0) / FrozenMatrix()  # type: ignore

    assert_type(5 / Vec(), Vec)
    assert_type(5 / FrozenVec(), FrozenVec)
    5 / Angle()         # type: ignore
    5 / FrozenAngle()   # type: ignore
    5 / Matrix()        # type: ignore
    5 / FrozenMatrix()  # type: ignore

    assert_type(5.0 / Vec(), Vec)
    assert_type(5.0 / FrozenVec(), FrozenVec)
    5.0 / Angle()         # type: ignore
    5.0 / FrozenAngle()   # type: ignore
    5.0 / Matrix()        # type: ignore
    5.0 / FrozenMatrix()  # type: ignore


def test_floor_division() -> None:
    """Test floor division operation types."""
    Vec() // Vec()            # type: ignore
    Vec() // FrozenVec()      # type: ignore
    Vec() // Angle()          # type: ignore
    Vec() // FrozenAngle()    # type: ignore
    Vec() // Matrix()         # type: ignore
    Vec() // FrozenMatrix()   # type: ignore
    Vec() // (1.0, 2.0, 3.0)  # type: ignore
    assert_type(Vec() // 5, Vec)
    assert_type(Vec() // 5.0, Vec)

    FrozenVec() // Vec()            # type: ignore
    FrozenVec() // FrozenVec()      # type: ignore
    FrozenVec() // Angle()          # type: ignore
    FrozenVec() // FrozenAngle()    # type: ignore
    FrozenVec() // Matrix()         # type: ignore
    FrozenVec() // FrozenMatrix()   # type: ignore
    FrozenVec() // (1.0, 2.0, 3.0)  # type: ignore
    assert_type(FrozenVec() // 5, FrozenVec)
    assert_type(FrozenVec() // 5.0, FrozenVec)

    Angle() // Vec()            # type: ignore
    Angle() // FrozenVec()      # type: ignore
    Angle() // Angle()          # type: ignore
    Angle() // FrozenAngle()    # type: ignore
    Angle() // Matrix()         # type: ignore
    Angle() // FrozenMatrix()   # type: ignore
    Angle() // (1.0, 2.0, 3.0)  # type: ignore
    Angle() // 5                # type: ignore
    Angle() // 5.0              # type: ignore

    FrozenAngle() // Vec()            # type: ignore
    FrozenAngle() // FrozenVec()      # type: ignore
    FrozenAngle() // Angle()          # type: ignore
    FrozenAngle() // FrozenAngle()    # type: ignore
    FrozenAngle() // Matrix()         # type: ignore
    FrozenAngle() // FrozenMatrix()   # type: ignore
    FrozenAngle() // (1.0, 2.0, 3.0)  # type: ignore
    FrozenAngle() // 5                # type: ignore
    FrozenAngle() // 5.0              # type: ignore

    Matrix() // Vec()             # type: ignore
    Matrix() // FrozenVec()       # type: ignore
    Matrix() // Angle()           # type: ignore
    Matrix() // FrozenAngle()     # type: ignore
    Matrix() // Matrix()          # type: ignore
    Matrix() // FrozenMatrix()    # type: ignore
    Matrix() // (1.0, 2.0, 3.0)   # type: ignore
    Matrix() // 5                 # type: ignore
    Matrix() // 5.0               # type: ignore

    FrozenMatrix() // Vec()             # type: ignore
    FrozenMatrix() // FrozenVec()       # type: ignore
    FrozenMatrix() // Angle()           # type: ignore
    FrozenMatrix() // FrozenAngle()     # type: ignore
    FrozenMatrix() // Matrix()          # type: ignore
    FrozenMatrix() // FrozenMatrix()    # type: ignore
    FrozenMatrix() // (1.0, 2.0, 3.0)   # type: ignore
    FrozenMatrix() // 5                 # type: ignore
    FrozenMatrix() // 5.0               # type: ignore

    (1.0, 2.0, 3.0) // Vec()           # type: ignore
    (1.0, 2.0, 3.0) // FrozenVec()     # type: ignore
    (1.0, 2.0, 3.0) // Angle()         # type: ignore
    (1.0, 2.0, 3.0) // FrozenAngle()   # type: ignore
    (1.0, 2.0, 3.0) // Matrix()        # type: ignore
    (1.0, 2.0, 3.0) // FrozenMatrix()  # type: ignore

    assert_type(5 // Vec(), Vec)
    assert_type(5 // FrozenVec(), FrozenVec)
    5 // Angle()         # type: ignore
    5 // FrozenAngle()   # type: ignore
    5 // Matrix()        # type: ignore
    5 // FrozenMatrix()  # type: ignore

    assert_type(5.0 // Vec(), Vec)
    assert_type(5.0 // FrozenVec(), FrozenVec)
    5.0 // Angle()          # type: ignore
    5.0 // FrozenAngle()    # type: ignore
    5.0 // Matrix()         # type: ignore
    5.0 // FrozenMatrix()   # type: ignore


def test_rotations() -> None:
    """Check rotation operation types."""
    Vec() @ Vec()  # type: ignore[operator]
    Vec() @ FrozenVec()  # type: ignore[operator]
    assert_type(Vec() @ Angle(), Vec)
    assert_type(Vec() @ FrozenAngle(), Vec)
    assert_type(Vec() @ Matrix(), Vec)
    assert_type(Vec() @ FrozenMatrix(), Vec)
    Vec() @ (1.0, 2.0, 3.0)  # type: ignore[operator]
    Vec() @ 5  # type: ignore[operator]
    Vec() @ 5.0  # type: ignore[operator]

    FrozenVec() @ Vec()  # type: ignore[operator]
    FrozenVec() @ FrozenVec()  # type: ignore[operator]
    assert_type(FrozenVec() @ Angle(), FrozenVec)
    assert_type(FrozenVec() @ FrozenAngle(), FrozenVec)
    assert_type(FrozenVec() @ Matrix(), FrozenVec)
    assert_type(FrozenVec() @ FrozenMatrix(), FrozenVec)
    FrozenVec() @ (1.0, 2.0, 3.0)  # type: ignore[operator]
    FrozenVec() @ 5  # type: ignore[operator]
    FrozenVec() @ 5.0  # type: ignore[operator]

    Angle() @ Vec()  # type: ignore[operator]
    Angle() @ FrozenVec()  # type: ignore[operator]
    assert_type(Angle() @ Angle(), Angle)
    assert_type(Angle() @ FrozenAngle(), Angle)
    assert_type(Angle() @ Matrix(), Angle)
    assert_type(Angle() @ FrozenMatrix(), Angle)
    Angle() @ (1.0, 2.0, 3.0)  # type: ignore[operator]
    Angle() @ 5  # type: ignore[operator]
    Angle() @ 5.0  # type: ignore[operator]

    FrozenAngle() @ Vec()  # type: ignore[operator]
    FrozenAngle() @ FrozenVec()  # type: ignore[operator]
    assert_type(FrozenAngle() @ Angle(), FrozenAngle)
    assert_type(FrozenAngle() @ FrozenAngle(), FrozenAngle)
    assert_type(FrozenAngle() @ Matrix(), FrozenAngle)
    assert_type(FrozenAngle() @ FrozenMatrix(), FrozenAngle)
    FrozenAngle() @ (1.0, 2.0, 3.0)  # type: ignore[operator]
    FrozenAngle() @ 5  # type: ignore[operator]
    FrozenAngle() @ 5.0  # type: ignore[operator]

    Matrix() @ Vec()  # type: ignore[operator]
    Matrix() @ FrozenVec()  # type: ignore[operator]
    assert_type(Matrix() @ Angle(), Matrix)
    assert_type(Matrix() @ FrozenAngle(), Matrix)
    assert_type(Matrix() @ Matrix(), Matrix)
    assert_type(Matrix() @ FrozenMatrix(), Matrix)
    Matrix() @ (1.0, 2.0, 3.0)  # type: ignore[operator]
    Matrix() @ 5  # type: ignore[operator]
    Matrix() @ 5.0  # type: ignore[operator]

    FrozenMatrix() @ Vec()  # type: ignore[operator]
    FrozenMatrix() @ FrozenVec()  # type: ignore[operator]
    assert_type(FrozenMatrix() @ Angle(), FrozenMatrix)
    assert_type(FrozenMatrix() @ FrozenAngle(), FrozenMatrix)
    assert_type(FrozenMatrix() @ Matrix(), FrozenMatrix)
    assert_type(FrozenMatrix() @ FrozenMatrix(), FrozenMatrix)
    FrozenMatrix() @ (1.0, 2.0, 3.0)  # type: ignore[operator]
    FrozenMatrix() @ 5  # type: ignore[operator]
    FrozenMatrix() @ 5.0  # type: ignore[operator]

    (1.0, 2.0, 3.0) @ Vec()   # type: ignore[operator]
    (1.0, 2.0, 3.0) @ FrozenVec()  # type: ignore[operator]
    assert_type((1.0, 2.0, 3.0) @ Angle(), Vec)
    assert_type((1.0, 2.0, 3.0) @ FrozenAngle(), Vec)
    assert_type((1.0, 2.0, 3.0) @ Matrix(), Vec)
    assert_type((1.0, 2.0, 3.0) @ FrozenMatrix(), Vec)

    5 @ Vec()           # type: ignore[operator]
    5 @ FrozenVec()     # type: ignore[operator]
    5 @ Angle()         # type: ignore[operator]
    5 @ FrozenAngle()   # type: ignore[operator]
    5 @ Matrix()        # type: ignore[operator]
    5 @ FrozenMatrix()  # type: ignore[operator]

    5.0 @ Vec()           # type: ignore[operator]
    5.0 @ FrozenVec()     # type: ignore[operator]
    5.0 @ Angle()         # type: ignore[operator]
    5.0 @ FrozenAngle()   # type: ignore[operator]
    5.0 @ Matrix()        # type: ignore[operator]
    5.0 @ FrozenMatrix()  # type: ignore[operator]


def test_dot() -> None:
    """Test that dot products can be called as a method and function."""
    mv = Vec()
    fv = FrozenVec()

    assert_type(mv.dot(mv), float)
    assert_type(mv.dot(fv), float)
    assert_type(fv.dot(mv), float)
    assert_type(fv.dot(fv), float)

    assert_type(Vec.dot(mv, mv), float)
    assert_type(Vec.dot(mv, fv), float)
    assert_type(Vec.dot(fv, mv), float)
    assert_type(Vec.dot(fv, fv), float)

    assert_type(FrozenVec.dot(mv, mv), float)
    assert_type(FrozenVec.dot(mv, fv), float)
    assert_type(FrozenVec.dot(fv, mv), float)
    assert_type(FrozenVec.dot(fv, fv), float)


def test_cross() -> None:
    """Test that cross products can be called as a method and function."""
    mv = Vec()
    fv = FrozenVec()

    assert_type(mv.cross(mv), Vec)
    assert_type(mv.cross(fv), Vec)
    assert_type(fv.cross(mv), FrozenVec)
    assert_type(fv.cross(fv), FrozenVec)

    assert_type(Vec.cross(mv, mv), Vec)
    assert_type(Vec.cross(mv, fv), Vec)
    assert_type(Vec.cross(fv, mv), Vec)
    assert_type(Vec.cross(fv, fv), Vec)

    assert_type(FrozenVec.cross(mv, mv), FrozenVec)
    assert_type(FrozenVec.cross(mv, fv), FrozenVec)
    assert_type(FrozenVec.cross(fv, mv), FrozenVec)
    assert_type(FrozenVec.cross(fv, fv), FrozenVec)
