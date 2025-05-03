srctools.math
-------------

.. automodule:: srctools.math
	:synopsis: Provides vector, Euler angles and rotation matrix classes


----------
Tolerances
----------
Repeated calculations, especially calculations involving rotation will inevitably acculumulate error,
making exact comparison unreliable.
However, it is quite useful to compare to values like ``(0, 0, 1)``, so to allow this comparison
operations will treat a difference of less than 10\ :sup:`-6` as equal.
This precision was chosen since it is the number of decimal points permitted in SMD files.
If exact comparisons are required, direct attribute comparisons can be used.

To allow use as dictionary keys, `FrozenVec`/`FrozenAngle` round to the same precision also when hashing.
Note however that such a lookup is not reliable, since it is possible that two similar values might
fall across the rounding threshold, causing them to be in different key "buckets".

--------------
Free Functions
--------------

.. autofunction:: parse_vec_str[X, Y, Z](val: str | Vec | FrozenVec | Angle | FrozenAngle, x: X = 0.0, y: Y = 0.0, z: Z = 0.0) -> tuple[float, float, float] | tuple[X, Y, Z]

.. autofunction:: format_float

.. autofunction:: lerp

.. autofunction:: quickhull

.. autofunction:: to_matrix

------------
Type aliases
------------

Several public type aliases are provided for convenience.

.. py:type:: AnyVec
	:canonical: Vec | FrozenVec | Vec_tuple | tuple[float, float, float]

	Type alias representing all values accepted as vectors.

.. py:type:: VecUnion
	:canonical: Vec | FrozenVec

	Type alias representing both vector types.

.. py:type:: AnyAngle
	:canonical: Angle | FrozenAngle

	Type alias representing either angle type.

.. py:type:: AnyMatrix
	:canonical: Matrix | FrozenMatrix

	Type alias representing either matrix type.

-------
Vectors
-------

.. finality = Bases can't be typing.final, but subclassing shouldn't be permitted.


.. autoclass:: VecBase()
	:members:
	:special-members:

.. autoclass:: FrozenVec
	:members:
	:special-members:

.. autoclass:: Vec
	:members:
	:special-members:

------------

.. autoclass:: Vec_tuple
	:members:

------------
Euler Angles
------------

.. autoclass:: AngleBase()
	:members:
	:special-members:

.. autoclass:: FrozenAngle
	:members:
	:special-members:

.. autoclass:: Angle
	:members:
	:special-members:

-----------------
Rotation Matrices
-----------------

.. autoclass:: MatrixBase()
	:members:
	:special-members:


.. autoclass:: FrozenMatrix
	:members:
	:special-members:


.. autoclass:: Matrix
	:members:
	:special-members:
