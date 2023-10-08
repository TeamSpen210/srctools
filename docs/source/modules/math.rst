srctools.math
-------------

.. automodule:: srctools.math
	:synopsis: Provides vector, Euler angles and rotation matrix classes


-------------------
Implicit Tolerances
-------------------
Repeated calculations, especially calculations involving rotation will inevitably acculumulate error, making exact comparison unreliable. However, it is quite useful to compare to values like ``(0, 0, 1)``, so to allow this comparison operations will treat a difference of less than 10\ :sup:`-6` as equal. This precision was chosen since it is the number of decimal points permitted in SMD files. If exact comparisons are required, direct attribute comparisons can be used. To allow use as dictionary keys, :py:meth:`Vec.as_tuple()` and :py:meth:`Angle.as_tuple()` round to the same precision also.

--------------
Free Functions
--------------

.. autofunction:: parse_vec_str

.. autofunction:: lerp

.. autofunction:: to_matrix

.. autofunction:: quickhull


-------
Vectors
-------

.. autoclass:: VecBase
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

.. autoclass:: AngleBase
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

.. autoclass:: MatrixBase
	:members:
	:special-members:


.. autoclass:: FrozenMatrix
	:members:
	:special-members:


.. autoclass:: Matrix
	:members:
	:special-members:
