srctools.tokenizer
------------------

.. automodule:: srctools.tokenizer
    :synopsis: A tokenizer used to parse all the text formats.


Constants
=========

.. autosrcenum:: Token
    :members:
    :undoc-members:
    :exclude-members: has_value

    .. autoproperty:: has_value

.. autodata:: BARE_DISALLOWED

.. autofunction:: escape_text

Errors
======

.. autoexception:: TokenSyntaxError
    :members:

.. autofunction:: format_exc_fileinfo
.. autoattribute:: BaseTokenizer.error_type
.. automethod:: BaseTokenizer.error


Main API
========

.. autoclass:: BaseTokenizer
    :members:
    :undoc-members:
    :exclude-members: error_type, error

.. autoclass:: Tokenizer
    :members:
    :undoc-members:

.. autoclass:: IterTokenizer
    :members:
    :undoc-members:
