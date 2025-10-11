# Bindings for pythoncapi-compat, implemented directly in Python.h in newer versions.

cdef extern from "pythoncapi_compat.h":
	ctypedef struct PyUnicodeWriter:
		pass

	void PyUnicodeWriter_Discard(PyUnicodeWriter *writer) noexcept
	PyUnicodeWriter* PyUnicodeWriter_Create(Py_ssize_t length) except NULL
	str PyUnicodeWriter_Finish(PyUnicodeWriter *writer)
	int PyUnicodeWriter_WriteChar(PyUnicodeWriter *writer, Py_UCS4 ch) except -1
	int PyUnicodeWriter_WriteStr(PyUnicodeWriter *writer, object obj) except -1
	int PyUnicodeWriter_WriteRepr(PyUnicodeWriter *writer, object obj) except -1
	int PyUnicodeWriter_WriteUTF8(PyUnicodeWriter *writer, const char *str, Py_ssize_t size) except -1
	int PyUnicodeWriter_WriteASCII(PyUnicodeWriter *writer, const char *str, Py_ssize_t size) except -1
	# int PyUnicodeWriter_WriteWideChar(PyUnicodeWriter *writer, const wchar_t *str, Py_ssize_t size) except -1
	int PyUnicodeWriter_WriteSubstring(PyUnicodeWriter *writer, unicode str, Py_ssize_t start, Py_ssize_t end) except -1
	int PyUnicodeWriter_Format(PyUnicodeWriter *writer, const char *format, ...) except -1


	ctypedef struct PyBytesWriter:
		pass

	void* PyBytesWriter_GetData(PyBytesWriter *writer) noexcept
	Py_ssize_t PyBytesWriter_GetSize(PyBytesWriter *writer) noexcept
	void PyBytesWriter_Discard(PyBytesWriter *writer) noexcept
	PyBytesWriter* PyBytesWriter_Create(Py_ssize_t size) except NULL
	bytes PyBytesWriter_FinishWithSize(PyBytesWriter *writer, Py_ssize_t size)
	bytes PyBytesWriter_Finish(PyBytesWriter *writer)
	bytes PyBytesWriter_FinishWithPointer(PyBytesWriter *writer, void *buf)
	int PyBytesWriter_Resize(PyBytesWriter *writer, Py_ssize_t size) except -1
	int PyBytesWriter_Grow(PyBytesWriter *writer, Py_ssize_t size) except -1
	void* PyBytesWriter_GrowAndUpdatePointer(PyBytesWriter *writer, Py_ssize_t size, void *buf) except NULL
	int PyBytesWriter_WriteBytes(PyBytesWriter *writer, const void *bytes, Py_ssize_t size) except -1
	int PyBytesWriter_Format(PyBytesWriter *writer, const char *format, ...) except -1
