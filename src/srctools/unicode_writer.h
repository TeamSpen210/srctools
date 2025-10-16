#ifdef PYPY_VERSION
// PyPy is missing _PyUnicodeWriter (since that's private), so include a basic version.

#include <stddef.h> // NULL

typedef struct PyUnicodeWriter {
	Py_ssize_t size;
	Py_ssize_t allocated;
	Py_UCS4 *buffer;
} PyUnicodeWriter;

static inline void PyUnicodeWriter_Discard(PyUnicodeWriter *writer) {
	if (writer != NULL) {
	    PyMem_Free(writer->buffer);
	    writer->buffer = NULL;
	    PyMem_Free(writer);
	}
}

// Ensure we have room for the specified amount of additional characters.
static int _PyUnicodeWriter_Prepare(PyUnicodeWriter *writer, Py_ssize_t size) {
	if (size <= (PY_SSIZE_T_MAX - writer->size)) {
		size += writer->size;
	} else {
        PyErr_NoMemory();
        return -1;
    }

	if (size <= writer->allocated) {
		writer->size = size;
		return 0;
	}
	Py_ssize_t alloc = writer->allocated;
	if (alloc < 4) {
		alloc = 4; // 1/2 = 0, so loop is infinite. Just jump to 4.
	}
	while (alloc < size) {
		if (alloc <= (PY_SSIZE_T_MAX - alloc / 2)) {
            alloc += alloc / 2;
        } else {
            PyErr_NoMemory();
            return -1;
        }
    }

	Py_UCS4 *buf = (Py_UCS4*)PyMem_Realloc(writer->buffer, alloc * sizeof(Py_UCS4));
	if (buf == NULL) {
		return -1;
	}
	writer->buffer = buf;
	writer->allocated = alloc;
	writer->size = size;
	return 0;
}

static inline PyUnicodeWriter* PyUnicodeWriter_Create(Py_ssize_t length) {
    if (length < 0) {
        PyErr_SetString(PyExc_ValueError, "length must be positive");
        return NULL;
    }

    PyUnicodeWriter *writer = (PyUnicodeWriter *)PyMem_Malloc(sizeof(PyUnicodeWriter));
    if (writer == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    writer->size = 0;
    writer->allocated = 0;
    writer->buffer = NULL;
    if (_PyUnicodeWriter_Prepare(writer, length) < 0) {
        PyUnicodeWriter_Discard(writer);
        return NULL;
    }
    writer->size = 0;
    return writer;
}

static inline PyObject* PyUnicodeWriter_Finish(PyUnicodeWriter *writer) {
	PyObject *str;
	if (writer->size > 0) {
	    str = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, writer->buffer, writer->size);
	} else { // Avoid passing a null pointer.
		str = PyUnicode_New(0, 0);
	}
    PyMem_Free(writer->buffer);
    PyMem_Free(writer);
    return str;
}

static inline int PyUnicodeWriter_WriteChar(PyUnicodeWriter *writer, Py_UCS4 ch) {
    if (ch > 0x10ffff) {
        PyErr_SetString(PyExc_ValueError, "character must be in range(0x110000)");
        return -1;
    }
    Py_ssize_t ind = writer->size;
    if (_PyUnicodeWriter_Prepare(writer, 1) < 0) {
        return -1;
    }
    writer->buffer[ind] = ch;
    return 0;
}

static inline int _PyUnicodeWriter_WriteStr(PyUnicodeWriter *writer, PyObject *str) {
	Py_ssize_t len = PyUnicode_GET_LENGTH(str);
	int kind = PyUnicode_KIND(str);
	void *data = PyUnicode_DATA(str);

	Py_ssize_t off = writer->size;
    if (_PyUnicodeWriter_Prepare(writer, len) < 0) {
        return -1;
    }
	for(Py_ssize_t i=0; i<len; i++) {
		writer->buffer[off+i] = PyUnicode_READ(kind, data, i);
	}
	return 0;
}

static inline int PyUnicodeWriter_WriteStr(PyUnicodeWriter *writer, PyObject *obj) {
    PyObject *str = PyObject_Str(obj);
    if (str == NULL) {
        return -1;
    }

    int res = _PyUnicodeWriter_WriteStr(writer, str);
    Py_DECREF(str);
    return res;
}

static inline int PyUnicodeWriter_WriteRepr(PyUnicodeWriter *writer, PyObject *obj) {
    PyObject *str = PyObject_Repr(obj);
    if (str == NULL) {
        return -1;
    }

    int res = _PyUnicodeWriter_WriteStr(writer, str);
    Py_DECREF(str);
    return res;
}

static inline int PyUnicodeWriter_WriteUTF8(PyUnicodeWriter *writer, const char *str, Py_ssize_t size) {
    if (size < 0) {
        size = (Py_ssize_t)strlen(str);
    }

    PyObject *str_obj = PyUnicode_FromStringAndSize(str, size);
    if (str_obj == NULL) {
        return -1;
    }

    int res = _PyUnicodeWriter_WriteStr(writer, str_obj);
    Py_DECREF(str_obj);
    return res;
}

static inline int PyUnicodeWriter_WriteASCII(PyUnicodeWriter *writer, const char *str, Py_ssize_t size) {
    if (size < 0) {
        size = (Py_ssize_t)strlen(str);
    }
	Py_ssize_t off = writer->size;
    if (_PyUnicodeWriter_Prepare(writer, size) < 0) {
        return -1;
    }

	for(Py_ssize_t i=0; i<size; i++) {
		writer->buffer[off+i] = str[i];
	}
}

// Not used yet, don't bother.
//static inline int PyUnicodeWriter_WriteWideChar(PyUnicodeWriter *writer, const wchar_t *str, Py_ssize_t size)
//static inline int PyUnicodeWriter_WriteSubstring(PyUnicodeWriter *writer, PyObject *str, Py_ssize_t start, Py_ssize_t end)

static inline int PyUnicodeWriter_Format(PyUnicodeWriter *writer, const char *format, ...) {
    va_list vargs;
    va_start(vargs, format);
    PyObject *str = PyUnicode_FromFormatV(format, vargs);
    va_end(vargs);
    if (str == NULL) {
        return -1;
    }

    int res = _PyUnicodeWriter_WriteStr(writer, str);
    Py_DECREF(str);
    return res;
}

#endif
