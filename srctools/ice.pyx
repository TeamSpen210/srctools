#cython: language_level=3, embedsignature=True, auto_pickle=False
"""Interface to the ICE encryption engine. 

This is used by Valve to encrypt certain script files.

The ICE class replicates the interface of the original API, allowing
direct encryption and decryption. The ICEReader and ICEWriter classes wrap a file
to allow transparently reading or writing data.
"""
cimport cython
from cpython.bytes cimport PyBytes_FromStringAndSize

# Original C API:
cdef extern from "ice/ice.h":
    struct ice_key_struct:
        pass
    ctypedef ice_key_struct ICE_KEY

    # Create an ICE key with a size of n.
    ICE_KEY *ice_key_create(int n)

    # Clean up an ICE key.
    void ice_key_destroy(ICE_KEY *ik)

    # Set the ICE key to use the given password.
    void ice_key_set(ICE_KEY *ik, unsigned char *k)

    # Encrypt/Decrypt 8-byte blocks of data.
    void ice_key_encrypt(ICE_KEY *ik, unsigned char *ptxt, unsigned char *ctxt)
    void ice_key_decrypt(ICE_KEY *ik, unsigned char *ctxt, unsigned char *ptxt)

    # Get the size of the key required, in bytes.
    int ice_key_key_size(ICE_KEY *ik)

    # Return the block size, in bytes.
    int ice_key_block_size(ICE_KEY *ik)


cdef class ICE(object):
    """Represents an ICE encrpyption key and session."""
    cdef ICE_KEY *key

    def __init__(self, int rounds=1, password: bytes=None) -> None:
        """Construct an encryption key, optionally setting the password."""
        self.key = ice_key_create(rounds)
        if password is not None:
            self.set_key(password)

    cpdef set_key(self, bytes password):
        """Set the encyption key to the given value."""
        key_len = ice_key_key_size(self.key)
        if len(password) != key_len:
            raise ValueError(
                f"Key must be {key_len} bytes "
                f"long, not {len(password)}!"
            )
        ice_key_set(self.key, password)

    cpdef bytes encrypt(self, const unsigned char[:] data):
        """Encrypt the given data."""
        # ICE requires 8-byte blocks.
        # So pad to that length first.

        cdef int size = data.shape[0]
        cdef int rem = size % 8

        if rem != 0:
            data = bytes(data) + bytes(8 - rem)
            size = data.shape[0]

        # Make a buffer of the right size.
        cdef bytes output = PyBytes_FromStringAndSize(NULL, size)

        cdef unsigned char[::1] cipher = output;

        cdef int off
        for off in range(0, size, 8):
            ice_key_encrypt(self.key, &data[off], &cipher[off])

        return output

    cpdef bytes decrypt(self, const unsigned char[::1] data):
        """Decrypt the given data."""
        # ICE requires 8-byte blocks.
        # So pad to that length first.

        cdef int size = len(data)

        if size % 8 != 0:
            raise ValueError('Blocks must be a multiple of 8 bytes.')

        # Make a buffer of the right size.
        cdef bytes output = PyBytes_FromStringAndSize(NULL, size)

        cdef unsigned char *c_plain = output

        cdef int off
        for off in range(0, size, 8):
            ice_key_decrypt(self.key, &data[off], c_plain +off)

        return output
