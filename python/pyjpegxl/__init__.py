"""PyJPEGXL — Python JPEG XL and JPEG encoding/decoding via libjxl and libjpeg-turbo."""

from pyjpegxl._pyjpegxl import (
    # JXL types
    Metadata,
    EncoderSpeed,
    # JXL codec
    decode,
    encode,
    decode_to_numpy,
    encode_from_numpy,
    # JPEG types
    JpegInfo,
    # JPEG codec
    jpeg_decode,
    jpeg_encode,
    jpeg_decode_to_numpy,
    jpeg_encode_from_numpy,
)
from pyjpegxl._io import (
    read,
    read_to_numpy,
    write,
    write_from_numpy,
)
from pyjpegxl._jpeg_io import (
    jpeg_read,
    jpeg_read_to_numpy,
    jpeg_write,
    jpeg_write_from_numpy,
)
from pyjpegxl._async import (
    # JXL async
    async_decode,
    async_encode,
    async_decode_to_numpy,
    async_encode_from_numpy,
    async_read,
    async_read_to_numpy,
    async_write,
    async_write_from_numpy,
    # JPEG async
    async_jpeg_decode,
    async_jpeg_encode,
    async_jpeg_decode_to_numpy,
    async_jpeg_encode_from_numpy,
    async_jpeg_read,
    async_jpeg_read_to_numpy,
    async_jpeg_write,
    async_jpeg_write_from_numpy,
)

__all__ = [
    # JXL — sync bytes API
    "decode",
    "encode",
    # JXL — sync NumPy API (zero-copy)
    "decode_to_numpy",
    "encode_from_numpy",
    # JXL — sync file I/O
    "read",
    "read_to_numpy",
    "write",
    "write_from_numpy",
    # JXL — async
    "async_decode",
    "async_encode",
    "async_decode_to_numpy",
    "async_encode_from_numpy",
    "async_read",
    "async_read_to_numpy",
    "async_write",
    "async_write_from_numpy",
    # JPEG — sync bytes API
    "jpeg_decode",
    "jpeg_encode",
    # JPEG — sync NumPy API
    "jpeg_decode_to_numpy",
    "jpeg_encode_from_numpy",
    # JPEG — sync file I/O
    "jpeg_read",
    "jpeg_read_to_numpy",
    "jpeg_write",
    "jpeg_write_from_numpy",
    # JPEG — async
    "async_jpeg_decode",
    "async_jpeg_encode",
    "async_jpeg_decode_to_numpy",
    "async_jpeg_encode_from_numpy",
    "async_jpeg_read",
    "async_jpeg_read_to_numpy",
    "async_jpeg_write",
    "async_jpeg_write_from_numpy",
    # Types
    "Metadata",
    "EncoderSpeed",
    "JpegInfo",
]
__version__ = "0.1.0"
