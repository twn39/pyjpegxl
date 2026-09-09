"""PyJPEGXL — Python JPEG XL and JPEG encoding/decoding via libjxl and libjpeg-turbo."""

from pyjpegxl._async import (
    # JXL async
    async_decode,
    async_decode_into,
    async_decode_to_numpy,
    async_encode,
    async_encode_from_numpy,
    # JPEG async
    async_jpeg_decode,
    async_jpeg_decode_into,
    async_jpeg_decode_to_numpy,
    async_jpeg_encode,
    async_jpeg_encode_from_numpy,
    async_jpeg_file_to_jxl,
    async_jpeg_probe,
    async_jpeg_probe_file,
    async_jpeg_read,
    async_jpeg_read_into,
    async_jpeg_read_to_numpy,
    # JPEG ↔ JXL transcoding async
    async_jpeg_to_jxl,
    async_jpeg_write,
    async_jpeg_write_from_numpy,
    async_jxl_file_to_jpeg,
    async_jxl_to_jpeg,
    async_probe,
    async_probe_file,
    async_read,
    async_read_into,
    async_read_to_numpy,
    async_write,
    async_write_from_numpy,
)
from pyjpegxl._io import (
    jpeg_file_to_jxl,
    jxl_file_to_jpeg,
    probe_file,
    read,
    read_into,
    read_to_numpy,
    write,
    write_from_numpy,
)
from pyjpegxl._jpeg_io import (
    jpeg_probe_file,
    jpeg_read,
    jpeg_read_into,
    jpeg_read_to_numpy,
    jpeg_write,
    jpeg_write_from_numpy,
)
from pyjpegxl._pyjpegxl import (
    EncoderSpeed,
    # JPEG types
    JpegInfo,
    # JXL types
    Metadata,
    # JXL codec
    decode,
    decode_into,
    decode_to_numpy,
    encode,
    encode_from_numpy,
    # Threading control
    get_num_threads,
    # JPEG codec
    jpeg_decode,
    jpeg_decode_into,
    jpeg_decode_to_numpy,
    jpeg_encode,
    jpeg_encode_from_numpy,
    # JPEG metadata probing
    jpeg_probe,
    # JPEG ↔ JXL lossless transcoding
    jpeg_to_jxl,
    jxl_to_jpeg,
    # Fast metadata probing
    probe,
    set_num_threads,
)
from pyjpegxl._sniff import (
    PrefixedStream,
    sniff_bytes,
    sniff_source,
    sniff_stream,
)
from pyjpegxl._unified import (
    imread,
    imwrite,
    probe_image,
)
from pyjpegxl.batch import (
    read_batch,
    transcode_batch,
)
from pyjpegxl.pillow import (
    from_pil,
    to_pil,
)
from pyjpegxl.torch import (
    decode_into_tensor,
    from_tensor,
    to_tensor,
)

__all__ = [
    # Unified polymorphic I/O & sniffing
    "imread",
    "imwrite",
    "probe_image",
    "sniff_bytes",
    "sniff_stream",
    "sniff_source",
    "PrefixedStream",
    # Ecosystem bridges
    "to_pil",
    "from_pil",
    "to_tensor",
    "from_tensor",
    "decode_into_tensor",
    # High-throughput batch operations
    "read_batch",
    "transcode_batch",
    # JXL — fast metadata probing
    "probe",
    "probe_file",
    "async_probe",
    "async_probe_file",
    # JXL — sync bytes API
    "decode",
    "encode",
    # JXL — sync NumPy API (zero-copy)
    "decode_to_numpy",
    "encode_from_numpy",
    # JXL — zero-allocation in-place decode
    "decode_into",
    "read_into",
    "async_decode_into",
    "async_read_into",
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
    # Concurrency control
    "set_num_threads",
    "get_num_threads",
    # JPEG — fast metadata probing
    "jpeg_probe",
    "jpeg_probe_file",
    "async_jpeg_probe",
    "async_jpeg_probe_file",
    # JPEG — sync bytes API
    "jpeg_decode",
    "jpeg_encode",
    # JPEG — sync NumPy API
    "jpeg_decode_to_numpy",
    "jpeg_encode_from_numpy",
    # JPEG — zero-allocation in-place decode
    "jpeg_decode_into",
    "jpeg_read_into",
    "async_jpeg_decode_into",
    "async_jpeg_read_into",
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
    # JPEG ↔ JXL lossless transcoding — bytes
    "jpeg_to_jxl",
    "jxl_to_jpeg",
    # JPEG ↔ JXL lossless transcoding — file I/O
    "jpeg_file_to_jxl",
    "jxl_file_to_jpeg",
    # JPEG ↔ JXL lossless transcoding — async
    "async_jpeg_to_jxl",
    "async_jxl_to_jpeg",
    "async_jpeg_file_to_jxl",
    "async_jxl_file_to_jpeg",
    # Types
    "Metadata",
    "EncoderSpeed",
    "JpegInfo",
    "__version__",
]

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("pyjpegxl")
except PackageNotFoundError:
    __version__ = "0.2.2"
