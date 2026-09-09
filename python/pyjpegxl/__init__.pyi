"""Type stubs for pyjpegxl."""

from __future__ import annotations

import io
import os
from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

# ===========================================================================
# JXL Types
# ===========================================================================

class Metadata:
    """Image metadata returned by JXL decode."""

    width: int
    height: int
    num_color_channels: int
    has_alpha: bool
    bits_per_sample: int
    intensity_target: float
    min_nits: float
    exif: bytes | None
    xmp: bytes | None
    icc: bytes | None
    icc_profile: bytes | None

class EncoderSpeed(IntEnum):
    """Encoder speed presets (fastest → slowest)."""

    Lightning = 1
    Thunder = 2
    Falcon = 3
    Cheetah = 4
    Hare = 5
    Wombat = 6
    Squirrel = 7
    Kitten = 8
    Tortoise = 9

# ===========================================================================
# JPEG Types
# ===========================================================================

class JpegInfo:
    """Image metadata returned by JPEG decode."""

    width: int
    height: int
    num_channels: int
    exif: bytes | None
    icc: bytes | None
    icc_profile: bytes | None

# ===========================================================================
# Fast Metadata Probing
# ===========================================================================

def probe(data: bytes) -> Metadata:
    """Fast metadata inspection without decoding pixel data.

    Executes in sub-millisecond time and allocates zero pixel buffer memory.
    """
    ...

def probe_file(path: str | os.PathLike) -> Metadata:
    """Fast metadata inspection of a JXL file without decoding pixel data."""
    ...

def jpeg_probe(data: bytes) -> JpegInfo:
    """Fast metadata inspection of a JPEG without decoding pixel data."""
    ...

def jpeg_probe_file(path: str | os.PathLike) -> JpegInfo:
    """Fast metadata inspection of a JPEG file without decoding pixel data."""
    ...

# ===========================================================================
# Concurrency & Thread Pool Control
# ===========================================================================

def set_num_threads(num_threads: int) -> None:
    """Set global thread count for libjxl encoding/decoding.

    Pass 0 for automatic detection (logical CPU count).
    Pass 1 for pure single-threaded execution without thread pool overhead.
    """
    ...

def get_num_threads() -> int:
    """Get current global thread count setting (0 = auto)."""
    ...

# ===========================================================================
# JXL — Sync bytes API
# ===========================================================================

def decode(data: bytes, *, dtype: str | None = None) -> tuple[Metadata, bytes]:
    """Decode JXL bytes → (Metadata, raw pixel bytes)."""
    ...

def encode(
    data: bytes,
    width: int,
    height: int,
    *,
    lossless: bool = False,
    quality: float = 1.0,
    speed: EncoderSpeed = EncoderSpeed.Squirrel,
    num_channels: int = 4,
    exif: bytes | None = None,
    xmp: bytes | None = None,
    icc: bytes | None = None,
    dtype: str | None = None,
    intensity_target: float | None = None,
) -> bytes:
    """Encode raw pixel bytes → JXL bytes."""
    ...

# ===========================================================================
# JXL — Sync NumPy API (zero-copy & zero-allocation)
# ===========================================================================

def decode_to_numpy(
    data: bytes,
    *,
    dtype: str | None = None,
) -> tuple[Metadata, npt.NDArray[np.uint8 | np.uint16 | np.float32]]:
    """Decode JXL bytes → (Metadata, ndarray shape (H,W,C)). Zero-copy."""
    ...

def decode_into(
    data: bytes,
    out: npt.NDArray[np.uint8 | np.uint16 | np.float32],
) -> Metadata:
    """Decode JXL bytes directly into preallocated writable NumPy array.

    Zero-allocation: eliminates output buffer allocation.
    `out` must be a C-contiguous writable ndarray with matching dimensions and dtype.
    """
    ...

def encode_from_numpy(
    array: npt.NDArray[np.uint8 | np.uint16 | np.float32],
    *,
    lossless: bool = False,
    quality: float = 1.0,
    speed: EncoderSpeed = EncoderSpeed.Squirrel,
    exif: bytes | None = None,
    xmp: bytes | None = None,
    icc: bytes | None = None,
    intensity_target: float | None = None,
) -> bytes:
    """Encode ndarray (H,W) or (H,W,C) (uint8, uint16, float32) → JXL bytes."""
    ...

# ===========================================================================
# JXL — Sync file I/O
# ===========================================================================

def read(path: str | os.PathLike, *, dtype: str | None = None) -> tuple[Metadata, bytes]:
    """Read a JXL file → (Metadata, raw pixel bytes)."""
    ...

def read_to_numpy(
    path: str | os.PathLike,
    *,
    dtype: str | None = None,
) -> tuple[Metadata, npt.NDArray[np.uint8 | np.uint16 | np.float32]]:
    """Read a JXL file → (Metadata, ndarray shape (H,W,C))."""
    ...

def read_into(
    path: str | os.PathLike,
    out: npt.NDArray[np.uint8 | np.uint16 | np.float32],
) -> Metadata:
    """Read a JXL file and decode directly into preallocated writable NumPy array."""
    ...

def write(
    path: str | os.PathLike,
    data: bytes,
    width: int,
    height: int,
    *,
    lossless: bool = False,
    quality: float = 1.0,
    speed: EncoderSpeed = EncoderSpeed.Squirrel,
    num_channels: int = 4,
    exif: bytes | None = None,
    xmp: bytes | None = None,
    icc: bytes | None = None,
    dtype: str | None = None,
    intensity_target: float | None = None,
) -> int:
    """Encode raw pixel bytes and write to a JXL file. Returns bytes written."""
    ...

def write_from_numpy(
    path: str | os.PathLike,
    array: npt.NDArray[np.uint8 | np.uint16 | np.float32],
    *,
    lossless: bool = False,
    quality: float = 1.0,
    speed: EncoderSpeed = EncoderSpeed.Squirrel,
    exif: bytes | None = None,
    xmp: bytes | None = None,
    icc: bytes | None = None,
    intensity_target: float | None = None,
) -> int:
    """Encode ndarray and write to a JXL file. Returns bytes written."""
    ...

# ===========================================================================
# JPEG — Sync bytes API
# ===========================================================================

def jpeg_decode(data: bytes) -> tuple[JpegInfo, bytes]:
    """Decode JPEG bytes → (JpegInfo, raw pixel bytes u8)."""
    ...

def jpeg_encode(
    data: bytes,
    width: int,
    height: int,
    *,
    quality: int = 95,
    num_channels: int = 3,
) -> bytes:
    """Encode raw pixel bytes → JPEG bytes."""
    ...

# ===========================================================================
# JPEG — Sync NumPy API
# ===========================================================================

def jpeg_decode_to_numpy(
    data: bytes,
    *,
    channels: int | None = None,
) -> tuple[JpegInfo, np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]]:
    """Decode JPEG bytes → (JpegInfo, ndarray shape (H,W,C) dtype uint8)."""
    ...

def jpeg_decode_into(
    data: bytes,
    out: npt.NDArray[np.uint8],
) -> JpegInfo:
    """Decode JPEG bytes directly into preallocated writable NumPy array."""
    ...

def jpeg_encode_from_numpy(
    array: npt.NDArray[np.uint8],
    *,
    quality: int = 95,
) -> bytes:
    """Encode ndarray (H,W,C) uint8 → JPEG bytes."""
    ...

# ===========================================================================
# JPEG — Sync file I/O
# ===========================================================================

def jpeg_read(path: str | os.PathLike, *, channels: int | None = None) -> tuple[JpegInfo, bytes]:
    """Read a JPEG file → (JpegInfo, raw pixel bytes)."""
    ...

def jpeg_read_to_numpy(
    path: str | os.PathLike,
    *,
    channels: int | None = None,
) -> tuple[JpegInfo, np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]]:
    """Read a JPEG file → (JpegInfo, ndarray shape (H,W,C) dtype uint8)."""
    ...

def jpeg_read_into(
    path: str | os.PathLike,
    out: npt.NDArray[np.uint8],
) -> JpegInfo:
    """Read a JPEG file directly into preallocated writable NumPy array."""
    ...

def jpeg_write(
    path: str | os.PathLike,
    data: bytes,
    width: int,
    height: int,
    *,
    quality: int = 95,
    num_channels: int = 3,
) -> int:
    """Encode raw pixel bytes and write to a JPEG file. Returns bytes written."""
    ...

def jpeg_write_from_numpy(
    path: str | os.PathLike,
    array: npt.NDArray[np.uint8],
    *,
    quality: int = 95,
) -> int:
    """Encode ndarray and write to a JPEG file. Returns bytes written."""
    ...

# ===========================================================================
# JXL — Async wrappers
# ===========================================================================

async def async_probe(data: bytes) -> Metadata: ...
async def async_probe_file(path: str | os.PathLike) -> Metadata: ...
async def async_decode(data: bytes, *, dtype: str | None = None) -> tuple[Metadata, bytes]: ...
async def async_decode_into(
    data: bytes,
    out: npt.NDArray[np.uint8 | np.uint16 | np.float32],
) -> Metadata: ...
async def async_encode(
    data: bytes,
    width: int,
    height: int,
    *,
    lossless: bool = False,
    quality: float = 1.0,
    speed: EncoderSpeed = EncoderSpeed.Squirrel,
    num_channels: int = 4,
    exif: bytes | None = None,
    xmp: bytes | None = None,
    icc: bytes | None = None,
    dtype: str | None = None,
    intensity_target: float | None = None,
) -> bytes: ...
async def async_decode_to_numpy(
    data: bytes,
    *,
    dtype: str | None = None,
) -> tuple[Metadata, npt.NDArray[np.uint8 | np.uint16 | np.float32]]: ...
async def async_encode_from_numpy(
    array: npt.NDArray[np.uint8 | np.uint16 | np.float32],
    *,
    lossless: bool = False,
    quality: float = 1.0,
    speed: EncoderSpeed = EncoderSpeed.Squirrel,
    exif: bytes | None = None,
    xmp: bytes | None = None,
    icc: bytes | None = None,
    intensity_target: float | None = None,
) -> bytes: ...
async def async_read(path: str | os.PathLike, *, dtype: str | None = None) -> tuple[Metadata, bytes]: ...
async def async_read_to_numpy(
    path: str | os.PathLike,
    *,
    dtype: str | None = None,
) -> tuple[Metadata, npt.NDArray[np.uint8 | np.uint16 | np.float32]]: ...
async def async_read_into(
    path: str | os.PathLike,
    out: npt.NDArray[np.uint8 | np.uint16 | np.float32],
) -> Metadata: ...
async def async_write(
    path: str | os.PathLike,
    data: bytes,
    width: int,
    height: int,
    *,
    lossless: bool = False,
    quality: float = 1.0,
    speed: EncoderSpeed = EncoderSpeed.Squirrel,
    num_channels: int = 4,
    exif: bytes | None = None,
    xmp: bytes | None = None,
    icc: bytes | None = None,
    dtype: str | None = None,
    intensity_target: float | None = None,
) -> int: ...
async def async_write_from_numpy(
    path: str | os.PathLike,
    array: npt.NDArray[np.uint8 | np.uint16 | np.float32],
    *,
    lossless: bool = False,
    quality: float = 1.0,
    speed: EncoderSpeed = EncoderSpeed.Squirrel,
    exif: bytes | None = None,
    xmp: bytes | None = None,
    icc: bytes | None = None,
    intensity_target: float | None = None,
) -> int: ...

# ===========================================================================
# JPEG — Async wrappers
# ===========================================================================

async def async_jpeg_probe(data: bytes) -> JpegInfo: ...
async def async_jpeg_probe_file(path: str | os.PathLike) -> JpegInfo: ...
async def async_jpeg_decode(data: bytes, *, channels: int | None = None) -> tuple[JpegInfo, bytes]: ...
async def async_jpeg_decode_into(data: bytes, out: npt.NDArray[np.uint8]) -> JpegInfo: ...
async def async_jpeg_encode(
    data: bytes,
    width: int,
    height: int,
    *,
    quality: int = 95,
    num_channels: int = 3,
) -> bytes: ...
async def async_jpeg_decode_to_numpy(
    data: bytes,
    *,
    channels: int | None = None,
) -> tuple[JpegInfo, np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]]: ...
async def async_jpeg_encode_from_numpy(
    array: npt.NDArray[np.uint8],
    *,
    quality: int = 95,
) -> bytes: ...
async def async_jpeg_read(path: str | os.PathLike, *, channels: int | None = None) -> tuple[JpegInfo, bytes]: ...
async def async_jpeg_read_to_numpy(
    path: str | os.PathLike,
    *,
    channels: int | None = None,
) -> tuple[JpegInfo, np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]]: ...
async def async_jpeg_read_into(path: str | os.PathLike, out: npt.NDArray[np.uint8]) -> JpegInfo: ...
async def async_jpeg_write(
    path: str | os.PathLike,
    data: bytes,
    width: int,
    height: int,
    *,
    quality: int = 95,
    num_channels: int = 3,
) -> int: ...
async def async_jpeg_write_from_numpy(
    path: str | os.PathLike,
    array: npt.NDArray[np.uint8],
    *,
    quality: int = 95,
) -> int: ...

# ===========================================================================
# JPEG ↔ JXL Lossless Transcoding — Sync bytes API
# ===========================================================================

def jpeg_to_jxl(data: bytes) -> bytes:
    """Losslessly transcode JPEG bytes → JXL bytes (bit-exact roundtrip)."""
    ...

def jxl_to_jpeg(data: bytes) -> bytes:
    """Reconstruct the original JPEG bytes from a JXL created via lossless transcoding."""
    ...

# ===========================================================================
# JPEG ↔ JXL Lossless Transcoding — Sync file I/O
# ===========================================================================

def jpeg_file_to_jxl(jpeg_path: str | os.PathLike, jxl_path: str | os.PathLike) -> int:
    """Losslessly transcode a JPEG file to JXL. Returns bytes written."""
    ...

def jxl_file_to_jpeg(jxl_path: str | os.PathLike, jpeg_path: str | os.PathLike) -> int:
    """Reconstruct original JPEG from a JXL file. Returns bytes written."""
    ...

# ===========================================================================
# JPEG ↔ JXL Lossless Transcoding — Async
# ===========================================================================

async def async_jpeg_to_jxl(data: bytes) -> bytes: ...
async def async_jxl_to_jpeg(data: bytes) -> bytes: ...
async def async_jpeg_file_to_jxl(
    jpeg_path: str | os.PathLike,
    jxl_path: str | os.PathLike,
) -> int: ...
async def async_jxl_file_to_jpeg(
    jxl_path: str | os.PathLike,
    jpeg_path: str | os.PathLike,
) -> int: ...

# ===========================================================================
# Unified Polymorphic I/O & Sniffing
# ===========================================================================

def sniff_bytes(header: bytes) -> str: ...
def sniff_stream(stream: io.IOBase) -> tuple[str, io.IOBase]: ...
def sniff_source(source: str | os.PathLike | bytes | io.IOBase) -> tuple[str, bytes]: ...

class PrefixedStream(io.RawIOBase):
    def __init__(self, prefix: bytes, raw_stream: io.IOBase) -> None: ...
    def read(self, size: int = -1) -> bytes: ...
    def readinto(self, b: bytearray | memoryview) -> int: ...
    def seekable(self) -> bool: ...
    def readable(self) -> bool: ...
    def writable(self) -> bool: ...
    def close(self) -> None: ...

def imread(
    source: str | os.PathLike | bytes | io.IOBase,
    *,
    dtype: str | None = None,
    channels: int | None = None,
) -> tuple[Metadata | JpegInfo, npt.NDArray[np.generic]]: ...
def imwrite(
    dest: str | os.PathLike | io.IOBase,
    image: npt.NDArray[np.generic],
    *,
    format: str | None = None,
    quality: float | None = None,
    lossless: bool = False,
    speed: EncoderSpeed = EncoderSpeed.Squirrel,
    **kwargs: Any,
) -> None: ...
def probe_image(
    source: str | os.PathLike | bytes | io.IOBase,
) -> Metadata | JpegInfo: ...

# ===========================================================================
# Ecosystem Bridges: Pillow & PyTorch
# ===========================================================================

def to_pil(
    image_or_array: npt.NDArray[np.generic] | bytes,
    metadata: Any = None,
    *,
    preserve_hdr: bool = True,
) -> Any: ...
def from_pil(image: Any) -> tuple[npt.NDArray[np.generic], dict[str, Any]]: ...
def to_tensor(
    array_or_bytes: npt.NDArray[np.generic] | bytes,
    *,
    permute_chw: bool = False,
    channels_last: bool = False,
    normalize: bool = False,
    device: Any = None,
) -> Any: ...
def from_tensor(tensor: Any) -> npt.NDArray[np.generic]: ...
def decode_into_tensor(
    data: bytes,
    tensor: Any,
    *,
    is_jpeg: bool | None = None,
) -> Metadata | JpegInfo: ...

# ===========================================================================
# High-Throughput Batch Operations
# ===========================================================================

def read_batch(
    sources: Sequence[str | os.PathLike | bytes],
    *,
    max_workers: int | None = None,
    dtype: str | None = None,
    channels: int | None = None,
) -> list[tuple[Metadata | JpegInfo, npt.NDArray[np.generic]]]: ...
def transcode_batch(
    sources: Sequence[str | os.PathLike],
    output_dir: str | os.PathLike,
    *,
    target_format: str = "jxl",
    max_workers: int | None = None,
) -> list[str]: ...

__version__: str
