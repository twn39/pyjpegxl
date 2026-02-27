"""Type stubs for pyjpegxl."""

from __future__ import annotations

import os
from enum import IntEnum
from typing import TYPE_CHECKING

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
    exif: bytes | None
    xmp: bytes | None

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

# ===========================================================================
# JXL — Sync bytes API
# ===========================================================================

def decode(data: bytes) -> tuple[Metadata, bytes]:
    """Decode JXL bytes → (Metadata, raw pixel bytes u8)."""
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
) -> bytes:
    """Encode raw pixel bytes → JXL bytes."""
    ...

# ===========================================================================
# JXL — Sync NumPy API (zero-copy)
# ===========================================================================

def decode_to_numpy(data: bytes) -> tuple[Metadata, np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]]:
    """Decode JXL bytes → (Metadata, ndarray shape (H,W,C) dtype uint8). Zero-copy."""
    ...

def encode_from_numpy(
    array: npt.NDArray[np.uint8],
    *,
    lossless: bool = False,
    quality: float = 1.0,
    speed: EncoderSpeed = EncoderSpeed.Squirrel,
    exif: bytes | None = None,
    xmp: bytes | None = None,
) -> bytes:
    """Encode ndarray (H,W,C) uint8 → JXL bytes. Zero-copy read."""
    ...

# ===========================================================================
# JXL — Sync file I/O
# ===========================================================================

def read(path: str | os.PathLike) -> tuple[Metadata, bytes]:
    """Read a JXL file → (Metadata, raw pixel bytes)."""
    ...

def read_to_numpy(path: str | os.PathLike) -> tuple[Metadata, np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]]:
    """Read a JXL file → (Metadata, ndarray shape (H,W,C) dtype uint8)."""
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
) -> int:
    """Encode raw pixel bytes and write to a JXL file. Returns bytes written."""
    ...

def write_from_numpy(
    path: str | os.PathLike,
    array: npt.NDArray[np.uint8],
    *,
    lossless: bool = False,
    quality: float = 1.0,
    speed: EncoderSpeed = EncoderSpeed.Squirrel,
    exif: bytes | None = None,
    xmp: bytes | None = None,
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

def jpeg_decode_to_numpy(data: bytes) -> tuple[JpegInfo, np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]]:
    """Decode JPEG bytes → (JpegInfo, ndarray shape (H,W,C) dtype uint8)."""
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

def jpeg_read(path: str | os.PathLike) -> tuple[JpegInfo, bytes]:
    """Read a JPEG file → (JpegInfo, raw pixel bytes)."""
    ...

def jpeg_read_to_numpy(
    path: str | os.PathLike,
) -> tuple[JpegInfo, np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]]:
    """Read a JPEG file → (JpegInfo, ndarray shape (H,W,C) dtype uint8)."""
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

async def async_decode(data: bytes) -> tuple[Metadata, bytes]: ...
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
) -> bytes: ...
async def async_decode_to_numpy(
    data: bytes,
) -> tuple[Metadata, np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]]: ...
async def async_encode_from_numpy(
    array: npt.NDArray[np.uint8],
    *,
    lossless: bool = False,
    quality: float = 1.0,
    speed: EncoderSpeed = EncoderSpeed.Squirrel,
    exif: bytes | None = None,
    xmp: bytes | None = None,
) -> bytes: ...
async def async_read(path: str | os.PathLike) -> tuple[Metadata, bytes]: ...
async def async_read_to_numpy(
    path: str | os.PathLike,
) -> tuple[Metadata, np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]]: ...
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
) -> int: ...
async def async_write_from_numpy(
    path: str | os.PathLike,
    array: npt.NDArray[np.uint8],
    *,
    lossless: bool = False,
    quality: float = 1.0,
    speed: EncoderSpeed = EncoderSpeed.Squirrel,
    exif: bytes | None = None,
    xmp: bytes | None = None,
) -> int: ...

# ===========================================================================
# JPEG — Async wrappers
# ===========================================================================

async def async_jpeg_decode(data: bytes) -> tuple[JpegInfo, bytes]: ...
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
) -> tuple[JpegInfo, np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]]: ...
async def async_jpeg_encode_from_numpy(
    array: npt.NDArray[np.uint8],
    *,
    quality: int = 95,
) -> bytes: ...
async def async_jpeg_read(path: str | os.PathLike) -> tuple[JpegInfo, bytes]: ...
async def async_jpeg_read_to_numpy(
    path: str | os.PathLike,
) -> tuple[JpegInfo, np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]]: ...
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
