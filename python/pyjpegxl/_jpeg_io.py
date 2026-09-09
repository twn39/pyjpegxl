"""File-level read/write helpers for JPEG images.

Thin wrappers around the core jpeg_encode/jpeg_decode functions
that handle file I/O.
"""

from __future__ import annotations

import os

import numpy as np

from pyjpegxl._pyjpegxl import (
    JpegInfo,
    jpeg_decode,
    jpeg_decode_into,
    jpeg_decode_to_numpy,
    jpeg_encode,
    jpeg_encode_from_numpy,
    jpeg_probe,
)


def jpeg_probe_file(path: str | os.PathLike) -> JpegInfo:
    """Fast metadata inspection of a JPEG file without decoding pixel data.

    Executes in sub-millisecond time with zero pixel buffer allocation.

    Args:
        path: Path to the .jpg/.jpeg file.

    Returns:
        JpegInfo containing width, height, channels, EXIF and ICC profile.
    """
    with open(path, "rb") as f:
        return jpeg_probe(f.read())


def jpeg_read(path: str | os.PathLike, *, channels: int | None = None) -> tuple[JpegInfo, bytes]:
    """Read a JPEG file and decode it to raw pixel bytes.

    Args:
        path: Path to the .jpg/.jpeg file.
        channels: Desired channel count (1=Gray, 3=RGB, 4=RGBA). Default is None
            (returns 1 for grayscale JPEG, 3 for color JPEG).

    Returns:
        A tuple of (JpegInfo, pixel bytes).
    """
    with open(path, "rb") as f:
        return jpeg_decode(f.read(), channels=channels)


def jpeg_read_to_numpy(path: str | os.PathLike, *, channels: int | None = None) -> tuple[JpegInfo, np.ndarray]:
    """Read a JPEG file and decode it to a NumPy array.

    Args:
        path: Path to the .jpg/.jpeg file.
        channels: Desired channel count (1=Gray, 3=RGB, 4=RGBA). Default is None.

    Returns:
        A tuple of (JpegInfo, ndarray of shape (H, W, C) dtype uint8).
    """
    with open(path, "rb") as f:
        return jpeg_decode_to_numpy(f.read(), channels=channels)


def jpeg_read_into(path: str | os.PathLike, out: np.ndarray) -> JpegInfo:
    """Read a JPEG file and decode directly into a preallocated writable NumPy array.

    Zero-allocation: eliminates output buffer allocation by decompressing
    directly into caller-provided memory.

    Args:
        path: Path to the .jpg/.jpeg file.
        out: Preallocated writable C-contiguous ndarray of dtype uint8,
            with matching dimensions (H, W) or (H, W, C).

    Returns:
        JpegInfo of the decoded image.
    """
    with open(path, "rb") as f:
        return jpeg_decode_into(f.read(), out)


def jpeg_write(
    path: str | os.PathLike,
    data: bytes,
    width: int,
    height: int,
    *,
    quality: int = 95,
    num_channels: int = 3,
) -> int:
    """Encode raw pixel data and write it to a JPEG file.

    Args:
        path: Destination file path. Parent directories are created automatically.
        data: Raw pixel bytes (uint8).
        width: Image width in pixels.
        height: Image height in pixels.
        quality: JPEG quality (1–100).
        num_channels: Number of channels (1=Gray, 3=RGB, 4=RGBA).

    Returns:
        Number of bytes written.
    """
    jpeg = jpeg_encode(data, width, height, quality=quality, num_channels=num_channels)
    out = os.fspath(path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        return f.write(jpeg)


def jpeg_write_from_numpy(
    path: str | os.PathLike,
    array: np.ndarray,
    *,
    quality: int = 95,
) -> int:
    """Encode a NumPy array and write it to a JPEG file.

    Automatically converts non-contiguous arrays to C-contiguous layout.

    Args:
        path: Destination file path. Parent directories are created automatically.
        array: Image as ndarray of shape (H, W) or (H, W, C), dtype uint8.
        quality: JPEG quality (1–100).

    Returns:
        Number of bytes written.
    """
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)

    jpeg = jpeg_encode_from_numpy(array, quality=quality)
    out = os.fspath(path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        return f.write(jpeg)
