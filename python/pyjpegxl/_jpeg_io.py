"""File-level read/write helpers for JPEG images.

Thin wrappers around the core jpeg_encode/jpeg_decode functions
that handle file I/O.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pyjpegxl._pyjpegxl import (
    JpegInfo,
    jpeg_decode,
    jpeg_encode,
    jpeg_decode_to_numpy,
    jpeg_encode_from_numpy,
)

if TYPE_CHECKING:
    import numpy as np


def jpeg_read(path: str | os.PathLike) -> tuple[JpegInfo, bytes]:
    """Read a JPEG file and decode it to raw pixel bytes.

    Args:
        path: Path to the .jpg/.jpeg file.

    Returns:
        A tuple of (JpegInfo, pixel bytes).
    """
    with open(path, "rb") as f:
        return jpeg_decode(f.read())


def jpeg_read_to_numpy(path: str | os.PathLike) -> tuple[JpegInfo, "np.ndarray"]:
    """Read a JPEG file and decode it to a NumPy array.

    Args:
        path: Path to the .jpg/.jpeg file.

    Returns:
        A tuple of (JpegInfo, ndarray of shape (H, W, 3) dtype uint8).
    """
    with open(path, "rb") as f:
        return jpeg_decode_to_numpy(f.read())


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
    array: "np.ndarray",
    *,
    quality: int = 95,
) -> int:
    """Encode a NumPy array and write it to a JPEG file.

    Args:
        path: Destination file path. Parent directories are created automatically.
        array: Image as ndarray of shape (H, W, C), dtype uint8, C-contiguous.
        quality: JPEG quality (1–100).

    Returns:
        Number of bytes written.
    """
    jpeg = jpeg_encode_from_numpy(array, quality=quality)
    out = os.fspath(path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        return f.write(jpeg)
