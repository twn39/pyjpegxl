"""File-level read/write helpers for JXL images.

These are thin wrappers around the core encode/decode functions
that handle file I/O so users don't have to.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pyjpegxl._pyjpegxl import (
    Metadata,
    EncoderSpeed,
    decode,
    encode,
    decode_to_numpy,
    encode_from_numpy,
)

if TYPE_CHECKING:
    import numpy as np


def read(path: str | os.PathLike) -> tuple[Metadata, bytes]:
    """Read a JXL file and decode it to raw pixel bytes.

    Args:
        path: Path to the .jxl file.

    Returns:
        A tuple of (Metadata, pixel bytes).
    """
    with open(path, "rb") as f:
        return decode(f.read())


def read_to_numpy(path: str | os.PathLike) -> tuple[Metadata, "np.ndarray"]:
    """Read a JXL file and decode it to a NumPy array.

    Args:
        path: Path to the .jxl file.

    Returns:
        A tuple of (Metadata, ndarray of shape (H, W, C) dtype uint8).
    """
    with open(path, "rb") as f:
        return decode_to_numpy(f.read())


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
    """Encode raw pixel data and write it to a JXL file.

    Args:
        path: Destination file path. Parent directories are created automatically.
        data: Raw pixel bytes (uint8).
        width: Image width in pixels.
        height: Image height in pixels.
        lossless: Use lossless compression.
        quality: Encoding quality (0.0–1.0). Ignored when lossless=True.
        speed: Encoder effort preset.
        num_channels: Number of channels (3=RGB, 4=RGBA, etc.).
        exif: Optional raw EXIF metadata bytes.
        xmp: Optional raw XMP metadata bytes.

    Returns:
        Number of bytes written.
    """
    jxl = encode(
        data,
        width,
        height,
        lossless=lossless,
        quality=quality,
        speed=speed,
        num_channels=num_channels,
        exif=exif,
        xmp=xmp,
    )
    out = os.fspath(path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        return f.write(jxl)


def write_from_numpy(
    path: str | os.PathLike,
    array: "np.ndarray",
    *,
    lossless: bool = False,
    quality: float = 1.0,
    speed: EncoderSpeed = EncoderSpeed.Squirrel,
    exif: bytes | None = None,
    xmp: bytes | None = None,
) -> int:
    """Encode a NumPy array and write it to a JXL file.

    Args:
        path: Destination file path. Parent directories are created automatically.
        array: Image as ndarray of shape (H, W, C), dtype uint8, C-contiguous.
        lossless: Use lossless compression.
        quality: Encoding quality (0.0–1.0). Ignored when lossless=True.
        speed: Encoder effort preset.
        exif: Optional raw EXIF metadata bytes.
        xmp: Optional raw XMP metadata bytes.

    Returns:
        Number of bytes written.
    """
    jxl = encode_from_numpy(
        array,
        lossless=lossless,
        quality=quality,
        speed=speed,
        exif=exif,
        xmp=xmp,
    )
    out = os.fspath(path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        return f.write(jxl)
