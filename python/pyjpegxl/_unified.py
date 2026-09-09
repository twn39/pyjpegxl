"""Unified polymorphic I/O for JPEG and JPEG XL images."""

from __future__ import annotations

import io
import os
from typing import Any

import numpy as np

from pyjpegxl._pyjpegxl import (
    EncoderSpeed,
    JpegInfo,
    Metadata,
    decode_to_numpy,
    encode_from_numpy,
    jpeg_decode_to_numpy,
    jpeg_encode_from_numpy,
    jpeg_probe,
    probe,
)
from pyjpegxl._sniff import sniff_source


def imread(
    source: str | os.PathLike | bytes | io.IOBase,
    *,
    dtype: str | None = None,
    channels: int | None = None,
) -> tuple[Metadata | JpegInfo, np.ndarray]:
    """Polymorphic image reader that auto-detects format (JXL or JPEG).

    Accepts file paths, raw bytes, bytearrays, memoryviews, or readable streams
    (including non-seekable streams).

    Args:
        source: Image source (path, bytes, or binary stream).
        dtype: Optional pixel data type for JXL ("uint8", "uint16", "float32").
        channels: Optional desired channel count for JPEG (1=Gray, 3=RGB, 4=RGBA).

    Returns:
        Tuple of (Metadata | JpegInfo, ndarray with image pixels).

    Raises:
        ValueError: If format is neither valid JPEG nor JPEG XL.
        TypeError: If source type is not supported.
    """
    fmt, data = sniff_source(source)
    if fmt == "jxl":
        return decode_to_numpy(data, dtype=dtype)
    elif fmt == "jpeg":
        return jpeg_decode_to_numpy(data, channels=channels)
    else:
        raise ValueError("Unsupported or corrupted image format: neither JPEG nor JPEG XL magic header found")


def imwrite(
    dest: str | os.PathLike | io.IOBase,
    image: np.ndarray,
    *,
    format: str | None = None,
    quality: float | None = None,
    lossless: bool = False,
    speed: EncoderSpeed = EncoderSpeed.Squirrel,
    **kwargs: Any,
) -> None:
    """Polymorphic image writer saving to JXL or JPEG.

    Format is auto-inferred from destination filename extension (e.g. '.jxl', '.jpg', '.jpeg')
    or can be explicitly passed via `format`.

    Args:
        dest: Output file path or writable binary stream.
        image: NumPy array representing image pixels (H, W) or (H, W, C).
        format: Optional format specification ('jxl' or 'jpeg').
        quality: Compression quality. For JXL, 0.0=lossless, <=15 Butteraugli distance,
            >15 standard percentage. For JPEG, integer 1-100 (default 90).
        lossless: Lossless compression (for JXL). Default is False.
        speed: Compression effort for JXL.
        **kwargs: Additional metadata parameters (exif, icc, xmp).

    Raises:
        ValueError: If format cannot be determined or is unsupported.
    """
    fmt = format.lower().lstrip(".") if format else None
    if not fmt:
        if isinstance(dest, (str, os.PathLike)):
            ext = os.path.splitext(os.fspath(dest))[1].lower().lstrip(".")
            if ext in ("jxl", "jpeg", "jpg"):
                fmt = "jpeg" if ext in ("jpg", "jpeg") else "jxl"
            else:
                raise ValueError(f"Cannot infer image format from extension: '{ext}'")
        else:
            raise ValueError("Must explicitly specify 'format' when writing to a stream")

    if fmt == "jxl":
        jxl_quality = quality if quality is not None else (0.0 if lossless else 1.0)
        encoded = encode_from_numpy(
            image,
            lossless=lossless,
            quality=jxl_quality,
            speed=speed,
            exif=kwargs.get("exif"),
            xmp=kwargs.get("xmp"),
            icc=kwargs.get("icc"),
        )
    elif fmt in ("jpeg", "jpg"):
        jpeg_quality = int(quality) if quality is not None else 90
        encoded = jpeg_encode_from_numpy(
            image,
            quality=jpeg_quality,
        )
    else:
        raise ValueError(f"Unsupported format: '{fmt}'. Expected 'jxl' or 'jpeg'")

    if isinstance(dest, (str, os.PathLike)):
        out_path = os.fspath(dest)
        parent_dir = os.path.dirname(out_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(encoded)
    elif hasattr(dest, "write"):
        dest.write(encoded)
    else:
        raise TypeError(f"Unsupported destination type: {type(dest).__name__}")


def probe_image(
    source: str | os.PathLike | bytes | io.IOBase,
) -> Metadata | JpegInfo:
    """Sub-millisecond metadata inspection for any supported image format.

    Args:
        source: Image file path, raw bytes, or stream.

    Returns:
        Metadata (for JXL) or JpegInfo (for JPEG).
    """
    fmt, data = sniff_source(source)
    if fmt == "jxl":
        return probe(data)
    elif fmt == "jpeg":
        return jpeg_probe(data)
    else:
        raise ValueError("Unsupported or corrupted image format: neither JPEG nor JPEG XL magic header found")
