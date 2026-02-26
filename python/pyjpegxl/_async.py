"""Async wrappers for pyjpegxl encode/decode functions.

These use `asyncio.to_thread` to run the GIL-released sync functions
in a separate thread, enabling true concurrent I/O and codec operations.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from pyjpegxl._pyjpegxl import (
    Metadata,
    EncoderSpeed,
    decode,
    encode,
    decode_to_numpy,
    encode_from_numpy,
    JpegInfo,
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

if TYPE_CHECKING:
    import numpy as np


# ---------------------------------------------------------------------------
# JXL async — codec
# ---------------------------------------------------------------------------


async def async_decode(data: bytes) -> tuple[Metadata, bytes]:
    """Async decode JXL bytes → (Metadata, pixel bytes)."""
    return await asyncio.to_thread(decode, data)


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
) -> bytes:
    """Async encode pixel bytes → JXL bytes."""
    return await asyncio.to_thread(
        encode,
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


async def async_decode_to_numpy(data: bytes) -> tuple[Metadata, "np.ndarray"]:
    """Async decode JXL bytes → (Metadata, numpy.ndarray)."""
    return await asyncio.to_thread(decode_to_numpy, data)


async def async_encode_from_numpy(
    array: "np.ndarray",
    *,
    lossless: bool = False,
    quality: float = 1.0,
    speed: EncoderSpeed = EncoderSpeed.Squirrel,
    exif: bytes | None = None,
    xmp: bytes | None = None,
) -> bytes:
    """Async encode numpy.ndarray → JXL bytes."""
    return await asyncio.to_thread(
        encode_from_numpy,
        array,
        lossless=lossless,
        quality=quality,
        speed=speed,
        exif=exif,
        xmp=xmp,
    )


# ---------------------------------------------------------------------------
# JXL async — file I/O
# ---------------------------------------------------------------------------


async def async_read(path: str | os.PathLike) -> tuple[Metadata, bytes]:
    """Async read a JXL file → (Metadata, pixel bytes)."""
    return await asyncio.to_thread(read, path)


async def async_read_to_numpy(path: str | os.PathLike) -> tuple[Metadata, "np.ndarray"]:
    """Async read a JXL file → (Metadata, numpy.ndarray)."""
    return await asyncio.to_thread(read_to_numpy, path)


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
) -> int:
    """Async encode pixel bytes and write to a JXL file."""
    return await asyncio.to_thread(
        write,
        path,
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


async def async_write_from_numpy(
    path: str | os.PathLike,
    array: "np.ndarray",
    *,
    lossless: bool = False,
    quality: float = 1.0,
    speed: EncoderSpeed = EncoderSpeed.Squirrel,
    exif: bytes | None = None,
    xmp: bytes | None = None,
) -> int:
    """Async encode numpy.ndarray and write to a JXL file."""
    return await asyncio.to_thread(
        write_from_numpy,
        path,
        array,
        lossless=lossless,
        quality=quality,
        speed=speed,
        exif=exif,
        xmp=xmp,
    )


# ---------------------------------------------------------------------------
# JPEG async — codec
# ---------------------------------------------------------------------------


async def async_jpeg_decode(data: bytes) -> tuple[JpegInfo, bytes]:
    """Async decode JPEG bytes → (JpegInfo, pixel bytes)."""
    return await asyncio.to_thread(jpeg_decode, data)


async def async_jpeg_encode(
    data: bytes,
    width: int,
    height: int,
    *,
    quality: int = 95,
    num_channels: int = 3,
) -> bytes:
    """Async encode pixel bytes → JPEG bytes."""
    return await asyncio.to_thread(
        jpeg_encode,
        data,
        width,
        height,
        quality=quality,
        num_channels=num_channels,
    )


async def async_jpeg_decode_to_numpy(data: bytes) -> tuple[JpegInfo, "np.ndarray"]:
    """Async decode JPEG bytes → (JpegInfo, numpy.ndarray)."""
    return await asyncio.to_thread(jpeg_decode_to_numpy, data)


async def async_jpeg_encode_from_numpy(
    array: "np.ndarray",
    *,
    quality: int = 95,
) -> bytes:
    """Async encode numpy.ndarray → JPEG bytes."""
    return await asyncio.to_thread(
        jpeg_encode_from_numpy,
        array,
        quality=quality,
    )


# ---------------------------------------------------------------------------
# JPEG async — file I/O
# ---------------------------------------------------------------------------


async def async_jpeg_read(path: str | os.PathLike) -> tuple[JpegInfo, bytes]:
    """Async read a JPEG file → (JpegInfo, pixel bytes)."""
    return await asyncio.to_thread(jpeg_read, path)


async def async_jpeg_read_to_numpy(path: str | os.PathLike) -> tuple[JpegInfo, "np.ndarray"]:
    """Async read a JPEG file → (JpegInfo, numpy.ndarray)."""
    return await asyncio.to_thread(jpeg_read_to_numpy, path)


async def async_jpeg_write(
    path: str | os.PathLike,
    data: bytes,
    width: int,
    height: int,
    *,
    quality: int = 95,
    num_channels: int = 3,
) -> int:
    """Async encode pixel bytes and write to a JPEG file."""
    return await asyncio.to_thread(
        jpeg_write,
        path,
        data,
        width,
        height,
        quality=quality,
        num_channels=num_channels,
    )


async def async_jpeg_write_from_numpy(
    path: str | os.PathLike,
    array: "np.ndarray",
    *,
    quality: int = 95,
) -> int:
    """Async encode numpy.ndarray and write to a JPEG file."""
    return await asyncio.to_thread(
        jpeg_write_from_numpy,
        path,
        array,
        quality=quality,
    )
