"""High-throughput multi-threaded batch operations releasing Python GIL."""

from __future__ import annotations

import concurrent.futures
import os
from collections.abc import Sequence

import numpy as np

from pyjpegxl._io import jpeg_file_to_jxl, jxl_file_to_jpeg
from pyjpegxl._pyjpegxl import JpegInfo, Metadata
from pyjpegxl._unified import imread


def read_batch(
    sources: Sequence[str | os.PathLike | bytes],
    *,
    max_workers: int | None = None,
    dtype: str | None = None,
    channels: int | None = None,
) -> list[tuple[Metadata | JpegInfo, np.ndarray]]:
    """Batch-decode a sequence of image paths or bytes using a worker thread pool.

    Leverages pyjpegxl's internal GIL-release to achieve true multi-core parallel
    decoding throughput across images.

    Args:
        sources: Sequence of image file paths or raw bytes.
        max_workers: Number of worker threads. Defaults to CPU core count.
        dtype: Optional pixel data type for JXL ("uint8", "uint16", "float32").
        channels: Optional desired channel count for JPEG (1=Gray, 3=RGB, 4=RGBA).

    Returns:
        List of (Metadata | JpegInfo, ndarray) tuples matching the input order.
    """

    def _read_one(src: str | os.PathLike | bytes) -> tuple[Metadata | JpegInfo, np.ndarray]:
        return imread(src, dtype=dtype, channels=channels)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_read_one, sources))


def transcode_batch(
    sources: Sequence[str | os.PathLike],
    output_dir: str | os.PathLike,
    *,
    target_format: str = "jxl",
    max_workers: int | None = None,
) -> list[str]:
    """Batch lossless transcoding of images (e.g., JPEG to JXL or JXL to JPEG).

    Args:
        sources: Sequence of input image file paths.
        output_dir: Directory where transcoded files will be written.
        target_format: Target format ('jxl' or 'jpeg'). Default is 'jxl'.
        max_workers: Number of worker threads. Defaults to CPU core count.

    Returns:
        List of generated destination file paths.

    Raises:
        ValueError: If target_format is not supported.
    """
    fmt = target_format.lower().lstrip(".")
    if fmt not in ("jxl", "jpeg", "jpg"):
        raise ValueError(f"Unsupported target format: '{target_format}'. Expected 'jxl' or 'jpeg'")

    ext = ".jxl" if fmt == "jxl" else ".jpg"
    out_dir_path = os.fspath(output_dir)
    os.makedirs(out_dir_path, exist_ok=True)

    def _transcode_one(src: str | os.PathLike) -> str:
        src_path = os.fspath(src)
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        dst_path = os.path.join(out_dir_path, base_name + ext)
        if fmt == "jxl":
            jpeg_file_to_jxl(src_path, dst_path)
        else:
            jxl_file_to_jpeg(src_path, dst_path)
        return dst_path

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_transcode_one, sources))
