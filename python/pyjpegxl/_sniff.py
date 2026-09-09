"""Image format sniffing and non-seekable stream wrapping utilities."""

from __future__ import annotations

import io
import os
from typing import Literal

ImageFormat = Literal["jpeg", "jxl", "unknown"]

# Magic signatures
JPEG_MAGIC = b"\xff\xd8\xff"
JXL_CONTAINER_MAGIC = b"\x00\x00\x00\x0cJXL \r\n\x87\n"  # 12 bytes
JXL_CODESTREAM_MAGIC = b"\xff\x0a"  # 2 bytes


def sniff_bytes(header: bytes) -> ImageFormat:
    """Sniff image format from raw bytes prefix (at least 12 bytes recommended).

    Args:
        header: Leading bytes of image data.

    Returns:
        "jpeg", "jxl", or "unknown".
    """
    if header.startswith(JPEG_MAGIC):
        return "jpeg"
    if header.startswith(JXL_CONTAINER_MAGIC) or header.startswith(JXL_CODESTREAM_MAGIC):
        return "jxl"
    return "unknown"


class PrefixedStream(io.RawIOBase):
    """Wrapper that prepends buffered bytes back to a stream (safe for non-seekable streams)."""

    def __init__(self, prefix: bytes, raw_stream: io.IOBase) -> None:
        super().__init__()
        self._prefix_io = io.BytesIO(prefix)
        self._raw_stream = raw_stream

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        prefix_data = self._prefix_io.read(size)
        if size > 0 and len(prefix_data) == size:
            return prefix_data
        needed = -1 if size < 0 else size - len(prefix_data)
        raw_data = self._raw_stream.read(needed)
        return prefix_data + (raw_data or b"")

    def readinto(self, b: bytearray | memoryview) -> int:  # type: ignore[override]
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def seekable(self) -> bool:
        return False

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def close(self) -> None:
        try:
            self._prefix_io.close()
        finally:
            self._raw_stream.close()


def sniff_stream(stream: io.IOBase) -> tuple[ImageFormat, io.IOBase]:
    """Sniff image format from an open stream without losing data.

    Utilizes `.peek()` if available, or reads and seeks back.
    If the stream is not seekable and does not support peek, returns
    a `PrefixedStream` wrapping the consumed bytes and the original stream.

    Args:
        stream: A readable binary stream.

    Returns:
        Tuple of (ImageFormat, usable_stream).
    """
    # 1. Try peek() if available (BufferedReader / BytesIO-like)
    if hasattr(stream, "peek"):
        try:
            header = stream.peek(32)[:32]  # type: ignore[attr-defined]
            fmt = sniff_bytes(header)
            return fmt, stream
        except Exception:
            pass

    # 2. Try read() + seek(0) if seekable
    try:
        if stream.seekable():
            pos = stream.tell()
            header = stream.read(32)
            stream.seek(pos)
            fmt = sniff_bytes(header)
            return fmt, stream
    except (io.UnsupportedOperation, AttributeError, OSError):
        pass

    # 3. Non-seekable stream without peek: read prefix and wrap with PrefixedStream
    header = stream.read(32)
    fmt = sniff_bytes(header)
    wrapped = PrefixedStream(header, stream)
    return fmt, wrapped


def sniff_source(source: str | os.PathLike | bytes | io.IOBase) -> tuple[ImageFormat, bytes]:
    """Sniff format and resolve source to full in-memory bytes.

    Args:
        source: File path, bytes, or binary stream.

    Returns:
        Tuple of (ImageFormat, full_bytes).
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as f:
            data = f.read()
    elif isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
    elif isinstance(source, io.IOBase):
        fmt, stream = sniff_stream(source)
        data = stream.read()
        if fmt != "unknown":
            return fmt, data
    else:
        raise TypeError(f"Unsupported source type: {type(source).__name__}")

    return sniff_bytes(data[:32]), data
