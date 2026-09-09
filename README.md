<div align="center">

# pyjpegxl

[![PyPI version](https://img.shields.io/pypi/v/pyjpegxl.svg)](https://pypi.org/project/pyjpegxl/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyjpegxl.svg)](https://pypi.org/project/pyjpegxl/)
[![CI Status](https://github.com/twn39/pyjpegxl/actions/workflows/ci.yml/badge.svg)](https://github.com/twn39/pyjpegxl/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

</div>

Python bindings for **JPEG XL** and **JPEG** encoding/decoding, powered by [libjxl](https://github.com/libjxl/libjxl) and [libjpeg-turbo](https://libjpeg-turbo.org/). Both libraries are statically linked — no system dependencies required.

**Features**:
- **JPEG XL + JPEG**: Full encode/decode/file I/O for both formats in one package.
- **High Bit Depth & HDR**: Full support for 8-bit (`uint8`), 16-bit (`uint16`), and 32-bit floating point (`float32`) pixels with HDR peak brightness (`intensity_target`).
- **Color Profiles & Metadata**: Preserves and embeds ICC color profiles, EXIF, and XMP metadata blocks.
- **Fast Metadata Probing**: Sub-millisecond inspection (`probe`, `probe_file`) with zero pixel buffer allocation.
- **Zero-Allocation In-Place Decoding**: Decode directly into preallocated NumPy arrays (`decode_into`, `read_into`), eliminating buffer copying overhead.
- **Lossless Transcoding**: Reversibly transcode JPEGs into 20% smaller JXLs, and losslessly reconstruct the exact original JPEG bit-for-bit.
- **NumPy Zero-Copy**: Directly encode from and decode to `numpy.ndarray` (2D grayscale or 3D color/alpha).
- **True Concurrency & Thread Control**: Releases the Python GIL during heavy operations, with configurable thread counts (`set_num_threads`) to avoid thread oversubscription.
- **Async API**: First-class `async`/`await` support via `asyncio.to_thread`.
- **Performance**: Fastest-in-class multi-threaded encoding and decoding with statically linked libjxl and libjpeg-turbo.

## Installation

```bash
pip install pyjpegxl
```

*(Note: Pre-built wheels are currently only available for select platforms. If a wheel is not available, pip will try to build it from source. You will need a Rust toolchain installed.)*

### Build from source

Requires Rust toolchain and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/twn39/pyjpegxl && cd pyjpegxl
# Install dependencies and build extension in-place
uv sync
```

## Quick Start

### Basic Usage (Bytes API)

```python
import pyjpegxl

# Decode
with open("image.jxl", "rb") as f:
    meta, pixels = pyjpegxl.decode(f.read())

print(f"{meta.width}x{meta.height}, channels={meta.num_color_channels}, bits={meta.bits_per_sample}")

# Encode
jxl_data = pyjpegxl.encode(pixels, width=meta.width, height=meta.height)

# Custom Encode (Lossless, Falcon effort)
jxl_data = pyjpegxl.encode(
    pixels,
    width=meta.width,
    height=meta.height,
    lossless=True,
    speed=pyjpegxl.EncoderSpeed.Falcon,
)
```

### NumPy Zero-Copy & High Bit Depth (8-bit, 16-bit, float32)

Move raw pixel buffers to and from NumPy arrays instantly without Python-level allocations:

```python
import pyjpegxl
import numpy as np

with open("image.jxl", "rb") as f:
    # Auto-detects bit depth: returns uint8, uint16, or float32 ndarray (H, W, C)
    meta, arr = pyjpegxl.decode_to_numpy(f.read())

print(arr.shape, arr.dtype)  # e.g. (1080, 1920, 3), dtype('uint8')

# Encode directly from a NumPy array (supports 2D (H, W) or 3D (H, W, C))
jxl_data = pyjpegxl.encode_from_numpy(arr, quality=1.0)

# Encode 16-bit or float32 HDR arrays
hdr_arr = np.random.rand(1080, 1920, 3).astype(np.float32)
hdr_jxl = pyjpegxl.encode_from_numpy(hdr_arr, intensity_target=1000.0)  # 1000 nits
```

### Fast Lightweight Metadata Probing

Inspect dimensions, bit depth, channels, HDR brightness, and metadata in **sub-millisecond** time with **zero pixel buffer allocation**:

```python
import pyjpegxl

# Probe bytes directly
meta = pyjpegxl.probe(jxl_bytes)
print(meta.width, meta.height, meta.bits_per_sample, meta.intensity_target)

# Probe file without reading pixel payload
meta = pyjpegxl.probe_file("large_image.jxl")
```

### Zero-Allocation In-Place Decoding

Write decoded pixels directly into a caller-preallocated writable NumPy array:

```python
import pyjpegxl
import numpy as np

# Preallocate buffer (e.g. reused in video processing loops or memory maps)
out_buffer = np.empty((1080, 1920, 3), dtype=np.uint8)

# Decodes directly into out_buffer with zero intermediate copying
meta = pyjpegxl.read_into("image.jxl", out_buffer)
```

### File I/O API

Read and write JXL files directly:

```python
import pyjpegxl

# Read a JXL file to a NumPy array (uint8 / uint16 / float32)
meta, arr = pyjpegxl.read_to_numpy("image.jxl")
print(arr.shape, arr.dtype)

# Write a NumPy array to a JXL file
pyjpegxl.write_from_numpy("output.jxl", arr, lossless=True)

# Bytes-level file I/O
meta, pixels = pyjpegxl.read("image.jxl")
pyjpegxl.write(
    "output.jxl",
    pixels,
    width=meta.width,
    height=meta.height,
    num_channels=meta.num_color_channels + int(meta.has_alpha),
)
```

### Color Profiles (ICC) & Metadata (EXIF/XMP)

Preserve, extract, or embed color profiles and metadata:

```python
import pyjpegxl

# Read and extract ICC profile / EXIF
meta, arr = pyjpegxl.read_to_numpy("photo.jxl")
icc_bytes = meta.icc  # raw ICC profile bytes (or None)
exif_bytes = meta.exif  # raw EXIF bytes (or None)

# Encode with custom ICC and EXIF metadata
pyjpegxl.write_from_numpy(
    "tagged.jxl",
    arr,
    icc=icc_bytes,
    exif=exif_bytes,
)
```

### Concurrency & Thread Pool Control

Control the internal multi-threading runner to prevent CPU oversubscription in worker pools:

```python
import pyjpegxl

# Get current setting (0 = auto-detect logical CPUs)
print(pyjpegxl.get_num_threads())

# Set to pure single-threaded execution (bypasses runner, 0 thread overhead)
# Ideal when running inside concurrent.futures.ThreadPoolExecutor or Gunicorn
pyjpegxl.set_num_threads(1)

# Set to specific thread count
pyjpegxl.set_num_threads(4)

# Reset back to auto
pyjpegxl.set_num_threads(0)
```

### Async API

Perfect for high-concurrency web servers like FastAPI or Starlette:

```python
import asyncio
import pyjpegxl


async def process_image():
    with open("image.jxl", "rb") as f:
        data = f.read()

    # Fast async metadata probing
    meta = await pyjpegxl.async_probe(data)

    # Non-blocking decode
    meta, arr = await pyjpegxl.async_decode_to_numpy(data)

    # Non-blocking encode
    out_jxl = await pyjpegxl.async_encode_from_numpy(arr)

    return out_jxl


asyncio.run(process_image())
```

### JPEG Quick Start

```python
import pyjpegxl

# Read JPEG → NumPy array
info, arr = pyjpegxl.jpeg_read_to_numpy("photo.jpg")
print(arr.shape)  # (H, W, 3)

# Write NumPy array → JPEG file
pyjpegxl.jpeg_write_from_numpy("output.jpg", arr, quality=95)

# In-memory encode/decode
jpeg_data = pyjpegxl.jpeg_encode_from_numpy(arr, quality=90)
info, decoded = pyjpegxl.jpeg_decode_to_numpy(jpeg_data)
```

### Direct JPEG ↔ JXL Lossless Transcoding

Repack JPEGs into smaller JXL files losslessly without ever decoding pixels, and revert them exactly bit-for-bit:

```python
import pyjpegxl

with open("photo.jpg", "rb") as f:
    jpeg_bytes = f.read()

# Transcode directly (lossless, usually 20% smaller)
jxl_bytes = pyjpegxl.jpeg_to_jxl(jpeg_bytes)

# Reconstruct the exact original JPEG bit-for-bit
restored_jpeg_bytes = pyjpegxl.jxl_to_jpeg(jxl_bytes)
assert jpeg_bytes == restored_jpeg_bytes

# Also available natively for File I/O
pyjpegxl.jpeg_file_to_jxl("photo.jpg", "smaller_version.jxl")
pyjpegxl.jxl_file_to_jpeg("smaller_version.jxl", "restored_photo.jpg")
```

## Concurrency and Performance

`pyjpegxl` natively releases the Global Interpreter Lock (GIL) and engages `ThreadsRunner` from `libjxl`. If you use `concurrent.futures.ThreadPoolExecutor` or `asyncio.gather()`, multiple images will encode and decode in parallel without blocking the main Python thread.

### Benchmarks (MacBook M-Series)

Benchmark processing `images/test.jpg` (decoded to Numpy arrays) among Python JXL wrappers on identical visual quality settings:

| Library | Decode Time (ms) | Peak Python Mem | Encode Time (ms) | Peak Python Mem |
| :--- | :--- | :--- | :--- | :--- |
| **`pyjpegxl`** | **36.78** | **0.0 MB** | **184.47** | **0.4 MB** |
| `pylibjxl` | 114.86 | 0.0 MB | 366.54 | 0.5 MB |
| `pillow-jxl` | 35.56 | 11.9 MB | 185.22 | 11.3 MB |

> `pyjpegxl` is fundamentally the fastest encoder and decoder, while matching the flawless memory performance of `pylibjxl` due to its zero-copy `IntoPyArray` bridging.

## API Reference

### Metadata Probing (Fast, Zero Pixel Allocation)
- `probe(data: bytes) -> Metadata`
- `probe_file(path: str | os.PathLike) -> Metadata`
- `async_probe(data: bytes) -> Metadata`
- `async_probe_file(path: str | os.PathLike) -> Metadata`

### In-Place Zero-Allocation Decoding
- `decode_into(data: bytes, out: np.ndarray) -> Metadata`
- `read_into(path: str | os.PathLike, out: np.ndarray) -> Metadata`
- `async_decode_into(data: bytes, out: np.ndarray) -> Metadata`
- `async_read_into(path: str | os.PathLike, out: np.ndarray) -> Metadata`

### Concurrency & Thread Control
- `set_num_threads(num_threads: int) -> None`: `0` for auto-detect, `1` for single-thread bypass, `>1` for explicit worker count.
- `get_num_threads() -> int`

### JXL Bytes API
- `decode(data: bytes, *, dtype: str | None = None) -> tuple[Metadata, bytes]`
- `encode(data, width, height, *, lossless=False, quality=1.0, speed=EncoderSpeed.Squirrel, num_channels=4, exif=None, xmp=None, icc=None, dtype=None, intensity_target=None) -> bytes`

### JXL NumPy API
- `decode_to_numpy(data: bytes, *, dtype: str | None = None) -> tuple[Metadata, np.ndarray]`
- `encode_from_numpy(array: np.ndarray, *, lossless=False, quality=1.0, speed=EncoderSpeed.Squirrel, exif=None, xmp=None, icc=None, intensity_target=None) -> bytes`

### JXL File I/O API
- `read(path, *, dtype: str | None = None) -> tuple[Metadata, bytes]`
- `read_to_numpy(path, *, dtype: str | None = None) -> tuple[Metadata, np.ndarray]`
- `write(path, data, width, height, **kwargs) -> int`
- `write_from_numpy(path, array, **kwargs) -> int`

### JPEG Bytes API
- `jpeg_decode(data: bytes) -> tuple[JpegInfo, bytes]`
- `jpeg_encode(data, width, height, *, quality=95, num_channels=3) -> bytes`

### JPEG NumPy API
- `jpeg_decode_to_numpy(data: bytes) -> tuple[JpegInfo, np.ndarray]`
- `jpeg_encode_from_numpy(array: np.ndarray, *, quality=95) -> bytes`

### JPEG File I/O API
- `jpeg_read(path) -> tuple[JpegInfo, bytes]`
- `jpeg_read_to_numpy(path) -> tuple[JpegInfo, np.ndarray]`
- `jpeg_write(path, data, width, height, **kwargs) -> int`
- `jpeg_write_from_numpy(path, array, **kwargs) -> int`

### Transcoding API
- `jpeg_to_jxl(data: bytes) -> bytes`
- `jxl_to_jpeg(data: bytes) -> bytes`
- `jpeg_file_to_jxl(jpeg_path: str, jxl_path: str) -> int`
- `jxl_file_to_jpeg(jxl_path: str, jpeg_path: str) -> int`

### Async API
All sync functions have async variants prefixed with `async_` (JXL) or `async_jpeg_` (JPEG).

### Types
- `Metadata`: Image metadata with properties:
  - `width: int`, `height: int`
  - `num_color_channels: int`, `has_alpha: bool`
  - `bits_per_sample: int` (e.g. 8, 16, 32)
  - `intensity_target: float` (HDR peak nits)
  - `min_nits: float`
  - `icc: bytes | None`, `exif: bytes | None`, `xmp: bytes | None`
- `JpegInfo`: JPEG image dimensions (`width`, `height`, `num_channels`).
- `EncoderSpeed`: JXL compression effort (`Lightning` → `Tortoise`).

## License

BSD 3-Clause
