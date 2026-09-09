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
- **Unified Polymorphic I/O & Sniffing**: `imread`, `imwrite`, and `probe_image` auto-detect formats (JXL container, JXL codestream, or JPEG). Seamlessly handles file paths, raw bytes, and non-seekable streams (`PrefixedStream`).
- **Pillow (PIL.Image) Deep Bridge & Plugin**: `to_pil()` and `from_pil()` zero-copy conversions with full ICC/EXIF metadata preservation. Built-in Pillow ImageFile plugin with automatic `entry_points` registration (`Image.open("pic.jxl")`).
- **PyTorch Tensor Zero-Copy & DataLoader Acceleration**: `to_tensor()` and `from_tensor()` preserving contiguous HWC memory layout; `decode_into_tensor()` enables zero-allocation in-place decoding into Pinned Memory or cross-process shared memory tensors.
- **High-Throughput Batch Pipeline**: `read_batch()` and `transcode_batch()` multi-threaded pipelines releasing the Python GIL for near-linear CPU core scaling.
- **Adaptive Quality Control**: Supports both Butteraugli distance (`<=15.0`, 0=lossless) and standard percentage quality (`>15.0`, e.g., 90, 95).
- **High Bit Depth & HDR**: Full support for 8-bit (`uint8`), 16-bit (`uint16`), and 32-bit floating point (`float32`) pixels with HDR peak brightness (`intensity_target`).
- **Color Profiles & Metadata**: Preserves and embeds ICC color profiles, EXIF, and XMP metadata blocks.
- **Fast Metadata Probing**: Sub-millisecond inspection (`probe`, `probe_file`, `probe_image`) with zero pixel buffer allocation.
- **Zero-Allocation In-Place Decoding**: Decode directly into preallocated NumPy arrays (`decode_into`, `read_into`), eliminating buffer copying overhead.
- **Lossless Transcoding**: Reversibly transcode JPEGs into 20% smaller JXLs, and losslessly reconstruct the exact original JPEG bit-for-bit.
- **PEP 561 Compliant**: Bundled `py.typed` marker and comprehensive `.pyi` type stubs.
- **Async API**: First-class `async`/`await` support via `asyncio.to_thread`.

## Installation

```bash
# Core package (minimal dependencies: only numpy)
pip install pyjpegxl

# With Pillow plugin support
pip install "pyjpegxl[pillow]"

# With PyTorch bridge support
pip install "pyjpegxl[torch]"

# Full ecosystem support
pip install "pyjpegxl[all]"
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

### Unified Polymorphic I/O (`imread` / `imwrite` / `probe_image`)

Automatically detects JPEG or JPEG XL (container or bare codestream) from paths, raw bytes, or streams:

```python
import pyjpegxl

# Auto-detects format and decodes to NumPy array
meta, arr = pyjpegxl.imread("photo.jxl")  # or photo.jpg, BytesIO, etc.
print(f"{meta.width}x{meta.height}, shape={arr.shape}")

# Auto-infers format from extension and writes directly
pyjpegxl.imwrite("compressed.jxl", arr, quality=1.0)  # Butteraugli quality
pyjpegxl.imwrite("web.jpg", arr, quality=90)  # Standard JPEG quality

# Fast sub-millisecond format inspection
info = pyjpegxl.probe_image("unknown_file")
```

### Pillow (PIL.Image) Integration & Plugin

Zero-copy bridge with full color profile (ICC) and EXIF preservation:

```python
from PIL import Image
import pyjpegxl

# 1. Native Pillow Plugin (works automatically via entry points)
# Directly open and save JXL images with standard Pillow!
img = Image.open("photo.jxl")
img.save("output.jxl", "JXL", quality=1.0)

# 2. Direct zero-copy bridge
meta, arr = pyjpegxl.imread("photo.jxl")
pil_img = pyjpegxl.to_pil(arr, metadata=meta)  # Attaches ICC & EXIF

# Extract C-contiguous NumPy array and metadata dict from PIL
arr, meta_dict = pyjpegxl.from_pil(pil_img)
```

### PyTorch Tensor & DataLoader Optimization

True zero-copy tensor wrapping and zero-IPC in-place decoding:

```python
import pyjpegxl
import torch

# 1. Zero-copy wrapping (default HWC layout avoids hidden memcpy/permute overhead)
meta, arr = pyjpegxl.imread("sample.jxl")
tensor = pyjpegxl.to_tensor(arr)  # Contiguous HWC Tensor sharing memory

# Optional: permute to (C, H, W) or normalize to [0.0, 1.0]
tensor_chw = pyjpegxl.to_tensor(arr, permute_chw=True, normalize=True)

# 2. High-performance DataLoader: direct decode into Pinned / Shared Memory
# Eliminates buffer allocation and multi-process IPC serialization overhead
pinned_tensor = torch.empty((1080, 1920, 3), dtype=torch.uint8, pin_memory=True)
pyjpegxl.decode_into_tensor(jxl_bytes, pinned_tensor)
```

### High-Throughput Batch Processing

Releases the Python GIL to achieve near-linear multi-core CPU scaling:

```python
import pyjpegxl

# Parallel decode a batch of images across CPU cores
files = ["img1.jxl", "img2.jpg", "img3.jxl"]
results = pyjpegxl.read_batch(files, max_workers=8)

# Parallel lossless dataset transcoding (e.g. JPEG to 20% smaller JXL)
transcoded = pyjpegxl.transcode_batch(["a.jpg", "b.jpg"], output_dir="jxl_dataset", target_format="jxl")
```

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

### Benchmarks & Competitor Comparison

Comprehensive benchmarks run on Apple Silicon (10 Cores), Python 3.12, evaluating real-world 1440×960 RGB images across **`pyjpegxl`**, **`pylibjxl` (v0.5.0)**, and **`pillow-jxl-plugin` (v1.3.8) / `Pillow`**.

> **Fairness & Methodology**: All libraries were tested under strictly matched CPU thread counts (10 worker threads) and identical compression effort tiers. Every benchmark performs warmup iterations to eliminate JIT/cold-cache bias and reports median (P50) latencies alongside industry-standard **MP/s** (Megapixels/second) throughput and Python heap allocation.

#### 1. JPEG XL Codec Comparison (1440×960 RGB, 10 Threads)

| Benchmark Scenario | `pyjpegxl` (Rust) | `pylibjxl` (C++) | `pillow-jxl` (Python/C) | Best Performer |
| :--- | :---: | :---: | :---: | :---: |
| **Decode to NumPy** | **42.90 ms** (32.2 MP/s) | 44.76 ms (30.9 MP/s) | 46.56 ms (29.7 MP/s) | 🏆 **`pyjpegxl`** |
| **Zero-Allocation Decode (`decode_into`)** | **41.93 ms** (33.0 MP/s) | 44.20 ms (with `out`) | *Unsupported* | 🏆 **`pyjpegxl`** |
| **Decode Peak Python Heap** | **0.00 MB** | **0.00 MB** | 11.88 MB | 🏆 **`pyjpegxl` & `pylibjxl`** |
| **Lossless Encode (Fastest, Effort=1)** | **2.01 ms** (688 MP/s) | **1.52 ms** (910 MP/s) | 3.83 ms (361 MP/s) | ⚡ **Microsecond-Tier** |
| **Lossless Encode (High, Effort=6/7)** | **289.17 ms** (eff=6) | 2,112.34 ms (eff=7) | 302.64 ms (eff=7) | 🏆 **`pyjpegxl`** |
| **Metadata Probing (`probe`)** | **49 μs** (0.049 ms) | **16 μs** (0.016 ms) | ~1,500 μs (requires open) | ⚡ **Sub-0.1ms** |

#### 2. JPEG Codec Comparison (1440×960 RGB)

| Operation | `pyjpegxl` (TurboJPEG) | `pylibjxl` (libjpeg-turbo) | `Pillow` | Best Performer |
| :--- | :---: | :---: | :---: | :---: |
| **JPEG Decode to NumPy** | 17.73 ms (78.0 MP/s) | **17.16 ms** (80.5 MP/s) | 18.53 ms (74.7 MP/s) | 🤝 Parity (~17.5 ms) |
| **JPEG Zero-Alloc Decode (`jpeg_decode_into`)** | **17.20 ms** (80.3 MP/s) | 17.15 ms | *Unsupported* | 🏆 **`pyjpegxl` & `pylibjxl`** |
| **JPEG Encode (Quality 95)** | **4.38 ms** (315.6 MP/s) | 6.01 ms (230.1 MP/s) | 5.27 ms (262.4 MP/s) | 🏆 **`pyjpegxl` (20-28% faster)** |
| **JPEG Marker Probe (Exif & ICC)** | **10 μs** (0.010 ms) | ~17,000 μs (full decode) | ~1,200 μs | 🏆 **`pyjpegxl` (Pure Rust)** |

#### 3. Concurrency & Multi-Threaded Batch Scaling (8 Images Batch)

`pyjpegxl` releases the GIL and provides thread-safe isolated runner instances (TLS), delivering near-linear throughput scaling when processing batches in parallel:

| Threads | Batch Time | Speedup | Aggregate Throughput |
| :---: | :---: | :---: | :---: |
| **1 Thread** | 2,274 ms | 1.00× | 4.9 MP/s (14.6 Raw MB/s) |
| **2 Threads** | 1,309 ms | 1.74× | 8.4 MP/s (25.3 Raw MB/s) |
| **4 Threads** | 864 ms | **2.63×** | **12.8 MP/s (38.4 Raw MB/s)** |

#### Reproducing the Benchmarks

You can reproduce all benchmark metrics on your hardware at any time:

```bash
# Run all benchmark categories and format output as a clean table
uv run python -m tests.test_benchmark --category all

# Output directly as GitHub-flavored Markdown
uv run python -m tests.test_benchmark --markdown

# Run specific category (e.g. competitor comparison or concurrency)
uv run python -m tests.test_benchmark --category compare
uv run python -m tests.test_benchmark --category concurrency
```

## API Reference

### Unified Polymorphic I/O & Sniffing
- `imread(source, *, dtype=None, channels=None) -> tuple[Metadata | JpegInfo, np.ndarray]`: Auto-detects format from path, bytes, or stream.
- `imwrite(dest, image, *, format=None, quality=None, lossless=False, speed=EncoderSpeed.Squirrel, **kwargs) -> None`: Auto-infers format or saves to JXL/JPEG.
- `probe_image(source) -> Metadata | JpegInfo`: Sub-millisecond metadata inspection for any supported format.
- `sniff_bytes(header: bytes) -> "jpeg" | "jxl" | "unknown"`
- `sniff_stream(stream) -> tuple[format, usable_stream]`: Safe peek/stream wrapping.
- `PrefixedStream`: Transparent wrapper for non-seekable binary streams.

### Ecosystem Bridges: Pillow & PyTorch
- `to_pil(image_or_array, metadata=None, *, preserve_hdr=True) -> PIL.Image.Image`: Zero-copy PIL conversion with ICC & EXIF preservation.
- `from_pil(image) -> tuple[np.ndarray, dict]`: Extracts C-contiguous NumPy array and metadata dict.
- `to_tensor(array_or_bytes, *, permute_chw=False, channels_last=False, normalize=False, device=None) -> torch.Tensor`: Zero-copy tensor wrapping (default contiguous HWC).
- `from_tensor(tensor) -> np.ndarray`: Converts GPU/CPU PyTorch Tensor back to NumPy HWC array.
- `decode_into_tensor(data: bytes, tensor, *, is_jpeg=None) -> Metadata | JpegInfo`: Zero-allocation decode directly into Pinned or Shared Memory Tensor.

### High-Throughput Batch Operations
- `read_batch(sources, *, max_workers=None, dtype=None, channels=None) -> list[tuple[Metadata | JpegInfo, np.ndarray]]`: Multi-core parallel batch decode releasing Python GIL.
- `transcode_batch(sources, output_dir, *, target_format="jxl", max_workers=None) -> list[str]`: Multi-core parallel lossless dataset transcoding.

### Metadata Probing (Fast, Zero Pixel Allocation)
- `probe(data: bytes) -> Metadata`
- `probe_file(path: str | os.PathLike) -> Metadata`
- `async_probe(data: bytes) -> Metadata`
- `async_probe_file(path: str | os.PathLike) -> Metadata`
- `jpeg_probe(data: bytes) -> JpegInfo`
- `jpeg_probe_file(path: str | os.PathLike) -> JpegInfo`
- `async_jpeg_probe(data: bytes) -> JpegInfo`
- `async_jpeg_probe_file(path: str | os.PathLike) -> JpegInfo`

### In-Place Zero-Allocation Decoding
- `decode_into(data: bytes, out: np.ndarray) -> Metadata`
- `read_into(path: str | os.PathLike, out: np.ndarray) -> Metadata`
- `async_decode_into(data: bytes, out: np.ndarray) -> Metadata`
- `async_read_into(path: str | os.PathLike, out: np.ndarray) -> Metadata`
- `jpeg_decode_into(data: bytes, out: np.ndarray) -> JpegInfo`
- `jpeg_read_into(path: str | os.PathLike, out: np.ndarray) -> JpegInfo`
- `async_jpeg_decode_into(data: bytes, out: np.ndarray) -> JpegInfo`
- `async_jpeg_read_into(path: str | os.PathLike, out: np.ndarray) -> JpegInfo`

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
- `jpeg_decode(data: bytes, *, channels: int | None = None) -> tuple[JpegInfo, bytes]`
- `jpeg_encode(data, width, height, *, quality=95, num_channels=3) -> bytes`

### JPEG NumPy API
- `jpeg_decode_to_numpy(data: bytes, *, channels: int | None = None) -> tuple[JpegInfo, np.ndarray]`
- `jpeg_encode_from_numpy(array: np.ndarray, *, quality=95) -> bytes`

### JPEG File I/O API
- `jpeg_read(path, *, channels: int | None = None) -> tuple[JpegInfo, bytes]`
- `jpeg_read_to_numpy(path, *, channels: int | None = None) -> tuple[JpegInfo, np.ndarray]`
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
- `JpegInfo`: JPEG image dimensions and metadata:
  - `width: int`, `height: int`, `num_channels: int`
  - `icc: bytes | None`, `exif: bytes | None`
- `EncoderSpeed`: JXL compression effort (`Lightning` → `Tortoise`).

## License

BSD 3-Clause
