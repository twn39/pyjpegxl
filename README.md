# pyjpegxl

Python bindings for **JPEG XL** and **JPEG** encoding/decoding, powered by [libjxl](https://github.com/libjxl/libjxl) and [libjpeg-turbo](https://libjpeg-turbo.org/). Both libraries are statically linked — no system dependencies required.

**v2 Features**:
- **JPEG XL + JPEG**: Full encode/decode/file I/O for both formats in one package.
- **NumPy Zero-Copy**: Directly encode from and decode to `numpy.ndarray` without memory duplication.
- **True Concurrency**: Releases the Python GIL during heavy encoding/decoding operations, enabling true multi-threading.
- **Async API**: First-class `async`/`await` support via `asyncio.to_thread`.
- **Performance**: Fastest-in-class multi-threaded encoding and decoding.

## Installation

```bash
pip install pyjpegxl
```

*(Note: Pre-built wheels are currently only available for select platforms. If a wheel is not available, pip will try to build it from source. You will need a Rust toolchain installed.)*

### Build from source

Requires Rust toolchain and [maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin
git clone https://github.com/user/pyjpegxl && cd pyjpegxl
maturin build --release
```

## Quick Start

### Basic Usage (Bytes API)

```python
import pyjpegxl

# Decode
with open("image.jxl", "rb") as f:
    meta, pixels = pyjpegxl.decode(f.read())

print(f"{meta.width}x{meta.height}, channels={meta.num_channels}")

# Encode
jxl_data = pyjpegxl.encode(pixels, width=meta.width, height=meta.height)

# Custom Encode
jxl_data = pyjpegxl.encode(
    pixels, width=meta.width, height=meta.height,
    lossless=True,
    speed=pyjpegxl.EncoderSpeed.Falcon,
)
```

### NumPy Zero-Copy API

Move raw bytes to and from NumPy arrays instantly without Python-level allocations.

```python
import pyjpegxl
import numpy as np

with open("image.jxl", "rb") as f:
    # Decode directly into a NumPy array (H, W, C)
    meta, arr = pyjpegxl.decode_to_numpy(f.read())

print(arr.shape, arr.dtype) # e.g. (1080, 1920, 3), dtype('uint8')

# Encode directly from a C-contiguous NumPy array
jxl_data = pyjpegxl.encode_from_numpy(arr, quality=1.0) # quality=1.0 is default for visually lossless
```

### File I/O API

Read and write JXL files directly — no manual `open()` needed.

```python
import pyjpegxl

# Read a JXL file to a NumPy array
meta, arr = pyjpegxl.read_to_numpy("image.jxl")
print(arr.shape, arr.dtype)

# Write a NumPy array to a JXL file
pyjpegxl.write_from_numpy("output.jxl", arr, lossless=True)

# Bytes-level file I/O
meta, pixels = pyjpegxl.read("image.jxl")
pyjpegxl.write("output.jxl", pixels, width=meta.width, height=meta.height,
               num_channels=meta.num_color_channels + int(meta.has_alpha))
```

### Async API

Perfect for high-concurrency web servers like FastAPI or Starlette.

```python
import asyncio
import pyjpegxl

async def process_image():
    with open("image.jxl", "rb") as f:
        data = f.read()
        
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

## Concurrency and Performance

`pyjpegxl` natively releases the Global Interpreter Lock (GIL) and engages `ThreadsRunner` from `libjxl`. If you use `concurrent.futures.ThreadPoolExecutor` or `asyncio.gather()`, multiple images will encode and decode perfectly in parallel without blocking the main Python thread.

### Benchmarks (MacBook M-Series)

Benchmark processing `images/test.jpg` (decoded to Numpy arrays) among Python JXL wrappers on identical visual quality settings:

| Library | Decode Time (ms) | Peak Python Mem | Encode Time (ms) | Peak Python Mem |
| :--- | :--- | :--- | :--- | :--- |
| **`pyjpegxl`** | **35.46** | **0.0 MB** | **180.85** | **0.4 MB** |
| `pylibjxl` | 110.84 | 0.0 MB | 349.83 | 0.5 MB |
| `pillow-jxl` | 38.39 | 11.9 MB | 189.19 | 10.6 MB |

> `pyjpegxl` is fundamentally the fastest encoder and decoder, while matching the flawless memory performance of `pylibjxl` due to its zero-copy `IntoPyArray` bridging.

## API Reference

### JXL Bytes API
- `decode(data: bytes) -> tuple[Metadata, bytes]`
- `encode(data, width, height, *, lossless=False, quality=1.0, speed=EncoderSpeed.Squirrel, num_channels=4) -> bytes`

### JXL NumPy API
- `decode_to_numpy(data: bytes) -> tuple[Metadata, np.ndarray]`
- `encode_from_numpy(array: np.ndarray, *, lossless=False, quality=1.0, speed=EncoderSpeed.Squirrel) -> bytes`

### JXL File I/O API
- `read(path) -> tuple[Metadata, bytes]`
- `read_to_numpy(path) -> tuple[Metadata, np.ndarray]`
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

### Async API
All sync functions have async variants prefixed with `async_` (JXL) or `async_jpeg_` (JPEG).

### Types
- `Metadata`: JXL image dimensions and channel information.
- `JpegInfo`: JPEG image dimensions and channel count.
- `EncoderSpeed`: JXL compression effort (`Lightning` → `Tortoise`).

## License

MIT
