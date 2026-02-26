"""Performance benchmarks for pyjpegxl using real images only."""

from __future__ import annotations

import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import pyjpegxl

IMAGES_DIR = Path(__file__).parent.parent / "images"
TEST_JXL = IMAGES_DIR / "test.jxl"
TEST_JPG = IMAGES_DIR / "test.jpg"

ITERATIONS = 10

# Skip completely if files are missing
pytestmark = pytest.mark.skipif(
    not TEST_JXL.exists(),
    reason="images/test.jxl not found",
)


def _fmt(label: str, elapsed: float, iters: int, data_bytes: int, peak_mb: float = 0.0) -> str:
    per_op = elapsed / iters * 1000
    throughput = data_bytes * iters / elapsed / 1e6
    mem_str = f", Peak Mem: {peak_mb:.1f} MB" if peak_mb > 0 else ""
    return f"{label}: {per_op:.2f} ms/op, {throughput:.1f} MB/s ({iters} iters, {elapsed:.3f}s total){mem_str}"

def run_bench(label, iters, data_bytes, func, *args, **kwargs):
    import gc
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(iters):
        func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    peak_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()
    print(_fmt(label, elapsed, iters, data_bytes, peak_mb=peak_mem))


# ---------------------------------------------------------------------------
# Core codec benchmarks
# ---------------------------------------------------------------------------

class TestBenchRealImage:
    """Benchmark with real JPEG and JXL images."""

    @pytest.fixture(autouse=True)
    def setup(self):
        meta, decoded = pyjpegxl.read_to_numpy(TEST_JXL)
        if meta.has_alpha:
            self.arr = decoded
        else:
            alpha = np.full((meta.height, meta.width, 1), 255, dtype=np.uint8)
            self.arr = np.ascontiguousarray(np.concatenate([decoded, alpha], axis=2))

        self.px = self.arr.tobytes()
        self.h, self.w, self.c = self.arr.shape

        with open(TEST_JXL, "rb") as f:
            self.jxl_source = f.read()

    def test_bench_encode_bytes(self):
        run_bench("encode(bytes, real image)", ITERATIONS, len(self.px), 
                  pyjpegxl.encode, self.px, self.w, self.h, num_channels=self.c)

    def test_bench_decode_bytes(self):
        run_bench("decode(bytes, test.jxl)", ITERATIONS, len(self.jxl_source), 
                  pyjpegxl.decode, self.jxl_source)

    def test_bench_encode_numpy_lossless(self):
        run_bench("encode_from_numpy(lossless, real image)", ITERATIONS, self.arr.nbytes, 
                  pyjpegxl.encode_from_numpy, self.arr, lossless=True)

    def test_bench_encode_numpy_lossy_std(self):
        run_bench("encode_from_numpy(lossy q1.0/std, real image)", ITERATIONS, self.arr.nbytes, 
                  pyjpegxl.encode_from_numpy, self.arr, quality=1.0)

    def test_bench_decode_numpy(self):
        run_bench("decode_to_numpy(test.jxl)", ITERATIONS, len(self.jxl_source), 
                  pyjpegxl.decode_to_numpy, self.jxl_source)

    def test_bench_concurrent_decode(self):
        """Compare sequential vs 4-thread concurrent decode using test.jxl."""
        n = 8

        # Sequential
        t0 = time.perf_counter()
        for _ in range(n):
            pyjpegxl.decode_to_numpy(self.jxl_source)
        seq_time = time.perf_counter() - t0

        # Concurrent (4 threads)
        def worker(data):
            return pyjpegxl.decode_to_numpy(data)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(worker, [self.jxl_source] * n))
        par_time = time.perf_counter() - t0

        speedup = seq_time / par_time
        print(
            f"concurrent decode (test.jxl): seq={seq_time:.3f}s, par(4)={par_time:.3f}s, "
            f"speedup={speedup:.2f}x"
        )

    def test_bench_concurrent_encode(self):
        """Compare sequential vs 4-thread concurrent encode on real image."""
        n = 8

        # Sequential
        t0 = time.perf_counter()
        for _ in range(n):
            pyjpegxl.encode_from_numpy(self.arr)
        seq_time = time.perf_counter() - t0

        # Concurrent (4 threads)
        def worker(a):
            return pyjpegxl.encode_from_numpy(a)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(worker, [self.arr] * n))
        par_time = time.perf_counter() - t0

        speedup = seq_time / par_time
        print(
            f"concurrent encode (real image): seq={seq_time:.3f}s, par(4)={par_time:.3f}s, "
            f"speedup={speedup:.2f}x"
        )


# ---------------------------------------------------------------------------
# Comparative Benchmarks: pyjpegxl vs pylibjxl
# ---------------------------------------------------------------------------

class TestBenchPylibjxlRealImage:
    """Benchmark comparing pyjpegxl vs pylibjxl on a real image."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import pylibjxl
        
        self.pylibjxl = pylibjxl

        meta, decoded = pyjpegxl.read_to_numpy(TEST_JXL)
        if meta.has_alpha:
            self.arr = decoded
        else:
            alpha = np.full((meta.height, meta.width, 1), 255, dtype=np.uint8)
            self.arr = np.ascontiguousarray(np.concatenate([decoded, alpha], axis=2))

        with open(TEST_JXL, "rb") as f:
            self.jxl_source = f.read()

    def test_compare_encode_numpy_lossy(self):
        run_bench("pyjpegxl encode (real, lossy ~std)", ITERATIONS, self.arr.nbytes, 
                  pyjpegxl.encode_from_numpy, self.arr, quality=1.0) # -> ~19MB file size for test.jpg
        run_bench("pylibjxl encode (real, lossy ~std)", ITERATIONS, self.arr.nbytes, 
                  self.pylibjxl.encode, self.arr, distance=6.0, lossless=False)

    def test_compare_encode_numpy_lossless(self):
        run_bench("pyjpegxl encode (real, lossless)", ITERATIONS, self.arr.nbytes, 
                  pyjpegxl.encode_from_numpy, self.arr, lossless=True)
        run_bench("pylibjxl encode (real, lossless)", ITERATIONS, self.arr.nbytes, 
                  self.pylibjxl.encode, self.arr, lossless=True)

    def test_compare_decode_numpy(self):
        run_bench("pyjpegxl decode (test.jxl)", ITERATIONS, len(self.jxl_source), 
                  pyjpegxl.decode_to_numpy, self.jxl_source)
        run_bench("pylibjxl decode (test.jxl)", ITERATIONS, len(self.jxl_source), 
                  self.pylibjxl.decode, self.jxl_source)

# ---------------------------------------------------------------------------
# Comparative Benchmarks: pyjpegxl vs pillow-jxl-plugin (Real Image)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TEST_JXL.exists(), reason="images/test.jxl not found")
class TestBenchPillowJxlRealImage:
    """Benchmark comparing pyjpegxl vs pillow-jxl-plugin on a real image."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import pillow_jxl
        import io
        from PIL import Image
        
        self.Image = Image
        self.io = io

        meta, decoded = pyjpegxl.read_to_numpy(TEST_JXL)
        if meta.has_alpha:
            self.arr = decoded
        else:
            alpha = np.full((meta.height, meta.width, 1), 255, dtype=np.uint8)
            self.arr = np.ascontiguousarray(np.concatenate([decoded, alpha], axis=2))

        with open(TEST_JXL, "rb") as f:
            self.jxl_source = f.read()

        # Helper functions to adapt Pillow to our run_bench API
        def pillow_encode_lossy(arr):
            img = self.Image.fromarray(arr)
            buf = self.io.BytesIO()
            img.save(buf, format='JXL', quality=90) # equivalent to pyjpegxl q=1.0
            return buf.getvalue()

        def pillow_encode_lossless(arr):
            img = self.Image.fromarray(arr)
            buf = self.io.BytesIO()
            img.save(buf, format='JXL', lossless=True)
            return buf.getvalue()

        def pillow_decode(data):
            buf = self.io.BytesIO(data)
            img = self.Image.open(buf)
            img.load() # Force decode
            return np.array(img)

        self.encode_lossy = pillow_encode_lossy
        self.encode_lossless = pillow_encode_lossless
        self.decode = pillow_decode

    def test_compare_encode_numpy_lossy_pillow(self):
        run_bench("pyjpegxl encode (real, lossy ~std)", ITERATIONS, self.arr.nbytes, 
                  pyjpegxl.encode_from_numpy, self.arr, quality=1.0) # -> ~19MB file size for test.jpg
        run_bench("pillow-jxl encode (real, lossy ~std)", ITERATIONS, self.arr.nbytes, 
                  self.encode_lossy, self.arr)

    def test_compare_encode_numpy_lossless_pillow(self):
        run_bench("pyjpegxl encode (real, lossless)", ITERATIONS, self.arr.nbytes, 
                  pyjpegxl.encode_from_numpy, self.arr, lossless=True)
        run_bench("pillow-jxl encode (real, lossless)", ITERATIONS, self.arr.nbytes, 
                  self.encode_lossless, self.arr)

    def test_compare_decode_numpy_pillow(self):
        run_bench("pyjpegxl decode (test.jxl)", ITERATIONS, len(self.jxl_source), 
                  pyjpegxl.decode_to_numpy, self.jxl_source)
        run_bench("pillow-jxl decode (test.jxl)", ITERATIONS, len(self.jxl_source), 
                  self.decode, self.jxl_source)


# ---------------------------------------------------------------------------
# JPEG Core Benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TEST_JPG.exists(), reason="images/test.jpg not found")
class TestBenchJPEG:
    """Core JPEG encode/decode benchmarks using pyjpegxl (turbojpeg)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        info, self.arr = pyjpegxl.jpeg_read_to_numpy(TEST_JPG)
        self.rgb_arr = np.ascontiguousarray(self.arr[..., :3]) if self.arr.shape[2] > 3 else self.arr
        self.px = self.rgb_arr.tobytes()
        self.h, self.w, self.c = self.rgb_arr.shape

        with open(TEST_JPG, "rb") as f:
            self.jpg_source = f.read()

    def test_bench_jpeg_decode_bytes(self):
        run_bench("jpeg_decode(bytes, test.jpg)", ITERATIONS, len(self.jpg_source),
                  pyjpegxl.jpeg_decode, self.jpg_source)

    def test_bench_jpeg_decode_numpy(self):
        run_bench("jpeg_decode_to_numpy(test.jpg)", ITERATIONS, len(self.jpg_source),
                  pyjpegxl.jpeg_decode_to_numpy, self.jpg_source)

    def test_bench_jpeg_encode_q95(self):
        run_bench("jpeg_encode_from_numpy(q95)", ITERATIONS, self.rgb_arr.nbytes,
                  pyjpegxl.jpeg_encode_from_numpy, self.rgb_arr, quality=95)

    def test_bench_jpeg_encode_q75(self):
        run_bench("jpeg_encode_from_numpy(q75)", ITERATIONS, self.rgb_arr.nbytes,
                  pyjpegxl.jpeg_encode_from_numpy, self.rgb_arr, quality=75)

    def test_bench_jpeg_concurrent_decode(self):
        """Compare sequential vs 4-thread concurrent JPEG decode."""
        n = 8

        t0 = time.perf_counter()
        for _ in range(n):
            pyjpegxl.jpeg_decode_to_numpy(self.jpg_source)
        seq_time = time.perf_counter() - t0

        def worker(data):
            return pyjpegxl.jpeg_decode_to_numpy(data)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(worker, [self.jpg_source] * n))
        par_time = time.perf_counter() - t0

        speedup = seq_time / par_time
        print(
            f"concurrent JPEG decode: seq={seq_time:.3f}s, par(4)={par_time:.3f}s, "
            f"speedup={speedup:.2f}x"
        )


# ---------------------------------------------------------------------------
# Comparative Benchmarks: JPEG — pyjpegxl vs pylibjxl vs Pillow
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TEST_JPG.exists(), reason="images/test.jpg not found")
class TestBenchJPEGCompare:
    """3-way JPEG benchmark: pyjpegxl (turbojpeg) vs pylibjxl (libjpeg-turbo) vs Pillow."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import pylibjxl
        import io
        from PIL import Image

        self.pylibjxl = pylibjxl
        self.Image = Image
        self.io = io

        info, self.rgb_arr = pyjpegxl.jpeg_read_to_numpy(TEST_JPG)
        self.rgb_arr = np.ascontiguousarray(self.rgb_arr[..., :3]) if self.rgb_arr.shape[2] > 3 else self.rgb_arr

        with open(TEST_JPG, "rb") as f:
            self.jpg_source = f.read()

        def pillow_encode(arr, quality=95):
            img = self.Image.fromarray(arr)
            buf = self.io.BytesIO()
            img.save(buf, format='JPEG', quality=quality)
            return buf.getvalue()

        def pillow_decode(data):
            buf = self.io.BytesIO(data)
            img = self.Image.open(buf)
            img.load()
            return np.array(img)

        self.pillow_encode = pillow_encode
        self.pillow_decode = pillow_decode

    def test_compare_jpeg_encode_q95(self):
        run_bench("pyjpegxl jpeg_encode (q95)", ITERATIONS, self.rgb_arr.nbytes,
                  pyjpegxl.jpeg_encode_from_numpy, self.rgb_arr, quality=95)
        run_bench("pylibjxl encode_jpeg (q95)", ITERATIONS, self.rgb_arr.nbytes,
                  self.pylibjxl.encode_jpeg, self.rgb_arr, quality=95)
        run_bench("Pillow JPEG encode (q95)", ITERATIONS, self.rgb_arr.nbytes,
                  self.pillow_encode, self.rgb_arr, quality=95)

    def test_compare_jpeg_encode_q75(self):
        run_bench("pyjpegxl jpeg_encode (q75)", ITERATIONS, self.rgb_arr.nbytes,
                  pyjpegxl.jpeg_encode_from_numpy, self.rgb_arr, quality=75)
        run_bench("pylibjxl encode_jpeg (q75)", ITERATIONS, self.rgb_arr.nbytes,
                  self.pylibjxl.encode_jpeg, self.rgb_arr, quality=75)
        run_bench("Pillow JPEG encode (q75)", ITERATIONS, self.rgb_arr.nbytes,
                  self.pillow_encode, self.rgb_arr, quality=75)

    def test_compare_jpeg_decode(self):
        run_bench("pyjpegxl jpeg_decode (test.jpg)", ITERATIONS, len(self.jpg_source),
                  pyjpegxl.jpeg_decode_to_numpy, self.jpg_source)
        run_bench("pylibjxl decode_jpeg (test.jpg)", ITERATIONS, len(self.jpg_source),
                  self.pylibjxl.decode_jpeg, self.jpg_source)
        run_bench("Pillow JPEG decode (test.jpg)", ITERATIONS, len(self.jpg_source),
                  self.pillow_decode, self.jpg_source)

