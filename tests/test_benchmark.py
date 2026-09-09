"""Comprehensive performance benchmarks for pyjpegxl.

Supports dual execution modes:
1. Pytest runner:
   uv run pytest tests/test_benchmark.py -v -s
2. Standalone CLI with formatted tables or Markdown output:
   uv run python -m tests.test_benchmark --category all --markdown
   uv run python -m tests.test_benchmark --iterations 10 --warmup 2
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import time
import tracemalloc
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyjpegxl
import pytest

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

IMAGES_DIR = Path(__file__).parent.parent / "images"
DEFAULT_TEST_JXL = IMAGES_DIR / "test.jxl"
DEFAULT_TEST_JPG = IMAGES_DIR / "test.jpg"


# ---------------------------------------------------------------------------
# Benchmark Measurement Engine
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    name: str
    category: str
    iterations: int
    warmup: int
    min_ms: float
    median_ms: float
    mean_ms: float
    std_ms: float
    p95_ms: float
    mp_per_sec: float | None = None
    raw_mb_per_sec: float | None = None
    peak_heap_mb: float = 0.0
    rss_delta_mb: float = 0.0
    extra: str = ""

    def summary(self) -> str:
        t_str = f"Median: {self.median_ms:.2f}ms (Min: {self.min_ms:.2f}ms, P95: {self.p95_ms:.2f}ms)"
        perf_parts = []
        if self.mp_per_sec:
            perf_parts.append(f"{self.mp_per_sec:.1f} MP/s")
        if self.raw_mb_per_sec:
            perf_parts.append(f"{self.raw_mb_per_sec:.1f} Raw MB/s")
        perf_str = f", {', '.join(perf_parts)}" if perf_parts else ""
        mem_str = f", Peak Heap: {self.peak_heap_mb:.2f}MB" if self.peak_heap_mb > 0.01 else ""
        return f"{self.name:<36} | {t_str}{perf_str}{mem_str}"


class BenchmarkRunner:
    """Executes callables with statistical rigor: warmups, GC isolation, and memory profiling."""

    def __init__(self, iterations: int = 10, warmup: int = 2):
        self.iterations = max(1, iterations)
        self.warmup = max(0, warmup)
        self.results: list[BenchmarkResult] = []

    def run(
        self,
        name: str,
        category: str,
        func: Callable[..., Any],
        *args: Any,
        pixels: int | None = None,
        raw_bytes: int | None = None,
        extra: str = "",
        **kwargs: Any,
    ) -> BenchmarkResult:
        # 1. Warmup phase
        for _ in range(self.warmup):
            func(*args, **kwargs)

        # 2. Memory baseline & GC collect
        gc.collect()
        rss_before = psutil.Process().memory_info().rss if HAS_PSUTIL else 0
        tracemalloc.start()

        # 3. Timed execution with GC disabled
        timings: list[float] = []
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()

        try:
            for _ in range(self.iterations):
                t0 = time.perf_counter()
                func(*args, **kwargs)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000.0)  # ms
        finally:
            if gc_was_enabled:
                gc.enable()

        peak_heap_bytes = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        rss_after = psutil.Process().memory_info().rss if HAS_PSUTIL else 0
        gc.collect()

        # 4. Statistical computation
        timings.sort()
        n = len(timings)
        min_ms = timings[0]
        median_ms = timings[n // 2] if n % 2 != 0 else (timings[n // 2 - 1] + timings[n // 2]) / 2.0
        mean_ms = sum(timings) / n
        variance = sum((t - mean_ms) ** 2 for t in timings) / n
        std_ms = variance**0.5
        p95_idx = min(n - 1, int(round(0.95 * (n - 1))))
        p95_ms = timings[p95_idx]

        median_sec = median_ms / 1000.0
        mp_per_sec = (pixels / 1e6) / median_sec if pixels and median_sec > 0 else None
        raw_mb_per_sec = (raw_bytes / 1e6) / median_sec if raw_bytes and median_sec > 0 else None

        res = BenchmarkResult(
            name=name,
            category=category,
            iterations=self.iterations,
            warmup=self.warmup,
            min_ms=min_ms,
            median_ms=median_ms,
            mean_ms=mean_ms,
            std_ms=std_ms,
            p95_ms=p95_ms,
            mp_per_sec=mp_per_sec,
            raw_mb_per_sec=raw_mb_per_sec,
            peak_heap_mb=peak_heap_bytes / (1024 * 1024),
            rss_delta_mb=max(0.0, (rss_after - rss_before) / (1024 * 1024)),
            extra=extra,
        )
        self.results.append(res)
        return res


# Global runner instance
runner = BenchmarkRunner(
    iterations=int(os.environ.get("BENCH_ITERS", 10)),
    warmup=int(os.environ.get("BENCH_WARMUP", 2)),
)


# ---------------------------------------------------------------------------
# Test Fixture & Image Generation Helpers
# ---------------------------------------------------------------------------


class BenchmarkContext:
    """Provides ready-to-use real or synthetic test images."""

    def __init__(self, jxl_path: Path | None = None, jpg_path: Path | None = None):
        self.jxl_path = jxl_path or DEFAULT_TEST_JXL
        self.jpg_path = jpg_path or DEFAULT_TEST_JPG

        # Load or generate JXL test data
        if self.jxl_path.exists():
            self.jxl_bytes = self.jxl_path.read_bytes()
            meta, self.rgb_arr = pyjpegxl.read_to_numpy(self.jxl_path)
            if self.rgb_arr.ndim == 3 and self.rgb_arr.shape[2] > 3:
                self.rgb_arr = np.ascontiguousarray(self.rgb_arr[..., :3])
        else:
            # Synthetic 1440x960 RGB gradient
            h, w = 960, 1440
            y, x = np.mgrid[0:h, 0:w]
            r = (x * 255 / w).astype(np.uint8)
            g = (y * 255 / h).astype(np.uint8)
            b = ((x + y) * 255 / (w + h)).astype(np.uint8)
            self.rgb_arr = np.ascontiguousarray(np.dstack([r, g, b]))
            self.jxl_bytes = pyjpegxl.encode_from_numpy(
                self.rgb_arr, lossless=False, quality=90, speed=pyjpegxl.EncoderSpeed.Lightning
            )

        self.height, self.width, self.channels = self.rgb_arr.shape
        self.pixels = self.width * self.height
        self.raw_bytes = self.rgb_arr.nbytes
        self.raw_pixel_bytes = self.rgb_arr.tobytes()

        # Load or generate JPEG test data
        if self.jpg_path.exists():
            self.jpg_bytes = self.jpg_path.read_bytes()
        else:
            self.jpg_bytes = pyjpegxl.jpeg_encode_from_numpy(self.rgb_arr, quality=90)


@pytest.fixture(scope="module")
def bench_ctx():
    return BenchmarkContext()


# ---------------------------------------------------------------------------
# Category 1: JXL Core Codec Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestBenchJxlCore:
    """Core JPEG XL decode and encode benchmarks."""

    def test_jxl_decode_to_numpy(self, bench_ctx: BenchmarkContext):
        res = runner.run(
            "JXL decode_to_numpy",
            "jxl_core",
            pyjpegxl.decode_to_numpy,
            bench_ctx.jxl_bytes,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
        )
        print(res.summary())

    def test_jxl_decode_bytes(self, bench_ctx: BenchmarkContext):
        res = runner.run(
            "JXL decode (raw bytes)",
            "jxl_core",
            pyjpegxl.decode,
            bench_ctx.jxl_bytes,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
        )
        print(res.summary())

    def test_jxl_decode_into(self, bench_ctx: BenchmarkContext):
        out = np.empty_like(bench_ctx.rgb_arr)
        res = runner.run(
            "JXL decode_into (zero-alloc)",
            "jxl_core",
            pyjpegxl.decode_into,
            bench_ctx.jxl_bytes,
            out,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
        )
        print(res.summary())

    def test_jxl_probe(self, bench_ctx: BenchmarkContext):
        res = runner.run(
            "JXL probe (metadata only)",
            "jxl_core",
            pyjpegxl.probe,
            bench_ctx.jxl_bytes,
        )
        print(res.summary())

    def test_jxl_encode_lightning(self, bench_ctx: BenchmarkContext):
        res = runner.run(
            "JXL encode (Lightning, lossy)",
            "jxl_core",
            pyjpegxl.encode_from_numpy,
            bench_ctx.rgb_arr,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
            lossless=False,
            quality=90,
            speed=pyjpegxl.EncoderSpeed.Lightning,
        )
        print(res.summary())

    def test_jxl_encode_cheetah(self, bench_ctx: BenchmarkContext):
        res = runner.run(
            "JXL encode (Cheetah, lossy)",
            "jxl_core",
            pyjpegxl.encode_from_numpy,
            bench_ctx.rgb_arr,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
            lossless=False,
            quality=90,
            speed=pyjpegxl.EncoderSpeed.Cheetah,
        )
        print(res.summary())

    def test_jxl_encode_squirrel(self, bench_ctx: BenchmarkContext):
        res = runner.run(
            "JXL encode (Squirrel, lossy)",
            "jxl_core",
            pyjpegxl.encode_from_numpy,
            bench_ctx.rgb_arr,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
            lossless=False,
            quality=90,
            speed=pyjpegxl.EncoderSpeed.Squirrel,
        )
        print(res.summary())

    def test_jxl_encode_lossless(self, bench_ctx: BenchmarkContext):
        res = runner.run(
            "JXL encode (Lightning, lossless)",
            "jxl_core",
            pyjpegxl.encode_from_numpy,
            bench_ctx.rgb_arr,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
            lossless=True,
            speed=pyjpegxl.EncoderSpeed.Lightning,
        )
        print(res.summary())


# ---------------------------------------------------------------------------
# Category 2: JPEG Core Codec Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestBenchJpegCore:
    """Core JPEG encode and decode benchmarks using TurboJPEG."""

    def test_jpeg_decode_to_numpy(self, bench_ctx: BenchmarkContext):
        res = runner.run(
            "JPEG decode_to_numpy",
            "jpeg_core",
            pyjpegxl.jpeg_decode_to_numpy,
            bench_ctx.jpg_bytes,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
        )
        print(res.summary())

    def test_jpeg_decode_into(self, bench_ctx: BenchmarkContext):
        out = np.empty_like(bench_ctx.rgb_arr)
        res = runner.run(
            "JPEG decode_into (zero-alloc)",
            "jpeg_core",
            pyjpegxl.jpeg_decode_into,
            bench_ctx.jpg_bytes,
            out,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
        )
        print(res.summary())

    def test_jpeg_probe(self, bench_ctx: BenchmarkContext):
        res = runner.run(
            "JPEG probe (pure-rust markers)",
            "jpeg_core",
            pyjpegxl.jpeg_probe,
            bench_ctx.jpg_bytes,
        )
        print(res.summary())

    def test_jpeg_encode_q95(self, bench_ctx: BenchmarkContext):
        res = runner.run(
            "JPEG encode (q95)",
            "jpeg_core",
            pyjpegxl.jpeg_encode_from_numpy,
            bench_ctx.rgb_arr,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
            quality=95,
        )
        print(res.summary())

    def test_jpeg_encode_q75(self, bench_ctx: BenchmarkContext):
        res = runner.run(
            "JPEG encode (q75)",
            "jpeg_core",
            pyjpegxl.jpeg_encode_from_numpy,
            bench_ctx.rgb_arr,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
            quality=75,
        )
        print(res.summary())


# ---------------------------------------------------------------------------
# Category 3: Zero-Allocation Optimization Focus
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestBenchZeroAllocation:
    """Quantifies zero-allocation decode advantage over standard allocation."""

    def test_zero_alloc_jxl_comparison(self, bench_ctx: BenchmarkContext):
        out = np.empty_like(bench_ctx.rgb_arr)

        res_std = runner.run(
            "JXL standard decode_to_numpy",
            "zero_alloc",
            pyjpegxl.decode_to_numpy,
            bench_ctx.jxl_bytes,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
        )
        res_into = runner.run(
            "JXL zero-alloc decode_into",
            "zero_alloc",
            pyjpegxl.decode_into,
            bench_ctx.jxl_bytes,
            out,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
        )

        speedup = res_std.median_ms / res_into.median_ms if res_into.median_ms > 0 else 1.0
        print(
            f"\n[Zero-Alloc JXL] Standard: {res_std.median_ms:.2f}ms vs Into: {res_into.median_ms:.2f}ms (Speedup: {speedup:.2f}x)"
        )

    def test_zero_alloc_jpeg_comparison(self, bench_ctx: BenchmarkContext):
        out = np.empty_like(bench_ctx.rgb_arr)

        res_std = runner.run(
            "JPEG standard decode_to_numpy",
            "zero_alloc",
            pyjpegxl.jpeg_decode_to_numpy,
            bench_ctx.jpg_bytes,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
        )
        res_into = runner.run(
            "JPEG zero-alloc decode_into",
            "zero_alloc",
            pyjpegxl.jpeg_decode_into,
            bench_ctx.jpg_bytes,
            out,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
        )

        speedup = res_std.median_ms / res_into.median_ms if res_into.median_ms > 0 else 1.0
        print(
            f"\n[Zero-Alloc JPEG] Standard: {res_std.median_ms:.2f}ms vs Into: {res_into.median_ms:.2f}ms (Speedup: {speedup:.2f}x)"
        )


# ---------------------------------------------------------------------------
# Category 4: Concurrency & Scaling (GIL Release)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestBenchConcurrency:
    """Evaluates multi-threaded throughput and GIL-release efficiency."""

    def test_concurrency_jxl_decode(self, bench_ctx: BenchmarkContext):
        batch_size = 8
        items = [bench_ctx.jxl_bytes] * batch_size
        total_pixels = bench_ctx.pixels * batch_size
        total_bytes = bench_ctx.raw_bytes * batch_size

        for num_threads in [1, 2, 4]:

            def batch_decode(data_list, threads=num_threads):
                with ThreadPoolExecutor(max_workers=threads) as pool:
                    return list(pool.map(pyjpegxl.decode_to_numpy, data_list))

            res = runner.run(
                f"JXL decode ({num_threads} threads, 8 imgs)",
                "concurrency",
                batch_decode,
                items,
                pixels=total_pixels,
                raw_bytes=total_bytes,
            )
            print(res.summary())

    def test_concurrency_jpeg_decode(self, bench_ctx: BenchmarkContext):
        batch_size = 8
        items = [bench_ctx.jpg_bytes] * batch_size
        total_pixels = bench_ctx.pixels * batch_size
        total_bytes = bench_ctx.raw_bytes * batch_size

        for num_threads in [1, 2, 4]:

            def batch_decode(data_list, threads=num_threads):
                with ThreadPoolExecutor(max_workers=threads) as pool:
                    return list(pool.map(pyjpegxl.jpeg_decode_to_numpy, data_list))

            res = runner.run(
                f"JPEG decode ({num_threads} threads, 8 imgs)",
                "concurrency",
                batch_decode,
                items,
                pixels=total_pixels,
                raw_bytes=total_bytes,
            )
            print(res.summary())


# ---------------------------------------------------------------------------
# Category 5: Lossless Transcoding
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestBenchTranscode:
    """Benchmarks lossless container transcoding vs re-encoding."""

    def test_transcode_jpeg_to_jxl(self, bench_ctx: BenchmarkContext):
        res_trans = runner.run(
            "Lossless jpeg_to_jxl",
            "transcode",
            pyjpegxl.jpeg_to_jxl,
            bench_ctx.jpg_bytes,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
        )

        def re_encode(data):
            _, arr = pyjpegxl.jpeg_decode_to_numpy(data)
            return pyjpegxl.encode_from_numpy(arr, lossless=True, speed=pyjpegxl.EncoderSpeed.Lightning)

        res_reenc = runner.run(
            "Re-encode (decode JPEG -> encode JXL)",
            "transcode",
            re_encode,
            bench_ctx.jpg_bytes,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
        )

        speedup = res_reenc.median_ms / res_trans.median_ms if res_trans.median_ms > 0 else 1.0
        print(f"\n[Lossless Transcode] jpeg_to_jxl is {speedup:.2f}x faster than re-encoding")


# ---------------------------------------------------------------------------
# Category 6: Competitor Comparisons (pyjpegxl vs pylibjxl vs Pillow)
# ---------------------------------------------------------------------------


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@pytest.mark.benchmark
class TestBenchComparative:
    """Side-by-side benchmark comparing pyjpegxl against pylibjxl and Pillow."""

    def test_compare_jxl_decode(self, bench_ctx: BenchmarkContext):
        cpu_count = os.cpu_count() or 4

        # pyjpegxl (uses all CPU cores by default)
        runner.run(
            f"pyjpegxl JXL decode ({cpu_count}t)",
            "compare_jxl_decode",
            pyjpegxl.decode_to_numpy,
            bench_ctx.jxl_bytes,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
        )

        # pylibjxl
        if has_module("pylibjxl"):
            import pylibjxl

            # 1. pylibjxl default top-level decode (defaults to 2 worker threads)
            runner.run(
                "pylibjxl JXL decode (default 2t)",
                "compare_jxl_decode",
                pylibjxl.decode,
                bench_ctx.jxl_bytes,
                pixels=bench_ctx.pixels,
                raw_bytes=bench_ctx.raw_bytes,
            )

            # 2. pylibjxl with matching CPU threads (apples-to-apples)
            jxl_multi = pylibjxl.JXL(threads=cpu_count)

            def pylibjxl_decode_multi(b):
                return jxl_multi.decode(b)

            runner.run(
                f"pylibjxl JXL decode (matched {cpu_count}t)",
                "compare_jxl_decode",
                pylibjxl_decode_multi,
                bench_ctx.jxl_bytes,
                pixels=bench_ctx.pixels,
                raw_bytes=bench_ctx.raw_bytes,
            )

        # pillow-jxl
        if has_module("pillow_jxl") and has_module("PIL"):
            import io

            import pillow_jxl  # noqa: F401
            from PIL import Image

            def pillow_jxl_dec(b):
                img = Image.open(io.BytesIO(b))
                img.load()
                return np.array(img)

            runner.run(
                "pillow-jxl JXL decode",
                "compare_jxl_decode",
                pillow_jxl_dec,
                bench_ctx.jxl_bytes,
                pixels=bench_ctx.pixels,
                raw_bytes=bench_ctx.raw_bytes,
            )

    def test_compare_jpeg_decode(self, bench_ctx: BenchmarkContext):
        # pyjpegxl
        runner.run(
            "pyjpegxl JPEG decode",
            "compare_jpeg_decode",
            pyjpegxl.jpeg_decode_to_numpy,
            bench_ctx.jpg_bytes,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
        )

        # pylibjxl
        if has_module("pylibjxl"):
            import pylibjxl

            runner.run(
                "pylibjxl JPEG decode",
                "compare_jpeg_decode",
                pylibjxl.decode_jpeg,
                bench_ctx.jpg_bytes,
                pixels=bench_ctx.pixels,
                raw_bytes=bench_ctx.raw_bytes,
            )

        # Pillow
        if has_module("PIL"):
            import io

            from PIL import Image

            def pillow_jpg_dec(b):
                img = Image.open(io.BytesIO(b))
                img.load()
                return np.array(img)

            runner.run(
                "Pillow JPEG decode",
                "compare_jpeg_decode",
                pillow_jpg_dec,
                bench_ctx.jpg_bytes,
                pixels=bench_ctx.pixels,
                raw_bytes=bench_ctx.raw_bytes,
            )

    def test_compare_jpeg_encode(self, bench_ctx: BenchmarkContext):
        # pyjpegxl
        runner.run(
            "pyjpegxl JPEG encode (q95)",
            "compare_jpeg_encode",
            pyjpegxl.jpeg_encode_from_numpy,
            bench_ctx.rgb_arr,
            quality=95,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
        )

        # pylibjxl
        if has_module("pylibjxl"):
            import pylibjxl

            runner.run(
                "pylibjxl JPEG encode (q95)",
                "compare_jpeg_encode",
                pylibjxl.encode_jpeg,
                bench_ctx.rgb_arr,
                quality=95,
                pixels=bench_ctx.pixels,
                raw_bytes=bench_ctx.raw_bytes,
            )

        # Pillow
        if has_module("PIL"):
            import io

            from PIL import Image

            def pillow_jpg_enc(arr):
                img = Image.fromarray(arr)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=95)
                return buf.getvalue()

            runner.run(
                "Pillow JPEG encode (q95)",
                "compare_jpeg_encode",
                pillow_jpg_enc,
                bench_ctx.rgb_arr,
                pixels=bench_ctx.pixels,
                raw_bytes=bench_ctx.raw_bytes,
            )

    def test_compare_jxl_encode(self, bench_ctx: BenchmarkContext):
        cpu_count = os.cpu_count() or 4

        # 1. Fast tier comparison (Lightning / effort=1, lossless)
        runner.run(
            f"pyjpegxl JXL encode (Lightning/eff=1, {cpu_count}t)",
            "compare_jxl_encode",
            pyjpegxl.encode_from_numpy,
            bench_ctx.rgb_arr,
            lossless=True,
            speed=pyjpegxl.EncoderSpeed.Lightning,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
        )

        if has_module("pylibjxl"):
            import pylibjxl

            jxl_enc_multi = pylibjxl.JXL(threads=cpu_count, effort=1, lossless=True)

            def pylibjxl_enc_eff1(arr):
                return jxl_enc_multi.encode(arr, effort=1, lossless=True)

            runner.run(
                f"pylibjxl JXL encode (eff=1, matched {cpu_count}t)",
                "compare_jxl_encode",
                pylibjxl_enc_eff1,
                bench_ctx.rgb_arr,
                pixels=bench_ctx.pixels,
                raw_bytes=bench_ctx.raw_bytes,
            )

        if has_module("pillow_jxl") and has_module("PIL"):
            import io

            import pillow_jxl  # noqa: F401
            from PIL import Image

            def pillow_jxl_enc(arr):
                img = Image.fromarray(arr)
                buf = io.BytesIO()
                img.save(buf, format="JXL", lossless=True)
                return buf.getvalue()

            runner.run(
                "pillow-jxl JXL encode (lossless)",
                "compare_jxl_encode",
                pillow_jxl_enc,
                bench_ctx.rgb_arr,
                pixels=bench_ctx.pixels,
                raw_bytes=bench_ctx.raw_bytes,
            )

        # 2. High effort comparison (Squirrel / effort=7, lossless)
        runner.run(
            f"pyjpegxl JXL encode (Squirrel/eff=6, {cpu_count}t)",
            "compare_jxl_encode",
            pyjpegxl.encode_from_numpy,
            bench_ctx.rgb_arr,
            lossless=True,
            speed=pyjpegxl.EncoderSpeed.Squirrel,
            pixels=bench_ctx.pixels,
            raw_bytes=bench_ctx.raw_bytes,
        )

        if has_module("pylibjxl"):
            import pylibjxl

            jxl_enc_eff7 = pylibjxl.JXL(threads=cpu_count, effort=7, lossless=True)

            def pylibjxl_enc_eff7(arr):
                return jxl_enc_eff7.encode(arr, effort=7, lossless=True)

            runner.run(
                f"pylibjxl JXL encode (eff=7, matched {cpu_count}t)",
                "compare_jxl_encode",
                pylibjxl_enc_eff7,
                bench_ctx.rgb_arr,
                pixels=bench_ctx.pixels,
                raw_bytes=bench_ctx.raw_bytes,
            )


# ---------------------------------------------------------------------------
# CLI Reporter & Markdown Generator
# ---------------------------------------------------------------------------


def render_table(results: list[BenchmarkResult], as_markdown: bool = False) -> str:
    """Renders results list into a cleanly formatted ASCII or Markdown table."""
    if not results:
        return "No benchmark results."

    headers = [
        "Benchmark",
        "Median (ms)",
        "Min (ms)",
        "P95 (ms)",
        "MP/s",
        "Raw MB/s",
        "Peak Heap",
    ]

    rows: list[list[str]] = []
    for r in results:
        mp_str = f"{r.mp_per_sec:.1f}" if r.mp_per_sec else "-"
        raw_str = f"{r.raw_mb_per_sec:.1f}" if r.raw_mb_per_sec else "-"
        heap_str = f"{r.peak_heap_mb:.2f} MB" if r.peak_heap_mb > 0.01 else "0.00 MB"
        rows.append(
            [
                r.name,
                f"{r.median_ms:.2f}",
                f"{r.min_ms:.2f}",
                f"{r.p95_ms:.2f}",
                mp_str,
                raw_str,
                heap_str,
            ]
        )

    if as_markdown:
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |"
        row_lines = ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join([header_line, sep_line] + row_lines)

    # ASCII table with dynamic column width
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    def fmt_row(vals: list[str]) -> str:
        return " | ".join(f"{v:<{col_widths[i]}}" if i == 0 else f"{v:>{col_widths[i]}}" for i, v in enumerate(vals))

    sep = "-+-".join("-" * w for w in col_widths)
    lines = [
        fmt_row(headers),
        sep,
        *[fmt_row(r) for r in rows],
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="pyjpegxl High-Performance Codec Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--category",
        "-c",
        choices=["all", "jxl", "jpeg", "zero-alloc", "concurrency", "transcode", "compare"],
        default="all",
        help="Benchmark category to run",
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=10,
        help="Number of iterations per benchmark run",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=2,
        help="Number of warmup iterations prior to timing",
    )
    parser.add_argument(
        "--markdown",
        "-m",
        action="store_true",
        help="Format output table as GitHub-flavored Markdown",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Custom image path to benchmark on (defaults to images/test.jxl)",
    )
    args = parser.parse_args()

    runner.iterations = args.iterations
    runner.warmup = args.warmup

    print("=" * 78)
    print(f"pyjpegxl Benchmark Runner (iters={args.iterations}, warmup={args.warmup})")
    print("=" * 78)

    ctx = BenchmarkContext(jxl_path=args.image)
    print(f"Test Image: {ctx.width}x{ctx.height} ({ctx.channels} channels), raw={ctx.raw_bytes / 1e6:.2f} MB")
    print("-" * 78)

    cat = args.category
    run_all = cat == "all"

    # Category 1: JXL
    if run_all or cat == "jxl":
        print("\n>>> Category: JXL Core Codec")
        bench = TestBenchJxlCore()
        bench.test_jxl_decode_to_numpy(ctx)
        bench.test_jxl_decode_bytes(ctx)
        bench.test_jxl_decode_into(ctx)
        bench.test_jxl_probe(ctx)
        bench.test_jxl_encode_lightning(ctx)
        bench.test_jxl_encode_cheetah(ctx)
        bench.test_jxl_encode_squirrel(ctx)
        bench.test_jxl_encode_lossless(ctx)

    # Category 2: JPEG
    if run_all or cat == "jpeg":
        print("\n>>> Category: JPEG Core Codec")
        bench = TestBenchJpegCore()
        bench.test_jpeg_decode_to_numpy(ctx)
        bench.test_jpeg_decode_into(ctx)
        bench.test_jpeg_probe(ctx)
        bench.test_jpeg_encode_q95(ctx)
        bench.test_jpeg_encode_q75(ctx)

    # Category 3: Zero Allocation
    if run_all or cat == "zero-alloc":
        print("\n>>> Category: Zero-Allocation Advantage")
        bench = TestBenchZeroAllocation()
        bench.test_zero_alloc_jxl_comparison(ctx)
        bench.test_zero_alloc_jpeg_comparison(ctx)

    # Category 4: Concurrency
    if run_all or cat == "concurrency":
        print("\n>>> Category: Concurrency & Scaling")
        bench = TestBenchConcurrency()
        bench.test_concurrency_jxl_decode(ctx)
        bench.test_concurrency_jpeg_decode(ctx)

    # Category 5: Transcode
    if run_all or cat == "transcode":
        print("\n>>> Category: Lossless Transcoding")
        bench = TestBenchTranscode()
        bench.test_transcode_jpeg_to_jxl(ctx)

    # Category 6: Competitor Comparisons
    if run_all or cat == "compare":
        print("\n>>> Category: Competitor Comparisons")
        bench = TestBenchComparative()
        bench.test_compare_jxl_decode(ctx)
        bench.test_compare_jxl_encode(ctx)
        bench.test_compare_jpeg_decode(ctx)
        bench.test_compare_jpeg_encode(ctx)

    print("\n" + "=" * 78)
    print("BENCHMARK SUMMARY REPORT")
    print("=" * 78 + "\n")
    print(render_table(runner.results, as_markdown=args.markdown))


if __name__ == "__main__":
    main()
