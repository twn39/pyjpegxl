"""Tests for pyjpegxl encode/decode — bytes, numpy, async, concurrency."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import pyjpegxl

IMAGES_DIR = Path(__file__).parent.parent / "images"
TEST_JXL = IMAGES_DIR / "test.jxl"

# Skip all tests if images are missing
pytestmark = pytest.mark.skipif(
    not TEST_JXL.exists(),
    reason="images/test.jxl not found"
)# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_image_data() -> tuple[np.ndarray, np.ndarray, bytes]:
    """Load test images once for all tests to speed up execution.
    Returns: (RGBA ndarray, RGB ndarray, JXL bytes)
    """
    meta, arr = pyjpegxl.read_to_numpy(TEST_JXL)

    if meta.has_alpha:
        rgba_arr = arr
        rgb_arr = np.ascontiguousarray(arr[..., :3])
    else:
        rgb_arr = arr
        alpha = np.full((meta.height, meta.width, 1), 255, dtype=np.uint8)
        rgba_arr = np.ascontiguousarray(np.concatenate([arr, alpha], axis=2))

    # Keep jxl_bytes for tests that need raw JXL data
    with open(TEST_JXL, "rb") as f:
        jxl_bytes = f.read()

    return rgba_arr, rgb_arr, jxl_bytes


# ---------------------------------------------------------------------------
# Basic bytes round-trip
# ---------------------------------------------------------------------------

class TestBytesAPI:
    def test_round_trip_rgba(self, real_image_data):
        rgba_arr, _, _ = real_image_data
        px = rgba_arr.tobytes()
        h, w, c = rgba_arr.shape
        jxl = pyjpegxl.encode(px, w, h, lossless=True, num_channels=c, speed=pyjpegxl.EncoderSpeed.Lightning)
        meta, decoded = pyjpegxl.decode(jxl)
        assert meta.width == w and meta.height == h
        assert meta.has_alpha is True
        assert decoded == px

    def test_round_trip_rgb(self, real_image_data):
        _, rgb_arr, _ = real_image_data
        px = rgb_arr.tobytes()
        h, w, c = rgb_arr.shape
        jxl = pyjpegxl.encode(px, w, h, lossless=True, num_channels=c, speed=pyjpegxl.EncoderSpeed.Lightning)
        meta, decoded = pyjpegxl.decode(jxl)
        assert meta.width == w and meta.height == h
        assert meta.has_alpha is False
        assert decoded == px

    def test_lossy_encode(self, real_image_data):
        _, rgb_arr, _ = real_image_data
        px = rgb_arr.tobytes()
        h, w, c = rgb_arr.shape
        jxl = pyjpegxl.encode(px, w, h, lossless=False, quality=1.0, num_channels=c, speed=pyjpegxl.EncoderSpeed.Lightning)
        meta, _ = pyjpegxl.decode(jxl)
        assert meta.width == w

    def test_decode_invalid(self):
        with pytest.raises(RuntimeError):
            pyjpegxl.decode(b"not jxl")

    def test_encode_wrong_size(self, real_image_data):
        _, _, jxl_bytes = real_image_data
        with pytest.raises(RuntimeError, match="Data length mismatch"):
            pyjpegxl.encode(jxl_bytes, 100, 100, num_channels=4)

    def test_metadata_repr(self, real_image_data):
        _, _, jxl_bytes = real_image_data
        meta, _ = pyjpegxl.decode(jxl_bytes)
        assert "width=" in repr(meta)

    def test_encoder_speed(self, real_image_data):
        _, rgb_arr, _ = real_image_data
        # Downsample drastically to make tests fast
        px = rgb_arr[::16, ::16].tobytes()
        h, w, c = rgb_arr[::16, ::16].shape
        for speed in [
            pyjpegxl.EncoderSpeed.Lightning,
            pyjpegxl.EncoderSpeed.Falcon,
            pyjpegxl.EncoderSpeed.Squirrel,
            pyjpegxl.EncoderSpeed.Tortoise,
        ]:
            jxl = pyjpegxl.encode(px, w, h, speed=speed, num_channels=c)
            meta, _ = pyjpegxl.decode(jxl)
            assert meta.width == w

    def test_metadata_roundtrip(self, real_image_data):
        _, rgb_arr, _ = real_image_data
        px = rgb_arr.tobytes()
        h, w, c = rgb_arr.shape
        
        # Dummy EXIF and XMP signatures
        fake_exif = b"Exif\x00\x00MM\x00*\x00\x00\x00\x08..."
        fake_xmp = b"http://ns.adobe.com/xap/1.0/\x00..."
        
        jxl = pyjpegxl.encode(
            px, w, h, 
            lossless=True, 
            num_channels=c, 
            speed=pyjpegxl.EncoderSpeed.Lightning,
            exif=fake_exif, 
            xmp=fake_xmp
        )
        meta, decoded = pyjpegxl.decode(jxl)
        
        assert meta.exif == fake_exif
        assert meta.xmp == fake_xmp


# ---------------------------------------------------------------------------
# NumPy zero-copy tests
# ---------------------------------------------------------------------------

class TestNumPy:
    def test_decode_to_numpy_shape(self, real_image_data):
        _, _, jxl_bytes = real_image_data
        meta, arr = pyjpegxl.decode_to_numpy(jxl_bytes)
        assert arr.shape == (meta.height, meta.width, meta.num_color_channels + int(meta.has_alpha))
        assert arr.dtype == np.uint8

    def test_numpy_round_trip_lossless(self, real_image_data):
        rgba_arr, _, _ = real_image_data
        jxl = pyjpegxl.encode_from_numpy(rgba_arr, lossless=True, speed=pyjpegxl.EncoderSpeed.Lightning)
        _, decoded = pyjpegxl.decode_to_numpy(jxl)
        np.testing.assert_array_equal(decoded, rgba_arr)

    def test_numpy_round_trip_rgb(self, real_image_data):
        _, rgb_arr, _ = real_image_data
        jxl = pyjpegxl.encode_from_numpy(rgb_arr, lossless=True, speed=pyjpegxl.EncoderSpeed.Lightning)
        _, decoded = pyjpegxl.decode_to_numpy(jxl)
        np.testing.assert_array_equal(decoded, rgb_arr)

    def test_encode_non_contiguous_raises(self, real_image_data):
        _, rgb_arr, _ = real_image_data
        # Fortran-order is not C-contiguous
        arr_f = np.asfortranarray(rgb_arr)
        with pytest.raises(RuntimeError, match="C-contiguous"):
            pyjpegxl.encode_from_numpy(arr_f)

    def test_encode_wrong_ndim_raises(self):
        arr_2d = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="3D"):
            pyjpegxl.encode_from_numpy(arr_2d)

    def test_numpy_array_writable(self, real_image_data):
        """Decoded numpy array should be writable (owned, not read-only)."""
        _, _, jxl_bytes = real_image_data
        # Use a small crop so decode is fast just for the test
        _, rgb_arr, _ = real_image_data
        crop = np.ascontiguousarray(rgb_arr[:8, :8])
        jxl = pyjpegxl.encode_from_numpy(crop, lossless=True, speed=pyjpegxl.EncoderSpeed.Lightning)
        _, decoded = pyjpegxl.decode_to_numpy(jxl)
        decoded[0, 0, 0] = 42  # Should not raise


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------

class TestAsync:
    @pytest.mark.asyncio
    async def test_async_decode(self, real_image_data):
        rgba_arr, _, _ = real_image_data
        jxl = pyjpegxl.encode_from_numpy(rgba_arr, lossless=True, speed=pyjpegxl.EncoderSpeed.Lightning)
        meta, decoded = await pyjpegxl.async_decode(jxl)
        assert meta.width == rgba_arr.shape[1]
        assert decoded == rgba_arr.tobytes()

    @pytest.mark.asyncio
    async def test_async_encode(self, real_image_data):
        _, rgb_arr, jxl_bytes = real_image_data
        px = rgb_arr.tobytes()
        h, w, c = rgb_arr.shape
        jxl = await pyjpegxl.async_encode(px, w, h, lossless=True, num_channels=c, speed=pyjpegxl.EncoderSpeed.Lightning)
        meta, decoded = pyjpegxl.decode(jxl)
        assert meta.width == w
        assert decoded == px

    @pytest.mark.asyncio
    async def test_async_numpy_round_trip(self, real_image_data):
        _, rgb_arr, _ = real_image_data
        jxl = await pyjpegxl.async_encode_from_numpy(rgb_arr, lossless=True, speed=pyjpegxl.EncoderSpeed.Lightning)
        meta, decoded = await pyjpegxl.async_decode_to_numpy(jxl)
        np.testing.assert_array_equal(decoded, rgb_arr)


# ---------------------------------------------------------------------------
# File I/O tests
# ---------------------------------------------------------------------------

class TestFileIO:
    def test_write_read_bytes(self, real_image_data, tmp_path):
        _, rgb_arr, _ = real_image_data
        px = rgb_arr.tobytes()
        h, w, c = rgb_arr.shape
        out = tmp_path / "test.jxl"
        n = pyjpegxl.write(out, px, w, h, lossless=True, num_channels=c, speed=pyjpegxl.EncoderSpeed.Lightning)
        assert n > 0
        assert out.exists()
        meta, decoded = pyjpegxl.read(out)
        assert meta.width == w and meta.height == h
        assert decoded == px

    def test_write_read_numpy(self, real_image_data, tmp_path):
        _, rgb_arr, _ = real_image_data
        out = tmp_path / "test_np.jxl"
        n = pyjpegxl.write_from_numpy(out, rgb_arr, lossless=True, speed=pyjpegxl.EncoderSpeed.Lightning)
        assert n > 0
        meta, decoded = pyjpegxl.read_to_numpy(out)
        np.testing.assert_array_equal(decoded, rgb_arr)

    def test_str_path(self, real_image_data, tmp_path):
        _, rgb_arr, _ = real_image_data
        out = str(tmp_path / "str_path.jxl")
        pyjpegxl.write_from_numpy(out, rgb_arr, lossless=True, speed=pyjpegxl.EncoderSpeed.Lightning)
        meta, decoded = pyjpegxl.read_to_numpy(out)
        np.testing.assert_array_equal(decoded, rgb_arr)

    def test_write_creates_parent_dirs(self, real_image_data, tmp_path):
        _, rgb_arr, _ = real_image_data
        out = tmp_path / "sub" / "dir" / "nested.jxl"
        pyjpegxl.write_from_numpy(out, rgb_arr, lossless=True, speed=pyjpegxl.EncoderSpeed.Lightning)
        assert out.exists()

    def test_read_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            pyjpegxl.read("/nonexistent/path.jxl")

    @pytest.mark.asyncio
    async def test_async_write_read(self, real_image_data, tmp_path):
        _, rgb_arr, _ = real_image_data
        out = tmp_path / "async.jxl"
        n = await pyjpegxl.async_write_from_numpy(out, rgb_arr, lossless=True, speed=pyjpegxl.EncoderSpeed.Lightning)
        assert n > 0
        meta, decoded = await pyjpegxl.async_read_to_numpy(out)
        np.testing.assert_array_equal(decoded, rgb_arr)


# ---------------------------------------------------------------------------
# JPEG codec tests
# ---------------------------------------------------------------------------

TEST_JPG = IMAGES_DIR / "test.jpg"


class TestJPEG:
    @pytest.fixture(scope="class")
    def jpeg_rgb_arr(self):
        """Get an RGB array from test.jpg or generate a synthetic one."""
        if TEST_JPG.exists():
            info, arr = pyjpegxl.jpeg_read_to_numpy(TEST_JPG)
            return arr
        else:
            return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    def test_bytes_round_trip(self, jpeg_rgb_arr):
        px = jpeg_rgb_arr.tobytes()
        h, w, c = jpeg_rgb_arr.shape
        jpeg = pyjpegxl.jpeg_encode(px, w, h, quality=100, num_channels=c)
        info, decoded = pyjpegxl.jpeg_decode(jpeg)
        assert info.width == w and info.height == h
        assert info.num_channels == 3
        assert len(decoded) == len(px)

    def test_numpy_round_trip(self, jpeg_rgb_arr):
        jpeg = pyjpegxl.jpeg_encode_from_numpy(jpeg_rgb_arr, quality=100)
        info, decoded = pyjpegxl.jpeg_decode_to_numpy(jpeg)
        assert decoded.shape == jpeg_rgb_arr.shape
        assert decoded.dtype == np.uint8
        # JPEG is lossy, so allow small differences at high quality
        assert np.mean(np.abs(decoded.astype(int) - jpeg_rgb_arr.astype(int))) < 3

    def test_quality_affects_size(self, jpeg_rgb_arr):
        low = pyjpegxl.jpeg_encode_from_numpy(jpeg_rgb_arr, quality=10)
        high = pyjpegxl.jpeg_encode_from_numpy(jpeg_rgb_arr, quality=95)
        assert len(low) < len(high)

    def test_decode_invalid(self):
        with pytest.raises(RuntimeError):
            pyjpegxl.jpeg_decode(b"not jpeg")

    def test_encode_wrong_size(self):
        with pytest.raises(RuntimeError, match="Data length mismatch"):
            pyjpegxl.jpeg_encode(b"short", 100, 100, num_channels=3)

    def test_jpeg_info_repr(self, jpeg_rgb_arr):
        jpeg = pyjpegxl.jpeg_encode_from_numpy(jpeg_rgb_arr, quality=80)
        info, _ = pyjpegxl.jpeg_decode(jpeg)
        assert "width=" in repr(info)

    def test_file_write_read_numpy(self, jpeg_rgb_arr, tmp_path):
        out = tmp_path / "test.jpg"
        n = pyjpegxl.jpeg_write_from_numpy(out, jpeg_rgb_arr, quality=95)
        assert n > 0
        assert out.exists()
        info, decoded = pyjpegxl.jpeg_read_to_numpy(out)
        assert decoded.shape == jpeg_rgb_arr.shape

    def test_file_write_read_bytes(self, jpeg_rgb_arr, tmp_path):
        px = jpeg_rgb_arr.tobytes()
        h, w, c = jpeg_rgb_arr.shape
        out = tmp_path / "test_bytes.jpg"
        n = pyjpegxl.jpeg_write(out, px, w, h, quality=95, num_channels=c)
        assert n > 0
        info, decoded = pyjpegxl.jpeg_read(out)
        assert info.width == w and info.height == h

    @pytest.mark.asyncio
    async def test_async_jpeg_round_trip(self, jpeg_rgb_arr, tmp_path):
        out = tmp_path / "async.jpg"
        n = await pyjpegxl.async_jpeg_write_from_numpy(out, jpeg_rgb_arr, quality=95)
        assert n > 0
        info, decoded = await pyjpegxl.async_jpeg_read_to_numpy(out)
        assert decoded.shape == jpeg_rgb_arr.shape

