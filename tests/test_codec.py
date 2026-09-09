"""Tests for pyjpegxl encode/decode — bytes, numpy, async, concurrency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyjpegxl
import pytest

IMAGES_DIR = Path(__file__).parent.parent / "images"
TEST_JXL = IMAGES_DIR / "test.jxl"

# Skip all tests if images are missing
pytestmark = pytest.mark.skipif(
    not TEST_JXL.exists(), reason="images/test.jxl not found"
)  # ---------------------------------------------------------------------------
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
        jxl = pyjpegxl.encode(
            px, w, h, lossless=False, quality=1.0, num_channels=c, speed=pyjpegxl.EncoderSpeed.Lightning
        )
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
            px, w, h, lossless=True, num_channels=c, speed=pyjpegxl.EncoderSpeed.Lightning, exif=fake_exif, xmp=fake_xmp
        )
        meta, decoded = pyjpegxl.decode(jxl)

        assert meta.exif == fake_exif
        assert meta.xmp == fake_xmp

    def test_version(self):
        assert pyjpegxl.__version__ == "0.2.2"

    def test_large_metadata_roundtrip(self, real_image_data):
        """Test EXIF/XMP larger than 64KB to verify dynamic buffer expansion."""
        _, rgb_arr, _ = real_image_data
        px = rgb_arr.tobytes()
        h, w, c = rgb_arr.shape

        # 100KB EXIF and 80KB XMP to trigger buffer reallocation in libjxl box parser
        large_exif = b"Exif\x00\x00MM\x00*\x00\x00\x00\x08" + b"E" * 100_000
        large_xmp = b"http://ns.adobe.com/xap/1.0/\x00" + b"X" * 80_000

        jxl = pyjpegxl.encode(
            px,
            w,
            h,
            lossless=True,
            num_channels=c,
            speed=pyjpegxl.EncoderSpeed.Lightning,
            exif=large_exif,
            xmp=large_xmp,
        )
        meta, _ = pyjpegxl.decode(jxl)
        assert meta.exif == large_exif
        assert meta.xmp == large_xmp


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
        arr_1d = np.zeros((10,), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="2D.*3D"):
            pyjpegxl.encode_from_numpy(arr_1d)
        arr_4d = np.zeros((2, 2, 2, 2), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="2D.*3D"):
            pyjpegxl.encode_from_numpy(arr_4d)

    def test_encode_2d_grayscale(self):
        arr_2d = np.full((16, 16), 128, dtype=np.uint8)
        jxl = pyjpegxl.encode_from_numpy(arr_2d, lossless=True, speed=pyjpegxl.EncoderSpeed.Lightning)
        meta, decoded = pyjpegxl.decode_to_numpy(jxl)
        assert meta.width == 16
        assert meta.height == 16
        assert meta.num_color_channels == 1
        assert not meta.has_alpha
        assert decoded.shape == (16, 16, 1)
        assert np.array_equal(decoded[:, :, 0], arr_2d)

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
        jxl = await pyjpegxl.async_encode(
            px, w, h, lossless=True, num_channels=c, speed=pyjpegxl.EncoderSpeed.Lightning
        )
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
    @classmethod
    def jpeg_rgb_arr(cls):
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


# ---------------------------------------------------------------------------
# JPEG ↔ JXL lossless transcoding tests
# ---------------------------------------------------------------------------


class TestTranscoding:
    @pytest.fixture(scope="class")
    @classmethod
    def jpeg_bytes(cls):
        """Get raw JPEG bytes from test.jpg or generate via turbojpeg."""
        if TEST_JPG.exists():
            with open(TEST_JPG, "rb") as f:
                return f.read()
        else:
            # Generate a synthetic JPEG via the jpeg_encode pipeline
            arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            return pyjpegxl.jpeg_encode_from_numpy(arr, quality=95)

    def test_jpeg_to_jxl_basic(self, jpeg_bytes):
        jxl = pyjpegxl.jpeg_to_jxl(jpeg_bytes)
        assert len(jxl) > 0
        # JXL files start with signature 0xFF0A or container 0x0000000C
        assert jxl[:2] == b"\xff\x0a" or jxl[:4] == b"\x00\x00\x00\x0c"

    def test_jpeg_to_jxl_smaller(self, jpeg_bytes):
        jxl = pyjpegxl.jpeg_to_jxl(jpeg_bytes)
        assert len(jxl) < len(jpeg_bytes), f"JXL ({len(jxl)} B) should be smaller than JPEG ({len(jpeg_bytes)} B)"

    def test_roundtrip_byte_exact(self, jpeg_bytes):
        """JPEG → JXL → JPEG must produce the exact same JPEG bytes."""
        jxl = pyjpegxl.jpeg_to_jxl(jpeg_bytes)
        reconstructed = pyjpegxl.jxl_to_jpeg(jxl)
        assert reconstructed == jpeg_bytes

    def test_jxl_to_jpeg_non_jpeg_source_raises(self, tmp_path):
        """jxl_to_jpeg on a JXL not from JPEG transcoding should raise."""
        # Use a real non-JPEG image (PNG)
        from PIL import Image

        png_path = Path(__file__).parent.parent / "images" / "test.png"
        rgb_arr = np.array(Image.open(png_path).convert("RGB"))

        jxl = pyjpegxl.encode_from_numpy(
            rgb_arr,
            lossless=True,
            speed=pyjpegxl.EncoderSpeed.Lightning,
        )
        with pytest.raises(RuntimeError):
            pyjpegxl.jxl_to_jpeg(jxl)

    def test_file_transcoding_roundtrip(self, jpeg_bytes, tmp_path):
        jpeg_in = tmp_path / "input.jpg"
        jxl_out = tmp_path / "transcoded.jxl"
        jpeg_out = tmp_path / "reconstructed.jpg"

        jpeg_in.write_bytes(jpeg_bytes)

        n1 = pyjpegxl.jpeg_file_to_jxl(jpeg_in, jxl_out)
        assert n1 > 0
        assert jxl_out.exists()

        n2 = pyjpegxl.jxl_file_to_jpeg(jxl_out, jpeg_out)
        assert n2 > 0
        assert jpeg_out.read_bytes() == jpeg_bytes

    @pytest.mark.asyncio
    async def test_async_transcoding(self, jpeg_bytes, tmp_path):
        jxl = await pyjpegxl.async_jpeg_to_jxl(jpeg_bytes)
        reconstructed = await pyjpegxl.async_jxl_to_jpeg(jxl)
        assert reconstructed == jpeg_bytes

        # File I/O async
        jpeg_in = tmp_path / "async_input.jpg"
        jxl_out = tmp_path / "async_transcoded.jxl"
        jpeg_out = tmp_path / "async_reconstructed.jpg"
        jpeg_in.write_bytes(jpeg_bytes)

        await pyjpegxl.async_jpeg_file_to_jxl(jpeg_in, jxl_out)
        await pyjpegxl.async_jxl_file_to_jpeg(jxl_out, jpeg_out)
        assert jpeg_out.read_bytes() == jpeg_bytes


# ---------------------------------------------------------------------------
# High Bit Depth, HDR & ICC Profile Tests
# ---------------------------------------------------------------------------


class TestHighBitDepthAndIcc:
    def test_uint16_lossless_roundtrip(self):
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 65536, size=(32, 32, 3), dtype=np.uint16)
        jxl = pyjpegxl.encode_from_numpy(
            arr,
            lossless=True,
            speed=pyjpegxl.EncoderSpeed.Lightning,
        )
        meta, decoded = pyjpegxl.decode_to_numpy(jxl, dtype="uint16")
        assert decoded.dtype == np.uint16
        assert meta.bits_per_sample == 16
        assert np.array_equal(arr, decoded)

    def test_float32_hdr_lossless_roundtrip(self):
        rng = np.random.default_rng(42)
        arr = rng.uniform(0.0, 5.0, size=(32, 32, 3)).astype(np.float32)
        jxl = pyjpegxl.encode_from_numpy(
            arr,
            lossless=True,
            speed=pyjpegxl.EncoderSpeed.Lightning,
        )
        meta, decoded = pyjpegxl.decode_to_numpy(jxl, dtype="float32")
        assert decoded.dtype == np.float32
        assert meta.bits_per_sample == 32
        # libjxl float encode/decode maintains high precision
        assert np.allclose(arr, decoded, atol=1e-2)

    def test_auto_dtype_detection(self):
        rng = np.random.default_rng(42)
        # 1. uint8
        u8_arr = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        u8_jxl = pyjpegxl.encode_from_numpy(u8_arr, lossless=True, speed=pyjpegxl.EncoderSpeed.Lightning)
        _, u8_dec = pyjpegxl.decode_to_numpy(u8_jxl)
        assert u8_dec.dtype == np.uint8

        # 2. uint16
        u16_arr = rng.integers(0, 65536, size=(16, 16, 3), dtype=np.uint16)
        u16_jxl = pyjpegxl.encode_from_numpy(u16_arr, lossless=True, speed=pyjpegxl.EncoderSpeed.Lightning)
        _, u16_dec = pyjpegxl.decode_to_numpy(u16_jxl)
        assert u16_dec.dtype == np.uint16

        # 3. float32
        f32_arr = rng.uniform(0.0, 1.0, size=(16, 16, 3)).astype(np.float32)
        f32_jxl = pyjpegxl.encode_from_numpy(f32_arr, lossless=True, speed=pyjpegxl.EncoderSpeed.Lightning)
        _, f32_dec = pyjpegxl.decode_to_numpy(f32_jxl)
        assert f32_dec.dtype == np.float32

    def test_icc_profile_roundtrip(self):
        # A minimal valid-like dummy ICC profile payload
        dummy_icc = b"TEST_ICC_PROFILE_HEADER" + bytes(range(100))
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

        jxl = pyjpegxl.encode_from_numpy(
            arr,
            lossless=True,
            speed=pyjpegxl.EncoderSpeed.Lightning,
            icc=dummy_icc,
        )
        meta, _ = pyjpegxl.decode_to_numpy(jxl)
        assert meta.icc == dummy_icc
        assert meta.icc_profile == dummy_icc

    def test_non_contiguous_array_handling(self, tmp_path):
        # Create non-contiguous slice
        full_arr = np.arange(64 * 64 * 3, dtype=np.uint8).reshape((64, 64, 3))
        sliced = full_arr[::2, ::2, :]
        assert not sliced.flags.c_contiguous

        # encode_from_numpy strictly enforces zero-copy C-contiguous layout
        with pytest.raises(RuntimeError, match="C-contiguous"):
            pyjpegxl.encode_from_numpy(sliced)

        # write_from_numpy and async_write_from_numpy auto-convert non-contiguous arrays
        out_path = tmp_path / "sliced.jxl"
        bytes_written = pyjpegxl.write_from_numpy(
            out_path, sliced, lossless=True, speed=pyjpegxl.EncoderSpeed.Lightning
        )
        assert bytes_written > 0
        _, read_dec = pyjpegxl.read_to_numpy(out_path)
        assert np.array_equal(sliced, read_dec)

    def test_file_io_high_bit_depth_and_icc(self, tmp_path):
        dummy_icc = b"CUSTOM_ICC_FOR_FILE_IO"
        rng = np.random.default_rng(42)
        arr_u16 = rng.integers(0, 65536, size=(24, 24, 3), dtype=np.uint16)

        out_path = tmp_path / "u16_icc.jxl"
        n = pyjpegxl.write_from_numpy(
            out_path,
            arr_u16,
            lossless=True,
            speed=pyjpegxl.EncoderSpeed.Lightning,
            icc=dummy_icc,
        )
        assert n > 0

        meta, dec = pyjpegxl.read_to_numpy(out_path)
        assert dec.dtype == np.uint16
        assert meta.bits_per_sample == 16
        assert meta.icc == dummy_icc
        assert np.array_equal(arr_u16, dec)

    @pytest.mark.asyncio
    async def test_async_high_bit_depth_and_icc(self, tmp_path):
        dummy_icc = b"ASYNC_ICC_PAYLOAD"
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 65536, size=(16, 16, 4), dtype=np.uint16)

        jxl = await pyjpegxl.async_encode_from_numpy(
            arr,
            lossless=True,
            speed=pyjpegxl.EncoderSpeed.Lightning,
            icc=dummy_icc,
        )
        meta, dec = await pyjpegxl.async_decode_to_numpy(jxl)
        assert dec.dtype == np.uint16
        assert meta.icc == dummy_icc
        assert np.array_equal(arr, dec)

        out_path = tmp_path / "async_u16.jxl"
        await pyjpegxl.async_write_from_numpy(
            out_path,
            arr,
            lossless=True,
            speed=pyjpegxl.EncoderSpeed.Lightning,
            icc=dummy_icc,
        )
        meta_read, dec_read = await pyjpegxl.async_read_to_numpy(out_path)
        assert dec_read.dtype == np.uint16
        assert meta_read.icc == dummy_icc
        assert np.array_equal(arr, dec_read)


class TestPerformanceAndConcurrency:
    """Tests for lightweight probing, zero-allocation decode_into, thread pool control, and intensity_target."""

    def test_probe_and_probe_file(self, real_image_data, tmp_path):
        _, _, jxl_bytes = real_image_data
        jxl_file = tmp_path / "probe_test.jxl"
        jxl_file.write_bytes(jxl_bytes)

        # Full decode metadata
        full_meta, _ = pyjpegxl.decode(jxl_bytes)

        # Probe bytes
        probe_meta = pyjpegxl.probe(jxl_bytes)
        assert probe_meta.width == full_meta.width
        assert probe_meta.height == full_meta.height
        assert probe_meta.num_color_channels == full_meta.num_color_channels
        assert probe_meta.has_alpha == full_meta.has_alpha
        assert probe_meta.bits_per_sample == full_meta.bits_per_sample

        # Probe file
        file_meta = pyjpegxl.probe_file(jxl_file)
        assert file_meta.width == full_meta.width
        assert file_meta.height == full_meta.height
        assert file_meta.bits_per_sample == full_meta.bits_per_sample

    def test_decode_into_uint8(self, real_image_data, tmp_path):
        _, expected_arr, jxl_bytes = real_image_data
        jxl_file = tmp_path / "decode_into.jxl"
        jxl_file.write_bytes(jxl_bytes)

        h, w, c = expected_arr.shape
        out = np.zeros((h, w, c), dtype=np.uint8)

        meta = pyjpegxl.decode_into(jxl_bytes, out)
        assert meta.width == w
        assert meta.height == h
        assert np.array_equal(out, expected_arr)

        # Test read_into
        out_file = np.zeros((h, w, c), dtype=np.uint8)
        meta_file = pyjpegxl.read_into(jxl_file, out_file)
        assert meta_file.width == w
        assert np.array_equal(out_file, expected_arr)

    def test_decode_into_uint16_and_float32(self):
        rng = np.random.default_rng(123)
        # 16-bit
        arr16 = rng.integers(0, 65536, size=(16, 16, 3), dtype=np.uint16)
        jxl16 = pyjpegxl.encode_from_numpy(arr16, lossless=True, speed=pyjpegxl.EncoderSpeed.Lightning)
        out16 = np.zeros_like(arr16)
        meta16 = pyjpegxl.decode_into(jxl16, out16)
        assert meta16.bits_per_sample == 16
        assert np.array_equal(out16, arr16)

        # float32
        arr_f32 = rng.random(size=(16, 16, 3), dtype=np.float32)
        jxl_f32 = pyjpegxl.encode_from_numpy(arr_f32, lossless=True, speed=pyjpegxl.EncoderSpeed.Lightning)
        out_f32 = np.zeros_like(arr_f32)
        meta_f32 = pyjpegxl.decode_into(jxl_f32, out_f32)
        assert meta_f32.bits_per_sample == 32
        assert np.allclose(out_f32, arr_f32, atol=1e-4)

    def test_decode_into_buffer_mismatch_raises(self, real_image_data):
        _, expected_arr, jxl_bytes = real_image_data
        h, w, c = expected_arr.shape

        # Wrong size
        too_small = np.zeros((h // 2, w // 2, c), dtype=np.uint8)
        with pytest.raises(RuntimeError):
            pyjpegxl.decode_into(jxl_bytes, too_small)

        # Wrong dtype
        wrong_dtype = np.zeros((h, w, c), dtype=np.int32)
        with pytest.raises(TypeError):
            pyjpegxl.decode_into(jxl_bytes, wrong_dtype)

    def test_set_and_get_num_threads(self, real_image_data):
        _, expected_arr, jxl_bytes = real_image_data
        # Test getting initial setting
        initial_threads = pyjpegxl.get_num_threads()
        assert isinstance(initial_threads, int)

        try:
            # Test pure single-threaded mode (runner bypass)
            pyjpegxl.set_num_threads(1)
            assert pyjpegxl.get_num_threads() == 1
            meta1, dec1 = pyjpegxl.decode_to_numpy(jxl_bytes)
            assert meta1.width == expected_arr.shape[1]

            # Test explicit multi-threaded mode
            pyjpegxl.set_num_threads(2)
            assert pyjpegxl.get_num_threads() == 2
            meta2, dec2 = pyjpegxl.decode_to_numpy(jxl_bytes)
            assert np.array_equal(dec1, dec2)

            # Test auto mode
            pyjpegxl.set_num_threads(0)
            assert pyjpegxl.get_num_threads() == 0
        finally:
            pyjpegxl.set_num_threads(initial_threads)

    def test_intensity_target_encoding(self):
        arr = np.full((16, 16, 3), 128, dtype=np.uint8)
        jxl = pyjpegxl.encode_from_numpy(
            arr,
            lossless=True,
            speed=pyjpegxl.EncoderSpeed.Lightning,
            intensity_target=1000.0,
        )
        meta = pyjpegxl.probe(jxl)
        assert meta.intensity_target == pytest.approx(1000.0, rel=1e-1)

    @pytest.mark.asyncio
    async def test_async_probe_and_decode_into(self, real_image_data, tmp_path):
        _, expected_arr, jxl_bytes = real_image_data
        jxl_file = tmp_path / "async_perf.jxl"
        jxl_file.write_bytes(jxl_bytes)

        # Async probe
        meta1 = await pyjpegxl.async_probe(jxl_bytes)
        assert meta1.width == expected_arr.shape[1]

        meta2 = await pyjpegxl.async_probe_file(jxl_file)
        assert meta2.width == expected_arr.shape[1]

        # Async decode_into
        out = np.zeros_like(expected_arr)
        meta3 = await pyjpegxl.async_decode_into(jxl_bytes, out)
        assert meta3.height == expected_arr.shape[0]
        assert np.array_equal(out, expected_arr)

        # Async read_into
        out_file = np.zeros_like(expected_arr)
        meta4 = await pyjpegxl.async_read_into(jxl_file, out_file)
        assert meta4.height == expected_arr.shape[0]
        assert np.array_equal(out_file, expected_arr)
