"""Tests for ecosystem convenience, data bridging, polymorphic I/O, and Pillow/PyTorch integrations."""

from __future__ import annotations

import io
import os
import sys
import unittest.mock

import numpy as np
import pyjpegxl
import pytest
from PIL import Image
from pyjpegxl import (
    PrefixedStream,
    decode_into_tensor,
    from_pil,
    from_tensor,
    imread,
    imwrite,
    probe_image,
    read_batch,
    sniff_bytes,
    sniff_source,
    sniff_stream,
    to_pil,
    to_tensor,
    transcode_batch,
)


@pytest.fixture
def sample_rgb_image():
    """Deterministic 64x64 RGB uint8 image."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :, 0] = np.arange(64, dtype=np.uint8)[:, None]
    arr[:, :, 1] = np.arange(64, dtype=np.uint8)[None, :]
    arr[:, :, 2] = 128
    return arr


@pytest.fixture
def sample_jxl_bytes(sample_rgb_image):
    return pyjpegxl.encode_from_numpy(sample_rgb_image, quality=1.0)


@pytest.fixture
def sample_jpeg_bytes(sample_rgb_image):
    return pyjpegxl.jpeg_encode_from_numpy(sample_rgb_image, quality=90)


# ===========================================================================
# 1. Sniffing and Stream Tests
# ===========================================================================


class TestSniffingAndStreams:
    def test_sniff_bytes(self, sample_jxl_bytes, sample_jpeg_bytes):
        assert sniff_bytes(sample_jpeg_bytes) == "jpeg"
        assert sniff_bytes(sample_jxl_bytes) == "jxl"
        assert sniff_bytes(b"\xff\x0a\x00\x00") == "jxl"  # bare codestream
        assert sniff_bytes(b"NOT_AN_IMAGE_HEADER") == "unknown"

    def test_prefixed_stream(self):
        prefix = b"12345"
        raw_io = io.BytesIO(b"67890")
        stream = PrefixedStream(prefix, raw_io)

        assert stream.readable() is True
        assert stream.seekable() is False
        assert stream.writable() is False

        # Read partial
        assert stream.read(3) == b"123"
        # Read past prefix
        assert stream.read(4) == b"4567"
        # Read rest
        assert stream.read() == b"890"
        assert stream.read(0) == b""

        # Readinto test
        raw2 = io.BytesIO(b"cd")
        s2 = PrefixedStream(b"ab", raw2)
        buf = bytearray(3)
        n = s2.readinto(buf)
        assert n == 3
        assert bytes(buf) == b"abc"
        s2.close()

    def test_sniff_stream_seekable(self, sample_jxl_bytes):
        bio = io.BytesIO(sample_jxl_bytes)
        fmt, stream = sniff_stream(bio)
        assert fmt == "jxl"
        assert stream.tell() == 0
        assert stream.read() == sample_jxl_bytes

    def test_sniff_stream_non_seekable(self, sample_jpeg_bytes):
        class NonSeekableStream(io.RawIOBase):
            def __init__(self, data: bytes):
                self._data = data
                self._pos = 0

            def read(self, size: int = -1) -> bytes:
                if size < 0:
                    chunk = self._data[self._pos :]
                    self._pos = len(self._data)
                    return chunk
                chunk = self._data[self._pos : self._pos + size]
                self._pos += len(chunk)
                return chunk

            def seekable(self) -> bool:
                return False

            def readable(self) -> bool:
                return True

        non_seek = NonSeekableStream(sample_jpeg_bytes)
        fmt, stream = sniff_stream(non_seek)
        assert fmt == "jpeg"
        # Reading from wrapped stream recovers full data
        recovered = stream.read()
        assert recovered == sample_jpeg_bytes

    def test_sniff_stream_buffered_peek(self, sample_jxl_bytes):
        # Test io.BufferedReader peek() path
        raw_io = io.BytesIO(sample_jxl_bytes)
        buf_reader = io.BufferedReader(raw_io)
        fmt, stream = sniff_stream(buf_reader)
        assert fmt == "jxl"
        assert stream.read() == sample_jxl_bytes

    def test_sniff_stream_seek_exception(self, sample_jpeg_bytes):
        # Test seekable() exception fallback path
        class ErrorStream(io.BytesIO):
            def seekable(self) -> bool:
                raise io.UnsupportedOperation("Not seekable")

        err_stream = ErrorStream(sample_jpeg_bytes)
        fmt, stream = sniff_stream(err_stream)
        assert fmt == "jpeg"
        assert stream.read() == sample_jpeg_bytes

    def test_sniff_source(self, tmp_path, sample_jxl_bytes, sample_jpeg_bytes):
        p_jxl = tmp_path / "test.jxl"
        p_jxl.write_bytes(sample_jxl_bytes)

        fmt, data = sniff_source(p_jxl)
        assert fmt == "jxl"
        assert data == sample_jxl_bytes

        fmt, data = sniff_source(memoryview(sample_jpeg_bytes))
        assert fmt == "jpeg"
        assert data == sample_jpeg_bytes

        # Test sniff_source with stream
        bio = io.BytesIO(sample_jxl_bytes)
        fmt, data = sniff_source(bio)
        assert fmt == "jxl"
        assert data == sample_jxl_bytes

        with pytest.raises(TypeError, match="Unsupported source type"):
            sniff_source(12345)  # type: ignore


# ===========================================================================
# 2. Unified Polymorphic I/O (imread / imwrite / probe_image)
# ===========================================================================


class TestUnifiedIO:
    def test_imread_jxl_and_jpeg(self, sample_jxl_bytes, sample_jpeg_bytes, sample_rgb_image):
        meta_jxl, arr_jxl = imread(sample_jxl_bytes)
        assert meta_jxl.width == 64
        assert meta_jxl.height == 64
        assert arr_jxl.shape == (64, 64, 3)

        info_jpeg, arr_jpeg = imread(sample_jpeg_bytes)
        assert info_jpeg.width == 64
        assert info_jpeg.height == 64
        assert arr_jpeg.shape == (64, 64, 3)

        with pytest.raises(ValueError, match="Unsupported or corrupted image format"):
            imread(b"corrupted_garbage_bytes")

    def test_imwrite_path_inference(self, tmp_path, sample_rgb_image):
        jxl_path = tmp_path / "subdir" / "out.jxl"
        imwrite(jxl_path, sample_rgb_image, quality=1.0)
        assert jxl_path.exists()
        meta, arr = imread(jxl_path)
        assert meta.width == 64

        jpg_path = tmp_path / "out.jpg"
        imwrite(jpg_path, sample_rgb_image, quality=85)
        assert jpg_path.exists()
        info, arr = imread(jpg_path)
        assert info.width == 64

        # Test writing to relative path with empty parent_dir
        local_tmp = "tmp_local_test.jxl"
        try:
            imwrite(local_tmp, sample_rgb_image, quality=1.0)
            assert os.path.exists(local_tmp)
        finally:
            if os.path.exists(local_tmp):
                os.remove(local_tmp)

        with pytest.raises(ValueError, match="Cannot infer image format"):
            imwrite(tmp_path / "unknown.bin", sample_rgb_image)

    def test_imwrite_to_stream(self, sample_rgb_image):
        bio = io.BytesIO()
        imwrite(bio, sample_rgb_image, format="jxl", quality=1.0)
        assert sniff_bytes(bio.getvalue()[:16]) == "jxl"

        bio_jpg = io.BytesIO()
        imwrite(bio_jpg, sample_rgb_image, format="jpeg", quality=90)
        assert bio_jpg.getvalue().startswith(b"\xff\xd8\xff")

        with pytest.raises(ValueError, match="Must explicitly specify 'format'"):
            imwrite(io.BytesIO(), sample_rgb_image)

        with pytest.raises(ValueError, match="Unsupported format"):
            imwrite(io.BytesIO(), sample_rgb_image, format="png")

        with pytest.raises(TypeError, match="Unsupported destination type"):
            imwrite(12345, sample_rgb_image, format="jxl")  # type: ignore

    def test_probe_image(self, tmp_path, sample_jxl_bytes, sample_jpeg_bytes):
        meta = probe_image(sample_jxl_bytes)
        assert meta.width == 64
        assert meta.height == 64

        info = probe_image(sample_jpeg_bytes)
        assert info.width == 64
        assert info.height == 64

        with pytest.raises(ValueError, match="Unsupported or corrupted"):
            probe_image(b"invalid")


# ===========================================================================
# 3. Pillow Bridge & Plugin Tests
# ===========================================================================


class TestPillowBridge:
    def test_to_pil_and_from_pil_rgb(self, sample_rgb_image):
        pil_img = to_pil(sample_rgb_image)
        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (64, 64)
        assert pil_img.mode == "RGB"

        arr, meta = from_pil(pil_img)
        assert arr.shape == (64, 64, 3)
        assert np.array_equal(arr, sample_rgb_image)

    def test_to_pil_grayscale(self):
        gray = np.zeros((32, 32), dtype=np.uint8)
        gray[10:20, 10:20] = 255
        pil_gray = to_pil(gray)
        assert pil_gray.mode == "L"

        gray_3d = gray[:, :, None]
        pil_gray_3d = to_pil(gray_3d)
        assert pil_gray_3d.mode == "L"

        gray_16 = np.zeros((16, 16), dtype=np.uint16)
        pil_16 = to_pil(gray_16, preserve_hdr=True)
        assert pil_16.mode == "I;16"

        pil_16_downscale = to_pil(gray_16, preserve_hdr=False)
        assert pil_16_downscale.mode == "L"

        gray_f32 = np.zeros((16, 16), dtype=np.float32)
        pil_f32 = to_pil(gray_f32, preserve_hdr=True)
        assert pil_f32.mode == "F"

    def test_to_pil_rgba_and_hdr_rgb(self):
        rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        pil_rgba = to_pil(rgba)
        assert pil_rgba.mode == "RGBA"

        # uint16 RGB downscaling
        rgb16 = np.zeros((16, 16, 3), dtype=np.uint16)
        pil_rgb16 = to_pil(rgb16)
        assert pil_rgb16.mode == "RGB"

        # uint16 RGBA downscaling
        rgba16 = np.zeros((16, 16, 4), dtype=np.uint16)
        pil_rgba16 = to_pil(rgba16)
        assert pil_rgba16.mode == "RGBA"

    def test_to_pil_from_bytes(self, sample_jxl_bytes):
        pil_img = to_pil(sample_jxl_bytes)
        assert pil_img.size == (64, 64)
        assert pil_img.mode == "RGB"

    def test_to_pil_metadata_attachment(self, sample_rgb_image):
        class DummyMeta:
            icc = b"DUMMY_ICC"
            exif = b"DUMMY_EXIF"

        pil_img = to_pil(sample_rgb_image, metadata=DummyMeta())
        assert pil_img.info.get("icc_profile") == b"DUMMY_ICC"
        assert pil_img.info.get("exif") == b"DUMMY_EXIF"

        class DummyIccOnly:
            icc = b"ONLY_ICC"
            exif = None

        class DummyExifOnly:
            icc = None
            exif = b"ONLY_EXIF"

        p1 = to_pil(sample_rgb_image, metadata=DummyIccOnly())
        assert p1.info.get("icc_profile") == b"ONLY_ICC"
        assert "exif" not in p1.info

        p2 = to_pil(sample_rgb_image, metadata=DummyExifOnly())
        assert "icc_profile" not in p2.info
        assert p2.info.get("exif") == b"ONLY_EXIF"

    def test_to_pil_errors(self):
        with pytest.raises(ValueError, match="Unsupported channel count"):
            to_pil(np.zeros((10, 10, 5), dtype=np.uint8))

        with pytest.raises(ValueError, match="Unsupported array shape"):
            to_pil(np.zeros((10, 10, 3, 2), dtype=np.uint8))

    def test_pillow_plugin_registration(self, tmp_path, sample_jxl_bytes):
        # Test duplicate register
        pyjpegxl.pillow.register()
        pyjpegxl.pillow.register()

        jxl_file = tmp_path / "test_plugin.jxl"
        jxl_file.write_bytes(sample_jxl_bytes)

        with Image.open(jxl_file) as img:
            assert img.format == "JXL"
            assert img.size == (64, 64)
            assert img.mode == "RGB"
            # Trigger load
            img.load()
            assert img.size == (64, 64)

            # Test saving via Pillow
            out_file = tmp_path / "saved_via_pil.jxl"
            img.save(out_file, "JXL", quality=1.0)
            assert out_file.exists()
            assert sniff_bytes(out_file.read_bytes()[:16]) == "jxl"

        # Test grayscale JXL opening via Pillow
        gray_arr = np.zeros((32, 32), dtype=np.uint8)
        gray_jxl = pyjpegxl.encode_from_numpy(gray_arr, quality=1.0)
        p_gray = tmp_path / "gray.jxl"
        p_gray.write_bytes(gray_jxl)
        with Image.open(p_gray) as img_g:
            assert img_g.mode == "L"
            img_g.load()

        # Test RGBA JXL opening with metadata via Pillow
        rgba_arr = np.zeros((16, 16, 4), dtype=np.uint8)
        rgba_jxl = pyjpegxl.encode_from_numpy(rgba_arr, quality=1.0, exif=b"SAMPLE_EXIF", icc=b"SAMPLE_ICC")
        p_rgba = tmp_path / "rgba.jxl"
        p_rgba.write_bytes(rgba_jxl)
        with Image.open(p_rgba) as img_rgba:
            assert img_rgba.mode == "RGBA"
            assert img_rgba.info.get("exif") == b"SAMPLE_EXIF"
            assert img_rgba.info.get("icc_profile") == b"SAMPLE_ICC"
            img_rgba.load()

    def test_pillow_missing_import_error(self, monkeypatch):
        # Temporarily mock PIL import failure
        with unittest.mock.patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            with pytest.raises(ImportError, match="Pillow is required"):
                pyjpegxl.pillow._check_pillow()


# ===========================================================================
# 4. PyTorch Bridge Tests
# ===========================================================================


class TestTorchBridge:
    def test_torch_missing_error(self):
        # PyTorch is not installed in current venv, should raise friendly ImportError
        with pytest.raises(ImportError, match="PyTorch is required"):
            to_tensor(np.zeros((10, 10, 3), dtype=np.uint8))

        with pytest.raises(ImportError, match="PyTorch is required"):
            from_tensor(None)

        with pytest.raises(ImportError, match="PyTorch is required"):
            decode_into_tensor(b"dummy", None)

    def test_torch_mocked_to_tensor_and_from_tensor(self, sample_rgb_image, sample_jxl_bytes):
        # Mock torch module to verify dimension handling and contiguous zero-copy
        class MockTensor:
            def __init__(self, arr, is_cuda=False, contiguous=True):
                self._arr = arr
                self.ndim = arr.ndim
                self.shape = arr.shape
                self._is_cuda = is_cuda
                self._contiguous = contiguous

            def permute(self, *dims):
                new_arr = np.transpose(self._arr, dims)
                return MockTensor(new_arr, self._is_cuda, contiguous=False)

            def contiguous(self):
                return MockTensor(np.ascontiguousarray(self._arr), self._is_cuda, contiguous=True)

            def is_contiguous(self):
                return self._contiguous

            def detach(self):
                return self

            def cpu(self):
                return MockTensor(self._arr, is_cuda=False, contiguous=self._contiguous)

            def numpy(self):
                return self._arr

            @property
            def is_cuda(self):
                return self._is_cuda

            def float(self):
                return MockTensor(self._arr.astype(np.float32), self._is_cuda, self._contiguous)

            def __truediv__(self, val):
                return MockTensor(self._arr / val, self._is_cuda, self._contiguous)

            def to(self, *args, **kwargs):
                return self

        mock_torch = unittest.mock.MagicMock()
        mock_torch.from_numpy.side_effect = lambda arr: MockTensor(arr)
        mock_torch.channels_last = "channels_last"

        with unittest.mock.patch("pyjpegxl.torch._check_torch", return_value=mock_torch):
            # 1. Standard HWC (default, zero-copy, contiguous)
            t = to_tensor(sample_rgb_image, permute_chw=False)
            assert t.shape == (64, 64, 3)
            assert t.is_contiguous() is True

            # 2. CHW permutation
            t_chw = to_tensor(sample_rgb_image, permute_chw=True)
            assert t_chw.shape == (3, 64, 64)

            # 3. from_tensor conversion back to HWC
            arr_back = from_tensor(t_chw)
            assert arr_back.shape == (64, 64, 3)

            # 4. Normalized tensor
            t_norm = to_tensor(sample_rgb_image, normalize=True)
            assert t_norm.shape == (64, 64, 3)

            # 5. to_tensor from raw bytes
            t_bytes = to_tensor(sample_jxl_bytes)
            assert t_bytes.shape == (64, 64, 3)

            # 6. to_tensor with 4D and channels_last
            arr_4d = np.zeros((2, 32, 32, 3), dtype=np.uint8)
            t_4d = to_tensor(arr_4d, channels_last=True)
            assert t_4d.ndim == 4

            # 7. to_tensor with device
            t_dev = to_tensor(sample_rgb_image, device="cpu")
            assert t_dev is not None

            # 8. from_tensor with 2D array and non-contiguous tensor
            t_2d_noncontig = MockTensor(np.zeros((32, 32), dtype=np.uint8), contiguous=False)
            arr_2d = from_tensor(t_2d_noncontig)
            assert arr_2d.shape == (32, 32)

    def test_torch_mocked_decode_into_tensor(self, sample_jxl_bytes, sample_jpeg_bytes):
        out_jxl = np.zeros((64, 64, 3), dtype=np.uint8)
        out_jpeg = np.zeros((64, 64, 3), dtype=np.uint8)

        class MockContiguousTensor:
            def __init__(self, arr, is_cuda=False, contiguous=True):
                self._arr = arr
                self.is_cuda = is_cuda
                self._contiguous = contiguous

            def is_contiguous(self):
                return self._contiguous

            def numpy(self):
                return self._arr

        mock_torch = unittest.mock.MagicMock()
        with unittest.mock.patch("pyjpegxl.torch._check_torch", return_value=mock_torch):
            # Target is not contiguous
            with pytest.raises(ValueError, match="Target tensor must be contiguous"):
                decode_into_tensor(sample_jxl_bytes, MockContiguousTensor(out_jxl, contiguous=False))

            # Target is on CUDA
            with pytest.raises(ValueError, match="Target tensor must be on CPU"):
                decode_into_tensor(sample_jxl_bytes, MockContiguousTensor(out_jxl, is_cuda=True))

            # Valid JXL decode into
            meta = decode_into_tensor(sample_jxl_bytes, MockContiguousTensor(out_jxl))
            assert meta.width == 64
            assert out_jxl.max() > 0

            # Valid JPEG decode into
            info = decode_into_tensor(sample_jpeg_bytes, MockContiguousTensor(out_jpeg))
            assert info.width == 64
            assert out_jpeg.max() > 0


# ===========================================================================
# 5. Batch Pipeline Tests
# ===========================================================================


class TestBatchPipeline:
    def test_read_batch(self, tmp_path, sample_rgb_image):
        p1 = tmp_path / "img1.jxl"
        p2 = tmp_path / "img2.jpg"
        p3 = tmp_path / "img3.jxl"

        imwrite(p1, sample_rgb_image, quality=1.0)
        imwrite(p2, sample_rgb_image, quality=90)
        imwrite(p3, sample_rgb_image, quality=1.0)

        results = read_batch([p1, p2, p3], max_workers=2)
        assert len(results) == 3
        for meta, arr in results:
            assert meta.width == 64
            assert arr.shape == (64, 64, 3)

    def test_transcode_batch(self, tmp_path, sample_rgb_image):
        in_dir = tmp_path / "inputs"
        in_dir.mkdir()
        jpg1 = in_dir / "pic1.jpg"
        jpg2 = in_dir / "pic2.jpg"

        imwrite(jpg1, sample_rgb_image, quality=90)
        imwrite(jpg2, sample_rgb_image, quality=90)

        out_dir = tmp_path / "outputs_jxl"
        jxl_files = transcode_batch([jpg1, jpg2], out_dir, target_format="jxl", max_workers=2)
        assert len(jxl_files) == 2
        for p in jxl_files:
            assert p.endswith(".jxl")
            assert os.path.exists(p)

        # Transcode back to JPEG
        back_dir = tmp_path / "outputs_jpeg"
        back_files = transcode_batch(jxl_files, back_dir, target_format="jpeg", max_workers=2)
        assert len(back_files) == 2
        for p in back_files:
            assert p.endswith(".jpg")
            assert os.path.exists(p)

        with pytest.raises(ValueError, match="Unsupported target format"):
            transcode_batch([jpg1], out_dir, target_format="gif")
