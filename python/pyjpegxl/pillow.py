"""Pillow (PIL.Image) zero-copy bridge and official ImageFile plugin registration."""

from __future__ import annotations

from typing import Any

import numpy as np

from pyjpegxl._pyjpegxl import EncoderSpeed, decode_to_numpy, encode_from_numpy, probe
from pyjpegxl._sniff import sniff_bytes


def _check_pillow() -> Any:
    """Ensure Pillow is installed."""
    try:
        from PIL import Image

        return Image
    except ImportError as e:
        raise ImportError(
            "Pillow is required for this operation. Install it with: pip install 'pyjpegxl[pillow]'"
        ) from e


def to_pil(
    image_or_array: np.ndarray | bytes,
    metadata: Any = None,
    *,
    preserve_hdr: bool = True,
) -> Any:
    """Convert decoded NumPy array or raw image bytes to a PIL.Image.Image instance.

    Automatically handles channel mappings (L, RGB, RGBA, I;16, F) and preserves
    ICC profile and EXIF metadata in `pil_img.info`.

    Args:
        image_or_array: Decoded NumPy array of shape (H, W) or (H, W, C), or raw JXL/JPEG bytes.
        metadata: Optional Metadata or JpegInfo instance containing ICC and EXIF info.
        preserve_hdr: Whether to keep high-bit-depth modes where Pillow supports them.

    Returns:
        PIL.Image.Image instance.
    """
    Image = _check_pillow()

    if isinstance(image_or_array, (bytes, bytearray, memoryview)):
        from pyjpegxl._unified import imread

        metadata, arr = imread(bytes(image_or_array))
    else:
        arr = np.ascontiguousarray(image_or_array)

    shape = arr.shape
    ndim = arr.ndim
    dtype = arr.dtype

    if ndim == 2 or (ndim == 3 and shape[2] == 1):
        if ndim == 3:
            arr = arr.squeeze(2)
        if dtype == np.uint8:
            pass
        elif dtype == np.uint16 and preserve_hdr:
            pass
        elif dtype == np.float32 and preserve_hdr:
            pass
        else:
            arr = (arr.astype(np.float32) / (65535.0 if dtype == np.uint16 else 1.0) * 255.0).astype(np.uint8)
    elif ndim == 3:
        channels = shape[2]
        if channels == 3:
            if dtype != np.uint8:
                arr = (
                    (arr.astype(np.float32) / (65535.0 if dtype == np.uint16 else 1.0) * 255.0)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
        elif channels == 4:
            if dtype != np.uint8:
                arr = (
                    (arr.astype(np.float32) / (65535.0 if dtype == np.uint16 else 1.0) * 255.0)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
        else:
            raise ValueError(f"Unsupported channel count for PIL conversion: {channels}")
    else:
        raise ValueError(f"Unsupported array shape for PIL conversion: {shape}")

    pil_img = Image.fromarray(arr)

    # Attach ICC profile and EXIF to pil_img.info
    if metadata is not None:
        icc = getattr(metadata, "icc", None)
        if icc:
            pil_img.info["icc_profile"] = icc
        exif = getattr(metadata, "exif", None)
        if exif:
            pil_img.info["exif"] = exif

    return pil_img


def from_pil(image: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract contiguous NumPy pixel array and metadata dictionary from PIL.Image.

    Args:
        image: PIL.Image.Image instance.

    Returns:
        Tuple of (ndarray of shape (H, W) or (H, W, C), metadata_dict).
    """
    _check_pillow()
    arr = np.ascontiguousarray(np.array(image))
    meta = {
        "icc": image.info.get("icc_profile"),
        "exif": image.info.get("exif"),
    }
    return arr, meta


def _accept(prefix: bytes) -> bool:
    """Quick magic sniffing for Pillow Image.open."""
    return sniff_bytes(prefix[:16]) == "jxl"


def _save(im: Any, fp: Any, filename: str | None = None) -> None:
    """Pillow save driver for JXL format."""
    arr, meta = from_pil(im)
    params = getattr(im, "encoderinfo", {})
    quality = params.get("quality", 1.0)
    lossless = params.get("lossless", False)
    speed = params.get("speed", EncoderSpeed.Squirrel)
    exif = params.get("exif", meta.get("exif"))
    icc = params.get("icc_profile", meta.get("icc"))

    encoded = encode_from_numpy(
        arr,
        quality=quality,
        lossless=lossless,
        speed=speed,
        exif=exif,
        icc=icc,
    )
    fp.write(encoded)


_registered = False


def register() -> None:
    """Register JXL plugin into Pillow's format registry.

    Enables standard `PIL.Image.open("pic.jxl")` and `img.save("pic.jxl", "JXL")`.
    """
    global _registered
    if _registered:
        return

    try:
        from PIL import Image, ImageFile
    except ImportError:
        return

    class JxlImageFile(ImageFile.ImageFile):
        format = "JXL"
        format_description = "JPEG XL image"

        def _open(self) -> None:
            raw_data = self.fp.read()
            # Fast probe
            meta = probe(raw_data)
            self._size = (meta.width, meta.height)
            total_channels = meta.num_color_channels + (1 if meta.has_alpha else 0)
            if total_channels == 1:
                self._mode = "L"
            elif total_channels == 3:
                self._mode = "RGB"
            elif total_channels == 4:
                self._mode = "RGBA"
            else:
                self._mode = "RGB"

            if meta.icc:
                self.info["icc_profile"] = meta.icc
            if meta.exif:
                self.info["exif"] = meta.exif

            # Decode pixel array
            _, arr = decode_to_numpy(raw_data, dtype="uint8")
            self._arr = arr

        def load(self) -> Any:
            if getattr(self, "_arr", None) is not None:
                arr = self._arr
                if arr.ndim == 3 and arr.shape[2] == 1:
                    arr = arr.squeeze(2)
                img = Image.fromarray(arr)
                self.im = img.im
                self._arr = None
            return super().load()

    Image.register_open(JxlImageFile.format, JxlImageFile, _accept)
    Image.register_save(JxlImageFile.format, _save)
    Image.register_extension(JxlImageFile.format, ".jxl")
    Image.register_mime(JxlImageFile.format, "image/jxl")
    _registered = True
