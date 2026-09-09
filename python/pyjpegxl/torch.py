"""PyTorch Tensor zero-copy bridge and zero-allocation in-place decoding."""

from __future__ import annotations

from typing import Any

import numpy as np

from pyjpegxl._pyjpegxl import JpegInfo, Metadata, decode_into, jpeg_decode_into
from pyjpegxl._sniff import sniff_bytes


def _check_torch() -> Any:
    """Ensure PyTorch is installed."""
    try:
        import torch

        return torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for this operation. Install it with: pip install 'pyjpegxl[torch]'"
        ) from e


def to_tensor(
    array_or_bytes: np.ndarray | bytes,
    *,
    permute_chw: bool = False,
    channels_last: bool = False,
    normalize: bool = False,
    device: Any = None,
) -> Any:
    """Convert decoded NumPy array or raw image bytes to a PyTorch Tensor.

    By default, keeps standard HWC contiguous layout (permute_chw=False) for true
    zero-copy memory sharing and zero hidden memcpy overhead.

    Args:
        array_or_bytes: NumPy array of shape (H, W) or (H, W, C), or raw JXL/JPEG bytes.
        permute_chw: If True, permutes dimensions to (C, H, W). Note that permuting
            yields a non-contiguous tensor unless .contiguous() is called. Default is False.
        channels_last: If True, ensures tensor memory is laid out in channels-last format.
        normalize: If True, scales pixel values to [0.0, 1.0] and casts to float32.
        device: Target PyTorch device (e.g., 'cuda', 'mps', 'cpu').

    Returns:
        torch.Tensor instance.
    """
    torch = _check_torch()

    if isinstance(array_or_bytes, (bytes, bytearray, memoryview)):
        from pyjpegxl._unified import imread

        _, arr = imread(bytes(array_or_bytes))
    else:
        arr = np.ascontiguousarray(array_or_bytes)

    orig_dtype = arr.dtype
    tensor = torch.from_numpy(arr)

    if permute_chw and tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)

    if normalize:
        scale = 255.0 if orig_dtype == np.uint8 else (65535.0 if orig_dtype == np.uint16 else 1.0)
        tensor = tensor.float() / scale

    if channels_last and tensor.ndim == 4:
        tensor = tensor.to(memory_format=torch.channels_last)

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def from_tensor(tensor: Any) -> np.ndarray:
    """Convert a PyTorch Tensor back to a C-contiguous NumPy array (H, W, C).

    Automatically handles GPU detachment, CPU transfer, and (C, H, W) -> (H, W, C)
    dimension permutation if needed.

    Args:
        tensor: PyTorch Tensor of shape (H, W), (H, W, C), or (C, H, W).

    Returns:
        C-contiguous NumPy array.
    """
    _check_torch()
    t = tensor.detach().cpu()

    # If tensor is in (C, H, W) format, permute back to (H, W, C)
    if t.ndim == 3 and t.shape[0] in (1, 3, 4) and (t.shape[0] < t.shape[1] and t.shape[0] < t.shape[2]):
        t = t.permute(1, 2, 0)

    if not t.is_contiguous():
        t = t.contiguous()

    return t.numpy()


def decode_into_tensor(
    data: bytes,
    tensor: Any,
    *,
    is_jpeg: bool | None = None,
) -> Metadata | JpegInfo:
    """Zero-allocation decode directly into a preallocated PyTorch CPU or Pinned-Memory Tensor.

    Eliminates intermediate buffer allocations by writing decoded pixel data directly
    into the underlying storage of the provided tensor.

    Args:
        data: Compressed JXL or JPEG image bytes.
        tensor: Preallocated writable CPU tensor with matching dimensions and dtype.
        is_jpeg: Explicitly indicate if data is JPEG. If None, auto-detects via magic header.

    Returns:
        Metadata or JpegInfo for the decoded image.
    """
    _check_torch()
    if not tensor.is_contiguous():
        raise ValueError("Target tensor must be contiguous")
    if tensor.is_cuda:
        raise ValueError("Target tensor must be on CPU (supports Pinned Memory)")

    arr = tensor.numpy()
    if is_jpeg is True or (is_jpeg is None and sniff_bytes(data[:16]) == "jpeg"):
        return jpeg_decode_into(data, arr)
    else:
        return decode_into(data, arr)
