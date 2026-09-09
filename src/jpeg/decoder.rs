use crate::jpeg::markers::parse_jpeg_markers;
use crate::jpeg::types::JpegInfo;
use numpy::{ndarray, IntoPyArray, PyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub struct JpegDecodeResult {
    pub info: JpegInfo,
    pub pixels: Vec<u8>,
}

pub fn jpeg_decode_internal(
    data: &[u8],
    requested_channels: Option<u32>,
) -> Result<JpegDecodeResult, String> {
    let mut decompressor = turbojpeg::Decompressor::new()
        .map_err(|e| format!("Failed to create JPEG decompressor: {e}"))?;

    let header = decompressor
        .read_header(data)
        .map_err(|e| format!("Failed to read JPEG header: {e}"))?;

    let width = header.width;
    let height = header.height;

    let (exif, icc) = parse_jpeg_markers(data);

    let num_channels = match requested_channels {
        Some(1) => 1,
        Some(3) => 3,
        Some(4) => 4,
        None => {
            if header.subsamp == turbojpeg::Subsamp::Gray {
                1
            } else {
                3
            }
        }
        Some(c) => return Err(format!("Unsupported channel count for JPEG decoding: {c}")),
    };

    let format = match num_channels {
        1 => turbojpeg::PixelFormat::GRAY,
        3 => turbojpeg::PixelFormat::RGB,
        4 => turbojpeg::PixelFormat::RGBA,
        _ => return Err(format!("Unsupported channel count for JPEG decoding: {num_channels}")),
    };

    let pitch = width * num_channels as usize;
    let mut pixels = vec![0u8; height * pitch];

    let image = turbojpeg::Image {
        pixels: pixels.as_mut_slice(),
        width,
        pitch,
        height,
        format,
    };

    decompressor
        .decompress(data, image)
        .map_err(|e| format!("Failed to decompress JPEG: {e}"))?;

    Ok(JpegDecodeResult {
        info: JpegInfo {
            width: width as u32,
            height: height as u32,
            num_channels,
            exif,
            icc,
        },
        pixels,
    })
}

/// Decode a JPEG image from bytes.
///
/// The GIL is released during decoding.
/// Returns a tuple of (JpegInfo, bytes).
#[pyfunction]
#[pyo3(signature = (data, *, channels = None))]
pub fn jpeg_decode<'py>(
    py: Python<'py>,
    data: &[u8],
    channels: Option<u32>,
) -> PyResult<(JpegInfo, Bound<'py, PyBytes>)> {
    let result = py
        .detach(|| jpeg_decode_internal(data, channels))
        .map_err(PyRuntimeError::new_err)?;
    Ok((result.info, PyBytes::new(py, &result.pixels)))
}

/// Decode a JPEG image, returning a NumPy array.
///
/// Returns (JpegInfo, ndarray) where ndarray has shape (H, W, C) and dtype uint8.
/// The GIL is released during decoding.
#[pyfunction]
#[pyo3(signature = (data, *, channels = None))]
pub fn jpeg_decode_to_numpy<'py>(
    py: Python<'py>,
    data: &[u8],
    channels: Option<u32>,
) -> PyResult<(JpegInfo, Bound<'py, PyArrayDyn<u8>>)> {
    let result = py
        .detach(|| jpeg_decode_internal(data, channels))
        .map_err(PyRuntimeError::new_err)?;

    let h = result.info.height as usize;
    let w = result.info.width as usize;
    let c = result.info.num_channels as usize;

    let shape = if c == 1 {
        ndarray::IxDyn(&[h, w, 1])
    } else {
        ndarray::IxDyn(&[h, w, c])
    };

    let array = ndarray::Array::from_shape_vec(shape, result.pixels)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape pixels: {e}")))?;

    Ok((result.info, array.into_pyarray(py)))
}

/// Decode a JPEG image directly into a pre-allocated writable NumPy array.
///
/// Eliminates buffer allocation by writing directly into the caller's memory.
/// The GIL is released during decoding.
#[pyfunction]
pub fn jpeg_decode_into<'py>(
    py: Python<'py>,
    data: &[u8],
    out: &Bound<'py, PyAny>,
) -> PyResult<JpegInfo> {
    let mut arr_u8 = out
        .extract::<PyReadwriteArrayDyn<'py, u8>>()
        .map_err(|_| PyRuntimeError::new_err("Expected writable NumPy array of dtype uint8."))?;

    let shape = arr_u8.shape();
    let (h, w, c) = if shape.len() == 2 {
        (shape[0], shape[1], 1u32)
    } else if shape.len() == 3 {
        (shape[0], shape[1], shape[2] as u32)
    } else {
        return Err(PyRuntimeError::new_err(format!(
            "Expected 2D (H, W) or 3D (H, W, C) array, got {}D",
            shape.len()
        )));
    };

    let format = match c {
        1 => turbojpeg::PixelFormat::GRAY,
        3 => turbojpeg::PixelFormat::RGB,
        4 => turbojpeg::PixelFormat::RGBA,
        _ => return Err(PyRuntimeError::new_err(format!("Unsupported channel count: {c}"))),
    };

    let mut view = arr_u8.as_array_mut();
    if !view.is_standard_layout() {
        return Err(PyRuntimeError::new_err(
            "Array must be C-contiguous. Use numpy.ascontiguousarray().",
        ));
    }

    let pitch = w * c as usize;
    let expected_len = h * pitch;
    let slice = view.as_slice_mut().ok_or_else(|| {
        PyRuntimeError::new_err(
            "Array must be C-contiguous and writable. Use numpy.ascontiguousarray().",
        )
    })?;
    if slice.len() < expected_len {
        return Err(PyRuntimeError::new_err(format!(
            "Buffer size too small: expected {} bytes, got {}",
            expected_len,
            slice.len()
        )));
    }
    let slice_ptr_val = slice.as_mut_ptr() as usize;
    let slice_len = slice.len();

    let (exif, icc) = parse_jpeg_markers(data);

    let info = py
        .detach(move || -> Result<JpegInfo, String> {
            let mut decompressor = turbojpeg::Decompressor::new()
                .map_err(|e| format!("Failed to create JPEG decompressor: {e}"))?;

            let header = decompressor
                .read_header(data)
                .map_err(|e| format!("Failed to read JPEG header: {e}"))?;

            if header.width != w || header.height != h {
                return Err(format!(
                    "JPEG dimensions ({}x{}) do not match buffer dimensions ({}x{})",
                    header.width, header.height, w, h
                ));
            }

            let slice = unsafe { std::slice::from_raw_parts_mut(slice_ptr_val as *mut u8, slice_len) };

            let image = turbojpeg::Image {
                pixels: slice,
                width: w,
                pitch,
                height: h,
                format,
            };

            decompressor
                .decompress(data, image)
                .map_err(|e| format!("Failed to decompress JPEG: {e}"))?;

            Ok(JpegInfo {
                width: w as u32,
                height: h as u32,
                num_channels: c,
                exif,
                icc,
            })
        })
        .map_err(PyRuntimeError::new_err)?;

    Ok(info)
}
