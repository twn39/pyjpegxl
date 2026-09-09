use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub fn jpeg_encode_internal(
    data: &[u8],
    width: u32,
    height: u32,
    quality: i32,
    num_channels: u32,
) -> Result<Vec<u8>, String> {
    let w = width as usize;
    let h = height as usize;
    let c = num_channels as usize;
    let expected_len = w * h * c;
    if data.len() != expected_len {
        return Err(format!(
            "Data length mismatch: expected {} bytes ({}x{}x{}), got {}",
            expected_len,
            w,
            h,
            c,
            data.len()
        ));
    }

    let (format, subsamp) = match num_channels {
        1 => (turbojpeg::PixelFormat::GRAY, turbojpeg::Subsamp::Gray),
        3 => (turbojpeg::PixelFormat::RGB, turbojpeg::Subsamp::Sub2x2),
        4 => (turbojpeg::PixelFormat::RGBA, turbojpeg::Subsamp::Sub2x2),
        _ => {
            return Err(format!(
                "Unsupported channel count for JPEG encoding: {num_channels} (expected 1, 3, or 4)"
            ))
        }
    };

    let image = turbojpeg::Image {
        pixels: data,
        width: w,
        pitch: w * c,
        height: h,
        format,
    };

    let mut compressor = turbojpeg::Compressor::new()
        .map_err(|e| format!("Failed to create JPEG compressor: {e}"))?;

    compressor
        .set_subsamp(subsamp)
        .map_err(|e| format!("Failed to set JPEG subsampling: {e}"))?;

    compressor
        .set_quality(quality.clamp(1, 100))
        .map_err(|e| format!("Failed to set JPEG quality: {e}"))?;

    let compressed = compressor
        .compress_to_vec(image)
        .map_err(|e| format!("Failed to compress JPEG: {e}"))?;

    Ok(compressed)
}

/// Encode raw pixel data to JPEG format.
///
/// The GIL is released during encoding.
#[pyfunction]
#[pyo3(signature = (data, width, height, *, quality = 95, num_channels = 3))]
pub fn jpeg_encode<'py>(
    py: Python<'py>,
    data: &[u8],
    width: u32,
    height: u32,
    quality: i32,
    num_channels: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    let jpeg = py
        .detach(|| jpeg_encode_internal(data, width, height, quality, num_channels))
        .map_err(PyRuntimeError::new_err)?;
    Ok(PyBytes::new(py, &jpeg))
}

/// Encode a NumPy array (H, W) or (H, W, C) of uint8 to JPEG.
///
/// The GIL is released during encoding.
#[pyfunction]
#[pyo3(signature = (array, *, quality = 95))]
pub fn jpeg_encode_from_numpy<'py>(
    py: Python<'py>,
    array: PyReadonlyArrayDyn<'py, u8>,
    quality: i32,
) -> PyResult<Bound<'py, PyBytes>> {
    let shape = array.shape();
    let (height, width, num_channels) = if shape.len() == 2 {
        (shape[0] as u32, shape[1] as u32, 1u32)
    } else if shape.len() == 3 {
        (shape[0] as u32, shape[1] as u32, shape[2] as u32)
    } else {
        return Err(PyRuntimeError::new_err(format!(
            "Expected 2D (H, W) or 3D (H, W, C) array, got {}D",
            shape.len()
        )));
    };

    let array_view = array.as_array();
    if !array_view.is_standard_layout() {
        return Err(PyRuntimeError::new_err(
            "Array must be C-contiguous. Use numpy.ascontiguousarray().",
        ));
    }
    let data = array_view.as_slice().ok_or_else(|| {
        PyRuntimeError::new_err(
            "Array is not contiguous or memory layout is invalid. Use numpy.ascontiguousarray().",
        )
    })?;

    let jpeg = py
        .detach(|| jpeg_encode_internal(data, width, height, quality, num_channels))
        .map_err(PyRuntimeError::new_err)?;

    Ok(PyBytes::new(py, &jpeg))
}
