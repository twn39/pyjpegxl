use crate::jxl::single_pass::{decode_jxl_into, decode_jxl_single_pass, DecodedPixels};
use crate::jxl::types::Metadata;
use jpegxl_sys::common::types::JxlDataType;
use numpy::{ndarray, IntoPyArray, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::ffi::c_void;

/// Decode a JPEG XL image from bytes.
///
/// If dtype is None, automatically detects source image bit depth (uint8, uint16, or float32 bytes).
/// The GIL is released during decoding for concurrency.
/// Returns a tuple of (Metadata, bytes).
#[pyfunction]
#[pyo3(signature = (data, *, dtype = None))]
pub fn decode<'py>(
    py: Python<'py>,
    data: &[u8],
    dtype: Option<&str>,
) -> PyResult<(Metadata, Bound<'py, PyBytes>)> {
    let result = py
        .detach(|| decode_jxl_single_pass(data, dtype))
        .map_err(PyRuntimeError::new_err)?;

    let bytes = PyBytes::new(py, result.pixels.as_bytes());
    Ok((result.meta, bytes))
}

/// Decode a JPEG XL image, returning a NumPy array.
///
/// Returns (Metadata, ndarray) where ndarray has shape (H, W, C).
/// If dtype is None, automatically detects source image bit depth (uint8, uint16, or float32).
/// The pixel buffer is transferred to NumPy via zero-copy ownership transfer.
/// The GIL is released during decoding.
#[pyfunction]
#[pyo3(signature = (data, *, dtype = None))]
pub fn decode_to_numpy<'py>(
    py: Python<'py>,
    data: &[u8],
    dtype: Option<&str>,
) -> PyResult<(Metadata, Bound<'py, PyAny>)> {
    let result = py
        .detach(|| decode_jxl_single_pass(data, dtype))
        .map_err(PyRuntimeError::new_err)?;

    let h = result.meta.height as usize;
    let w = result.meta.width as usize;
    let c = result.total_channels as usize;

    match result.pixels {
        DecodedPixels::Uint8(p) => {
            let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[h, w, c]), p)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape pixels: {e}")))?;
            Ok((result.meta, array.into_pyarray(py).into_any()))
        }
        DecodedPixels::Uint16(p) => {
            let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[h, w, c]), p)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape pixels: {e}")))?;
            Ok((result.meta, array.into_pyarray(py).into_any()))
        }
        DecodedPixels::Float(p) => {
            let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[h, w, c]), p)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape pixels: {e}")))?;
            Ok((result.meta, array.into_pyarray(py).into_any()))
        }
    }
}

/// Decode a JPEG XL image directly into a pre-allocated, writable, C-contiguous NumPy array.
///
/// Eliminates intermediate buffer allocation (zero-copy into caller's memory).
/// The GIL is released during decoding.
#[pyfunction]
pub fn decode_into<'py>(
    py: Python<'py>,
    data: &[u8],
    out: &Bound<'py, PyAny>,
) -> PyResult<Metadata> {
    if let Ok(mut arr_u8) = out.extract::<PyReadwriteArrayDyn<'py, u8>>() {
        let shape = arr_u8.shape();
        let (h, w, c) = extract_dims(shape)?;
        let mut array_view = arr_u8.as_array_mut();
        if !array_view.is_standard_layout() {
            return Err(PyRuntimeError::new_err(
                "Array must be C-contiguous. Use numpy.ascontiguousarray().",
            ));
        }
        let out_byte_len = array_view.len();
        let out_ptr_val = array_view.as_mut_ptr() as usize;

        py.detach(move || {
            let out_ptr = out_ptr_val as *mut c_void;
            decode_jxl_into(
                data,
                out_ptr,
                out_byte_len,
                JxlDataType::Uint8,
                c,
                w,
                h,
            )
        })
        .map_err(PyRuntimeError::new_err)
    } else if let Ok(mut arr_u16) = out.extract::<PyReadwriteArrayDyn<'py, u16>>() {
        let shape = arr_u16.shape();
        let (h, w, c) = extract_dims(shape)?;
        let mut array_view = arr_u16.as_array_mut();
        if !array_view.is_standard_layout() {
            return Err(PyRuntimeError::new_err(
                "Array must be C-contiguous. Use numpy.ascontiguousarray().",
            ));
        }
        let out_byte_len = array_view.len() * std::mem::size_of::<u16>();
        let out_ptr_val = array_view.as_mut_ptr() as usize;

        py.detach(move || {
            let out_ptr = out_ptr_val as *mut c_void;
            decode_jxl_into(
                data,
                out_ptr,
                out_byte_len,
                JxlDataType::Uint16,
                c,
                w,
                h,
            )
        })
        .map_err(PyRuntimeError::new_err)
    } else if let Ok(mut arr_f32) = out.extract::<PyReadwriteArrayDyn<'py, f32>>() {
        let shape = arr_f32.shape();
        let (h, w, c) = extract_dims(shape)?;
        let mut array_view = arr_f32.as_array_mut();
        if !array_view.is_standard_layout() {
            return Err(PyRuntimeError::new_err(
                "Array must be C-contiguous. Use numpy.ascontiguousarray().",
            ));
        }
        let out_byte_len = array_view.len() * std::mem::size_of::<f32>();
        let out_ptr_val = array_view.as_mut_ptr() as usize;

        py.detach(move || {
            let out_ptr = out_ptr_val as *mut c_void;
            decode_jxl_into(
                data,
                out_ptr,
                out_byte_len,
                JxlDataType::Float,
                c,
                w,
                h,
            )
        })
        .map_err(PyRuntimeError::new_err)
    } else {
        Err(PyTypeError::new_err(
            "Expected writable NumPy array of dtype uint8, uint16, or float32.",
        ))
    }
}

fn extract_dims(shape: &[usize]) -> PyResult<(u32, u32, u32)> {
    if shape.len() == 2 {
        Ok((shape[0] as u32, shape[1] as u32, 1u32))
    } else if shape.len() == 3 {
        Ok((shape[0] as u32, shape[1] as u32, shape[2] as u32))
    } else {
        Err(PyRuntimeError::new_err(format!(
            "Expected 2D (H, W) or 3D (H, W, C) array, got {}D",
            shape.len()
        )))
    }
}
