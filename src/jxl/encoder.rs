use crate::common::runner::with_threads_runner;
use crate::jxl::types::EncoderSpeed;
use jpegxl_rs::encode::{ColorEncoding, EncoderFrame, Metadata as JxlMetadata};
use jpegxl_rs::encoder_builder;
use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

macro_rules! impl_encode_internal {
    ($fn_name:ident, $t:ty, $luma_encoding:expr) => {
        pub fn $fn_name(
            data: &[$t],
            width: u32,
            height: u32,
            lossless: bool,
            quality: f32,
            speed: EncoderSpeed,
            num_channels: u32,
            exif: Option<&[u8]>,
            xmp: Option<&[u8]>,
            icc: Option<&[u8]>,
            intensity_target: Option<f32>,
        ) -> Result<Vec<u8>, String> {
            let expected_len = (width * height * num_channels) as usize;
            if data.len() != expected_len {
                return Err(format!(
                    "Data length mismatch: expected {} elements ({}x{}x{}), got {}",
                    expected_len,
                    width,
                    height,
                    num_channels,
                    data.len()
                ));
            }

            let has_alpha = num_channels == 2 || num_channels == 4;

            with_threads_runner(|runner| {
                let mut encoder = if let Some(r) = runner {
                    encoder_builder()
                        .parallel_runner(r)
                        .speed(speed.into())
                        .has_alpha(has_alpha)
                        .build()
                } else {
                    encoder_builder()
                        .speed(speed.into())
                        .has_alpha(has_alpha)
                        .build()
                }
                .map_err(|e| format!("Failed to create encoder: {e}"))?;

                if num_channels == 1 || num_channels == 2 {
                    encoder.color_encoding = Some($luma_encoding);
                }

                if lossless {
                    encoder.lossless = Some(true);
                    encoder.uses_original_profile = true;
                    encoder.quality = 0.0;
                } else {
                    encoder.quality = quality;
                }

                if let Some(it) = intensity_target {
                    encoder.target_intensity = Some(it);
                }

                if let Some(e) = exif {
                    encoder
                        .add_metadata(&JxlMetadata::Exif(e), true)
                        .map_err(|e| format!("Failed adding exif: {e}"))?;
                }
                if let Some(x) = xmp {
                    encoder
                        .add_metadata(&JxlMetadata::Xmp(x), true)
                        .map_err(|e| format!("Failed adding xmp: {e}"))?;
                }
                if let Some(i) = icc {
                    encoder
                        .add_metadata(&JxlMetadata::Custom(*b"prof", i), false)
                        .map_err(|e| format!("Failed adding icc: {e}"))?;
                }

                let frame = EncoderFrame::new(data).num_channels(num_channels);
                let result = encoder
                    .encode_frame::<$t, $t>(&frame, width, height)
                    .map_err(|e| format!("Failed to encode: {e}"))?;

                Ok(result.data)
            })
        }
    };
}

impl_encode_internal!(encode_internal_u8, u8, ColorEncoding::SrgbLuma);
impl_encode_internal!(encode_internal_u16, u16, ColorEncoding::SrgbLuma);
impl_encode_internal!(encode_internal_f32, f32, ColorEncoding::LinearSrgbLuma);

/// Encode raw pixel data to JPEG XL format.
///
/// The GIL is released during encoding for concurrency.
#[pyfunction]
#[pyo3(signature = (data, width, height, *, lossless = false, quality = 1.0, speed = EncoderSpeed::Squirrel, num_channels = 4, exif = None, xmp = None, icc = None, intensity_target = None))]
pub fn encode<'py>(
    py: Python<'py>,
    data: &[u8],
    width: u32,
    height: u32,
    lossless: bool,
    quality: f32,
    speed: EncoderSpeed,
    num_channels: u32,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    icc: Option<&[u8]>,
    intensity_target: Option<f32>,
) -> PyResult<Bound<'py, PyBytes>> {
    let jxl = py
        .detach(|| {
            encode_internal_u8(
                data,
                width,
                height,
                lossless,
                quality,
                speed,
                num_channels,
                exif,
                xmp,
                icc,
                intensity_target,
            )
        })
        .map_err(PyRuntimeError::new_err)?;
    Ok(PyBytes::new(py, &jxl))
}

/// Encode a NumPy array (H, W) or (H, W, C) to JPEG XL.
///
/// Supports arrays of uint8, uint16, and float32.
/// Array must be C-contiguous.
/// The GIL is released during encoding.
#[pyfunction]
#[pyo3(signature = (array, *, lossless = false, quality = 1.0, speed = EncoderSpeed::Squirrel, exif = None, xmp = None, icc = None, intensity_target = None))]
pub fn encode_from_numpy<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    lossless: bool,
    quality: f32,
    speed: EncoderSpeed,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    icc: Option<&[u8]>,
    intensity_target: Option<f32>,
) -> PyResult<Bound<'py, PyBytes>> {
    if let Ok(arr_u8) = array.extract::<PyReadonlyArrayDyn<'py, u8>>() {
        let (height, width, num_channels) = parse_array_shape(arr_u8.shape())?;
        let view = arr_u8.as_array();
        if !view.is_standard_layout() {
            return Err(PyRuntimeError::new_err(
                "Array must be C-contiguous. Use numpy.ascontiguousarray().",
            ));
        }
        let data = view.as_slice().ok_or_else(|| {
            PyRuntimeError::new_err(
                "Array is not contiguous or memory layout is invalid. Use numpy.ascontiguousarray().",
            )
        })?;
        let jxl = py
            .detach(|| {
                encode_internal_u8(
                    data,
                    width,
                    height,
                    lossless,
                    quality,
                    speed,
                    num_channels,
                    exif,
                    xmp,
                    icc,
                    intensity_target,
                )
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok(PyBytes::new(py, &jxl))
    } else if let Ok(arr_u16) = array.extract::<PyReadonlyArrayDyn<'py, u16>>() {
        let (height, width, num_channels) = parse_array_shape(arr_u16.shape())?;
        let view = arr_u16.as_array();
        if !view.is_standard_layout() {
            return Err(PyRuntimeError::new_err(
                "Array must be C-contiguous. Use numpy.ascontiguousarray().",
            ));
        }
        let data = view.as_slice().ok_or_else(|| {
            PyRuntimeError::new_err(
                "Array is not contiguous or memory layout is invalid. Use numpy.ascontiguousarray().",
            )
        })?;
        let jxl = py
            .detach(|| {
                encode_internal_u16(
                    data,
                    width,
                    height,
                    lossless,
                    quality,
                    speed,
                    num_channels,
                    exif,
                    xmp,
                    icc,
                    intensity_target,
                )
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok(PyBytes::new(py, &jxl))
    } else if let Ok(arr_f32) = array.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        let (height, width, num_channels) = parse_array_shape(arr_f32.shape())?;
        let view = arr_f32.as_array();
        if !view.is_standard_layout() {
            return Err(PyRuntimeError::new_err(
                "Array must be C-contiguous. Use numpy.ascontiguousarray().",
            ));
        }
        let data = view.as_slice().ok_or_else(|| {
            PyRuntimeError::new_err(
                "Array is not contiguous or memory layout is invalid. Use numpy.ascontiguousarray().",
            )
        })?;
        let jxl = py
            .detach(|| {
                encode_internal_f32(
                    data,
                    width,
                    height,
                    lossless,
                    quality,
                    speed,
                    num_channels,
                    exif,
                    xmp,
                    icc,
                    intensity_target,
                )
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok(PyBytes::new(py, &jxl))
    } else {
        Err(PyTypeError::new_err(
            "Unsupported array dtype. Supported dtypes are uint8, uint16, and float32.",
        ))
    }
}

fn parse_array_shape(shape: &[usize]) -> PyResult<(u32, u32, u32)> {
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
