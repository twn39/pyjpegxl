use crate::jpeg::markers::parse_jpeg_markers;
use crate::jpeg::types::JpegInfo;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub fn jpeg_probe_internal(data: &[u8]) -> Result<JpegInfo, String> {
    let mut decompressor = turbojpeg::Decompressor::new()
        .map_err(|e| format!("Failed to create JPEG decompressor: {e}"))?;

    let header = decompressor
        .read_header(data)
        .map_err(|e| format!("Failed to read JPEG header: {e}"))?;

    let (exif, icc) = parse_jpeg_markers(data);

    let num_channels = match header.subsamp {
        turbojpeg::Subsamp::Gray => 1,
        _ => 3,
    };

    Ok(JpegInfo {
        width: header.width as u32,
        height: header.height as u32,
        num_channels,
        exif,
        icc,
    })
}

/// Fast metadata inspection of a JPEG image without decoding pixel data.
///
/// The GIL is released during probing.
#[pyfunction]
pub fn jpeg_probe<'py>(py: Python<'py>, data: &[u8]) -> PyResult<JpegInfo> {
    py.detach(|| jpeg_probe_internal(data))
        .map_err(PyRuntimeError::new_err)
}
