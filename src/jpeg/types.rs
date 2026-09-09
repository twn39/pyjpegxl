use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Simple metadata for decoded JPEG images.
#[pyclass(get_all, from_py_object)]
#[derive(Clone, Debug)]
pub struct JpegInfo {
    pub width: u32,
    pub height: u32,
    pub num_channels: u32,
    pub exif: Option<Vec<u8>>,
    pub icc: Option<Vec<u8>>,
}

#[pymethods]
impl JpegInfo {
    #[getter]
    fn icc_profile<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.icc.as_ref().map(|b| PyBytes::new(py, b))
    }

    fn __repr__(&self) -> String {
        format!(
            "JpegInfo(width={}, height={}, num_channels={}, has_exif={}, has_icc={})",
            self.width,
            self.height,
            self.num_channels,
            self.exif.is_some(),
            self.icc.is_some()
        )
    }
}
