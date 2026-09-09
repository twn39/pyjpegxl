use jpegxl_rs::encode::EncoderSpeed as JxlEncoderSpeed;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Image metadata returned by decode.
#[pyclass(get_all, from_py_object)]
#[derive(Clone, Debug)]
pub struct Metadata {
    pub width: u32,
    pub height: u32,
    pub num_color_channels: u32,
    pub has_alpha: bool,
    pub exif: Option<Vec<u8>>,
    pub xmp: Option<Vec<u8>>,
    pub icc: Option<Vec<u8>>,
    pub bits_per_sample: u32,
    pub intensity_target: f32,
    pub min_nits: f32,
}

#[pymethods]
impl Metadata {
    #[getter]
    fn icc_profile<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.icc.as_ref().map(|b| PyBytes::new(py, b))
    }

    fn __repr__(&self) -> String {
        format!(
            "Metadata(width={}, height={}, num_color_channels={}, has_alpha={}, bits_per_sample={}, has_exif={}, has_xmp={}, has_icc={}, intensity_target={}, min_nits={})",
            self.width,
            self.height,
            self.num_color_channels,
            self.has_alpha,
            self.bits_per_sample,
            self.exif.is_some(),
            self.xmp.is_some(),
            self.icc.is_some(),
            self.intensity_target,
            self.min_nits
        )
    }
}

/// Encoder speed presets (fastest → slowest).
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum EncoderSpeed {
    Lightning = 1,
    Thunder = 2,
    Falcon = 3,
    Cheetah = 4,
    Hare = 5,
    Wombat = 6,
    Squirrel = 7,
    Kitten = 8,
    Tortoise = 9,
}

impl From<EncoderSpeed> for JxlEncoderSpeed {
    fn from(s: EncoderSpeed) -> Self {
        match s {
            EncoderSpeed::Lightning => JxlEncoderSpeed::Lightning,
            EncoderSpeed::Thunder => JxlEncoderSpeed::Thunder,
            EncoderSpeed::Falcon => JxlEncoderSpeed::Falcon,
            EncoderSpeed::Cheetah => JxlEncoderSpeed::Cheetah,
            EncoderSpeed::Hare => JxlEncoderSpeed::Hare,
            EncoderSpeed::Wombat => JxlEncoderSpeed::Wombat,
            EncoderSpeed::Squirrel => JxlEncoderSpeed::Squirrel,
            EncoderSpeed::Kitten => JxlEncoderSpeed::Kitten,
            EncoderSpeed::Tortoise => JxlEncoderSpeed::Tortoise,
        }
    }
}
