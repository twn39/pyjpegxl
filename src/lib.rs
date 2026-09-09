#![allow(clippy::type_complexity, clippy::too_many_arguments)]

pub mod common;
pub mod jpeg;
pub mod jxl;
pub mod transcode;

use pyo3::prelude::*;

// Threading controls
use common::runner::{get_num_threads, set_num_threads};

// JXL types and functions
use jxl::decoder::{decode, decode_into, decode_to_numpy};
use jxl::encoder::{encode, encode_from_numpy};
use jxl::prober::probe;
use jxl::types::{EncoderSpeed, Metadata};

// JPEG types and functions
use jpeg::decoder::{jpeg_decode, jpeg_decode_into, jpeg_decode_to_numpy};
use jpeg::encoder::{jpeg_encode, jpeg_encode_from_numpy};
use jpeg::prober::jpeg_probe;
use jpeg::types::JpegInfo;

// Lossless Transcoding functions
use transcode::lossless::{jpeg_to_jxl, jxl_to_jpeg};

/// Python module for JPEG XL and JPEG encoding, decoding, and lossless transcoding.
#[pymodule]
fn _pyjpegxl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // JXL types
    m.add_class::<Metadata>()?;
    m.add_class::<EncoderSpeed>()?;

    // JXL functions
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode_to_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(encode_from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(probe, m)?)?;
    m.add_function(wrap_pyfunction!(decode_into, m)?)?;

    // Concurrency controls
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;

    // JPEG types
    m.add_class::<JpegInfo>()?;

    // JPEG functions
    m.add_function(wrap_pyfunction!(jpeg_decode, m)?)?;
    m.add_function(wrap_pyfunction!(jpeg_encode, m)?)?;
    m.add_function(wrap_pyfunction!(jpeg_decode_to_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(jpeg_encode_from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(jpeg_probe, m)?)?;
    m.add_function(wrap_pyfunction!(jpeg_decode_into, m)?)?;

    // JPEG ↔ JXL lossless transcoding
    m.add_function(wrap_pyfunction!(jpeg_to_jxl, m)?)?;
    m.add_function(wrap_pyfunction!(jxl_to_jpeg, m)?)?;

    Ok(())
}
