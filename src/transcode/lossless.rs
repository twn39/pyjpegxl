use crate::common::guards::{DecoderGuard, EncoderGuard};
use crate::common::runner::get_shared_c_runner;
use jpegxl_sys::common::types::JxlBool;
use jpegxl_sys::decode::{
    JxlDecoderCreate, JxlDecoderProcessInput, JxlDecoderReleaseJPEGBuffer, JxlDecoderSetInput,
    JxlDecoderSetJPEGBuffer, JxlDecoderSetParallelRunner, JxlDecoderStatus,
    JxlDecoderSubscribeEvents,
};
use jpegxl_sys::encoder::encode::{
    JxlEncoderAddJPEGFrame, JxlEncoderCloseInput, JxlEncoderCreate, JxlEncoderFrameSettingsCreate,
    JxlEncoderProcessOutput, JxlEncoderSetParallelRunner, JxlEncoderStatus,
    JxlEncoderStoreJPEGMetadata, JxlEncoderUseContainer,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::ptr;

/// Lossless transcode: JPEG bytes → JXL bytes.
/// Uses JxlEncoderStoreJPEGMetadata + JxlEncoderAddJPEGFrame so the original
/// JPEG can be reconstructed bit-for-bit from the resulting JXL.
pub fn jpeg_to_jxl_internal(jpeg_data: &[u8]) -> Result<Vec<u8>, String> {
    unsafe {
        // Create encoder
        let enc = JxlEncoderCreate(ptr::null());
        if enc.is_null() {
            return Err("Failed to create JXL encoder".into());
        }
        let _enc_guard = EncoderGuard(enc);

        // Attach shared runner if configured
        let _runner_ref = get_shared_c_runner();
        if let Some(ref r) = _runner_ref {
            if JxlEncoderSetParallelRunner(
                enc,
                jpegxl_sys::threads::thread_parallel_runner::JxlThreadParallelRunner,
                r.as_ptr(),
            ) != JxlEncoderStatus::Success
            {
                return Err("Failed to set parallel runner".into());
            }
        }

        // Use container format (required for JPEG metadata storage)
        JxlEncoderUseContainer(enc, JxlBool::True);

        // Enable JPEG reconstruction metadata storage
        if JxlEncoderStoreJPEGMetadata(enc, JxlBool::True) != JxlEncoderStatus::Success {
            return Err("Failed to enable JPEG metadata storage".into());
        }

        // Create frame settings
        let frame_settings = JxlEncoderFrameSettingsCreate(enc, ptr::null());
        if frame_settings.is_null() {
            return Err("Failed to create frame settings".into());
        }

        // Add the JPEG frame (lossless transcoding)
        let status = JxlEncoderAddJPEGFrame(frame_settings, jpeg_data.as_ptr(), jpeg_data.len());
        if status != JxlEncoderStatus::Success {
            return Err("Failed to add JPEG frame for transcoding".into());
        }

        // Signal no more frames
        JxlEncoderCloseInput(enc);

        // Collect output
        let mut output = Vec::with_capacity(jpeg_data.len());
        let chunk_size = 65536usize;
        loop {
            let offset = output.len();
            output.resize(offset + chunk_size, 0u8);
            let mut next_out = output.as_mut_ptr().add(offset);
            let mut avail_out = chunk_size;

            let status = JxlEncoderProcessOutput(enc, &mut next_out, &mut avail_out);
            let bytes_written = chunk_size - avail_out;
            output.truncate(offset + bytes_written);

            match status {
                JxlEncoderStatus::Success => break,
                JxlEncoderStatus::NeedMoreOutput => continue,
                JxlEncoderStatus::Error => {
                    return Err("JXL encoder error during output".into());
                }
            }
        }

        Ok(output)
    }
}

/// Lossless reconstruct: JXL bytes → original JPEG bytes.
/// Only works for JXL files that were created via lossless JPEG transcoding.
pub fn jxl_to_jpeg_internal(jxl_data: &[u8]) -> Result<Vec<u8>, String> {
    unsafe {
        let dec = JxlDecoderCreate(ptr::null());
        if dec.is_null() {
            return Err("Failed to create JXL decoder".into());
        }
        let _dec_guard = DecoderGuard(dec);

        // Attach shared runner if configured
        let _runner_ref = get_shared_c_runner();
        if let Some(ref r) = _runner_ref {
            let _ = JxlDecoderSetParallelRunner(
                dec,
                jpegxl_sys::threads::thread_parallel_runner::JxlThreadParallelRunner,
                r.as_ptr(),
            );
        }

        // We must subscribe to FULLIMAGE along with JPEGRECONSTRUCTION.
        let events = (JxlDecoderStatus::JPEGReconstruction as std::os::raw::c_int)
            | (JxlDecoderStatus::FullImage as std::os::raw::c_int);
        if JxlDecoderSubscribeEvents(dec, events) != JxlDecoderStatus::Success {
            return Err("Failed to subscribe to decoder events".into());
        }

        if JxlDecoderSetInput(dec, jxl_data.as_ptr(), jxl_data.len()) != JxlDecoderStatus::Success {
            return Err("Failed to set decoder input".into());
        }

        let mut jpeg_buf: Vec<u8> = vec![0u8; jxl_data.len() * 2];
        let mut jpeg_buf_offset = 0usize;
        let mut got_jpeg_reconstruction = false;

        loop {
            let status = JxlDecoderProcessInput(dec);
            match status {
                JxlDecoderStatus::JPEGReconstruction => {
                    got_jpeg_reconstruction = true;
                    let buf_ptr = jpeg_buf.as_mut_ptr().add(jpeg_buf_offset);
                    let buf_len = jpeg_buf.len() - jpeg_buf_offset;
                    if JxlDecoderSetJPEGBuffer(dec, buf_ptr, buf_len) != JxlDecoderStatus::Success {
                        return Err("Failed to set JPEG output buffer".into());
                    }
                }
                JxlDecoderStatus::JPEGNeedMoreOutput => {
                    let remaining = JxlDecoderReleaseJPEGBuffer(dec);
                    let written = (jpeg_buf.len() - jpeg_buf_offset) - remaining;
                    jpeg_buf_offset += written;

                    let new_size = jpeg_buf.len() * 2;
                    jpeg_buf.resize(new_size, 0u8);

                    let buf_ptr = jpeg_buf.as_mut_ptr().add(jpeg_buf_offset);
                    let buf_len = jpeg_buf.len() - jpeg_buf_offset;
                    if JxlDecoderSetJPEGBuffer(dec, buf_ptr, buf_len) != JxlDecoderStatus::Success {
                        return Err("Failed to set grown JPEG buffer".into());
                    }
                }
                JxlDecoderStatus::NeedImageOutBuffer => {
                    if got_jpeg_reconstruction {
                        let remaining = JxlDecoderReleaseJPEGBuffer(dec);
                        let written = (jpeg_buf.len() - jpeg_buf_offset) - remaining;
                        jpeg_buf_offset += written;
                        jpeg_buf.truncate(jpeg_buf_offset);
                    }
                    break;
                }
                JxlDecoderStatus::FullImage | JxlDecoderStatus::Success => {
                    if got_jpeg_reconstruction {
                        let remaining = JxlDecoderReleaseJPEGBuffer(dec);
                        let written = (jpeg_buf.len() - jpeg_buf_offset) - remaining;
                        jpeg_buf_offset += written;
                        jpeg_buf.truncate(jpeg_buf_offset);
                    }
                    break;
                }
                JxlDecoderStatus::Error => {
                    return Err("JXL Decoder error during JPEG reconstruction".into());
                }
                JxlDecoderStatus::NeedMoreInput => {
                    return Err("Incomplete JXL data for JPEG reconstruction".into());
                }
                _ => {}
            }
        }

        if !got_jpeg_reconstruction || jpeg_buf.is_empty() {
            return Err("No JPEG reconstruction data found in JXL file".into());
        }

        Ok(jpeg_buf)
    }
}

/// Transcode JPEG bytes to JXL bytes (lossless, bit-exact roundtrip).
///
/// The GIL is released during transcoding.
#[pyfunction]
pub fn jpeg_to_jxl<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let jxl = py
        .detach(|| jpeg_to_jxl_internal(data))
        .map_err(PyRuntimeError::new_err)?;
    Ok(PyBytes::new(py, &jxl))
}

/// Reconstruct the original JPEG from a JXL that was created via lossless transcoding.
///
/// The GIL is released during reconstruction.
#[pyfunction]
pub fn jxl_to_jpeg<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let jpeg = py
        .detach(|| jxl_to_jpeg_internal(data))
        .map_err(PyRuntimeError::new_err)?;
    Ok(PyBytes::new(py, &jpeg))
}
