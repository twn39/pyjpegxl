use jpegxl_rs::encode::{EncoderFrame, EncoderSpeed as JxlEncoderSpeed};
use jpegxl_rs::{decoder_builder, encoder_builder, ThreadsRunner};
use numpy::{ndarray, IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use jpegxl_sys::common::types::JxlBool;
use jpegxl_sys::decode::{
    JxlDecoderCreate, JxlDecoderDestroy, JxlDecoderGetBoxSizeRaw, JxlDecoderGetBoxType,
    JxlDecoderProcessInput, JxlDecoderReleaseBoxBuffer, JxlDecoderReleaseInput,
    JxlDecoderReleaseJPEGBuffer, JxlDecoderSetBoxBuffer, JxlDecoderSetDecompressBoxes,
    JxlDecoderSetInput, JxlDecoderSetJPEGBuffer, JxlDecoderStatus, JxlDecoderSubscribeEvents,
};
use jpegxl_sys::encoder::encode::{
    JxlEncoderAddJPEGFrame, JxlEncoderCloseInput, JxlEncoderCreate, JxlEncoderDestroy,
    JxlEncoderFrameSettingsCreate, JxlEncoderProcessOutput, JxlEncoderSetParallelRunner,
    JxlEncoderStatus, JxlEncoderStoreJPEGMetadata, JxlEncoderUseContainer,
};
use jpegxl_sys::threads::thread_parallel_runner::{
    JxlThreadParallelRunner, JxlThreadParallelRunnerCreate,
    JxlThreadParallelRunnerDefaultNumWorkerThreads, JxlThreadParallelRunnerDestroy,
};
use std::ptr;

// ---------------------------------------------------------------------------
// RAII Guards for jpegxl-sys FFI types to prevent memory leaks on panic
// ---------------------------------------------------------------------------
macro_rules! define_guard {
    ($name:ident, $destroy:path) => {
        struct $name<T>(*mut T);
        impl<T> Drop for $name<T> {
            fn drop(&mut self) {
                // Safety: C FFI destroy functions are safe to call on pointers allocated by create functions
                unsafe { $destroy(self.0 as _) }
            }
        }
    };
}
define_guard!(EncoderGuard, JxlEncoderDestroy);
define_guard!(DecoderGuard, JxlDecoderDestroy);
define_guard!(RunnerGuard, JxlThreadParallelRunnerDestroy);

/// Image metadata returned by decode.
#[pyclass(get_all)]
#[derive(Clone)]
struct Metadata {
    width: u32,
    height: u32,
    num_color_channels: u32,
    has_alpha: bool,
    exif: Option<Vec<u8>>,
    xmp: Option<Vec<u8>>,
}

#[pymethods]
impl Metadata {
    fn __repr__(&self) -> String {
        format!(
            "Metadata(width={}, height={}, num_color_channels={}, has_alpha={}, has_exif={}, has_xmp={})",
            self.width, self.height, self.num_color_channels, self.has_alpha, self.exif.is_some(), self.xmp.is_some()
        )
    }
}

/// Encoder speed presets (fastest → slowest).
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
enum EncoderSpeed {
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

// ---------------------------------------------------------------------------
// Internal helpers (no Python objects, safe to call without GIL)
// ---------------------------------------------------------------------------

struct DecodeResult {
    meta: Metadata,
    pixels: Vec<u8>,
    total_channels: u32,
}

fn extract_metadata(data: &[u8]) -> (Option<Vec<u8>>, Option<Vec<u8>>) {
    let mut exif = None;
    let mut xmp = None;

    unsafe {
        let dec = JxlDecoderCreate(ptr::null());
        if dec.is_null() {
            return (None, None);
        }
        let _dec_guard = DecoderGuard(dec);

        let events = (JxlDecoderStatus::Box as std::os::raw::c_int)
            | (JxlDecoderStatus::BoxComplete as std::os::raw::c_int);
        if JxlDecoderSubscribeEvents(dec, events) != JxlDecoderStatus::Success {
            return (None, None);
        }

        JxlDecoderSetDecompressBoxes(dec, JxlBool::True);

        if JxlDecoderSetInput(dec, data.as_ptr(), data.len()) != JxlDecoderStatus::Success {
            return (None, None);
        }

        let mut current_box_type = [0u8; 4];
        let mut current_box_data = Vec::new();
        let mut getting_box = false;

        loop {
            let status = JxlDecoderProcessInput(dec);
            match status {
                JxlDecoderStatus::Success | JxlDecoderStatus::Error => break,
                JxlDecoderStatus::NeedMoreInput => break, // We provided everything
                JxlDecoderStatus::Box => {
                    let mut box_type = jpegxl_sys::common::types::JxlBoxType([0; 4]);
                    if JxlDecoderGetBoxType(dec, &mut box_type, JxlBool::True)
                        == JxlDecoderStatus::Success
                    {
                        current_box_type = [
                            box_type.0[0] as u8,
                            box_type.0[1] as u8,
                            box_type.0[2] as u8,
                            box_type.0[3] as u8,
                        ];

                        // We only care about Exif and xml
                        if &current_box_type == b"Exif" || &current_box_type == b"xml " {
                            let mut size = 0;
                            if JxlDecoderGetBoxSizeRaw(dec, &mut size) == JxlDecoderStatus::Success
                            {
                                current_box_data.resize(size as usize, 0);
                                JxlDecoderSetBoxBuffer(
                                    dec,
                                    current_box_data.as_mut_ptr(),
                                    size as usize,
                                );
                                getting_box = true;
                            } else {
                                // Dynamic size, allocate a large enough buffer or implement progressive reading
                                // For simplicity, assume sizes are known or we can just skip
                            }
                        }
                    }
                }
                JxlDecoderStatus::BoxNeedMoreOutput => {
                    // Buffer was not large enough. Realistically we should grow `current_box_data` and call `JxlDecoderSetBoxBuffer` again.
                    // For now, if we don't handle it, we must release buffer.
                    JxlDecoderReleaseBoxBuffer(dec);
                    getting_box = false;
                }
                JxlDecoderStatus::BoxComplete => {
                    if getting_box {
                        let released = JxlDecoderReleaseBoxBuffer(dec);
                        // The remaining valid size is original buffer len minus released unused bytes
                        let valid_size = current_box_data.len().saturating_sub(released);
                        current_box_data.truncate(valid_size);

                        if &current_box_type == b"Exif" {
                            exif = Some(current_box_data.clone());
                        } else if &current_box_type == b"xml " {
                            xmp = Some(current_box_data.clone());
                        }
                        getting_box = false;
                    }
                }
                _ => {}
            }

            // If we found both, we can exit early!
            if exif.is_some() && xmp.is_some() {
                break;
            }
        }
    }

    (exif, xmp)
}

fn decode_internal(data: &[u8]) -> Result<DecodeResult, String> {
    let runner = ThreadsRunner::default();
    let decoder = decoder_builder()
        .parallel_runner(&runner)
        .build()
        .map_err(|e| format!("Failed to create decoder: {e}"))?;

    let (meta, pixel_data) = decoder
        .decode_with::<u8>(data)
        .map_err(|e| format!("Failed to decode: {e}"))?;

    let total_channels = meta.num_color_channels + u32::from(meta.has_alpha_channel);

    let (exif, xmp) = extract_metadata(data);

    let metadata = Metadata {
        width: meta.width,
        height: meta.height,
        num_color_channels: meta.num_color_channels,
        has_alpha: meta.has_alpha_channel,
        exif,
        xmp,
    };

    Ok(DecodeResult {
        meta: metadata,
        pixels: pixel_data,
        total_channels,
    })
}

fn encode_internal(
    data: &[u8],
    width: u32,
    height: u32,
    lossless: bool,
    quality: f32,
    speed: EncoderSpeed,
    num_channels: u32,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
) -> Result<Vec<u8>, String> {
    let expected_len = (width * height * num_channels) as usize;
    if data.len() != expected_len {
        return Err(format!(
            "Data length mismatch: expected {} bytes ({}x{}x{}), got {}",
            expected_len,
            width,
            height,
            num_channels,
            data.len()
        ));
    }

    let has_alpha = num_channels == 2 || num_channels == 4;

    let runner = ThreadsRunner::default();
    let mut encoder = encoder_builder()
        .parallel_runner(&runner)
        .speed(speed.into())
        .has_alpha(has_alpha)
        .build()
        .map_err(|e| format!("Failed to create encoder: {e}"))?;

    if lossless {
        encoder.lossless = Some(true);
        encoder.uses_original_profile = true;
        encoder.quality = 0.0;
    } else {
        encoder.quality = quality;
    }

    if let Some(e) = exif {
        encoder
            .add_metadata(&jpegxl_rs::encode::Metadata::Exif(e), true)
            .map_err(|e| format!("Failed adding exif: {e}"))?;
    }
    if let Some(x) = xmp {
        encoder
            .add_metadata(&jpegxl_rs::encode::Metadata::Xmp(x), true)
            .map_err(|e| format!("Failed adding xmp: {e}"))?;
    }

    let frame = EncoderFrame::new(data).num_channels(num_channels);
    let result = encoder
        .encode_frame::<u8, u8>(&frame, width, height)
        .map_err(|e| format!("Failed to encode: {e}"))?;

    Ok(result.data)
}

// ---------------------------------------------------------------------------
// Python API — bytes
// ---------------------------------------------------------------------------

/// Decode a JPEG XL image from bytes.
///
/// The GIL is released during decoding for concurrency.
/// Returns a tuple of (Metadata, bytes).
#[pyfunction]
fn decode<'py>(py: Python<'py>, data: &[u8]) -> PyResult<(Metadata, Bound<'py, PyBytes>)> {
    let result = py
        .allow_threads(|| decode_internal(data))
        .map_err(PyRuntimeError::new_err)?;
    Ok((result.meta, PyBytes::new(py, &result.pixels)))
}

/// Encode raw pixel data to JPEG XL format.
///
/// The GIL is released during encoding for concurrency.
#[pyfunction]
#[pyo3(signature = (data, width, height, *, lossless = false, quality = 1.0, speed = EncoderSpeed::Squirrel, num_channels = 4, exif = None, xmp = None))]
fn encode<'py>(
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
) -> PyResult<Bound<'py, PyBytes>> {
    let jxl = py
        .allow_threads(|| {
            encode_internal(
                data,
                width,
                height,
                lossless,
                quality,
                speed,
                num_channels,
                exif,
                xmp,
            )
        })
        .map_err(PyRuntimeError::new_err)?;
    Ok(PyBytes::new(py, &jxl))
}

// ---------------------------------------------------------------------------
// Python API — NumPy (zero-copy)
// ---------------------------------------------------------------------------

/// Decode a JPEG XL image, returning a NumPy array.
///
/// Returns (Metadata, ndarray) where ndarray has shape (H, W, C) and dtype uint8.
/// The pixel buffer is transferred to NumPy via zero-copy ownership transfer.
/// The GIL is released during decoding.
#[pyfunction]
fn decode_to_numpy<'py>(
    py: Python<'py>,
    data: &[u8],
) -> PyResult<(Metadata, Bound<'py, PyArrayDyn<u8>>)> {
    let result = py
        .allow_threads(|| decode_internal(data))
        .map_err(PyRuntimeError::new_err)?;

    let h = result.meta.height as usize;
    let w = result.meta.width as usize;
    let c = result.total_channels as usize;

    // Zero-copy: ownership of Vec<u8> transfers to NumPy
    let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[h, w, c]), result.pixels)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape pixels: {e}")))?;

    Ok((result.meta, array.into_pyarray(py)))
}

/// Encode a NumPy array (H, W, C) of uint8 to JPEG XL.
///
/// Reads from the NumPy array via zero-copy (if C-contiguous).
/// The GIL is released during encoding.
#[pyfunction]
#[pyo3(signature = (array, *, lossless = false, quality = 1.0, speed = EncoderSpeed::Squirrel, exif = None, xmp = None))]
fn encode_from_numpy<'py>(
    py: Python<'py>,
    array: PyReadonlyArrayDyn<'py, u8>,
    lossless: bool,
    quality: f32,
    speed: EncoderSpeed,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
) -> PyResult<Bound<'py, PyBytes>> {
    let shape = array.shape();
    if shape.len() != 3 {
        return Err(PyRuntimeError::new_err(format!(
            "Expected 3D array (H, W, C), got {}D",
            shape.len()
        )));
    }
    let height = shape[0] as u32;
    let width = shape[1] as u32;
    let num_channels = shape[2] as u32;

    // Get contiguous data — zero-copy if already C-contiguous
    let array_view = array.as_array();
    if !array_view.is_standard_layout() {
        return Err(PyRuntimeError::new_err(
            "Array must be C-contiguous. Use numpy.ascontiguousarray().",
        ));
    }
    let data = array_view.as_slice().unwrap();

    let jxl = py
        .allow_threads(|| {
            encode_internal(
                data,
                width,
                height,
                lossless,
                quality,
                speed,
                num_channels,
                exif,
                xmp,
            )
        })
        .map_err(PyRuntimeError::new_err)?;

    Ok(PyBytes::new(py, &jxl))
}

// ---------------------------------------------------------------------------
// JPEG codec via turbojpeg (static-linked libjpeg-turbo)
// ---------------------------------------------------------------------------

/// Simple metadata for decoded JPEG images.
#[pyclass(get_all)]
#[derive(Clone)]
struct JpegInfo {
    width: u32,
    height: u32,
    num_channels: u32,
}

#[pymethods]
impl JpegInfo {
    fn __repr__(&self) -> String {
        format!(
            "JpegInfo(width={}, height={}, num_channels={})",
            self.width, self.height, self.num_channels
        )
    }
}

struct JpegDecodeResult {
    info: JpegInfo,
    pixels: Vec<u8>,
}

fn jpeg_decode_internal(data: &[u8]) -> Result<JpegDecodeResult, String> {
    let mut decompressor = turbojpeg::Decompressor::new()
        .map_err(|e| format!("Failed to create JPEG decompressor: {e}"))?;

    let header = decompressor
        .read_header(data)
        .map_err(|e| format!("Failed to read JPEG header: {e}"))?;

    let width = header.width;
    let height = header.height;

    // Always decompress to RGB (3 channels)
    let num_channels: u32 = 3;
    let pitch = width * num_channels as usize;
    let mut pixels = vec![0u8; height * pitch];

    let image = turbojpeg::Image {
        pixels: pixels.as_mut_slice(),
        width,
        pitch,
        height,
        format: turbojpeg::PixelFormat::RGB,
    };

    decompressor
        .decompress(data, image)
        .map_err(|e| format!("Failed to decompress JPEG: {e}"))?;

    Ok(JpegDecodeResult {
        info: JpegInfo {
            width: width as u32,
            height: height as u32,
            num_channels,
        },
        pixels,
    })
}

fn jpeg_encode_internal(
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

    let format = match num_channels {
        1 => turbojpeg::PixelFormat::GRAY,
        3 => turbojpeg::PixelFormat::RGB,
        4 => turbojpeg::PixelFormat::RGBA,
        _ => {
            return Err(format!(
                "Unsupported channel count: {num_channels} (must be 1, 3 or 4)"
            ))
        }
    };

    let pitch = w * c;
    let image = turbojpeg::Image {
        pixels: data,
        width: w,
        pitch,
        height: h,
        format,
    };

    let mut compressor = turbojpeg::Compressor::new()
        .map_err(|e| format!("Failed to create JPEG compressor: {e}"))?;
    compressor
        .set_quality(quality)
        .map_err(|e| format!("Failed to set quality: {e}"))?;

    let jpeg_data = compressor
        .compress_to_vec(image)
        .map_err(|e| format!("Failed to compress JPEG: {e}"))?;

    Ok(jpeg_data)
}

// ---------------------------------------------------------------------------
// JPEG ↔ JXL lossless transcoding (raw jpegxl-sys FFI)
// ---------------------------------------------------------------------------

/// Lossless transcode: JPEG bytes → JXL bytes.
/// Uses JxlEncoderStoreJPEGMetadata + JxlEncoderAddJPEGFrame so the original
/// JPEG can be reconstructed bit-for-bit from the resulting JXL.
fn jpeg_to_jxl_internal(jpeg_data: &[u8]) -> Result<Vec<u8>, String> {
    unsafe {
        // Create thread runner
        let num_threads = JxlThreadParallelRunnerDefaultNumWorkerThreads();
        let runner = JxlThreadParallelRunnerCreate(ptr::null(), num_threads);
        if runner.is_null() {
            return Err("Failed to create thread runner".into());
        }
        let _runner_guard = RunnerGuard(runner);

        // Create encoder
        let enc = JxlEncoderCreate(ptr::null());
        if enc.is_null() {
            return Err("Failed to create JXL encoder".into());
        }
        let _enc_guard = EncoderGuard(enc);

        // Set parallel runner
        if JxlEncoderSetParallelRunner(enc, JxlThreadParallelRunner, runner)
            != JxlEncoderStatus::Success
        {
            return Err("Failed to set parallel runner".into());
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
fn jxl_to_jpeg_internal(jxl_data: &[u8]) -> Result<Vec<u8>, String> {
    unsafe {
        let dec = JxlDecoderCreate(ptr::null());
        if dec.is_null() {
            return Err("Failed to create JXL decoder".into());
        }
        let _dec_guard = DecoderGuard(dec);

        // We must subscribe to FULLIMAGE along with JPEGRECONSTRUCTION.
        // If we don't subscribe to FULLIMAGE, the decoder stops after metadata.
        let events = (JxlDecoderStatus::JPEGReconstruction as std::os::raw::c_int)
            | (JxlDecoderStatus::FullImage as std::os::raw::c_int);
        if JxlDecoderSubscribeEvents(dec, events) != JxlDecoderStatus::Success {
            return Err("Failed to subscribe to decoder events".into());
        }

        // Set input
        if JxlDecoderSetInput(dec, jxl_data.as_ptr(), jxl_data.len()) != JxlDecoderStatus::Success {
            return Err("Failed to set decoder input".into());
        }

        // Initial JPEG buffer — we'll grow it as needed
        let mut jpeg_buf: Vec<u8> = Vec::with_capacity(jxl_data.len() * 2);
        jpeg_buf.resize(jpeg_buf.capacity(), 0u8);
        let mut jpeg_buf_offset = 0usize;
        let mut got_jpeg_reconstruction = false;

        loop {
            let status = JxlDecoderProcessInput(dec);
            match status {
                JxlDecoderStatus::JPEGReconstruction => {
                    got_jpeg_reconstruction = true;
                    // Set the JPEG output buffer
                    let buf_ptr = jpeg_buf.as_mut_ptr().add(jpeg_buf_offset);
                    let buf_len = jpeg_buf.len() - jpeg_buf_offset;
                    if JxlDecoderSetJPEGBuffer(dec, buf_ptr, buf_len) != JxlDecoderStatus::Success {
                        return Err("Failed to set JPEG output buffer".into());
                    }
                }
                JxlDecoderStatus::JPEGNeedMoreOutput => {
                    // Release current buffer to find how much was written
                    let remaining = JxlDecoderReleaseJPEGBuffer(dec);
                    let written = (jpeg_buf.len() - jpeg_buf_offset) - remaining;
                    jpeg_buf_offset += written;

                    // Grow the buffer
                    let new_size = jpeg_buf.len() * 2;
                    jpeg_buf.resize(new_size, 0u8);

                    // Set buffer again from where we left off
                    let buf_ptr = jpeg_buf.as_mut_ptr().add(jpeg_buf_offset);
                    let buf_len = jpeg_buf.len() - jpeg_buf_offset;
                    if JxlDecoderSetJPEGBuffer(dec, buf_ptr, buf_len) != JxlDecoderStatus::Success {
                        return Err("Failed to set grown JPEG buffer".into());
                    }
                }
                JxlDecoderStatus::NeedImageOutBuffer => {
                    // The decoder wants to decode pixels! This happens after metadata.
                    // If we have JPEG Reconstruction data, we would have received it already.
                    // So if we reach here, we can stop regardless.
                    if got_jpeg_reconstruction {
                        let remaining = JxlDecoderReleaseJPEGBuffer(dec);
                        let written = (jpeg_buf.len() - jpeg_buf_offset) - remaining;
                        jpeg_buf_offset += written;
                        jpeg_buf.truncate(jpeg_buf_offset);
                    }
                    break;
                }
                JxlDecoderStatus::FullImage | JxlDecoderStatus::Success => {
                    // Decoder finished processing.
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
                _ => {
                    // Ignore other events like BasicInfo, ColorEncoding, etc.
                    // Just let the decoder continue.
                }
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
fn jpeg_to_jxl<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let jxl = py
        .allow_threads(|| jpeg_to_jxl_internal(data))
        .map_err(PyRuntimeError::new_err)?;
    Ok(PyBytes::new(py, &jxl))
}

/// Reconstruct the original JPEG from a JXL that was created via lossless transcoding.
///
/// The GIL is released during reconstruction.
#[pyfunction]
fn jxl_to_jpeg<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let jpeg = py
        .allow_threads(|| jxl_to_jpeg_internal(data))
        .map_err(PyRuntimeError::new_err)?;
    Ok(PyBytes::new(py, &jpeg))
}

// ---------------------------------------------------------------------------
// JPEG Python API — bytes
// ---------------------------------------------------------------------------

/// Decode a JPEG image from bytes.
///
/// The GIL is released during decoding.
/// Returns a tuple of (JpegInfo, bytes).
#[pyfunction]
fn jpeg_decode<'py>(py: Python<'py>, data: &[u8]) -> PyResult<(JpegInfo, Bound<'py, PyBytes>)> {
    let result = py
        .allow_threads(|| jpeg_decode_internal(data))
        .map_err(PyRuntimeError::new_err)?;
    Ok((result.info, PyBytes::new(py, &result.pixels)))
}

/// Encode raw pixel data to JPEG format.
///
/// The GIL is released during encoding.
#[pyfunction]
#[pyo3(signature = (data, width, height, *, quality = 95, num_channels = 3))]
fn jpeg_encode<'py>(
    py: Python<'py>,
    data: &[u8],
    width: u32,
    height: u32,
    quality: i32,
    num_channels: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    let jpeg = py
        .allow_threads(|| jpeg_encode_internal(data, width, height, quality, num_channels))
        .map_err(PyRuntimeError::new_err)?;
    Ok(PyBytes::new(py, &jpeg))
}

// ---------------------------------------------------------------------------
// JPEG Python API — NumPy
// ---------------------------------------------------------------------------

/// Decode a JPEG image, returning a NumPy array.
///
/// Returns (JpegInfo, ndarray) where ndarray has shape (H, W, C) and dtype uint8.
/// The GIL is released during decoding.
#[pyfunction]
fn jpeg_decode_to_numpy<'py>(
    py: Python<'py>,
    data: &[u8],
) -> PyResult<(JpegInfo, Bound<'py, PyArrayDyn<u8>>)> {
    let result = py
        .allow_threads(|| jpeg_decode_internal(data))
        .map_err(PyRuntimeError::new_err)?;

    let h = result.info.height as usize;
    let w = result.info.width as usize;
    let c = result.info.num_channels as usize;

    let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[h, w, c]), result.pixels)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape pixels: {e}")))?;

    Ok((result.info, array.into_pyarray(py)))
}

/// Encode a NumPy array (H, W, C) of uint8 to JPEG.
///
/// The GIL is released during encoding.
#[pyfunction]
#[pyo3(signature = (array, *, quality = 95))]
fn jpeg_encode_from_numpy<'py>(
    py: Python<'py>,
    array: PyReadonlyArrayDyn<'py, u8>,
    quality: i32,
) -> PyResult<Bound<'py, PyBytes>> {
    let shape = array.shape();
    if shape.len() != 3 {
        return Err(PyRuntimeError::new_err(format!(
            "Expected 3D array (H, W, C), got {}D",
            shape.len()
        )));
    }
    let height = shape[0] as u32;
    let width = shape[1] as u32;
    let num_channels = shape[2] as u32;

    let array_view = array.as_array();
    if !array_view.is_standard_layout() {
        return Err(PyRuntimeError::new_err(
            "Array must be C-contiguous. Use numpy.ascontiguousarray().",
        ));
    }
    let data = array_view.as_slice().unwrap();

    let jpeg = py
        .allow_threads(|| jpeg_encode_internal(data, width, height, quality, num_channels))
        .map_err(PyRuntimeError::new_err)?;

    Ok(PyBytes::new(py, &jpeg))
}

// ---------------------------------------------------------------------------
// Python module registration
// ---------------------------------------------------------------------------

/// Python module for JPEG XL and JPEG encoding and decoding.
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
    // JPEG types
    m.add_class::<JpegInfo>()?;
    // JPEG functions
    m.add_function(wrap_pyfunction!(jpeg_decode, m)?)?;
    m.add_function(wrap_pyfunction!(jpeg_encode, m)?)?;
    m.add_function(wrap_pyfunction!(jpeg_decode_to_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(jpeg_encode_from_numpy, m)?)?;
    // JPEG ↔ JXL lossless transcoding
    m.add_function(wrap_pyfunction!(jpeg_to_jxl, m)?)?;
    m.add_function(wrap_pyfunction!(jxl_to_jpeg, m)?)?;
    Ok(())
}
