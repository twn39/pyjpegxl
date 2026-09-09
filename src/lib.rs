#![allow(clippy::type_complexity, clippy::too_many_arguments)]

use jpegxl_rs::decode::Pixels;
use jpegxl_rs::encode::{EncoderFrame, EncoderSpeed as JxlEncoderSpeed};
use jpegxl_rs::{decoder_builder, encoder_builder, ThreadsRunner};
use numpy::{
    ndarray, IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyReadwriteArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use jpegxl_sys::common::types::{JxlBool, JxlDataType, JxlEndianness, JxlPixelFormat};
use jpegxl_sys::decode::{
    JxlColorProfileTarget, JxlDecoderCreate, JxlDecoderDestroy, JxlDecoderGetBasicInfo,
    JxlDecoderGetBoxSizeRaw, JxlDecoderGetBoxType, JxlDecoderGetColorAsICCProfile,
    JxlDecoderGetICCProfileSize, JxlDecoderImageOutBufferSize, JxlDecoderProcessInput,
    JxlDecoderReleaseBoxBuffer, JxlDecoderReleaseJPEGBuffer, JxlDecoderSetBoxBuffer,
    JxlDecoderSetDecompressBoxes, JxlDecoderSetImageOutBuffer, JxlDecoderSetInput,
    JxlDecoderSetJPEGBuffer, JxlDecoderSetParallelRunner, JxlDecoderStatus,
    JxlDecoderSubscribeEvents,
};
use jpegxl_sys::encoder::encode::{
    JxlEncoderAddJPEGFrame, JxlEncoderCloseInput, JxlEncoderCreate, JxlEncoderDestroy,
    JxlEncoderFrameSettingsCreate, JxlEncoderProcessOutput, JxlEncoderSetParallelRunner,
    JxlEncoderStatus, JxlEncoderStoreJPEGMetadata, JxlEncoderUseContainer,
};
use jpegxl_sys::metadata::codestream_header::JxlBasicInfo;
use jpegxl_sys::threads::thread_parallel_runner::{
    JxlThreadParallelRunner, JxlThreadParallelRunnerCreate,
    JxlThreadParallelRunnerDefaultNumWorkerThreads, JxlThreadParallelRunnerDestroy,
};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

static GLOBAL_NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

#[pyfunction]
fn set_num_threads(n: usize) {
    GLOBAL_NUM_THREADS.store(n, Ordering::SeqCst);
}

#[pyfunction]
fn get_num_threads() -> usize {
    GLOBAL_NUM_THREADS.load(Ordering::SeqCst)
}

fn get_runner() -> Option<ThreadsRunner<'static>> {
    let threads = GLOBAL_NUM_THREADS.load(Ordering::Relaxed);
    if threads == 1 {
        None
    } else if threads > 1 {
        ThreadsRunner::new(None, Some(threads))
    } else {
        Some(ThreadsRunner::default())
    }
}

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
#[pyclass(get_all, from_py_object)]
#[derive(Clone)]
struct Metadata {
    width: u32,
    height: u32,
    num_color_channels: u32,
    has_alpha: bool,
    exif: Option<Vec<u8>>,
    xmp: Option<Vec<u8>>,
    icc: Option<Vec<u8>>,
    bits_per_sample: u32,
    intensity_target: f32,
    min_nits: f32,
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

struct DecodeResult<T> {
    meta: Metadata,
    pixels: Vec<T>,
    total_channels: u32,
}

enum AutoDecodeResult {
    Uint8(DecodeResult<u8>),
    Uint16(DecodeResult<u16>),
    Float(DecodeResult<f32>),
}

fn extract_metadata(data: &[u8]) -> (Option<Vec<u8>>, Option<Vec<u8>>, Option<Vec<u8>>) {
    let mut exif = None;
    let mut xmp = None;
    let mut icc = None;

    unsafe {
        let dec = JxlDecoderCreate(ptr::null());
        if dec.is_null() {
            return (None, None, None);
        }
        let _dec_guard = DecoderGuard(dec);

        let events = (JxlDecoderStatus::Box as std::os::raw::c_int)
            | (JxlDecoderStatus::BoxComplete as std::os::raw::c_int);
        if JxlDecoderSubscribeEvents(dec, events) != JxlDecoderStatus::Success {
            return (None, None, None);
        }

        JxlDecoderSetDecompressBoxes(dec, JxlBool::True);

        if JxlDecoderSetInput(dec, data.as_ptr(), data.len()) != JxlDecoderStatus::Success {
            return (None, None, None);
        }

        let mut current_box_type = [0u8; 4];
        let mut current_box_data = Vec::new();
        let mut current_box_offset = 0usize;
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

                        // We care about Exif, xml, and prof boxes
                        if &current_box_type == b"Exif"
                            || &current_box_type == b"xml "
                            || &current_box_type == b"prof"
                        {
                            let mut size = 0;
                            let initial_size = if JxlDecoderGetBoxSizeRaw(dec, &mut size)
                                == JxlDecoderStatus::Success
                                && size > 0
                            {
                                size as usize
                            } else {
                                65536 // 64KB initial chunk for dynamic or unknown box size
                            };
                            current_box_data.resize(initial_size, 0);
                            current_box_offset = 0;
                            if JxlDecoderSetBoxBuffer(
                                dec,
                                current_box_data.as_mut_ptr(),
                                initial_size,
                            ) == JxlDecoderStatus::Success
                            {
                                getting_box = true;
                            }
                        }
                    }
                }
                JxlDecoderStatus::BoxNeedMoreOutput if getting_box => {
                    let remaining = JxlDecoderReleaseBoxBuffer(dec);
                    let written = (current_box_data.len() - current_box_offset) - remaining;
                    current_box_offset += written;

                    // Grow buffer dynamically
                    let new_size = (current_box_data.len() * 2).max(current_box_data.len() + 65536);
                    current_box_data.resize(new_size, 0);

                    let buf_ptr = current_box_data.as_mut_ptr().add(current_box_offset);
                    let buf_len = current_box_data.len() - current_box_offset;
                    if JxlDecoderSetBoxBuffer(dec, buf_ptr, buf_len) != JxlDecoderStatus::Success {
                        getting_box = false;
                    }
                }
                JxlDecoderStatus::BoxComplete if getting_box => {
                    let released = JxlDecoderReleaseBoxBuffer(dec);
                    let written =
                        (current_box_data.len() - current_box_offset).saturating_sub(released);
                    current_box_offset += written;
                    current_box_data.truncate(current_box_offset);

                    if &current_box_type == b"Exif" {
                        exif = Some(current_box_data.clone());
                    } else if &current_box_type == b"xml " {
                        xmp = Some(current_box_data.clone());
                    } else if &current_box_type == b"prof" {
                        icc = Some(current_box_data.clone());
                    }
                    getting_box = false;
                }
                _ => {}
            }

            // If we found all three, we can exit early!
            if exif.is_some() && xmp.is_some() && icc.is_some() {
                break;
            }
        }
    }

    (exif, xmp, icc)
}

fn probe_internal(data: &[u8]) -> Result<Metadata, String> {
    unsafe {
        let dec = JxlDecoderCreate(ptr::null());
        if dec.is_null() {
            return Err("Failed to create JXL decoder".into());
        }
        let _dec_guard = DecoderGuard(dec);

        let events = (JxlDecoderStatus::BasicInfo as std::os::raw::c_int)
            | (JxlDecoderStatus::ColorEncoding as std::os::raw::c_int)
            | (JxlDecoderStatus::Box as std::os::raw::c_int)
            | (JxlDecoderStatus::BoxComplete as std::os::raw::c_int);
        if JxlDecoderSubscribeEvents(dec, events) != JxlDecoderStatus::Success {
            return Err("Failed to subscribe to decoder events".into());
        }

        JxlDecoderSetDecompressBoxes(dec, JxlBool::True);

        if JxlDecoderSetInput(dec, data.as_ptr(), data.len()) != JxlDecoderStatus::Success {
            return Err("Failed to set decoder input".into());
        }

        let mut width = 0u32;
        let mut height = 0u32;
        let mut num_color_channels = 3u32;
        let mut has_alpha = false;
        let mut bits_per_sample = 8u32;
        let mut intensity_target = 255.0f32;
        let mut min_nits = 0.0f32;

        let mut exif = None;
        let mut xmp = None;
        let mut icc_box = None;
        let mut decoder_icc = None;

        let mut current_box_type = [0u8; 4];
        let mut current_box_data = Vec::new();
        let mut current_box_offset = 0usize;
        let mut getting_box = false;

        loop {
            let status = JxlDecoderProcessInput(dec);
            match status {
                JxlDecoderStatus::BasicInfo => {
                    let mut basic_info = std::mem::MaybeUninit::<JxlBasicInfo>::uninit();
                    if JxlDecoderGetBasicInfo(dec, basic_info.as_mut_ptr())
                        == JxlDecoderStatus::Success
                    {
                        let info = basic_info.assume_init();
                        width = info.xsize;
                        height = info.ysize;
                        num_color_channels = info.num_color_channels;
                        has_alpha = info.alpha_bits > 0;
                        bits_per_sample = info.bits_per_sample;
                        intensity_target = info.intensity_target;
                        min_nits = info.min_nits;
                    }
                }
                JxlDecoderStatus::ColorEncoding => {
                    let mut icc_size = 0usize;
                    if JxlDecoderGetICCProfileSize(dec, JxlColorProfileTarget::Data, &mut icc_size)
                        == JxlDecoderStatus::Success
                        && icc_size > 0
                    {
                        let mut icc_buf = vec![0u8; icc_size];
                        if JxlDecoderGetColorAsICCProfile(
                            dec,
                            JxlColorProfileTarget::Data,
                            icc_buf.as_mut_ptr(),
                            icc_size,
                        ) == JxlDecoderStatus::Success
                        {
                            decoder_icc = Some(icc_buf);
                        }
                    }
                }
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

                        if &current_box_type == b"Exif"
                            || &current_box_type == b"xml "
                            || &current_box_type == b"prof"
                        {
                            let mut size = 0;
                            let initial_size = if JxlDecoderGetBoxSizeRaw(dec, &mut size)
                                == JxlDecoderStatus::Success
                                && size > 0
                            {
                                size as usize
                            } else {
                                65536
                            };
                            current_box_data.resize(initial_size, 0);
                            current_box_offset = 0;
                            if JxlDecoderSetBoxBuffer(
                                dec,
                                current_box_data.as_mut_ptr(),
                                initial_size,
                            ) == JxlDecoderStatus::Success
                            {
                                getting_box = true;
                            }
                        }
                    }
                }
                JxlDecoderStatus::BoxNeedMoreOutput if getting_box => {
                    let remaining = JxlDecoderReleaseBoxBuffer(dec);
                    let written = (current_box_data.len() - current_box_offset) - remaining;
                    current_box_offset += written;
                    let new_size = (current_box_data.len() * 2).max(current_box_data.len() + 65536);
                    current_box_data.resize(new_size, 0);
                    let buf_ptr = current_box_data.as_mut_ptr().add(current_box_offset);
                    let buf_len = current_box_data.len() - current_box_offset;
                    if JxlDecoderSetBoxBuffer(dec, buf_ptr, buf_len) != JxlDecoderStatus::Success {
                        getting_box = false;
                    }
                }
                JxlDecoderStatus::BoxComplete if getting_box => {
                    let released = JxlDecoderReleaseBoxBuffer(dec);
                    let written =
                        (current_box_data.len() - current_box_offset).saturating_sub(released);
                    current_box_offset += written;
                    current_box_data.truncate(current_box_offset);

                    if &current_box_type == b"Exif" {
                        exif = Some(current_box_data.clone());
                    } else if &current_box_type == b"xml " {
                        xmp = Some(current_box_data.clone());
                    } else if &current_box_type == b"prof" {
                        icc_box = Some(current_box_data.clone());
                    }
                    getting_box = false;
                }
                JxlDecoderStatus::NeedImageOutBuffer
                | JxlDecoderStatus::FullImage
                | JxlDecoderStatus::Success => {
                    break;
                }
                JxlDecoderStatus::Error => {
                    return Err("JXL decoder error during probe".into());
                }
                JxlDecoderStatus::NeedMoreInput => {
                    break;
                }
                _ => {}
            }
        }

        if width == 0 || height == 0 {
            return Err("Failed to parse basic image information".into());
        }

        let final_icc = icc_box.or(decoder_icc);

        Ok(Metadata {
            width,
            height,
            num_color_channels,
            has_alpha,
            exif,
            xmp,
            icc: final_icc,
            bits_per_sample,
            intensity_target,
            min_nits,
        })
    }
}

fn decode_into_internal(
    data: &[u8],
    out_ptr: *mut u8,
    out_byte_len: usize,
    data_type: JxlDataType,
    expected_channels: u32,
    expected_w: u32,
    expected_h: u32,
) -> Result<Metadata, String> {
    unsafe {
        let dec = JxlDecoderCreate(ptr::null());
        if dec.is_null() {
            return Err("Failed to create JXL decoder".into());
        }
        let _dec_guard = DecoderGuard(dec);

        let num_threads = GLOBAL_NUM_THREADS.load(Ordering::Relaxed);
        let mut _runner_guard = None;
        if num_threads != 1 {
            let workers = if num_threads > 1 {
                num_threads
            } else {
                JxlThreadParallelRunnerDefaultNumWorkerThreads()
            };
            let runner = JxlThreadParallelRunnerCreate(ptr::null(), workers);
            if !runner.is_null() {
                JxlDecoderSetParallelRunner(dec, JxlThreadParallelRunner, runner);
                _runner_guard = Some(RunnerGuard(runner));
            }
        }

        let events = (JxlDecoderStatus::BasicInfo as std::os::raw::c_int)
            | (JxlDecoderStatus::ColorEncoding as std::os::raw::c_int)
            | (JxlDecoderStatus::Box as std::os::raw::c_int)
            | (JxlDecoderStatus::BoxComplete as std::os::raw::c_int)
            | (JxlDecoderStatus::FullImage as std::os::raw::c_int);
        if JxlDecoderSubscribeEvents(dec, events) != JxlDecoderStatus::Success {
            return Err("Failed to subscribe to decoder events".into());
        }

        JxlDecoderSetDecompressBoxes(dec, JxlBool::True);

        if JxlDecoderSetInput(dec, data.as_ptr(), data.len()) != JxlDecoderStatus::Success {
            return Err("Failed to set decoder input".into());
        }

        let mut width = 0u32;
        let mut height = 0u32;
        let mut num_color_channels = 3u32;
        let mut has_alpha = false;
        let mut bits_per_sample = 8u32;
        let mut intensity_target = 255.0f32;
        let mut min_nits = 0.0f32;

        let mut exif = None;
        let mut xmp = None;
        let mut icc_box = None;
        let mut decoder_icc = None;

        let mut current_box_type = [0u8; 4];
        let mut current_box_data = Vec::new();
        let mut current_box_offset = 0usize;
        let mut getting_box = false;
        let mut buffer_set = false;

        loop {
            let status = JxlDecoderProcessInput(dec);
            match status {
                JxlDecoderStatus::BasicInfo => {
                    let mut basic_info = std::mem::MaybeUninit::<JxlBasicInfo>::uninit();
                    if JxlDecoderGetBasicInfo(dec, basic_info.as_mut_ptr())
                        == JxlDecoderStatus::Success
                    {
                        let info = basic_info.assume_init();
                        width = info.xsize;
                        height = info.ysize;
                        num_color_channels = info.num_color_channels;
                        has_alpha = info.alpha_bits > 0;
                        bits_per_sample = info.bits_per_sample;
                        intensity_target = info.intensity_target;
                        min_nits = info.min_nits;

                        if width != expected_w || height != expected_h {
                            return Err(format!(
                                "Image dimensions ({}x{}) do not match buffer dimensions ({}x{})",
                                width, height, expected_w, expected_h
                            ));
                        }
                    }
                }
                JxlDecoderStatus::ColorEncoding => {
                    let mut icc_size = 0usize;
                    if JxlDecoderGetICCProfileSize(dec, JxlColorProfileTarget::Data, &mut icc_size)
                        == JxlDecoderStatus::Success
                        && icc_size > 0
                    {
                        let mut icc_buf = vec![0u8; icc_size];
                        if JxlDecoderGetColorAsICCProfile(
                            dec,
                            JxlColorProfileTarget::Data,
                            icc_buf.as_mut_ptr(),
                            icc_size,
                        ) == JxlDecoderStatus::Success
                        {
                            decoder_icc = Some(icc_buf);
                        }
                    }
                }
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

                        if &current_box_type == b"Exif"
                            || &current_box_type == b"xml "
                            || &current_box_type == b"prof"
                        {
                            let mut size = 0;
                            let initial_size = if JxlDecoderGetBoxSizeRaw(dec, &mut size)
                                == JxlDecoderStatus::Success
                                && size > 0
                            {
                                size as usize
                            } else {
                                65536
                            };
                            current_box_data.resize(initial_size, 0);
                            current_box_offset = 0;
                            if JxlDecoderSetBoxBuffer(
                                dec,
                                current_box_data.as_mut_ptr(),
                                initial_size,
                            ) == JxlDecoderStatus::Success
                            {
                                getting_box = true;
                            }
                        }
                    }
                }
                JxlDecoderStatus::BoxNeedMoreOutput if getting_box => {
                    let remaining = JxlDecoderReleaseBoxBuffer(dec);
                    let written = (current_box_data.len() - current_box_offset) - remaining;
                    current_box_offset += written;
                    let new_size = (current_box_data.len() * 2).max(current_box_data.len() + 65536);
                    current_box_data.resize(new_size, 0);
                    let buf_ptr = current_box_data.as_mut_ptr().add(current_box_offset);
                    let buf_len = current_box_data.len() - current_box_offset;
                    if JxlDecoderSetBoxBuffer(dec, buf_ptr, buf_len) != JxlDecoderStatus::Success {
                        getting_box = false;
                    }
                }
                JxlDecoderStatus::BoxComplete if getting_box => {
                    let released = JxlDecoderReleaseBoxBuffer(dec);
                    let written =
                        (current_box_data.len() - current_box_offset).saturating_sub(released);
                    current_box_offset += written;
                    current_box_data.truncate(current_box_offset);

                    if &current_box_type == b"Exif" {
                        exif = Some(current_box_data.clone());
                    } else if &current_box_type == b"xml " {
                        xmp = Some(current_box_data.clone());
                    } else if &current_box_type == b"prof" {
                        icc_box = Some(current_box_data.clone());
                    }
                    getting_box = false;
                }
                JxlDecoderStatus::NeedImageOutBuffer => {
                    let format = JxlPixelFormat {
                        num_channels: expected_channels,
                        data_type,
                        endianness: JxlEndianness::Native,
                        align: 0,
                    };
                    let mut required_size = 0usize;
                    if JxlDecoderImageOutBufferSize(dec, &format, &mut required_size)
                        != JxlDecoderStatus::Success
                    {
                        return Err("Failed to calculate image output buffer size".into());
                    }
                    if out_byte_len < required_size {
                        return Err(format!(
                            "Buffer size too small: provided {} bytes, need {} bytes",
                            out_byte_len, required_size
                        ));
                    }
                    if JxlDecoderSetImageOutBuffer(dec, &format, out_ptr as *mut _, out_byte_len)
                        != JxlDecoderStatus::Success
                    {
                        return Err("Failed to set image output buffer".into());
                    }
                    buffer_set = true;
                }
                JxlDecoderStatus::FullImage | JxlDecoderStatus::Success => {
                    break;
                }
                JxlDecoderStatus::Error => {
                    return Err("JXL decoder error during decode_into".into());
                }
                JxlDecoderStatus::NeedMoreInput => {
                    return Err("Incomplete JXL data".into());
                }
                _ => {}
            }
        }

        if !buffer_set {
            return Err("Decoder did not produce image frames".into());
        }

        let final_icc = icc_box.or(decoder_icc);

        Ok(Metadata {
            width,
            height,
            num_color_channels,
            has_alpha,
            exif,
            xmp,
            icc: final_icc,
            bits_per_sample,
            intensity_target,
            min_nits,
        })
    }
}

fn decode_auto(data: &[u8]) -> Result<AutoDecodeResult, String> {
    let runner = get_runner();
    let decoder = if let Some(ref r) = runner {
        decoder_builder()
            .parallel_runner(r)
            .icc_profile(true)
            .build()
    } else {
        decoder_builder().icc_profile(true).build()
    }
    .map_err(|e| format!("Failed to create decoder: {e}"))?;

    let (meta, pixels) = decoder
        .decode(data)
        .map_err(|e| format!("Failed to decode: {e}"))?;

    let total_channels = meta.num_color_channels + u32::from(meta.has_alpha_channel);
    let (exif, xmp, icc_box) = extract_metadata(data);
    let final_icc = icc_box.or(meta.icc_profile);

    let bits_per_sample = match &pixels {
        Pixels::Uint8(_) => 8,
        Pixels::Uint16(_) => 16,
        Pixels::Float(_) => 32,
        Pixels::Float16(_) => 16,
    };

    let metadata = Metadata {
        width: meta.width,
        height: meta.height,
        num_color_channels: meta.num_color_channels,
        has_alpha: meta.has_alpha_channel,
        exif,
        xmp,
        icc: final_icc,
        bits_per_sample,
        intensity_target: meta.intensity_target,
        min_nits: meta.min_nits,
    };

    match pixels {
        Pixels::Uint8(p) => Ok(AutoDecodeResult::Uint8(DecodeResult {
            meta: metadata,
            pixels: p,
            total_channels,
        })),
        Pixels::Uint16(p) => Ok(AutoDecodeResult::Uint16(DecodeResult {
            meta: metadata,
            pixels: p,
            total_channels,
        })),
        Pixels::Float(p) => Ok(AutoDecodeResult::Float(DecodeResult {
            meta: metadata,
            pixels: p,
            total_channels,
        })),
        Pixels::Float16(p) => {
            let p_f32: Vec<f32> = p.into_iter().map(f32::from).collect();
            Ok(AutoDecodeResult::Float(DecodeResult {
                meta: metadata,
                pixels: p_f32,
                total_channels,
            }))
        }
    }
}

macro_rules! impl_decode_internal {
    ($fn_name:ident, $t:ty, $bits:expr) => {
        fn $fn_name(data: &[u8]) -> Result<DecodeResult<$t>, String> {
            let runner = get_runner();
            let decoder = if let Some(ref r) = runner {
                decoder_builder()
                    .parallel_runner(r)
                    .icc_profile(true)
                    .build()
            } else {
                decoder_builder().icc_profile(true).build()
            }
            .map_err(|e| format!("Failed to create decoder: {e}"))?;

            let (meta, pixel_data) = decoder
                .decode_with::<$t>(data)
                .map_err(|e| format!("Failed to decode: {e}"))?;

            let total_channels = meta.num_color_channels + u32::from(meta.has_alpha_channel);
            let (exif, xmp, icc_box) = extract_metadata(data);
            let final_icc = icc_box.or(meta.icc_profile);

            let metadata = Metadata {
                width: meta.width,
                height: meta.height,
                num_color_channels: meta.num_color_channels,
                has_alpha: meta.has_alpha_channel,
                exif,
                xmp,
                icc: final_icc,
                bits_per_sample: $bits,
                intensity_target: meta.intensity_target,
                min_nits: meta.min_nits,
            };

            Ok(DecodeResult {
                meta: metadata,
                pixels: pixel_data,
                total_channels,
            })
        }
    };
}

impl_decode_internal!(decode_internal_u8, u8, 8);
impl_decode_internal!(decode_internal_u16, u16, 16);
impl_decode_internal!(decode_internal_f32, f32, 32);

macro_rules! impl_encode_internal {
    ($fn_name:ident, $t:ty, $luma_encoding:expr) => {
        fn $fn_name(
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

            let runner = get_runner();
            let mut encoder = if let Some(ref r) = runner {
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
                    .add_metadata(&jpegxl_rs::encode::Metadata::Exif(e), true)
                    .map_err(|e| format!("Failed adding exif: {e}"))?;
            }
            if let Some(x) = xmp {
                encoder
                    .add_metadata(&jpegxl_rs::encode::Metadata::Xmp(x), true)
                    .map_err(|e| format!("Failed adding xmp: {e}"))?;
            }
            if let Some(i) = icc {
                encoder
                    .add_metadata(&jpegxl_rs::encode::Metadata::Custom(*b"prof", i), false)
                    .map_err(|e| format!("Failed adding icc: {e}"))?;
            }

            let frame = EncoderFrame::new(data).num_channels(num_channels);
            let result = encoder
                .encode_frame::<$t, $t>(&frame, width, height)
                .map_err(|e| format!("Failed to encode: {e}"))?;

            Ok(result.data)
        }
    };
}

impl_encode_internal!(
    encode_internal_u8,
    u8,
    jpegxl_rs::encode::ColorEncoding::SrgbLuma
);
impl_encode_internal!(
    encode_internal_u16,
    u16,
    jpegxl_rs::encode::ColorEncoding::SrgbLuma
);
impl_encode_internal!(
    encode_internal_f32,
    f32,
    jpegxl_rs::encode::ColorEncoding::LinearSrgbLuma
);

// ---------------------------------------------------------------------------
// Python API — bytes
// ---------------------------------------------------------------------------

/// Decode a JPEG XL image from bytes.
///
/// The GIL is released during decoding for concurrency.
/// Returns a tuple of (Metadata, bytes).
#[pyfunction]
#[pyo3(signature = (data, *, dtype = None))]
fn decode<'py>(
    py: Python<'py>,
    data: &[u8],
    dtype: Option<&str>,
) -> PyResult<(Metadata, Bound<'py, PyBytes>)> {
    match dtype {
        Some("uint16") => {
            let result = py
                .detach(|| decode_internal_u16(data))
                .map_err(PyRuntimeError::new_err)?;
            let bytes_slice = unsafe {
                std::slice::from_raw_parts(
                    result.pixels.as_ptr() as *const u8,
                    result.pixels.len() * std::mem::size_of::<u16>(),
                )
            };
            Ok((result.meta, PyBytes::new(py, bytes_slice)))
        }
        Some("float32") => {
            let result = py
                .detach(|| decode_internal_f32(data))
                .map_err(PyRuntimeError::new_err)?;
            let bytes_slice = unsafe {
                std::slice::from_raw_parts(
                    result.pixels.as_ptr() as *const u8,
                    result.pixels.len() * std::mem::size_of::<f32>(),
                )
            };
            Ok((result.meta, PyBytes::new(py, bytes_slice)))
        }
        _ => {
            let result = py
                .detach(|| decode_internal_u8(data))
                .map_err(PyRuntimeError::new_err)?;
            Ok((result.meta, PyBytes::new(py, &result.pixels)))
        }
    }
}

/// Probe a JPEG XL image to extract metadata without decoding pixel data.
///
/// Fast execution (< 0.5ms) with zero pixel buffer allocation.
/// The GIL is released during probing.
#[pyfunction]
fn probe<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Metadata> {
    py.detach(|| probe_internal(data))
        .map_err(PyRuntimeError::new_err)
}

/// Decode a JPEG XL image directly into a pre-allocated, writable, C-contiguous NumPy array.
///
/// Eliminates intermediate buffer allocation (zero-copy into caller's memory).
/// The GIL is released during decoding.
#[pyfunction]
fn decode_into<'py>(py: Python<'py>, data: &[u8], out: &Bound<'py, PyAny>) -> PyResult<Metadata> {
    if let Ok(mut arr_u8) = out.extract::<PyReadwriteArrayDyn<'py, u8>>() {
        let shape = arr_u8.shape();
        let (h, w, c) = if shape.len() == 2 {
            (shape[0] as u32, shape[1] as u32, 1u32)
        } else if shape.len() == 3 {
            (shape[0] as u32, shape[1] as u32, shape[2] as u32)
        } else {
            return Err(PyRuntimeError::new_err(
                "Destination array must be 2D or 3D",
            ));
        };

        let slice = arr_u8.as_slice_mut().map_err(|_| {
            PyRuntimeError::new_err("Destination array must be C-contiguous and writable")
        })?;
        let ptr_addr = slice.as_mut_ptr() as usize;
        let byte_len = std::mem::size_of_val(slice);

        py.detach(move || {
            let ptr = ptr_addr as *mut u8;
            decode_into_internal(data, ptr, byte_len, JxlDataType::Uint8, c, w, h)
        })
        .map_err(PyRuntimeError::new_err)
    } else if let Ok(mut arr_u16) = out.extract::<PyReadwriteArrayDyn<'py, u16>>() {
        let shape = arr_u16.shape();
        let (h, w, c) = if shape.len() == 2 {
            (shape[0] as u32, shape[1] as u32, 1u32)
        } else if shape.len() == 3 {
            (shape[0] as u32, shape[1] as u32, shape[2] as u32)
        } else {
            return Err(PyRuntimeError::new_err(
                "Destination array must be 2D or 3D",
            ));
        };

        let slice = arr_u16.as_slice_mut().map_err(|_| {
            PyRuntimeError::new_err("Destination array must be C-contiguous and writable")
        })?;
        let ptr_addr = slice.as_mut_ptr() as usize;
        let byte_len = std::mem::size_of_val(slice);

        py.detach(move || {
            let ptr = ptr_addr as *mut u8;
            decode_into_internal(data, ptr, byte_len, JxlDataType::Uint16, c, w, h)
        })
        .map_err(PyRuntimeError::new_err)
    } else if let Ok(mut arr_f32) = out.extract::<PyReadwriteArrayDyn<'py, f32>>() {
        let shape = arr_f32.shape();
        let (h, w, c) = if shape.len() == 2 {
            (shape[0] as u32, shape[1] as u32, 1u32)
        } else if shape.len() == 3 {
            (shape[0] as u32, shape[1] as u32, shape[2] as u32)
        } else {
            return Err(PyRuntimeError::new_err(
                "Destination array must be 2D or 3D",
            ));
        };

        let slice = arr_f32.as_slice_mut().map_err(|_| {
            PyRuntimeError::new_err("Destination array must be C-contiguous and writable")
        })?;
        let ptr_addr = slice.as_mut_ptr() as usize;
        let byte_len = std::mem::size_of_val(slice);

        py.detach(move || {
            let ptr = ptr_addr as *mut u8;
            decode_into_internal(data, ptr, byte_len, JxlDataType::Float, c, w, h)
        })
        .map_err(PyRuntimeError::new_err)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Destination array must be a writable numpy array of uint8, uint16, or float32",
        ))
    }
}

/// Encode raw pixel data to JPEG XL format.
///
/// The GIL is released during encoding for concurrency.
#[pyfunction]
#[pyo3(signature = (data, width, height, *, lossless = false, quality = 1.0, speed = EncoderSpeed::Squirrel, num_channels = 4, exif = None, xmp = None, icc = None, intensity_target = None))]
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

// ---------------------------------------------------------------------------
// Python API — NumPy (zero-copy)
// ---------------------------------------------------------------------------

/// Decode a JPEG XL image, returning a NumPy array.
///
/// Returns (Metadata, ndarray) where ndarray has shape (H, W, C).
/// If dtype is None, automatically detects source image bit depth (uint8, uint16, or float32).
/// The pixel buffer is transferred to NumPy via zero-copy ownership transfer.
/// The GIL is released during decoding.
#[pyfunction]
#[pyo3(signature = (data, *, dtype = None))]
fn decode_to_numpy<'py>(
    py: Python<'py>,
    data: &[u8],
    dtype: Option<&str>,
) -> PyResult<(Metadata, Bound<'py, PyAny>)> {
    match dtype {
        Some("uint8") => {
            let result = py
                .detach(|| decode_internal_u8(data))
                .map_err(PyRuntimeError::new_err)?;
            let h = result.meta.height as usize;
            let w = result.meta.width as usize;
            let c = result.total_channels as usize;
            let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[h, w, c]), result.pixels)
                .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to reshape pixels: {e}"))
            })?;
            Ok((result.meta, array.into_pyarray(py).into_any()))
        }
        Some("uint16") => {
            let result = py
                .detach(|| decode_internal_u16(data))
                .map_err(PyRuntimeError::new_err)?;
            let h = result.meta.height as usize;
            let w = result.meta.width as usize;
            let c = result.total_channels as usize;
            let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[h, w, c]), result.pixels)
                .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to reshape pixels: {e}"))
            })?;
            Ok((result.meta, array.into_pyarray(py).into_any()))
        }
        Some("float32") => {
            let result = py
                .detach(|| decode_internal_f32(data))
                .map_err(PyRuntimeError::new_err)?;
            let h = result.meta.height as usize;
            let w = result.meta.width as usize;
            let c = result.total_channels as usize;
            let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[h, w, c]), result.pixels)
                .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to reshape pixels: {e}"))
            })?;
            Ok((result.meta, array.into_pyarray(py).into_any()))
        }
        _ => {
            // Auto-detect bit depth and data type from JPEG XL codestream
            let result = py
                .detach(|| decode_auto(data))
                .map_err(PyRuntimeError::new_err)?;
            match result {
                AutoDecodeResult::Uint8(res) => {
                    let h = res.meta.height as usize;
                    let w = res.meta.width as usize;
                    let c = res.total_channels as usize;
                    let array =
                        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[h, w, c]), res.pixels)
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!("Failed to reshape pixels: {e}"))
                            })?;
                    Ok((res.meta, array.into_pyarray(py).into_any()))
                }
                AutoDecodeResult::Uint16(res) => {
                    let h = res.meta.height as usize;
                    let w = res.meta.width as usize;
                    let c = res.total_channels as usize;
                    let array =
                        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[h, w, c]), res.pixels)
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!("Failed to reshape pixels: {e}"))
                            })?;
                    Ok((res.meta, array.into_pyarray(py).into_any()))
                }
                AutoDecodeResult::Float(res) => {
                    let h = res.meta.height as usize;
                    let w = res.meta.width as usize;
                    let c = res.total_channels as usize;
                    let array =
                        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[h, w, c]), res.pixels)
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!("Failed to reshape pixels: {e}"))
                            })?;
                    Ok((res.meta, array.into_pyarray(py).into_any()))
                }
            }
        }
    }
}

/// Encode a NumPy array (H, W) or (H, W, C) of uint8, uint16, or float32 to JPEG XL.
///
/// Automatically supports 2D grayscale arrays. Reads via zero-copy (if C-contiguous).
/// The GIL is released during encoding.
#[pyfunction]
#[pyo3(signature = (array, *, lossless = false, quality = 1.0, speed = EncoderSpeed::Squirrel, exif = None, xmp = None, icc = None, intensity_target = None))]
fn encode_from_numpy<'py>(
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
        let shape = arr_u8.shape();
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

        let array_view = arr_u8.as_array();
        if !array_view.is_standard_layout() {
            return Err(PyRuntimeError::new_err(
                "Array must be C-contiguous. Use numpy.ascontiguousarray().",
            ));
        }
        let data = array_view.as_slice().unwrap();

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
        let shape = arr_u16.shape();
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

        let array_view = arr_u16.as_array();
        if !array_view.is_standard_layout() {
            return Err(PyRuntimeError::new_err(
                "Array must be C-contiguous. Use numpy.ascontiguousarray().",
            ));
        }
        let data = array_view.as_slice().unwrap();

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
        let shape = arr_f32.shape();
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

        let array_view = arr_f32.as_array();
        if !array_view.is_standard_layout() {
            return Err(PyRuntimeError::new_err(
                "Array must be C-contiguous. Use numpy.ascontiguousarray().",
            ));
        }
        let data = array_view.as_slice().unwrap();

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
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported array dtype. Supported dtypes are uint8, uint16, and float32.",
        ))
    }
}

// ---------------------------------------------------------------------------
// JPEG codec via turbojpeg (static-linked libjpeg-turbo)
// ---------------------------------------------------------------------------

/// Simple metadata for decoded JPEG images.
#[pyclass(get_all, from_py_object)]
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
        let mut jpeg_buf: Vec<u8> = vec![0u8; jxl_data.len() * 2];
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
        .detach(|| jpeg_to_jxl_internal(data))
        .map_err(PyRuntimeError::new_err)?;
    Ok(PyBytes::new(py, &jxl))
}

/// Reconstruct the original JPEG from a JXL that was created via lossless transcoding.
///
/// The GIL is released during reconstruction.
#[pyfunction]
fn jxl_to_jpeg<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let jpeg = py
        .detach(|| jxl_to_jpeg_internal(data))
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
        .detach(|| jpeg_decode_internal(data))
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
        .detach(|| jpeg_encode_internal(data, width, height, quality, num_channels))
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
        .detach(|| jpeg_decode_internal(data))
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
        .detach(|| jpeg_encode_internal(data, width, height, quality, num_channels))
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
    // JPEG ↔ JXL lossless transcoding
    m.add_function(wrap_pyfunction!(jpeg_to_jxl, m)?)?;
    m.add_function(wrap_pyfunction!(jxl_to_jpeg, m)?)?;
    Ok(())
}
