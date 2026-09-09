use crate::common::guards::DecoderGuard;
use crate::jxl::types::Metadata;
use jpegxl_sys::common::types::JxlBool;
use jpegxl_sys::decode::{
    JxlColorProfileTarget, JxlDecoderCreate, JxlDecoderGetBasicInfo, JxlDecoderGetBoxSizeRaw,
    JxlDecoderGetBoxType, JxlDecoderGetColorAsICCProfile, JxlDecoderGetICCProfileSize,
    JxlDecoderProcessInput, JxlDecoderReleaseBoxBuffer, JxlDecoderSetBoxBuffer,
    JxlDecoderSetDecompressBoxes, JxlDecoderSetInput, JxlDecoderStatus, JxlDecoderSubscribeEvents,
};
use jpegxl_sys::metadata::codestream_header::JxlBasicInfo;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::ptr;

pub fn probe_internal(data: &[u8]) -> Result<Metadata, String> {
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
                    let new_size =
                        (current_box_data.len() * 2).max(current_box_data.len() + 65536);
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
                JxlDecoderStatus::Success
                | JxlDecoderStatus::Error
                | JxlDecoderStatus::NeedMoreInput => break,
                _ => {}
            }

            if width > 0 && exif.is_some() && xmp.is_some() && icc_box.is_some() {
                break;
            }
        }

        if width == 0 || height == 0 {
            return Err("Invalid or unreadable JPEG XL data".into());
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

/// Probe a JPEG XL image to extract metadata without decoding pixel data.
///
/// Fast execution (< 0.5ms) with zero pixel buffer allocation.
/// The GIL is released during probing.
#[pyfunction]
pub fn probe<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Metadata> {
    py.detach(|| probe_internal(data))
        .map_err(PyRuntimeError::new_err)
}
