use crate::common::guards::DecoderGuard;
use crate::common::runner::get_shared_c_runner;
use crate::jxl::types::Metadata;
use jpegxl_sys::common::types::{JxlBool, JxlDataType, JxlEndianness, JxlPixelFormat};
use jpegxl_sys::decode::{
    JxlColorProfileTarget, JxlDecoderCreate, JxlDecoderGetBasicInfo, JxlDecoderGetBoxSizeRaw,
    JxlDecoderGetBoxType, JxlDecoderGetColorAsICCProfile, JxlDecoderGetICCProfileSize,
    JxlDecoderImageOutBufferSize, JxlDecoderProcessInput, JxlDecoderReleaseBoxBuffer,
    JxlDecoderSetBoxBuffer, JxlDecoderSetDecompressBoxes, JxlDecoderSetImageOutBuffer,
    JxlDecoderSetInput, JxlDecoderSetParallelRunner, JxlDecoderStatus, JxlDecoderSubscribeEvents,
};
use jpegxl_sys::metadata::codestream_header::JxlBasicInfo;
use std::ffi::c_void;
use std::ptr;

pub enum DecodedPixels {
    Uint8(Vec<u8>),
    Uint16(Vec<u16>),
    Float(Vec<f32>),
}

impl DecodedPixels {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            DecodedPixels::Uint8(v) => v.as_slice(),
            DecodedPixels::Uint16(v) => unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of::<u16>())
            },
            DecodedPixels::Float(v) => unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of::<f32>())
            },
        }
    }
}

pub struct SinglePassResult {
    pub meta: Metadata,
    pub pixels: DecodedPixels,
    pub total_channels: u32,
}

enum BufferTarget {
    Allocated(Option<DecodedPixels>),
    Preallocated {
        out_ptr: *mut c_void,
        out_byte_len: usize,
        expected_w: u32,
        expected_h: u32,
        expected_channels: u32,
    },
}

/// Core single-pass decoding engine with full ISOBMFF box support before and after image codestreams.
pub fn decode_jxl_single_pass(
    data: &[u8],
    requested_dtype: Option<&str>,
) -> Result<SinglePassResult, String> {
    let mut target = BufferTarget::Allocated(None);
    let (meta, total_channels) = run_decoder_loop(data, requested_dtype, &mut target)?;

    let pixels = match target {
        BufferTarget::Allocated(Some(p)) => p,
        _ => return Err("Failed to retrieve decoded pixel data".into()),
    };

    Ok(SinglePassResult {
        meta,
        pixels,
        total_channels,
    })
}

/// Decode directly into preallocated buffer without intermediate allocations.
pub fn decode_jxl_into(
    data: &[u8],
    out_ptr: *mut c_void,
    out_byte_len: usize,
    data_type: JxlDataType,
    expected_channels: u32,
    expected_w: u32,
    expected_h: u32,
) -> Result<Metadata, String> {
    let requested_dtype = match data_type {
        JxlDataType::Uint8 => Some("uint8"),
        JxlDataType::Uint16 => Some("uint16"),
        JxlDataType::Float => Some("float32"),
        _ => return Err("Unsupported data type for decode_into".into()),
    };

    let mut target = BufferTarget::Preallocated {
        out_ptr,
        out_byte_len,
        expected_w,
        expected_h,
        expected_channels,
    };

    let (meta, _) = run_decoder_loop(data, requested_dtype, &mut target)?;
    Ok(meta)
}

fn run_decoder_loop(
    data: &[u8],
    requested_dtype: Option<&str>,
    target: &mut BufferTarget,
) -> Result<(Metadata, u32), String> {
    unsafe {
        let dec = JxlDecoderCreate(ptr::null());
        if dec.is_null() {
            return Err("Failed to create JXL decoder".into());
        }
        let _dec_guard = DecoderGuard(dec);

        // Attach shared thread runner if configured
        let _runner_ref = get_shared_c_runner();
        if let Some(ref r) = _runner_ref {
            if JxlDecoderSetParallelRunner(
                dec,
                jpegxl_sys::threads::thread_parallel_runner::JxlThreadParallelRunner,
                r.as_ptr(),
            ) != JxlDecoderStatus::Success
            {
                return Err("Failed to set parallel runner on decoder".into());
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
        let mut total_channels = 4u32;

        let mut selected_dtype = JxlDataType::Uint8;
        let mut buffer_set = false;
        let mut image_decoded = false;

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
                        total_channels = num_color_channels + u32::from(has_alpha);

                        // Determine pixel data type
                        selected_dtype = match requested_dtype {
                            Some("uint8") => JxlDataType::Uint8,
                            Some("uint16") => JxlDataType::Uint16,
                            Some("float32") => JxlDataType::Float,
                            _ => {
                                // Auto-detect bit depth
                                if bits_per_sample <= 8 {
                                    JxlDataType::Uint8
                                } else if bits_per_sample <= 16 {
                                    JxlDataType::Uint16
                                } else {
                                    JxlDataType::Float
                                }
                            }
                        };

                        if let BufferTarget::Preallocated {
                            expected_w,
                            expected_h,
                            expected_channels,
                            ..
                        } = target
                        {
                            if width != *expected_w || height != *expected_h {
                                return Err(format!(
                                    "Image dimensions ({}x{}) do not match buffer dimensions ({}x{})",
                                    width, height, expected_w, expected_h
                                ));
                            }
                            if total_channels != *expected_channels {
                                return Err(format!(
                                    "Image channels ({}) do not match buffer channels ({})",
                                    total_channels, expected_channels
                                ));
                            }
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
                JxlDecoderStatus::NeedImageOutBuffer => {
                    let format = JxlPixelFormat {
                        num_channels: total_channels,
                        data_type: selected_dtype,
                        endianness: JxlEndianness::Native,
                        align: 0,
                    };
                    let mut required_size = 0usize;
                    if JxlDecoderImageOutBufferSize(dec, &format, &mut required_size)
                        != JxlDecoderStatus::Success
                    {
                        return Err("Failed to calculate image output buffer size".into());
                    }

                    match target {
                        BufferTarget::Preallocated {
                            out_ptr,
                            out_byte_len,
                            ..
                        } => {
                            if *out_byte_len < required_size {
                                return Err(format!(
                                    "Buffer size too small: provided {} bytes, need {} bytes",
                                    *out_byte_len, required_size
                                ));
                            }
                            if JxlDecoderSetImageOutBuffer(dec, &format, *out_ptr, *out_byte_len)
                                != JxlDecoderStatus::Success
                            {
                                return Err("Failed to set image output buffer".into());
                            }
                        }
                        BufferTarget::Allocated(ref mut alloc_ref) => match selected_dtype {
                            JxlDataType::Uint8 => {
                                let mut buf = vec![0u8; required_size];
                                if JxlDecoderSetImageOutBuffer(
                                    dec,
                                    &format,
                                    buf.as_mut_ptr() as *mut c_void,
                                    required_size,
                                ) != JxlDecoderStatus::Success
                                {
                                    return Err("Failed to set image output buffer".into());
                                }
                                *alloc_ref = Some(DecodedPixels::Uint8(buf));
                            }
                            JxlDataType::Uint16 => {
                                let num_elements = required_size / 2;
                                let mut buf = vec![0u16; num_elements];
                                if JxlDecoderSetImageOutBuffer(
                                    dec,
                                    &format,
                                    buf.as_mut_ptr() as *mut c_void,
                                    required_size,
                                ) != JxlDecoderStatus::Success
                                {
                                    return Err("Failed to set image output buffer".into());
                                }
                                *alloc_ref = Some(DecodedPixels::Uint16(buf));
                            }
                            JxlDataType::Float => {
                                let num_elements = required_size / 4;
                                let mut buf = vec![0.0f32; num_elements];
                                if JxlDecoderSetImageOutBuffer(
                                    dec,
                                    &format,
                                    buf.as_mut_ptr() as *mut c_void,
                                    required_size,
                                ) != JxlDecoderStatus::Success
                                {
                                    return Err("Failed to set image output buffer".into());
                                }
                                *alloc_ref = Some(DecodedPixels::Float(buf));
                            }
                            _ => return Err("Unsupported pixel data type".into()),
                        },
                    }
                    buffer_set = true;
                }
                JxlDecoderStatus::FullImage => {
                    image_decoded = true;
                    // Do NOT break here! Continue loop until Success so any trailing Box chunks (Exif/XMP) are parsed!
                }
                JxlDecoderStatus::Success => {
                    break;
                }
                JxlDecoderStatus::Error => {
                    return Err("JXL decoder error encountered".into());
                }
                JxlDecoderStatus::NeedMoreInput => {
                    if image_decoded {
                        // All image data was processed
                        break;
                    } else {
                        return Err("Incomplete JXL data".into());
                    }
                }
                _ => {}
            }
        }

        if !buffer_set || !image_decoded {
            return Err("Decoder did not finish producing full image frame".into());
        }

        let final_icc = icc_box.or(decoder_icc);

        let actual_bits = match selected_dtype {
            JxlDataType::Uint8 => 8,
            JxlDataType::Uint16 => 16,
            JxlDataType::Float => 32,
            _ => bits_per_sample,
        };

        Ok((
            Metadata {
                width,
                height,
                num_color_channels,
                has_alpha,
                exif,
                xmp,
                icc: final_icc,
                bits_per_sample: actual_bits,
                intensity_target,
                min_nits,
            },
            total_channels,
        ))
    }
}
