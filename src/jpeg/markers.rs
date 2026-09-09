use std::collections::BTreeMap;

/// Fast, zero-dependency JPEG APP marker parser for EXIF and ICC profile.
pub fn parse_jpeg_markers(data: &[u8]) -> (Option<Vec<u8>>, Option<Vec<u8>>) {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return (None, None);
    }

    let mut exif = None;
    let mut icc_chunks: BTreeMap<u8, Vec<u8>> = BTreeMap::new();
    let mut expected_icc_chunks = 0u8;

    let mut i = 2usize;
    while i + 4 <= data.len() {
        if data[i] != 0xFF {
            break;
        }

        // Skip extra 0xFF padding bytes
        while i < data.len() && data[i] == 0xFF {
            i += 1;
        }
        if i >= data.len() {
            break;
        }

        let marker = data[i];
        i += 1;

        // Standalone markers without length
        if marker == 0xD8 || marker == 0xD9 || (0xD0..=0xD7).contains(&marker) {
            continue;
        }

        // Start of Scan (SOS) - compressed image data follows, stop parsing headers
        if marker == 0xDA {
            break;
        }

        if i + 2 > data.len() {
            break;
        }

        let length = u16::from_be_bytes([data[i], data[i + 1]]) as usize;
        if length < 2 || i + length > data.len() {
            break;
        }

        let payload = &data[i + 2..i + length];
        i += length;

        // APP1: EXIF
        if marker == 0xE1 && exif.is_none() && payload.starts_with(b"Exif\0\0") {
            exif = Some(payload[6..].to_vec());
        }

        // APP2: ICC Profile
        if marker == 0xE2 && payload.starts_with(b"ICC_PROFILE\0") && payload.len() >= 14 {
            let chunk_num = payload[12];
            let total_chunks = payload[13];
            if chunk_num >= 1 && (expected_icc_chunks == 0 || total_chunks == expected_icc_chunks) {
                expected_icc_chunks = total_chunks;
                icc_chunks.insert(chunk_num, payload[14..].to_vec());
            }
        }
    }

    let icc = if expected_icc_chunks > 0 && icc_chunks.len() == expected_icc_chunks as usize {
        let mut full_icc = Vec::new();
        for chunk_idx in 1..=expected_icc_chunks {
            if let Some(chunk) = icc_chunks.get(&chunk_idx) {
                full_icc.extend_from_slice(chunk);
            }
        }
        if full_icc.is_empty() {
            None
        } else {
            Some(full_icc)
        }
    } else {
        None
    };

    (exif, icc)
}
