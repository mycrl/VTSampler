use crate::VTFormat;

/// CPU-side pixel buffers passed to [`crate::VTImage::from_cpu`].
///
/// Each variant stores plane pointers and **row stride in bytes** (not always `width * bytes_per_pixel`).
///
/// # Example (YUV420P / I420)
///
/// ```
/// use vtsampler::PixelData;
///
/// let width = 1920u32;
/// let height = 1080u32;
/// let y_size = (width * height) as usize;
/// let uv_size = (width as usize / 2 * height as usize / 2);
/// # let y_plane = vec![0u8; y_size];
/// # let u_plane = vec![0u8; uv_size];
/// # let v_plane = vec![0u8; uv_size];
///
/// let data = PixelData::YUV420P {
///     buffer: [&y_plane, &u_plane, &v_plane],
///     stride: [width as usize, width as usize / 2, width as usize / 2],
/// };
/// assert_eq!(data.format(), vtsampler::VTFormat::YUV420P);
/// ```
pub enum PixelData<'a> {
    /// Packed 8-bit RGBA rows.
    RGBA {
        /// Contiguous or padded frame data (at least `stride * height` bytes).
        buffer: &'a [u8],
        /// Bytes between the start of consecutive rows (≥ `width * 4`).
        stride: usize,
    },
    /// Packed 8-bit BGRA rows.
    BGRA {
        /// Contiguous or padded frame data (at least `stride * height` bytes).
        buffer: &'a [u8],
        /// Bytes between the start of consecutive rows (≥ `width * 4`).
        stride: usize,
    },
    /// NV12: index 0 = Y plane, index 1 = interleaved UV.
    NV12 {
        /// `[y_plane, uv_plane]`.
        buffer: [&'a [u8]; 2],
        /// Row pitch in bytes for each plane.
        stride: [usize; 2],
    },
    /// YUV420P (I420): separate Y, U, and V planes.
    YUV420P {
        /// `[y, u, v]`.
        buffer: [&'a [u8]; 3],
        /// Row pitch in bytes per plane (U/V are typically `width / 2`).
        stride: [usize; 3],
    },
}

impl PixelData<'_> {
    /// Returns the [`VTFormat`] that matches this memory layout.
    pub fn format(&self) -> VTFormat {
        match self {
            Self::RGBA { .. } => VTFormat::RGBA,
            Self::BGRA { .. } => VTFormat::BGRA,
            Self::NV12 { .. } => VTFormat::NV12,
            Self::YUV420P { .. } => VTFormat::YUV420P,
        }
    }
}
