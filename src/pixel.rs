/// CPU-side pixel payload referenced by [`crate::VTImage`].
pub enum PixelData<'a> {
    RGBA {
        buffer: &'a [u8],
        stride: usize,
    },
    BGRA {
        buffer: &'a [u8],
        stride: usize,
    },
    NV12 {
        buffer: [&'a [u8]; 2],
        stride: [usize; 2],
    },
    YUV420P {
        buffer: [&'a [u8]; 3],
        stride: [usize; 3],
    },
}

impl PixelData<'_> {
    pub fn format(&self) -> crate::VTFormat {
        match self {
            Self::RGBA { .. } => crate::VTFormat::RGBA,
            Self::BGRA { .. } => crate::VTFormat::BGRA,
            Self::NV12 { .. } => crate::VTFormat::NV12,
            Self::YUV420P { .. } => crate::VTFormat::YUV420P,
        }
    }
}
