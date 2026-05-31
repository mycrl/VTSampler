use wgpu::TextureFormat;

/// Logical pixel format for conversion and scaling.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum VTFormat {
    RGBA,
    BGRA,
    YUV420P,
    NV12,
}

impl VTFormat {
    pub fn all() -> &'static [Self] {
        &[Self::RGBA, Self::BGRA, Self::NV12, Self::YUV420P]
    }

    /// Number of GPU texture planes used internally (NV12/YUV420P are multi-plane).
    pub fn plane_count(self) -> usize {
        match self {
            Self::RGBA | Self::BGRA => 1,
            Self::NV12 => 2,
            Self::YUV420P => 3,
        }
    }

    /// WGSL storage/texture view formats per plane (split-plane layout).
    #[rustfmt::skip]
    pub fn plane_formats(self) -> &'static [TextureFormat] {
        match self {
            Self::RGBA => &[TextureFormat::Rgba8Unorm],
            Self::BGRA => &[TextureFormat::Bgra8Unorm],
            Self::NV12 => &[TextureFormat::R8Unorm, TextureFormat::Rg8Unorm],
            Self::YUV420P => {
                &[TextureFormat::R8Unorm, TextureFormat::R8Unorm, TextureFormat::R8Unorm]
            }
        }
    }

    pub fn shader_name(self) -> &'static str {
        match self {
            Self::RGBA => "rgba",
            Self::BGRA => "bgra",
            Self::NV12 => "nv12",
            Self::YUV420P => "yuv420p",
        }
    }
}

/// YUV/RGB matrix selection for conversion shaders.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub enum VTColorSpace {
    #[default]
    Bt601Limited,
    Bt709Limited,
}

/// Filtering used when input and output dimensions differ.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub enum VTScaleFilter {
    #[default]
    Nearest,
    Bilinear,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub struct VTProcessOptions {
    pub color_space: VTColorSpace,
    pub scale_filter: VTScaleFilter,
}

/// Errors surfaced by the conversion pipeline.
#[derive(Debug, thiserror::Error)]
pub enum VTSampleError {
    #[error("no suitable GPU adapter")]
    NotFoundAdapter,
    #[error("failed to request GPU device")]
    RequestDeviceFailed,
    #[error("unsupported pixel format")]
    UnsupportedFormat,
    #[error("unsupported image backing")]
    UnsupportedBacking,
    #[error("texture is missing required usage flags")]
    MissingTextureUsage,
    #[error("shader compile: {0}")]
    ShaderCompile(String),
    #[error("template: {0}")]
    Template(String),
    #[error(transparent)]
    Bridge(#[from] crate::bridge::BridgeError),
}
