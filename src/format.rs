use wgpu::TextureFormat;

/// Logical pixel format handled by the conversion pipeline.
///
/// Each variant corresponds to a layout understood by [`crate::VTSampler`]. Multi-plane
/// formats use split wgpu textures unless noted (Windows combined NV12 via
/// [`crate::VTImage::from_nv12_texture`]).
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum VTFormat {
    /// 8-bit RGBA, single plane ([`TextureFormat::Rgba8Unorm`]).
    RGBA,
    /// 8-bit BGRA, single plane ([`TextureFormat::Bgra8Unorm`]).
    BGRA,
    /// 4:2:0 planar (I420-style): Y, U, V as three [`TextureFormat::R8Unorm`] planes.
    YUV420P,
    /// 4:2:0 semi-planar: Y ([`TextureFormat::R8Unorm`]) + interleaved UV ([`TextureFormat::Rg8Unorm`]).
    NV12,
}

impl VTFormat {
    /// All formats that can be used as conversion input or output.
    pub fn all() -> &'static [Self] {
        &[Self::RGBA, Self::BGRA, Self::NV12, Self::YUV420P]
    }

    /// Number of GPU texture planes in the **split-plane** layout.
    ///
    /// Combined NV12 (single texture, plane aspects) still reports `2` here; use
    /// [`crate::VTImage::from_nv12_texture`] on Windows for that backing.
    pub fn plane_count(self) -> usize {
        match self {
            Self::RGBA | Self::BGRA => 1,
            Self::NV12 => 2,
            Self::YUV420P => 3,
        }
    }

    /// [`wgpu::TextureFormat`] for each plane in split-plane layout.
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

    /// Name used when generating WGSL from templates (internal).
    pub fn shader_name(self) -> &'static str {
        match self {
            Self::RGBA => "rgba",
            Self::BGRA => "bgra",
            Self::NV12 => "nv12",
            Self::YUV420P => "yuv420p",
        }
    }
}

/// YUV ↔ RGB matrix used during format conversion.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub enum VTColorSpace {
    /// ITU-R BT.601 limited range (default).
    #[default]
    Bt601Limited,
    /// ITU-R BT.709 limited range.
    Bt709Limited,
}

/// Filter used when [`crate::VTImage::width`] / [`crate::VTImage::height`] differ between input and output.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub enum VTScaleFilter {
    /// Nearest-neighbor sampling (default).
    #[default]
    Nearest,
    /// Bilinear sampling in the compute shader.
    Bilinear,
}

/// Options for a single [`crate::VTSampler::process`] or [`crate::VTSampler::encode`] call.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub struct VTProcessOptions {
    /// Color matrix applied when converting between YUV and RGB.
    pub color_space: VTColorSpace,
    /// Resampling mode when input and output sizes differ.
    pub scale_filter: VTScaleFilter,
}

/// Error type for [`crate::VTSampler`] and related APIs.
#[derive(Debug, thiserror::Error)]
pub enum VTSampleError {
    /// No GPU adapter was found (only when using the built-in headless device).
    #[error("no suitable GPU adapter")]
    NotFoundAdapter,
    /// [`wgpu::Adapter::request_device`] failed.
    #[error("failed to request GPU device")]
    RequestDeviceFailed,
    /// The format pair or plane layout is not implemented.
    #[error("unsupported pixel format")]
    UnsupportedFormat,
    /// The [`crate::VTImage`] backing cannot be used for this operation.
    #[error("unsupported image backing")]
    UnsupportedBacking,
    /// A texture is missing `COPY_SRC`, `COPY_DST`, `STORAGE_BINDING`, or `TEXTURE_BINDING`.
    #[error("texture is missing required usage flags")]
    MissingTextureUsage,
    /// Generated WGSL failed to compile.
    #[error("shader compile: {0}")]
    ShaderCompile(String),
    /// Minijinja template rendering failed.
    #[error("template: {0}")]
    Template(String),
    /// Platform bridge error (D3D11, Core Video, or Metal).
    #[error(transparent)]
    Bridge(#[from] crate::bridge::BridgeError),
}
