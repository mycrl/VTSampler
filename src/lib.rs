//! Cross-platform GPU video format conversion and scaling (wgpu compute).
//!
//! Public API: one [`VTImage`] in, one [`VTImage`] out via [`VTSampler::process`] or
//! [`VTSampler::encode`]. CPU pixels and render-target textures are handled internally
//! with a scratch pool to avoid extra copies when GPU usages already match.

pub mod bridge;

mod format;
mod gpu;
mod image;
mod pixel;
mod pool;
mod process;
mod sampler;
mod shader;

pub use bridge::BridgeError;

#[cfg(windows)]
pub use bridge::{VtD3d11Bridge, VtD3d11Device, VtD3d11Pool, vt_format_to_dxgi_wgpu};

#[cfg(target_os = "macos")]
pub use bridge::{CVPixelBufferRef, VtMetalCache};
pub use format::{VTColorSpace, VTFormat, VTProcessOptions, VTSampleError, VTScaleFilter};
pub use image::{VTImage, VTImageOwned, VTTextureRole};
pub use pixel::PixelData;
pub use sampler::{VTSampler, VTSamplerBuilder};
pub use shader::PipelineKey;

pub use wgpu;
