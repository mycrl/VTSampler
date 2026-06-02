//! # VTSampler
//!
//! Cross-platform GPU **video format conversion** and **scaling** built on [wgpu] compute shaders.
//! Think of it as a portable alternative to D3D11 Video Processor: one [`VTImage`] in, one [`VTImage`] out.
//!
//! [wgpu]: https://docs.rs/wgpu
//!
//! ## Quick start
//!
//! ```no_run
//! use vtsampler::{PixelData, VTFormat, VTImage, VTProcessOptions, VTSamplerBuilder};
//!
//! # async fn example() -> Result<(), vtsampler::VTSampleError> {
//! let mut sampler = VTSamplerBuilder::default().build().await?;
//!
//! let pixels = PixelData::RGBA {
//!     buffer: &[],
//!     stride: 1920 * 4,
//! };
//! let input = VTImage::from_cpu(&pixels, 1920, 1080);
//!
//! // `output_texture` must be created with STORAGE_BINDING (+ COPY_DST if render target).
//! # let output_texture = todo!();
//! let output = VTImage::from_render_target(&output_texture, VTFormat::BGRA);
//!
//! sampler.process(&input, &output, VTProcessOptions::default())?;
//! # Ok(())
//! # }
//! ```
//!
//! Share the application's wgpu device (recommended when you already render with wgpu):
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use vtsampler::VTSamplerBuilder;
//! # async fn example(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Result<(), vtsampler::VTSampleError> {
//! let mut sampler = VTSamplerBuilder::default()
//!     .with_arc_device(device, queue)
//!     .build()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Core types
//!
//! | Type | Role |
//! |------|------|
//! | [`VTSampler`] | Runs conversion / scale passes (owns shader cache + scratch pool). |
//! | [`VTImage`] | Logical frame: [`VTFormat`] + size + backing (CPU, wgpu, or native). |
//! | [`VTProcessOptions`] | Color matrix ([`VTColorSpace`]) and scaler ([`VTScaleFilter`]). |
//! | [`PixelData`] | CPU plane layout for [`VTImage::from_cpu`]. |
//!
//! ## Supported formats
//!
//! | [`VTFormat`] | Planes | Notes |
//! |--------------|--------|-------|
//! | `RGBA` | 1 | 8-bit per channel, tightly packed rows via [`PixelData::RGBA`]. |
//! | `BGRA` | 1 | Same as RGBA with B/G order. |
//! | `NV12` | 2 (split) or 1 (combined, Windows) | Y plane + interleaved UV; see [`VTImage::from_nv12_planes`]. |
//! | `YUV420P` | 3 | Separate Y, U, V planes (I420-style). |
//!
//! Any supported input format can be converted to any supported output format. When
//! `input.width/height != output.width/height`, scaling is applied using [`VTScaleFilter`].
//!
//! ## Backing kinds ([`VTImage`])
//!
//! - **CPU** — [`VTImage::from_cpu`]: uploaded to GPU scratch textures for the pass.
//! - **wgpu** — [`VTImage::from_texture`], [`VTImage::from_render_target`],
//!   [`VTImage::from_nv12_planes`], [`VTImage::from_yuv420p_planes`].
//! - **Windows** — [`VTImage::from_d3d11`], [`VTImage::from_d3d11_bridge`],
//!   [`VTImage::from_nv12_texture`] (combined NV12 on DXGI).
//! - **macOS** — `VTImage::from_cv_pixel_buffer` (uploaded via Metal inside the pass).
//!
//! Use [`VTTextureRole::Renderable`] (via [`VTImage::from_render_target`]) when the output
//! is drawn in a render pass and may not have `STORAGE_BINDING`. The processor copies through
//! scratch storage when needed.
//!
//! ## Texture usage requirements
//!
//! | Role | Typical `wgpu::TextureUsages` |
//! |------|-------------------------------|
//! | Compute input | `TEXTURE_BINDING` |
//! | Compute output (intermediate) | `STORAGE_BINDING` |
//! | Render target output | `STORAGE_BINDING` \| `TEXTURE_BINDING` \| `COPY_DST` |
//! | CPU → GPU upload target | `COPY_DST` (internal scratch) |
//!
//! Missing flags return [`VTSampleError::MissingTextureUsage`].
//!
//! ## `process` vs `encode`
//!
//! - [`VTSampler::process`] — Creates a command encoder, records the pass, and **submits** to the queue.
//! - [`VTSampler::encode`] — Records only into **your** [`wgpu::CommandEncoder`] so you can batch with
//!   rendering or native bridge copies in one submission.
//!
//! ## Platform bridges ([`bridge`])
//!
//! **Windows (DX12 wgpu backend):** [`VtD3d11Device`], [`VtD3d11Bridge`], [`VtD3d11Pool`] import
//! shared D3D11 textures into wgpu. Requires `D3D11_RESOURCE_MISC_SHARED` textures.
//!
//! **macOS (Metal wgpu backend):** `VtMetalCache` maps `CVPixelBufferRef` to Metal and copies into
//! wgpu plane textures.
//!
//! ## Limitations
//!
//! - Color matrices: [`VTColorSpace::Bt601Limited`] and [`VTColorSpace::Bt709Limited`] only.
//! - No GPU → CPU readback in this crate; use your own staging path if needed.
//! - [`VTFormat::YUV420P`] is not supported as a D3D11 bridge DXGI format (use split wgpu planes or NV12).
//! - Android / WebGPU are not tested; Windows and macOS bridges are feature-gated.
//!
//! ## Feature flags
//!
//! This crate has no optional Cargo features today. Windows and macOS bridge code is compiled
//! via `cfg(windows)` and `cfg(target_os = "macos")` respectively.
//!
//! ## Examples
//!
//! ```sh
//! cargo run --example simple
//! ```
//!
//! The `simple` example decodes a JPEG to YUV420P with **ffmpeg**, converts to BGRA via VTSampler,
//! and blits to a window. It requires `ffmpeg` on `PATH` and network access for the sample URL.

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

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

/// Windows D3D11 ↔ wgpu bridge types.
#[cfg(windows)]
#[cfg_attr(docsrs, doc(cfg(windows)))]
pub use bridge::{VtD3d11Bridge, VtD3d11Device, VtD3d11Pool, vt_format_to_dxgi_wgpu};

/// macOS Core Video ↔ Metal ↔ wgpu bridge types.
#[cfg(target_os = "macos")]
#[cfg_attr(docsrs, doc(cfg(target_os = "macos")))]
pub use bridge::{CVPixelBufferRef, VtMetalCache};
pub use format::{VTColorSpace, VTFormat, VTProcessOptions, VTSampleError, VTScaleFilter};
pub use image::{VTImage, VTImageOwned, VTTextureRole};
pub use pixel::PixelData;
pub use sampler::{VTSampler, VTSamplerBuilder};
pub use shader::PipelineKey;

/// Re-exported for convenience when you share devices or inspect capabilities with VTSampler.
pub use wgpu;
