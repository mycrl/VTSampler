//! Platform bridges: import native GPU textures into wgpu.
//!
//! | Platform | Module | wgpu backend |
//! |----------|--------|----------------|
//! | Windows | [`d3d11`] | **DX12** |
//! | macOS | `metal` module | **Metal** |
//!
//! Use these when integrating with D3D11 capture/encode or `CVPixelBuffer` from
//! AVFoundation / ScreenCaptureKit, or pass native handles via [`crate::VTImage::from_d3d11`]
//! (Windows) and `VTImage::from_cv_pixel_buffer` (macOS).

#[cfg(windows)]
pub mod d3d11;

#[cfg(target_os = "macos")]
pub mod metal;

/// Errors from D3D11 or Core Video / Metal interop.
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    /// Win32 API error.
    #[cfg(windows)]
    #[error(transparent)]
    Windows(#[from] windows::core::Error),
    /// Core Video returned a non-success status.
    #[cfg(target_os = "macos")]
    #[error("core video error code {0}")]
    CoreVideo(i32),
    /// wgpu was not created with the DX12 backend.
    #[error("wgpu dx12 backend is not available")]
    NotFoundDxBackend,
    /// DXGI shared handle is null or invalid.
    #[cfg(windows)]
    #[error("invalid d3d shared handle")]
    InvalidSharedHandle,
    /// wgpu was not created with the Metal backend.
    #[error("wgpu metal backend is not available")]
    NotFoundMetalBackend,
    /// Pixel format is not supported by the bridge (e.g. YUV420P on D3D11).
    #[error("unsupported pixel format for bridge")]
    UnsupportedFormat,
}

#[cfg(windows)]
pub use d3d11::{VtD3d11Bridge, VtD3d11Device, VtD3d11Pool, vt_format_to_dxgi_wgpu};

#[cfg(target_os = "macos")]
pub use metal::{CVPixelBufferRef, VtMetalCache};
