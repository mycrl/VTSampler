//! Platform texture bridges (D3D11 / CVPixelBuffer+Metal → wgpu).

#[cfg(windows)]
pub mod d3d11;

#[cfg(target_os = "macos")]
pub mod metal;

#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[cfg(windows)]
    #[error(transparent)]
    Windows(#[from] windows::core::Error),
    #[cfg(target_os = "macos")]
    #[error("core video error code {0}")]
    CoreVideo(i32),
    #[error("wgpu dx12 backend is not available")]
    NotFoundDxBackend,
    #[cfg(windows)]
    #[error("invalid d3d shared handle")]
    InvalidSharedHandle,
    #[error("wgpu metal backend is not available")]
    NotFoundMetalBackend,
    #[error("unsupported pixel format for bridge")]
    UnsupportedFormat,
}

#[cfg(windows)]
pub use d3d11::{VtD3d11Bridge, VtD3d11Device, VtD3d11Pool, vt_format_to_dxgi_wgpu};

#[cfg(target_os = "macos")]
pub use metal::{CVPixelBufferRef, VtMetalCache};
