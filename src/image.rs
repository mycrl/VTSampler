use wgpu::Texture;

use crate::{VTFormat, pixel::PixelData};

/// How a GPU texture participates in a process pass (affects copies vs in-place compute).
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub enum VTTextureRole {
    /// Compute may read/write directly when [`wgpu::TextureUsages`] allow it.
    #[default]
    General,
    /// Output is consumed by a render pass (swapchain, etc.); may use an internal scratch copy.
    Renderable,
}

/// A single video frame: [`VTFormat`], dimensions, and backing storage.
///
/// Create instances with the `from_*` constructors, then pass to
/// [`crate::VTSampler::process`] or [`crate::VTSampler::encode`].
///
/// # Example
///
/// ```no_run
/// use vtsampler::{PixelData, VTFormat, VTImage, VTProcessOptions, VTSamplerBuilder};
///
/// # async fn demo() -> Result<(), vtsampler::VTSampleError> {
/// # let mut sampler = VTSamplerBuilder::default().build().await?;
/// let pixels = PixelData::RGBA { buffer: &[], stride: 4 };
/// let input = VTImage::from_cpu(&pixels, 1, 1);
/// # let tex = todo!();
/// let output = VTImage::from_render_target(&tex, VTFormat::BGRA);
/// sampler.process(&input, &output, VTProcessOptions::default())?;
/// # Ok(())
/// # }
/// ```
pub struct VTImage<'a> {
    /// Pixel layout of this frame.
    pub format: VTFormat,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    pub(crate) backing: VTImageBacking<'a>,
}

pub(crate) enum VTImageBacking<'a> {
    Cpu(&'a PixelData<'a>),
    Gpu {
        texture: &'a Texture,
        role: VTTextureRole,
    },
    Nv12Split {
        y: &'a Texture,
        uv: &'a Texture,
        role: VTTextureRole,
    },
    Yuv420pSplit {
        y: &'a Texture,
        u: &'a Texture,
        v: &'a Texture,
        role: VTTextureRole,
    },
    #[cfg(windows)]
    Nv12Combined {
        texture: &'a Texture,
        role: VTTextureRole,
    },
    #[cfg(windows)]
    D3d11 {
        device: &'a crate::bridge::d3d11::VtD3d11Device<'a>,
        texture: &'a windows::Win32::Graphics::Direct3D11::ID3D11Texture2D,
        array_index: u32,
    },
    #[cfg(target_os = "macos")]
    CvPixelBuffer {
        buffer: crate::bridge::metal::CVPixelBufferRef,
    },
}

impl<'a> VTImage<'a> {
    /// Wraps CPU memory; uploaded to internal GPU scratch textures during processing.
    ///
    /// `width` and `height` must match the logical frame size implied by `data` and strides.
    pub fn from_cpu(data: &'a PixelData<'a>, width: u32, height: u32) -> Self {
        Self {
            format: data.format(),
            width,
            height,
            backing: VTImageBacking::Cpu(data),
        }
    }

    /// Wraps an existing wgpu texture (dimensions taken from the texture).
    ///
    /// The texture should include `TEXTURE_BINDING` for inputs and `STORAGE_BINDING` for
    /// compute outputs when possible.
    pub fn from_texture(texture: &'a Texture, format: VTFormat, role: VTTextureRole) -> Self {
        Self {
            format,
            width: texture.width(),
            height: texture.height(),
            backing: VTImageBacking::Gpu { texture, role },
        }
    }

    /// Same as [`Self::from_texture`] with [`VTTextureRole::Renderable`] for swapchain / RT output.
    ///
    /// Use when the texture is drawn in a render pass and may lack `STORAGE_BINDING`.
    pub fn from_render_target(texture: &'a Texture, format: VTFormat) -> Self {
        Self::from_texture(texture, format, VTTextureRole::Renderable)
    }

    /// NV12 in split-plane layout: separate Y and UV textures.
    ///
    /// `width` / `height` are the luma (Y) plane dimensions; UV plane size is half in each axis.
    pub fn from_nv12_planes(
        y: &'a Texture,
        uv: &'a Texture,
        width: u32,
        height: u32,
        role: VTTextureRole,
    ) -> Self {
        Self {
            format: VTFormat::NV12,
            width,
            height,
            backing: VTImageBacking::Nv12Split { y, uv, role },
        }
    }

    /// YUV420P (I420) as three separate R8 textures (Y, U, V).
    pub fn from_yuv420p_planes(
        y: &'a Texture,
        u: &'a Texture,
        v: &'a Texture,
        width: u32,
        height: u32,
        role: VTTextureRole,
    ) -> Self {
        Self {
            format: VTFormat::YUV420P,
            width,
            height,
            backing: VTImageBacking::Yuv420pSplit { y, u, v, role },
        }
    }

    /// Returns how this image participates in the next process pass.
    pub fn role(&self) -> VTTextureRole {
        match &self.backing {
            VTImageBacking::Cpu(_) => VTTextureRole::General,
            VTImageBacking::Gpu { role, .. }
            | VTImageBacking::Nv12Split { role, .. }
            | VTImageBacking::Yuv420pSplit { role, .. }
            | VTImageBacking::Nv12Combined { role, .. } => *role,
            #[cfg(windows)]
            VTImageBacking::D3d11 { .. } => VTTextureRole::General,
            #[cfg(target_os = "macos")]
            VTImageBacking::CvPixelBuffer { .. } => VTTextureRole::General,
        }
    }

    /// References a D3D11 texture; copied into an internal bridge during [`crate::VTSampler::process`].
    ///
    /// # Platform
    ///
    /// Windows only. Requires a wgpu **DX12** device (see [`crate::VtD3d11Bridge`]).
    ///
    /// # Arguments
    ///
    /// * `array_index` — Subresource index for texture arrays / DXGI flip models.
    #[cfg(windows)]
    pub fn from_d3d11(
        device: &'a crate::bridge::d3d11::VtD3d11Device<'a>,
        texture: &'a windows::Win32::Graphics::Direct3D11::ID3D11Texture2D,
        format: VTFormat,
        array_index: u32,
    ) -> Self {
        let mut desc = windows::Win32::Graphics::Direct3D11::D3D11_TEXTURE2D_DESC::default();
        unsafe {
            texture.GetDesc(&mut desc);
        }
        Self {
            format,
            width: desc.Width,
            height: desc.Height,
            backing: VTImageBacking::D3d11 {
                device,
                texture,
                array_index,
            },
        }
    }

    /// Wraps the wgpu side of a [`crate::VtD3d11Bridge`] (call [`crate::VtD3d11Bridge::copy_from`] first if needed).
    ///
    /// For [`VTFormat::NV12`], uses combined NV12 plane aspects on `bridge.wgpu`.
    #[cfg(windows)]
    pub fn from_d3d11_bridge(bridge: &'a crate::bridge::d3d11::VtD3d11Bridge, format: VTFormat) -> Self {
        let width = bridge.wgpu.width();
        let height = bridge.wgpu.height();
        let backing = match format {
            VTFormat::NV12 => VTImageBacking::Nv12Combined {
                texture: &bridge.wgpu,
                role: VTTextureRole::General,
            },
            _ => VTImageBacking::Gpu {
                texture: &bridge.wgpu,
                role: VTTextureRole::General,
            },
        };
        Self {
            format,
            width,
            height,
            backing,
        }
    }

    /// References a Core Video pixel buffer; uploaded via Metal during processing.
    ///
    /// # Platform
    ///
    /// macOS only. Requires a wgpu **Metal** device. `format` / `width` / `height` must match the buffer.
    #[cfg(target_os = "macos")]
    pub fn from_cv_pixel_buffer(
        buffer: crate::bridge::metal::CVPixelBufferRef,
        format: VTFormat,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            format,
            width,
            height,
            backing: VTImageBacking::CvPixelBuffer { buffer },
        }
    }

    /// Combined NV12 on a single wgpu/DXGI texture (plane 0 = Y, plane 1 = UV).
    ///
    /// # Platform
    ///
    /// Typical on Windows when interoping with D3D11 NV12 shared textures.
    #[cfg(windows)]
    pub fn from_nv12_texture(texture: &'a Texture, role: VTTextureRole) -> Self {
        Self {
            format: VTFormat::NV12,
            width: texture.width(),
            height: texture.height(),
            backing: VTImageBacking::Nv12Combined { texture, role },
        }
    }
}

/// GPU-backed image allocated by [`crate::VTSampler::allocate`].
///
/// Holds scratch textures suitable as an intermediate conversion result; call
/// [`Self::as_image`] to obtain a [`VTImage`] reference for processing.
pub struct VTImageOwned {
    pub(crate) scratch: crate::pool::ScratchTextures,
}

impl VTImageOwned {
    pub(crate) fn new(scratch: crate::pool::ScratchTextures) -> Self {
        Self { scratch }
    }

    /// Borrows the allocated planes as a [`VTImage`] with [`VTTextureRole::General`].
    pub fn as_image(&self) -> VTImage<'_> {
        match self.scratch.format {
            VTFormat::RGBA | VTFormat::BGRA => VTImage::from_texture(
                &self.scratch.planes[0],
                self.scratch.format,
                VTTextureRole::General,
            ),
            VTFormat::NV12 => VTImage::from_nv12_planes(
                &self.scratch.planes[0],
                &self.scratch.planes[1],
                self.scratch.width,
                self.scratch.height,
                VTTextureRole::General,
            ),
            VTFormat::YUV420P => VTImage::from_yuv420p_planes(
                &self.scratch.planes[0],
                &self.scratch.planes[1],
                &self.scratch.planes[2],
                self.scratch.width,
                self.scratch.height,
                VTTextureRole::General,
            ),
        }
    }
}
