use wgpu::Texture;

use crate::{VTFormat, pixel::PixelData};

/// Describes how a GPU texture participates in a process pass.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub enum VTTextureRole {
    /// Default: may be used as compute input/output when usages allow.
    #[default]
    General,
    /// Output will be consumed by a render pass (swapchain, RT, etc.).
    Renderable,
}

/// User-facing "one image" (format + size + backing).
pub struct VTImage<'a> {
    pub format: VTFormat,
    pub width: u32,
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
    /// D3D11 texture copied into an internal bridge at process time.
    #[cfg(windows)]
    D3d11 {
        device: &'a crate::bridge::d3d11::VtD3d11Device<'a>,
        texture: &'a windows::Win32::Graphics::Direct3D11::ID3D11Texture2D,
        array_index: u32,
    },
    /// Core Video pixel buffer uploaded via Metal at process time.
    #[cfg(target_os = "macos")]
    CvPixelBuffer {
        buffer: crate::bridge::metal::CVPixelBufferRef,
    },
}

impl<'a> VTImage<'a> {
    pub fn from_cpu(data: &'a PixelData<'a>, width: u32, height: u32) -> Self {
        Self {
            format: data.format(),
            width,
            height,
            backing: VTImageBacking::Cpu(data),
        }
    }

    pub fn from_texture(texture: &'a Texture, format: VTFormat, role: VTTextureRole) -> Self {
        Self {
            format,
            width: texture.width(),
            height: texture.height(),
            backing: VTImageBacking::Gpu { texture, role },
        }
    }

    /// Output (or input) texture intended for rendering after processing.
    pub fn from_render_target(texture: &'a Texture, format: VTFormat) -> Self {
        Self::from_texture(texture, format, VTTextureRole::Renderable)
    }

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

    /// Role of this image in a process pass (see [`VTTextureRole`]).
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

    /// Reference a D3D11 texture; copied through an internal bridge during [`crate::VTSampler::process`].
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

    /// Use a [`VtD3d11Bridge`] after [`VtD3d11Bridge::copy_from`] (or as a process output target).
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

    /// Reference a `CVPixelBuffer`; uploaded via Metal during [`crate::VTSampler::process`].
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

/// GPU image allocated by [`crate::VTSampler::allocate`].
pub struct VTImageOwned {
    pub(crate) scratch: crate::pool::ScratchTextures,
}

impl VTImageOwned {
    pub(crate) fn new(scratch: crate::pool::ScratchTextures) -> Self {
        Self { scratch }
    }

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
