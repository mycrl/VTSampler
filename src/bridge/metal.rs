//! CVPixelBuffer → Metal → wgpu import and plane upload.

use std::ptr::{NonNull, null_mut};
use std::sync::{Arc, Mutex};

use metal::{MTLTextureType, Texture as MtlTexture};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::kCFAllocatorDefault;
use objc2_core_video::{
    CVMetalTexture, CVMetalTextureCache, CVMetalTextureCacheCreate,
    CVMetalTextureCacheCreateTextureFromImage, CVMetalTextureCacheFlush, CVMetalTextureGetTexture,
    CVPixelBuffer, CVPixelBufferGetHeight, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth,
    kCVPixelFormatType_32BGRA, kCVPixelFormatType_32RGBA,
    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
    kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, kCVReturnSuccess,
};
use objc2_metal::{MTLDevice as Objc2MTLDevice, MTLPixelFormat as Objc2MTLPixelFormat};

use wgpu::{
    CommandEncoder, Device, Extent3d, ImageCopyTexture, Origin3d, Texture, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, hal::{Api, CopyExtent, api::Metal},
};

use crate::{VTFormat, bridge::BridgeError, gpu::plane_size};

pub type CVPixelBufferRef = *mut CVPixelBuffer;

pub struct VtMetalCache {
    cache: Retained<CVMetalTextureCache>,
    wgpu_device: Arc<Device>,
}

impl VtMetalCache {
    pub fn new(wgpu_device: Arc<Device>) -> Result<Self, BridgeError> {
        let mut raw_mtl = None;
        wgpu_device.as_hal::<Metal, _, _>(|device| {
            if let Some(device) = device {
                raw_mtl = Some(device.raw_device().lock().clone());
            }
        });
        let raw_mtl = raw_mtl.ok_or(BridgeError::NotFoundMetalBackend)?;

        let device: Retained<ProtocolObject<dyn Objc2MTLDevice>> =
            unsafe { Retained::from_raw(raw_mtl.into_ptr().cast()).unwrap() };

        let mut cache = null_mut();
        let code = unsafe {
            CVMetalTextureCacheCreate(
                kCFAllocatorDefault,
                None,
                device.as_ref(),
                None,
                NonNull::new(&mut cache).unwrap(),
            )
        };
        if code != kCVReturnSuccess || cache.is_null() {
            return Err(BridgeError::CoreVideo(code));
        }

        Ok(Self {
            cache: unsafe { Retained::from_raw(cache).unwrap() },
            wgpu_device,
        })
    }

    pub fn flush(&self) {
        unsafe {
            CVMetalTextureCacheFlush(&self.cache, 0);
        }
    }

    /// Upload pixel buffer contents into wgpu plane textures (via Metal interop + copy).
    pub fn upload_to_planes(
        &self,
        encoder: &mut CommandEncoder,
        buffer: CVPixelBufferRef,
        format: VTFormat,
        width: u32,
        height: u32,
        dst_planes: &[Texture],
    ) -> Result<(), BridgeError> {
        let pixel_format = unsafe { CVPixelBufferGetPixelFormatType(&*buffer) };
        match format {
            VTFormat::BGRA | VTFormat::RGBA => {
                let mtl_format = match format {
                    VTFormat::BGRA => Objc2MTLPixelFormat::BGRA8Unorm,
                    VTFormat::RGBA => Objc2MTLPixelFormat::RGBA8Unorm,
                    _ => unreachable!(),
                };
                let mtl_tex = self.create_metal_texture(buffer, mtl_format, width, height, 0)?;
                self.copy_metal_to_wgpu(encoder, mtl_tex, format, &dst_planes[0], width, height)?;
            }
            VTFormat::NV12 => {
                if pixel_format != kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
                    && pixel_format != kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
                {
                    return Err(BridgeError::UnsupportedFormat);
                }
                let y_tex = self.create_metal_texture(
                    buffer,
                    Objc2MTLPixelFormat::R8Unorm,
                    width,
                    height,
                    0,
                )?;
                let (uvw, uvh) = plane_size(format, width, height, 1);
                let uv_tex = self.create_metal_texture(
                    buffer,
                    Objc2MTLPixelFormat::RG8Unorm,
                    uvw,
                    uvh,
                    1,
                )?;
                self.copy_metal_to_wgpu(encoder, y_tex, VTFormat::NV12, &dst_planes[0], width, height)?;
                self.copy_metal_to_wgpu(encoder, uv_tex, VTFormat::NV12, &dst_planes[1], uvw, uvh)?;
            }
            VTFormat::YUV420P => return Err(BridgeError::UnsupportedFormat),
        }
        self.flush();
        Ok(())
    }

    fn create_metal_texture(
        &self,
        buffer: CVPixelBufferRef,
        mtl_format: Objc2MTLPixelFormat,
        width: u32,
        height: u32,
        plane_index: usize,
    ) -> Result<MtlTexture, BridgeError> {
        let mut cv_texture = null_mut();
        let code = unsafe {
            CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault,
                &self.cache,
                &*buffer,
                None,
                mtl_format,
                width as usize,
                height as usize,
                plane_index,
                NonNull::new(&mut cv_texture).unwrap(),
            )
        };
        if code != kCVReturnSuccess || cv_texture.is_null() {
            return Err(BridgeError::CoreVideo(code));
        }
        let cv_texture = unsafe { Retained::<CVMetalTexture>::from_raw(cv_texture).unwrap() };
        if let Some(texture) = unsafe { CVMetalTextureGetTexture(&cv_texture) } {
            Ok(unsafe {
                MtlTexture::from_ptr(Retained::into_raw(texture).cast()).to_owned()
            })
        } else {
            Err(BridgeError::CoreVideo(-1))
        }
    }

    fn copy_metal_to_wgpu(
        &self,
        encoder: &mut CommandEncoder,
        mtl_src: MtlTexture,
        format: VTFormat,
        dst: &Texture,
        width: u32,
        height: u32,
    ) -> Result<(), BridgeError> {
        let wgpu_format = format.plane_formats()[0];
        let src = unsafe {
            self.wgpu_device.create_texture_from_hal::<Metal>(
                <Metal as Api>::Device::texture_from_raw(
                    mtl_src,
                    wgpu_format,
                    MTLTextureType::D2,
                    1,
                    1,
                    CopyExtent {
                        width,
                        height,
                        depth: 1,
                    },
                ),
                &TextureDescriptor {
                    label: Some("vtsampler_metal_import"),
                    size: Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: wgpu_format,
                    usage: TextureUsages::COPY_SRC,
                    view_formats: &[],
                },
            )
        };
        encoder.copy_texture_to_texture(
            ImageCopyTexture {
                texture: &src,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            ImageCopyTexture {
                texture: dst,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }
}

pub fn cv_pixel_format_to_vt(pixel_format: u32) -> Result<VTFormat, BridgeError> {
    match pixel_format {
        kCVPixelFormatType_32RGBA => Ok(VTFormat::RGBA),
        kCVPixelFormatType_32BGRA => Ok(VTFormat::BGRA),
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        | kCVPixelFormatType_420YpCbCr8BiPlanarFullRange => Ok(VTFormat::NV12),
        _ => Err(BridgeError::UnsupportedFormat),
    }
}

pub fn cv_pixel_buffer_size(buffer: CVPixelBufferRef) -> (u32, u32) {
    (
        unsafe { CVPixelBufferGetWidth(&*buffer) } as u32,
        unsafe { CVPixelBufferGetHeight(&*buffer) } as u32,
    )
}

