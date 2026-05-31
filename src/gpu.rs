use wgpu::{
    Extent3d, ImageCopyTexture, ImageDataLayout, Origin3d, Queue, Texture, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor, TextureViewDimension,
};

use crate::{VTFormat, format::VTSampleError, pixel::PixelData};

pub fn create_plane_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: TextureFormat,
    extra_usage: TextureUsages,
) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some("vtsampler_plane"),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        usage: TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::COPY_SRC
            | TextureUsages::COPY_DST
            | extra_usage,
        view_formats: &[],
        format,
        size: Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
    })
}

pub fn plane_size(format: VTFormat, width: u32, height: u32, plane: usize) -> (u32, u32) {
    match format {
        VTFormat::RGBA | VTFormat::BGRA => (width, height),
        VTFormat::NV12 => {
            if plane == 0 {
                (width, height)
            } else {
                (width / 2, height / 2)
            }
        }
        VTFormat::YUV420P => {
            if plane == 0 {
                (width, height)
            } else {
                (width / 2, height / 2)
            }
        }
    }
}

pub fn has_storage(texture: &Texture) -> bool {
    texture.usage().contains(TextureUsages::STORAGE_BINDING)
}

pub fn has_copy_dst(texture: &Texture) -> bool {
    texture.usage().contains(TextureUsages::COPY_DST)
}

pub fn has_copy_src(texture: &Texture) -> bool {
    texture.usage().contains(TextureUsages::COPY_SRC)
}

pub fn has_sample(texture: &Texture) -> bool {
    texture.usage().contains(TextureUsages::TEXTURE_BINDING)
}

/// Upload CPU pixels into existing GPU plane textures (single upload path).
pub fn upload_cpu(
    queue: &Queue,
    format: VTFormat,
    width: u32,
    height: u32,
    data: &PixelData<'_>,
    planes: &[Texture],
) -> Result<(), VTSampleError> {
    match (format, data) {
        (VTFormat::RGBA, PixelData::RGBA { buffer, stride }) => {
            write_plane(queue, &planes[0], buffer, *stride as u32, width, height);
        }
        (VTFormat::BGRA, PixelData::BGRA { buffer, stride }) => {
            write_plane(queue, &planes[0], buffer, *stride as u32, width, height);
        }
        (VTFormat::NV12, PixelData::NV12 { buffer, stride }) => {
            write_plane(
                queue,
                &planes[0],
                buffer[0],
                stride[0] as u32,
                width,
                height,
            );
            let (pw, ph) = plane_size(format, width, height, 1);
            write_plane(queue, &planes[1], buffer[1], stride[1] as u32, pw, ph);
        }
        (VTFormat::YUV420P, PixelData::YUV420P { buffer, stride }) => {
            write_plane(
                queue,
                &planes[0],
                buffer[0],
                stride[0] as u32,
                width,
                height,
            );
            let (pw, ph) = plane_size(format, width, height, 1);
            write_plane(queue, &planes[1], buffer[1], stride[1] as u32, pw, ph);
            write_plane(queue, &planes[2], buffer[2], stride[2] as u32, pw, ph);
        }
        _ => return Err(VTSampleError::UnsupportedFormat),
    }
    Ok(())
}

fn write_plane(
    queue: &Queue,
    texture: &Texture,
    buffer: &[u8],
    stride: u32,
    width: u32,
    height: u32,
) {
    queue.write_texture(
        ImageCopyTexture {
            aspect: TextureAspect::All,
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
        },
        buffer,
        ImageDataLayout {
            bytes_per_row: Some(stride),
            rows_per_image: Some(height),
            offset: 0,
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

pub fn copy_plane(
    encoder: &mut wgpu::CommandEncoder,
    src: &Texture,
    dst: &Texture,
    width: u32,
    height: u32,
) {
    copy_plane_aspect(
        encoder,
        src,
        TextureAspect::All,
        dst,
        TextureAspect::All,
        width,
        height,
    );
}

/// GPU→GPU copy for a single plane (used for split NV12 and combined NV12 targets).
pub fn copy_plane_aspect(
    encoder: &mut wgpu::CommandEncoder,
    src: &Texture,
    src_aspect: TextureAspect,
    dst: &Texture,
    dst_aspect: TextureAspect,
    width: u32,
    height: u32,
) {
    encoder.copy_texture_to_texture(
        ImageCopyTexture {
            texture: src,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: src_aspect,
        },
        ImageCopyTexture {
            texture: dst,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: dst_aspect,
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

pub fn create_sample_view(texture: &Texture, plane_format: TextureFormat) -> TextureView {
    let format = texture.format();
    let aspect = if format == TextureFormat::NV12 {
        if plane_format == TextureFormat::R8Unorm {
            TextureAspect::Plane0
        } else {
            TextureAspect::Plane1
        }
    } else {
        TextureAspect::All
    };

    texture.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2),
        format: Some(plane_format),
        aspect,
        ..Default::default()
    })
}

pub fn create_storage_view(texture: &Texture, plane_format: TextureFormat) -> TextureView {
    create_sample_view(texture, plane_format)
}
