//! Frame processing: resolve backing → compute → optional copy (minimal copies).

use std::sync::Arc;

use smallvec::SmallVec;

use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindingResource, CommandEncoder, ComputePassDescriptor,
    Texture, TextureView,
};

use crate::{
    VTImage, VTProcessOptions,
    format::VTSampleError,
    gpu::{
        copy_plane, copy_plane_aspect, create_sample_view, create_storage_view, has_copy_dst,
        has_copy_src, has_sample, has_storage, plane_size, upload_cpu,
    },
    image::{VTImageBacking, VTTextureRole},
    pool::ScratchPool,
    shader::{PipelineKey, ShaderRegistry},
};

pub struct Processor<'a> {
    pub device: &'a wgpu::Device,
    #[cfg(target_os = "macos")]
    pub device_arc: &'a Arc<wgpu::Device>,
    pub queue: &'a wgpu::Queue,
    pub shaders: &'a mut ShaderRegistry,
    pub pool: &'a ScratchPool,
    #[cfg(windows)]
    pub d3d11_pool: &'a crate::bridge::d3d11::VtD3d11Pool,
    #[cfg(target_os = "macos")]
    pub metal_cache: &'a std::sync::Mutex<Option<crate::bridge::metal::VtMetalCache>>,
}

impl Processor<'_> {
    pub fn encode(
        &mut self,
        input: &VTImage<'_>,
        output: &VTImage<'_>,
        encoder: &mut CommandEncoder,
        opts: VTProcessOptions,
    ) -> Result<(), VTSampleError> {
        let need_scale = input.width != output.width || input.height != output.height;
        let pipeline = self.shaders.get(
            self.device,
            PipelineKey {
                input: input.format,
                output: output.format,
                need_scale,
                color_space: opts.color_space,
                scale_filter: opts.scale_filter,
            },
        )?;

        let input_handle = self.pool.acquire(input.format, input.width, input.height);
        let output_handle = self
            .pool
            .acquire(output.format, output.width, output.height);
        let input_scratch = input_handle.lock().expect("scratch lock");
        let output_scratch = output_handle.lock().expect("scratch lock");

        let (input_views, pre_copies): (InputViews<'_>, Vec<GpuCopy<'_>>) = match &input.backing {
            #[cfg(windows)]
            VTImageBacking::D3d11 {
                device,
                texture,
                array_index,
            } => {
                let bridge = self.d3d11_pool.acquire(
                    device,
                    self.device,
                    input.width,
                    input.height,
                    input.format,
                    wgpu::TextureUsages::TEXTURE_BINDING,
                )?;
                bridge.copy_from(device, texture, *array_index)?;
                (InputViews::D3d11Bridge(bridge), Vec::new())
            }
            #[cfg(windows)]
            #[cfg(target_os = "macos")]
            VTImageBacking::CvPixelBuffer { buffer } => {
                let mut guard = self.metal_cache.lock().expect("metal cache lock");
                if guard.is_none() {
                    *guard = Some(crate::bridge::metal::VtMetalCache::new(self.device_arc.clone())?);
                }
                guard
                    .as_mut()
                    .unwrap()
                    .upload_to_planes(
                        encoder,
                        *buffer,
                        input.format,
                        input.width,
                        input.height,
                        &input_scratch.planes,
                    )?;
                let planes: SmallVec<[&Texture; 3]> = input_scratch.planes.iter().collect();
                (InputViews::Planes(planes), Vec::new())
            }
            VTImageBacking::Cpu(data) => {
                upload_cpu(
                    self.queue,
                    input.format,
                    input.width,
                    input.height,
                    data,
                    &input_scratch.planes,
                )?;
                let planes: SmallVec<[&Texture; 3]> = input_scratch.planes.iter().collect();
                (InputViews::Planes(planes), Vec::new())
            }
            #[cfg(windows)]
            VTImageBacking::Nv12Combined { texture, .. } if has_sample(texture) => {
                (InputViews::Nv12Combined(texture), Vec::new())
            }
            backing => {
                let planes = collect_planes(backing)?;
                // Presentation / render-target textures are poor compute sources; copy first.
                let use_scratch = input.role() == VTTextureRole::Renderable
                    || planes.len() != input.format.plane_count()
                    || !planes.iter().all(|t| has_sample(t));
                if !use_scratch {
                    (InputViews::Planes(planes), Vec::new())
                } else {
                    let mut copies = Vec::new();
                    for (i, src) in planes.iter().enumerate() {
                        let (w, h) = plane_size(input.format, input.width, input.height, i);
                        copies.push(GpuCopy {
                            src,
                            dst: &input_scratch.planes[i],
                            width: w,
                            height: h,
                        });
                    }
                    let scratch_planes: SmallVec<[&Texture; 3]> =
                        input_scratch.planes.iter().collect();
                    (InputViews::Planes(scratch_planes), copies)
                }
            }
        };

        for c in &pre_copies {
            if has_copy_src(c.src) && has_copy_dst(c.dst) {
                copy_plane(encoder, c.src, c.dst, c.width, c.height);
            } else {
                return Err(VTSampleError::MissingTextureUsage);
            }
        }

        let (storage_planes, finish): (SmallVec<[&Texture; 3]>, Option<OutputFinish<'_>>) =
            match collect_planes(&output.backing) {
                Ok(planes) if can_write_output_direct(output, &planes) => {
                    (planes, None)
                }
                Ok(planes) => {
                    let storage: SmallVec<[&Texture; 3]> = output_scratch.planes.iter().collect();
                    let mut dst: SmallVec<[(&Texture, u32, u32); 3]> = SmallVec::new();
                    match &output.backing {
                        #[cfg(windows)]
                        VTImageBacking::Nv12Combined { texture, .. } => {
                            dst.push((texture, output.width, output.height));
                        }
                        _ => {
                            for (i, tex) in planes.iter().enumerate() {
                                let (w, h) =
                                    plane_size(output.format, output.width, output.height, i);
                                dst.push((tex, w, h));
                            }
                        }
                    }
                    (storage, Some(OutputFinish::CopyPlanes { dst }))
                }
                Err(_) => {
                    let storage: SmallVec<[&Texture; 3]> = output_scratch.planes.iter().collect();
                    (storage, None)
                }
            };

        let mut views: SmallVec<[TextureView; 6]> = SmallVec::new();
        match input_views {
            InputViews::Planes(planes) => {
                for (i, tex) in planes.iter().enumerate() {
                    views.push(create_sample_view(tex, input.format.plane_formats()[i]));
                }
            }
            #[cfg(windows)]
            InputViews::Nv12Combined(tex) => {
                views.push(create_sample_view(tex, wgpu::TextureFormat::R8Unorm));
                views.push(create_sample_view(tex, wgpu::TextureFormat::Rg8Unorm));
            }
            #[cfg(windows)]
            InputViews::D3d11Bridge(bridge) => {
                push_d3d11_bridge_views(&mut views, input, &bridge.wgpu);
            }
        }
        for (i, tex) in storage_planes.iter().enumerate() {
            views.push(create_storage_view(tex, output.format.plane_formats()[i]));
        }

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("vtsampler_bind_group"),
            layout: &pipeline.layout,
            entries: &views
                .iter()
                .enumerate()
                .map(|(i, view)| BindGroupEntry {
                    binding: i as u32,
                    resource: BindingResource::TextureView(view),
                })
                .collect::<Vec<_>>(),
        });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("vtsampler_compute"),
                ..Default::default()
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(output.width.div_ceil(16), output.height.div_ceil(16), 1);
        }

        if let Some(OutputFinish::CopyPlanes { dst }) = finish {
            if output.role() == VTTextureRole::Renderable {
                for (tex, _, _) in &dst {
                    if !has_copy_dst(tex) {
                        return Err(VTSampleError::MissingTextureUsage);
                    }
                }
            }
            match &output.backing {
                #[cfg(windows)]
                VTImageBacking::Nv12Combined { texture, .. } => {
                    if let (Some(y_src), Some(uv_src)) =
                        (storage_planes.first(), storage_planes.get(1))
                    {
                        use wgpu::TextureAspect;
                        let (yw, yh) = plane_size(output.format, output.width, output.height, 0);
                        let (uvw, uvh) = plane_size(output.format, output.width, output.height, 1);
                        copy_plane_aspect(
                            encoder,
                            y_src,
                            TextureAspect::All,
                            texture,
                            TextureAspect::Plane0,
                            yw,
                            yh,
                        );
                        copy_plane_aspect(
                            encoder,
                            uv_src,
                            TextureAspect::All,
                            texture,
                            TextureAspect::Plane1,
                            uvw,
                            uvh,
                        );
                    }
                }
                _ => {
                    for (i, (dst_tex, w, h)) in dst.iter().enumerate() {
                        if let Some(src) = storage_planes.get(i)
                            && has_copy_src(src)
                            && has_copy_dst(dst_tex)
                        {
                            copy_plane(encoder, src, dst_tex, *w, *h);
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

enum InputViews<'a> {
    Planes(SmallVec<[&'a Texture; 3]>),
    #[cfg(windows)]
    Nv12Combined(&'a Texture),
    #[cfg(windows)]
    D3d11Bridge(Arc<crate::bridge::d3d11::VtD3d11Bridge>),
}

struct GpuCopy<'a> {
    src: &'a Texture,
    dst: &'a Texture,
    width: u32,
    height: u32,
}

enum OutputFinish<'a> {
    CopyPlanes {
        dst: SmallVec<[(&'a Texture, u32, u32); 3]>,
    },
}

/// Direct compute output when every plane has STORAGE (typical intermediate textures).
#[cfg(windows)]
fn push_d3d11_bridge_views(views: &mut SmallVec<[TextureView; 6]>, input: &VTImage<'_>, tex: &Texture) {
    if input.format == crate::VTFormat::NV12 && has_sample(tex) {
        views.push(create_sample_view(tex, wgpu::TextureFormat::R8Unorm));
        views.push(create_sample_view(tex, wgpu::TextureFormat::Rg8Unorm));
    } else {
        views.push(create_sample_view(tex, input.format.plane_formats()[0]));
    }
}

fn can_write_output_direct(image: &VTImage<'_>, planes: &[&Texture]) -> bool {
    planes.len() == image.format.plane_count() && planes.iter().all(|t| has_storage(t))
}

fn collect_planes<'a>(
    backing: &'a VTImageBacking<'a>,
) -> Result<SmallVec<[&'a Texture; 3]>, VTSampleError> {
    let mut planes: SmallVec<[&Texture; 3]> = SmallVec::new();
    match backing {
        VTImageBacking::Gpu { texture, .. } => planes.push(texture),
        VTImageBacking::Nv12Split { y, uv, .. } => {
            planes.push(y);
            planes.push(uv);
        }
        VTImageBacking::Yuv420pSplit { y, u, v, .. } => {
            planes.push(y);
            planes.push(u);
            planes.push(v);
        }
        #[cfg(windows)]
        VTImageBacking::Nv12Combined { .. } => return Err(VTSampleError::UnsupportedBacking),
        #[cfg(windows)]
        #[cfg(windows)]
        VTImageBacking::D3d11 { .. } => return Err(VTSampleError::UnsupportedBacking),
        #[cfg(target_os = "macos")]
        VTImageBacking::CvPixelBuffer { .. } => return Err(VTSampleError::UnsupportedBacking),
        VTImageBacking::Cpu(_) => return Err(VTSampleError::UnsupportedBacking),
    }
    Ok(planes)
}
