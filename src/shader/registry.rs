use std::collections::HashMap;

use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    ComputePipeline, ComputePipelineDescriptor, Device, PipelineCompilationOptions,
    ShaderStages, StorageTextureAccess, TextureSampleType, TextureViewDimension,
};

use crate::{
    VTFormat,
    format::{VTColorSpace, VTSampleError, VTScaleFilter},
    shader::{compile_wgsl, create_shader_module},
};

/// Identifies a cached compute pipeline (input/output format, scale, color matrix, filter).
///
/// Exposed for debugging or custom tooling; [`crate::VTSampler`] builds keys automatically
/// from [`crate::VTImage`] dimensions and [`crate::VTProcessOptions`].
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct PipelineKey {
    /// Source pixel format.
    pub input: VTFormat,
    /// Destination pixel format.
    pub output: VTFormat,
    /// `true` when input and output widths/heights differ.
    pub need_scale: bool,
    /// YUV ↔ RGB matrix variant.
    pub color_space: VTColorSpace,
    /// Resampling filter when `need_scale` is true.
    pub scale_filter: VTScaleFilter,
}

pub(crate) struct PipelineSet {
    pub layout: BindGroupLayout,
    pub pipeline: ComputePipeline,
}

pub struct ShaderRegistry {
    pipelines: HashMap<PipelineKey, PipelineSet>,
}

impl ShaderRegistry {
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
        }
    }

    pub fn get(
        &mut self,
        device: &Device,
        key: PipelineKey,
    ) -> Result<&PipelineSet, VTSampleError> {
        if let std::collections::hash_map::Entry::Vacant(e) = self.pipelines.entry(key) {
            e.insert(Self::build_pipeline(device, key)?);
        }
        Ok(self.pipelines.get(&key).unwrap())
    }

    fn build_pipeline(device: &Device, key: PipelineKey) -> Result<PipelineSet, VTSampleError> {
        let source = compile_wgsl(
            key.input,
            key.output,
            key.need_scale,
            key.color_space,
            key.scale_filter,
        )?;
        let module = create_shader_module(device, &source);
        let layout = Self::create_bind_group_layout(device, key.input, key.output);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("vtsampler_process"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("vtsampler_process_layout"),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        Ok(PipelineSet { layout, pipeline })
    }

    fn create_bind_group_layout(
        device: &Device,
        input: VTFormat,
        output: VTFormat,
    ) -> BindGroupLayout {
        let mut entries = Vec::new();
        for i in 0..input.plane_count() {
            entries.push(BindGroupLayoutEntry {
                binding: i as u32,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }
        for (i, format) in output.plane_formats().iter().enumerate() {
            entries.push(BindGroupLayoutEntry {
                binding: (input.plane_count() + i) as u32,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: *format,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            });
        }
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("vtsampler_bind_group_layout"),
            entries: &entries,
        })
    }
}
