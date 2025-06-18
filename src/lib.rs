use std::{borrow::Cow, collections::HashMap, sync::OnceLock};

use smallvec::SmallVec;
use wgpu::{
    Adapter, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, DeviceDescriptor, Extent3d, Instance, InstanceDescriptor,
    MemoryHints, Origin3d, PipelineCompilationOptions, PowerPreference, Queue,
    RequestAdapterOptions, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    StorageTextureAccess, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureView, TextureViewDescriptor, TextureViewDimension,
};

pub use wgpu;

#[derive(Debug)]
struct GlobalInstance {
    #[allow(dead_code)]
    instance: Instance,
    #[allow(dead_code)]
    adapter: Adapter,
    device: Device,
    queue: Queue,
}

// A global instance used to manage GPU resources. Multiple instances can share
// a single GPU device.
static GLOBAL_INSTANCE: OnceLock<GlobalInstance> = OnceLock::new();

// Get the global instance. If it already exists, return it; otherwise, create
// a new instance.
async fn get_global_instance() -> Result<&'static GlobalInstance, VTSampleError> {
    if let Some(instance) = GLOBAL_INSTANCE.get() {
        return Ok(instance);
    }

    let instance = Instance::new(&InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::LowPower,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        })
        .await
        .map_err(|_| VTSampleError::NotFoundAdapter)?;

    let (device, queue) = adapter
        .request_device(&DeviceDescriptor {
            memory_hints: MemoryHints::MemoryUsage,
            required_features: adapter.features(),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .map_err(|_| VTSampleError::RequestDeviceFailed)?;

    // Ignore the return value because the global instance is guaranteed to be unique.
    GLOBAL_INSTANCE
        .set(GlobalInstance {
            instance,
            adapter,
            device,
            queue,
        })
        .unwrap();

    Ok(GLOBAL_INSTANCE.get().unwrap())
}

trait DeviceExt {
    fn create_tex(
        &self,
        width: u32,
        height: u32,
        format: TextureFormat,
        usage: TextureUsages,
    ) -> Texture;
}

impl DeviceExt for Device {
    fn create_tex(
        &self,
        width: u32,
        height: u32,
        format: TextureFormat,
        usage: TextureUsages,
    ) -> Texture {
        self.create_texture(&TextureDescriptor {
            label: None,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | usage,
            view_formats: &[],
            format,
            size: Extent3d {
                depth_or_array_layers: 1,
                width,
                height,
            },
        })
    }
}

trait QueueExt {
    fn copy_buffer(&self, texture: &Texture, buffer: &[u8], width: u32, height: u32);
}

impl QueueExt for Queue {
    fn copy_buffer(&self, texture: &Texture, buffer: &[u8], width: u32, height: u32) {
        self.write_texture(
            TexelCopyTextureInfo {
                aspect: TextureAspect::All,
                texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
            },
            buffer,
            TexelCopyBufferLayout {
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
                offset: 0,
            },
            texture.size(),
        )
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum VTResourceType {
    Texture,
    PixelBuffer,
}

/// Video texture resource trait
pub trait VTResource {
    const TYPE: VTResourceType;

    /// Get the width of the resource
    fn get_width(&self) -> u32;

    /// Get the height of the resource
    fn get_height(&self) -> u32;

    /// Get the format of the resource
    fn get_format(&self) -> VTFormat;

    /// Create texture views
    fn create_views(&self, views: &mut SmallVec<[TextureView; 10]>);
}

/// Video texture format enumeration
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum VTFormat {
    /// RGBA format, 4 bytes per pixel
    RGBA,
    /// BGRA format, 4 bytes per pixel
    BGRA,
    /// YUV420P format, contains Y, U, V planes
    YUV420P,
    /// NV12 format, contains Y plane and interleaved UV plane
    NV12,
}

impl VTFormat {
    pub fn all() -> &'static [Self] {
        &[
            VTFormat::RGBA,
            VTFormat::BGRA,
            VTFormat::NV12,
            VTFormat::YUV420P,
        ]
    }

    #[rustfmt::skip]
    pub fn view_formats(&self) -> &'static [TextureFormat] {
        match self {
            VTFormat::RGBA => &[TextureFormat::Rgba8Unorm],
            VTFormat::BGRA => &[TextureFormat::Bgra8Unorm],
            // Since NV12 consists of two planes, two views are required:
            //
            // one view stores the Y channel, and the other view stores the UV
            // mixed channel.
            VTFormat::NV12 => &[TextureFormat::R8Unorm, TextureFormat::Rg8Unorm],
            // YUV420P consists of three planes, so three views are required:
            //
            // one view stores the Y channel, one view stores the U channel, and
            // one view stores the V channel.
            VTFormat::YUV420P => &[TextureFormat::R8Unorm, TextureFormat::R8Unorm, TextureFormat::R8Unorm],
        }
    }
}

/// Errors that may occur when creating a video texture processor
#[derive(Debug, Clone, Copy)]
pub enum VTSampleError {
    /// No suitable GPU adapter found
    NotFoundAdapter,
    /// Failed to request device
    RequestDeviceFailed,
}

impl std::error::Error for VTSampleError {}

impl std::fmt::Display for VTSampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VTSampleError::{:?}", self)
    }
}

/// Video texture processor builder
#[derive(Default)]
pub struct VTSamplerBuilder<'a> {
    device: Option<&'a Device>,
    queue: Option<&'a Queue>,
}

impl<'a> VTSamplerBuilder<'a> {
    /// Set GPU device and command queue
    pub fn with_device(mut self, device: &'a Device, queue: &'a Queue) -> Self {
        self.device = Some(device);
        self.queue = Some(queue);
        self
    }

    /// Build video texture processor
    pub async fn build(self) -> Result<VTSampler<'a>, VTSampleError> {
        // Use provided device and queue if available, otherwise create new instance
        let (device, queue) = if let (Some(device), Some(queue)) = (self.device, self.queue) {
            (device, queue)
        } else {
            let instance = get_global_instance().await?;
            (&instance.device, &instance.queue)
        };

        // Create a shader table for the current device to be used for creating
        // shader pipelines.
        //
        // The table stores all possible shaders to avoid duplicate creation.
        let mut shader_table = HashMap::new();
        {
            for need_scale in [true, false] {
                for input in VTFormat::all() {
                    for output in VTFormat::all() {
                        shader_table.insert(
                            (*input, *output, need_scale),
                            device.create_shader_module(ShaderModuleDescriptor {
                                label: None,
                                source: ShaderSource::Wgsl(Cow::Borrowed(&compile_shader(
                                    *input, *output, need_scale,
                                ))),
                            }),
                        );
                    }
                }
            }
        }

        Ok(VTSampler {
            device,
            queue,
            shader_table,
        })
    }
}

/// GPU video texture sampler
pub struct VTSampler<'a> {
    shader_table: HashMap<(VTFormat, VTFormat, bool), ShaderModule>,
    device: &'a Device,
    queue: &'a Queue,
}

impl<'a> VTSampler<'a> {
    /// Create pixel buffer
    pub fn create_pixel_buffer(&self, format: VTFormat, width: u32, height: u32) -> PixelBuffer {
        PixelBuffer::new(&self.device, &self.queue, format, width, height)
    }

    /// Create a video texture processor task
    pub fn create_task<I, O>(&self, input: &I, output: &O) -> VTSampleTask
    where
        I: VTResource,
        O: VTResource,
    {
        VTSampleTask::new(&self.shader_table, &self.device, &self.queue, input, output)
    }
}

pub struct VTSampleTask<'a> {
    device: &'a Device,
    queue: &'a Queue,
    bind_group: BindGroup,
    pipeline: ComputePipeline,
    workgroups: (u32, u32),
}

impl<'a> VTSampleTask<'a> {
    fn new<I, O>(
        shader_table: &HashMap<(VTFormat, VTFormat, bool), ShaderModule>,
        device: &'a Device,
        queue: &'a Queue,
        input: &I,
        output: &O,
    ) -> Self
    where
        I: VTResource,
        O: VTResource,
    {
        let bind_group_layout = {
            let mut entries: SmallVec<[BindGroupLayoutEntry; 10]> = SmallVec::with_capacity(10);

            for i in 0..input.get_format().view_formats().len() {
                entries.push(BindGroupLayoutEntry {
                    count: None,
                    binding: i as u32,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                });
            }

            for format in output.get_format().view_formats() {
                entries.push(BindGroupLayoutEntry {
                    count: None,
                    binding: entries.len() as u32,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        view_dimension: TextureViewDimension::D2,
                        access: StorageTextureAccess::WriteOnly,
                        format: *format,
                    },
                });
            }

            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &entries,
            })
        };

        let bind_group = {
            let mut views: SmallVec<[TextureView; 10]> = SmallVec::with_capacity(10);
            let mut entries: SmallVec<[BindGroupEntry; 10]> = SmallVec::with_capacity(10);

            input.create_views(&mut views);
            output.create_views(&mut views);

            for (i, view) in views.iter().enumerate() {
                entries.push(BindGroupEntry {
                    binding: i as u32,
                    resource: BindingResource::TextureView(view),
                });
            }

            device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &entries,
            })
        };

        Self {
            device,
            queue,
            bind_group,
            pipeline: device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                cache: None,
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    }),
                ),
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions::default(),
                module: shader_table
                    .get(&(
                        input.get_format(),
                        output.get_format(),
                        input.get_width() != output.get_width()
                            || input.get_height() != output.get_height(),
                    ))
                    .expect(
                        "No corresponding entry was found in the shader table. \
                    This is highly unlikely to exist and is a bug.",
                    ),
            }),
            workgroups: (
                (output.get_width() + 15) / 16,
                (output.get_height() + 15) / 16,
            ),
        }
    }

    pub fn run(&self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(self.workgroups.0, self.workgroups.1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }
}

impl VTResource for Texture {
    const TYPE: VTResourceType = VTResourceType::Texture;

    fn get_width(&self) -> u32 {
        self.width()
    }

    fn get_height(&self) -> u32 {
        self.height()
    }

    fn get_format(&self) -> VTFormat {
        match self.format() {
            TextureFormat::NV12 => VTFormat::NV12,
            TextureFormat::Rgba8Unorm => VTFormat::RGBA,
            TextureFormat::Bgra8Unorm => VTFormat::BGRA,
            f => unreachable!("unsupport format={:?}", f),
        }
    }

    fn create_views(&self, views: &mut SmallVec<[TextureView; 10]>) {
        let format = self.get_format();

        for view_format in format.view_formats() {
            views.push(self.create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2),
                format: Some(*view_format),
                // NV12 is a bit different because it doesn't support YUV420P
                // textures, so we won't consider YUV420P here.
                //
                // If it is a single texture of NV12, then two planes need to be
                // divided on the single texture. Other complex situations are
                // not considered here.
                aspect: if let VTFormat::NV12 = format {
                    if view_format == &TextureFormat::R8Unorm {
                        TextureAspect::Plane0
                    } else {
                        TextureAspect::Plane1
                    }
                } else {
                    TextureAspect::All
                },
                ..Default::default()
            }));
        }
    }
}

/// Pixel data enumeration, supporting different pixel formats
///
/// This enum represents different pixel data formats commonly used in video
/// processing.
///
/// Each variant contains the raw pixel data and stride information for proper
/// memory access.
pub enum PixelData<'a> {
    /// RGBA format pixel data
    ///
    /// Memory layout:
    ///
    /// - Each pixel is stored as 4 consecutive bytes: [R, G, B, A]
    /// - Pixels are arranged in row-major order
    /// - Total bytes per pixel = 4
    /// - Total bytes per row = width * 4
    ///
    /// Example for a 2x2 image:
    ///
    /// ```
    /// [R, G, B, A, R, G, B, A]
    /// [R, G, B, A, R, G, B, A]
    /// ```
    RGBA {
        /// Pixel data buffer containing raw RGBA values
        ///
        /// The buffer size should be at least: height * stride
        buffer: &'a [u8],
        /// Number of bytes per row (stride)
        ///
        /// Must be >= width * 4
        ///
        /// May include padding bytes for memory alignment
        stride: usize,
    },
    /// BGRA format pixel data
    ///
    /// Memory layout:
    ///
    /// - Each pixel is stored as 4 consecutive bytes: [B, G, R, A]
    /// - Pixels are arranged in row-major order
    /// - Total bytes per pixel = 4
    /// - Total bytes per row = width * 4
    ///
    /// Example for a 2x2 image:
    ///
    /// ```
    /// [B, G, R, A, B, G, R, A]
    /// [B, G, R, A, B, G, R, A]
    /// ```
    BGRA {
        /// Pixel data buffer containing raw BGRA values
        ///
        /// The buffer size should be at least: height * stride
        buffer: &'a [u8],
        /// Number of bytes per row (stride)
        ///
        /// Must be >= width * 4
        ///
        /// May include padding bytes for memory alignment
        stride: usize,
    },
    /// NV12 format pixel data
    ///
    /// Memory layout:
    ///
    /// - Y plane: Full resolution luminance values
    ///   - Each pixel is 1 byte
    ///   - Total bytes per row = width
    ///
    /// - UV plane: Half resolution chrominance values
    ///   - Each 2x2 block shares one U and one V value
    ///   - U and V values are interleaved: [U1,V1, U2,V2, ...]
    ///   - Total bytes per row = width
    ///
    /// Example for a 4x4 image:
    ///
    /// Y plane (4x4):
    ///
    /// ```
    /// [Y, Y, Y, Y]
    /// [Y, Y, Y, Y]
    /// [Y, Y, Y, Y]
    /// [Y, Y, Y, Y]
    /// ```
    ///
    /// UV plane (2x2):
    ///
    /// ```
    /// [U, V, U, V]
    /// [U, V, U, V]
    /// ```
    ///
    /// Where each UV pair corresponds to a 2x2 block in the Y plane
    NV12 {
        /// Pixel data buffer array containing Y and UV planes
        ///
        /// - buffer[0]: Y plane data (size = height * stride[0])
        /// - buffer[1]: UV plane data (size = (height/2) * stride[1])
        buffer: [&'a [u8]; 2],
        /// Number of bytes per row for each plane
        ///
        /// - stride[0]: Y plane stride (must be >= width)
        /// - stride[1]: UV plane stride (must be >= width)
        ///
        /// May include padding bytes for memory alignment
        stride: [usize; 2],
    },
    /// YUV420P format pixel data
    ///
    /// Memory layout:
    ///
    /// - Y plane: Full resolution luminance values
    ///   - Each pixel is 1 byte
    ///   - Total bytes per row = width
    ///
    /// - U plane: Quarter resolution U chrominance values
    ///   - Each 2x2 block shares one U value
    ///   - Total bytes per row = width/2
    ///
    /// - V plane: Quarter resolution V chrominance values
    ///   - Each 2x2 block shares one V value
    ///   - Total bytes per row = width/2
    ///
    /// Example for a 4x4 image:
    ///
    /// Y plane (4x4):
    ///
    /// ```
    /// [Y, Y, Y, Y]
    /// [Y, Y, Y, Y]
    /// [Y, Y, Y, Y]
    /// [Y, Y, Y, Y]
    /// ```
    ///
    /// U plane (2x2):
    ///
    /// ```
    /// [U, U]
    /// [U, U]
    /// ```
    ///
    /// V plane (2x2):
    ///
    /// ```
    /// [V, V]
    /// [V, V]
    /// ```
    ///
    /// Where each U/V value corresponds to a 2x2 block in the Y plane
    YUV420P {
        /// Pixel data buffer array containing Y, U, and V planes
        ///
        /// - buffer[0]: Y plane data (size = height * stride[0])
        /// - buffer[1]: U plane data (size = (height/2) * stride[1])
        /// - buffer[2]: V plane data (size = (height/2) * stride[2])
        buffer: [&'a [u8]; 3],
        /// Number of bytes per row for each plane
        ///
        /// - stride[0]: Y plane stride (must be >= width)
        /// - stride[1]: U plane stride (must be >= width/2)
        /// - stride[2]: V plane stride (must be >= width/2)
        ///
        /// May include padding bytes for memory alignment
        stride: [usize; 3],
    },
}

pub enum PixelBufferTextures<'a> {
    RGBA(&'a Texture),
    BGRA(&'a Texture),
    /// Y/UV
    NV12(&'a Texture, &'a Texture),
    /// Y/U/V
    YUV420P(&'a Texture, &'a Texture, &'a Texture),
}

/// Pixel buffer, managing GPU textures and pixel data
pub struct PixelBuffer<'a> {
    queue: &'a Queue,
    width: u32,
    height: u32,
    format: VTFormat,
    textures: Vec<Texture>,
}

impl<'a> VTResource for PixelBuffer<'a> {
    const TYPE: VTResourceType = VTResourceType::PixelBuffer;

    fn get_width(&self) -> u32 {
        self.width
    }

    fn get_height(&self) -> u32 {
        self.height
    }

    fn get_format(&self) -> VTFormat {
        self.format
    }

    fn create_views(&self, views: &mut SmallVec<[TextureView; 10]>) {
        for (i, format) in self.get_format().view_formats().iter().enumerate() {
            views.push(self.textures[i].create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2),
                aspect: TextureAspect::All,
                format: Some(*format),
                ..Default::default()
            }));
        }
    }
}

impl<'a> PixelBuffer<'a> {
    pub(crate) fn new(
        device: &Device,
        queue: &'a Queue,
        format: VTFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let usage = TextureUsages::COPY_DST;

        Self {
            queue,
            width,
            height,
            format,
            textures: {
                match format {
                    VTFormat::RGBA => {
                        vec![device.create_tex(width, height, TextureFormat::Rgba8Unorm, usage)]
                    }
                    VTFormat::BGRA => {
                        vec![device.create_tex(width, height, TextureFormat::Bgra8Unorm, usage)]
                    }
                    VTFormat::NV12 => {
                        vec![
                            device.create_tex(width, height, TextureFormat::R8Unorm, usage),
                            device.create_tex(
                                width / 2,
                                height / 2,
                                TextureFormat::Rg8Unorm,
                                usage,
                            ),
                        ]
                    }
                    VTFormat::YUV420P => {
                        vec![
                            device.create_tex(width, height, TextureFormat::R8Unorm, usage),
                            device.create_tex(width / 2, height / 2, TextureFormat::R8Unorm, usage),
                            device.create_tex(width / 2, height / 2, TextureFormat::R8Unorm, usage),
                        ]
                    }
                }
            },
        }
    }

    /// Write pixel data to pixel buffer
    pub fn write(&self, pixel_data: &PixelData) {
        match pixel_data {
            PixelData::RGBA { buffer, stride } | PixelData::BGRA { buffer, stride } => {
                self.queue
                    .copy_buffer(&self.textures[0], buffer, *stride as u32, self.height);
            }
            PixelData::NV12 { buffer, stride } => {
                for i in 0..2 {
                    self.queue.copy_buffer(
                        &self.textures[i],
                        buffer[i],
                        stride[i] as u32,
                        self.height,
                    );
                }
            }
            PixelData::YUV420P { buffer, stride } => {
                for i in 0..3 {
                    self.queue.copy_buffer(
                        &self.textures[i],
                        &buffer[i],
                        stride[i] as u32,
                        if i == 0 { self.height } else { self.height / 2 },
                    );
                }
            }
        }
    }

    #[rustfmt::skip]
    pub fn get_textures(&self) -> PixelBufferTextures {
        match self.format {
            VTFormat::RGBA => PixelBufferTextures::RGBA(&self.textures[0]),
            VTFormat::BGRA => PixelBufferTextures::BGRA(&self.textures[0]),
            VTFormat::NV12 => PixelBufferTextures::NV12(&self.textures[0], &self.textures[1]),
            VTFormat::YUV420P => PixelBufferTextures::YUV420P(&self.textures[0], &self.textures[1], &self.textures[2]),
        }
    }
}

// Compile shader code
fn compile_shader(input: VTFormat, output: VTFormat, need_scale: bool) -> String {
    let input_size = match input {
        VTFormat::RGBA | VTFormat::BGRA => 1,
        VTFormat::NV12 => 2,
        VTFormat::YUV420P => 3,
    };

    let output_types = match output {
        VTFormat::RGBA => &["rgba"][..],
        VTFormat::BGRA => &["bgra"],
        VTFormat::NV12 => &["r", "rg"],
        VTFormat::YUV420P => &["r", "r", "r"],
    };

    // Generate WGSL shader code
    format!(
        r#"
        {}

        {}

        {}

        @compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) position: vec3<u32>) 
        {{
            let max_position = textureDimensions(output_0);
            if (position.x >= max_position.x || position.y >= max_position.y) 
            {{
                return;
            }}

            let coords = vec2<i32>(position.xy);

            {}

            {}

            {}
        }}
    "#,
        // Generate input texture bindings
        {
            (0..input_size)
                .into_iter()
                .map(|i| {
                    format!(
                        r#"
                        @group(0) 
                        @binding({i}) 
                        var input_{i}: texture_2d<f32>;
                    "#
                    )
                })
                .collect::<String>()
        },
        // Generate output texture bindings
        {
            output_types
                .iter()
                .enumerate()
                .map(|(i, ty)| {
                    format!(
                        r#"
                        @group(0) 
                        @binding({}) 
                        var output_{i}: texture_storage_2d<{ty}8unorm, write>;
                    "#,
                        input_size + i
                    )
                })
                .collect::<String>()
        },
        {
            if need_scale {
                r#"
                fn scale(coords: vec2<i32>, input_size: vec2<u32>, output_size: vec2<u32>) -> vec2<i32> 
                {{
                    return vec2<i32>(
                        clamp(i32(round(f32(coords.x) * (f32(input_size.x) / f32(output_size.x)))), 0, i32(input_size.x) - 1),
                        clamp(i32(round(f32(coords.y) * (f32(input_size.y) / f32(output_size.y)))), 0, i32(input_size.y) - 1)
                    );
                }}
            "#
            } else {
                ""
            }
        },
        // Generate input sampling code
        {
            match input {
                // RGBA/BGRA format directly reads RGB components
                VTFormat::RGBA | VTFormat::BGRA => {
                    // If the input and output sizes are different, we need to scale the input
                    if need_scale {
                        r#"
                        let input_size = textureDimensions(input_0);
                        let rgba = textureLoad(input_0, scale(coords, input_size, max_position), 0);
                        let r = rgba.r;
                        let g = rgba.g;
                        let b = rgba.b;
                        let a = rgba.a;
                    "#
                    } else {
                        r#"
                        let r = textureLoad(input_0, coords, 0).r;
                        let g = textureLoad(input_0, coords, 0).g;
                        let b = textureLoad(input_0, coords, 0).b;
                    "#
                    }
                }
                // NV12/YUV420P format reads YUV components
                VTFormat::NV12 | VTFormat::YUV420P => {
                    // If the input and output sizes are different, we need to scale the input
                    if need_scale {
                        if input == VTFormat::NV12 {
                            r#"
                            let input_size = textureDimensions(input_0);
                            let y = textureLoad(input_0, scale(coords, input_size, max_position), 0).r;
                            let uv = textureLoad(input_1, scale(coords / 2, input_size / 2, max_position / 2), 0).rg;
                            let u = uv.r - 0.5;
                            let v = uv.g - 0.5;
                        "#
                        } else {
                            r#"
                            let input_size = textureDimensions(input_0);
                            let y = textureLoad(input_0, scale(coords, input_size, max_position), 0).r;
                            let u = textureLoad(input_1, scale(coords / 2, input_size / 2, max_position / 2), 0).r - 0.5;
                            let v = textureLoad(input_2, scale(coords / 2, input_size / 2, max_position / 2), 0).r - 0.5;
                        "#
                        }
                    } else {
                        if input == VTFormat::NV12 {
                            r#"
                            let y = textureLoad(input_0, coords, 0).r;
                            let u = textureLoad(input_1, coords / 2, 0).r - 0.5;
                            let v = textureLoad(input_1, coords / 2, 0).g - 0.5;
                        "#
                        } else {
                            r#"
                            let y = textureLoad(input_0, coords, 0).r;
                            let u = textureLoad(input_1, coords / 2, 0).r - 0.5;
                            let v = textureLoad(input_2, coords / 2, 0).r - 0.5;
                        "#
                        }
                    }
                }
            }
        },
        // Generate color space conversion code
        {
            match input {
                // RGBA/BGRA to YUV conversion
                VTFormat::RGBA | VTFormat::BGRA => match output {
                    VTFormat::RGBA | VTFormat::BGRA => "",
                    VTFormat::NV12 | VTFormat::YUV420P => {
                        r#"
                        let y = 0.299 * r + 0.587 * g + 0.114 * b;
                        let u = -0.169 * r - 0.331 * g + 0.5 * b + 0.5;
                        let v = 0.5 * r - 0.419 * g - 0.081 * b + 0.5;
                    "#
                    }
                },
                // YUV to RGBA/BGRA conversion
                VTFormat::NV12 | VTFormat::YUV420P => match output {
                    VTFormat::NV12 | VTFormat::YUV420P => "",
                    VTFormat::RGBA | VTFormat::BGRA => {
                        r#"
                        let r = y + 1.5748 * v;
                        let g = y - 0.187324 * u - 0.468124 * v;
                        let b = y + 1.8556 * u;
                    "#
                    }
                },
            }
        },
        // Generate output storage code
        {
            match output {
                // RGBA/BGRA format directly stores RGB components
                VTFormat::RGBA | VTFormat::BGRA => r#"
                    textureStore(output_0, coords, vec4<f32>(r, g, b, 1.0));
                "#
                .to_string(),
                // NV12/YUV420P format stores YUV components separately
                VTFormat::NV12 | VTFormat::YUV420P => {
                    format!(
                        r#"
                        textureStore(output_0, coords, vec4<f32>(y, 0.0, 0.0, 0.0));

                        if (coords.x % 2 != 0 || coords.y % 2 != 0) 
                        {{
                            return;
                        }}

                        {}
                    "#,
                        match output {
                            VTFormat::NV12 =>
                                "textureStore(output_1, coords / 2, vec4<f32>(u + 0.5, v + 0.5, 0.0, 0.0));",
                            VTFormat::YUV420P =>
                                r#"
                                textureStore(output_1, coords / 2, vec4<f32>(u + 0.5, 0.0, 0.0, 0.0));
                                textureStore(output_2, coords / 2, vec4<f32>(v + 0.5, 0.0, 0.0, 0.0));
                            "#,
                            _ => unreachable!(),
                        }
                    )
                }
            }
        }
    )
}
