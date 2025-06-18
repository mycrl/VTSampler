use std::{borrow::Cow, process::Command, sync::Arc};

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use vtsampler::{
    PixelData, VTFormat, VTSamplerBuilder,
    wgpu::{
        util::{BufferInitDescriptor, DeviceExt},
        *,
    },
};

use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

#[derive(Default)]
struct App {
    window: Option<Window>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                WindowAttributes::default().with_inner_size(PhysicalSize::new(600, 800)),
            )
            .unwrap();

        let output = Command::new("ffmpeg")
            .args([
                "-i",
                "https://w.wallhaven.cc/full/w5/wallhaven-w58l7p.jpg",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "rawvideo",
                "-",
            ])
            .output()
            .unwrap();

        pollster::block_on(render_texture(3000, 4000, &output.stdout, &window)).unwrap();
        self.window.replace(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut App::default())?;
    Ok(())
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
}

impl Vertex {
    pub const INDICES: &'static [u16] = &[0, 1, 2, 2, 1, 3];

    pub const VERTICES: &'static [Vertex] = &[
        Vertex::new([-1.0, -1.0], [0.0, 0.0]),
        Vertex::new([1.0, -1.0], [1.0, 0.0]),
        Vertex::new([-1.0, 1.0], [0.0, 1.0]),
        Vertex::new([1.0, 1.0], [1.0, 1.0]),
    ];

    pub const fn new(position: [f32; 2], tex_coords: [f32; 2]) -> Self {
        Self {
            position,
            tex_coords,
        }
    }

    pub fn desc<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    shader_location: 1,
                    format: VertexFormat::Float32x2,
                    offset: std::mem::size_of::<[f32; 2]>() as BufferAddress,
                },
            ],
        }
    }
}

async fn render_texture(width: u32, height: u32, buffer: &[u8], window: &Window) -> Result<()> {
    let window_size = window.inner_size();

    let instance = Instance::new(&InstanceDescriptor::default());
    let surface = instance.create_surface(window)?;
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::LowPower,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            ..Default::default()
        })
        .await?;

    let (device, queue) = adapter
        .request_device(&DeviceDescriptor {
            memory_hints: MemoryHints::MemoryUsage,
            required_features: adapter.features(),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await?;

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    {
        let mut config = surface
            .get_default_config(&adapter, window_size.width, window_size.height)
            .unwrap();

        config.present_mode = PresentMode::Immediate;
        config.format = TextureFormat::Bgra8Unorm;
        config.alpha_mode = CompositeAlphaMode::Opaque;
        config.usage = TextureUsages::RENDER_ATTACHMENT;
        surface.configure(&device, &config);
    }

    let output_texture = device.create_texture(&TextureDescriptor {
        label: None,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        format: TextureFormat::Bgra8Unorm,
        view_formats: &[],
        size: Extent3d {
            depth_or_array_layers: 1,
            width: window_size.width,
            height: window_size.height,
        },
    });

    conversion(
        device.clone(),
        queue.clone(),
        width as usize,
        height as usize,
        buffer,
        &output_texture,
        window_size,
    )
    .await?;

    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(Vertex::VERTICES),
        usage: BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(Vertex::INDICES),
        usage: BufferUsages::INDEX,
    });

    let sampler = device.create_sampler(&SamplerDescriptor {
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mipmap_filter: FilterMode::Nearest,
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        ..Default::default()
    });

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let texture_view = output_texture.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2),
        format: Some(TextureFormat::Bgra8Unorm),
        aspect: TextureAspect::All,
        ..Default::default()
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&sampler),
            },
        ],
    });

    let pipeline =
        device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: None,
                layout: Some(&device.create_pipeline_layout(
                    &PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    },
                )),
                vertex: VertexState {
                    entry_point: Some("main"),
                    module: &device
                        .create_shader_module(ShaderModuleDescriptor {
                            label: None,
                            source: ShaderSource::Wgsl(Cow::Borrowed(r#"
                                struct VertexOutput {
                                    @builtin(position) position: vec4<f32>,
                                    @location(0) coords: vec2<f32>,
                                };

                                @vertex fn main(@location(0) position: vec2<f32>, @location(1) coords: vec2<f32>) -> VertexOutput {
                                    var output: VertexOutput;
                                    output.position = vec4<f32>(position, 0.0, 1.0);
                                    output.coords = vec2<f32>(coords.x, 1.0 - coords.y);
                                    return output;
                                }
                            "#)),
                        }),
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[Vertex::desc()],
                },
                fragment: Some(FragmentState {
                    entry_point: Some("main"),
                    module: &device.create_shader_module(ShaderModuleDescriptor {
                        label: None,
                        source: ShaderSource::Wgsl(Cow::Borrowed(r#"
                            @group(0) @binding(0) var texture_: texture_2d<f32>;
                            @group(0) @binding(1) var sampler_: sampler;

                            @fragment fn main(@location(0) coords: vec2<f32>) -> @location(0) vec4<f32> {
                                return textureSample(texture_, sampler_, coords);
                            }
                        "#)),
                    }),
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(ColorTargetState {
                        blend: Some(BlendState::REPLACE),
                        write_mask: ColorWrites::ALL,
                        format: TextureFormat::Bgra8Unorm,
                    })],
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleStrip,
                    strip_index_format: Some(IndexFormat::Uint16),
                    ..Default::default()
                },
                multisample: MultisampleState::default(),
                depth_stencil: None,
                multiview: None,
                cache: None,
            });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    let output = surface.get_current_texture()?;
    let view = output
        .texture
        .create_view(&TextureViewDescriptor::default());

    {
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            ..Default::default()
        });

        render_pass.set_pipeline(&pipeline);
        render_pass.set_bind_group(0, Some(&bind_group), &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(index_buffer.slice(..), IndexFormat::Uint16);
        render_pass.draw_indexed(0..Vertex::INDICES.len() as u32, 0, 0..1);
    }

    queue.submit(Some(encoder.finish()));
    output.present();

    Ok(())
}

async fn conversion(
    device: Arc<Device>,
    queue: Arc<Queue>,
    width: usize,
    height: usize,
    buffer: &[u8],
    output_texture: &Texture,
    output_size: PhysicalSize<u32>,
) -> Result<()> {
    let sampler = VTSamplerBuilder::default()
        .with_device(&device, &queue)
        .build()
        .await?;

    let yuv420p_texture =
        sampler.create_pixel_buffer(VTFormat::YUV420P, width as u32, height as u32);

    let nv12_texture =
        sampler.create_pixel_buffer(VTFormat::NV12, output_size.width, output_size.height);

    let rgba_texture = device.create_texture(&TextureDescriptor {
        label: None,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        format: TextureFormat::Rgba8Unorm,
        view_formats: &[],
        size: Extent3d {
            depth_or_array_layers: 1,
            width: output_size.width,
            height: output_size.height,
        },
    });

    yuv420p_texture.write(&PixelData::YUV420P {
        buffer: [
            &buffer[..(width * height)],
            &buffer[(width * height)..((width * height) + (width / 2 * height / 2))],
            &buffer[((width * height) + (width / 2 * height / 2))..],
        ],
        stride: [width, width / 2, width / 2],
    });

    sampler.create_task(&yuv420p_texture, &nv12_texture).run();
    sampler.create_task(&nv12_texture, &rgba_texture).run();
    sampler.create_task(&rgba_texture, output_texture).run();
    Ok(())
}
