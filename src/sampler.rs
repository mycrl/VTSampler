use std::sync::{Arc, OnceLock};

use wgpu::{
    Adapter, CommandEncoder, Device, DeviceDescriptor, Instance, InstanceDescriptor, MemoryHints,
    PowerPreference, Queue, RequestAdapterOptions,
};

use crate::{
    VTFormat, VTImage, VTImageOwned, VTProcessOptions, format::VTSampleError, pool::ScratchPool,
    process::Processor, shader::ShaderRegistry,
};

#[cfg(windows)]
use crate::bridge::d3d11::VtD3d11Pool;
#[cfg(target_os = "macos")]
use crate::bridge::metal::VtMetalCache;
#[cfg(target_os = "macos")]
use std::sync::Mutex;

struct GlobalInstance {
    #[allow(dead_code)]
    instance: Instance,
    #[allow(dead_code)]
    adapter: Adapter,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

static GLOBAL: OnceLock<GlobalInstance> = OnceLock::new();

async fn global_instance() -> Result<&'static GlobalInstance, VTSampleError> {
    if let Some(g) = GLOBAL.get() {
        return Ok(g);
    }

    let instance = Instance::new(InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::LowPower,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .ok_or(VTSampleError::NotFoundAdapter)?;

    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                memory_hints: MemoryHints::MemoryUsage,
                required_features: adapter.features(),
                required_limits: adapter.limits(),
                ..Default::default()
            },
            None,
        )
        .await
        .map_err(|_| VTSampleError::RequestDeviceFailed)?;

    GLOBAL
        .set(GlobalInstance {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
        .ok();

    Ok(GLOBAL.get().unwrap())
}

/// Builds a [`VTSampler`] on a shared or external GPU device.
#[derive(Default)]
pub struct VTSamplerBuilder {
    device: Option<Arc<Device>>,
    queue: Option<Arc<Queue>>,
}

impl VTSamplerBuilder {
    /// Use an existing wgpu device (typically shared with the application renderer).
    pub fn with_arc_device(mut self, device: Arc<Device>, queue: Arc<Queue>) -> Self {
        self.device = Some(device);
        self.queue = Some(queue);
        self
    }

    pub async fn build(self) -> Result<VTSampler, VTSampleError> {
        let (device, queue) = if let (Some(d), Some(q)) = (self.device, self.queue) {
            (d, q)
        } else {
            let g = global_instance().await?;
            (g.device.clone(), g.queue.clone())
        };

        Ok(VTSampler {
            shaders: ShaderRegistry::new(),
            pool: ScratchPool::new(device.clone()),
            #[cfg(windows)]
            d3d11_pool: VtD3d11Pool::new(),
            #[cfg(target_os = "macos")]
            metal_cache: Mutex::new(None),
            device,
            queue,
        })
    }
}

/// GPU video convert + scale engine (one input image → one output image per call).
pub struct VTSampler {
    device: Arc<Device>,
    queue: Arc<Queue>,
    shaders: ShaderRegistry,
    pool: ScratchPool,
    #[cfg(windows)]
    d3d11_pool: VtD3d11Pool,
    #[cfg(target_os = "macos")]
    metal_cache: Mutex<Option<VtMetalCache>>,
}

impl VTSampler {
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn allocate(&self, format: VTFormat, width: u32, height: u32) -> VTImageOwned {
        crate::pool::ScratchPool::allocate_owned(&self.device, format, width, height)
    }

    pub fn process(
        &mut self,
        input: &VTImage<'_>,
        output: &VTImage<'_>,
        opts: VTProcessOptions,
    ) -> Result<(), VTSampleError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vtsampler_process"),
            });
        self.encode(input, output, &mut encoder, opts)?;
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    pub fn encode(
        &mut self,
        input: &VTImage<'_>,
        output: &VTImage<'_>,
        encoder: &mut CommandEncoder,
        opts: VTProcessOptions,
    ) -> Result<(), VTSampleError> {
        Processor {
            device: &self.device,
            #[cfg(target_os = "macos")]
            device_arc: &self.device,
            queue: &self.queue,
            shaders: &mut self.shaders,
            pool: &self.pool,
            #[cfg(windows)]
            d3d11_pool: &self.d3d11_pool,
            #[cfg(target_os = "macos")]
            metal_cache: &self.metal_cache,
        }
        .encode(input, output, encoder, opts)
    }
}
