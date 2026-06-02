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

/// Configures how a [`VTSampler`] obtains its [`wgpu::Device`] and [`wgpu::Queue`].
///
/// # Examples
///
/// **Headless device** (tools, tests):
///
/// ```no_run
/// # async fn demo() -> Result<(), vtsampler::VTSampleError> {
/// let _sampler = vtsampler::VTSamplerBuilder::default().build().await?;
/// # Ok(())
/// # }
/// ```
///
/// **Shared with an existing renderer** (recommended for apps):
///
/// ```no_run
/// # use std::sync::Arc;
/// # async fn demo(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Result<(), vtsampler::VTSampleError> {
/// let _sampler = vtsampler::VTSamplerBuilder::default()
///     .with_arc_device(device, queue)
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Default)]
pub struct VTSamplerBuilder {
    device: Option<Arc<Device>>,
    queue: Option<Arc<Queue>>,
}

impl VTSamplerBuilder {
    /// Uses your application's wgpu device instead of creating a global headless instance.
    ///
    /// # Notes
    ///
    /// * `device` and `queue` must belong to the same adapter.
    /// * On Windows, use a **DX12** backend for [`crate::VtD3d11Bridge`].
    /// * On macOS, use **Metal** for `VtMetalCache` and `VTImage::from_cv_pixel_buffer`.
    pub fn with_arc_device(mut self, device: Arc<Device>, queue: Arc<Queue>) -> Self {
        self.device = Some(device);
        self.queue = Some(queue);
        self
    }

    /// Creates a [`VTSampler`] (async due to wgpu adapter/device initialization).
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

/// GPU engine for video **format conversion** and **scaling**.
///
/// One [`VTImage`] input and one [`VTImage`] output per call. Compute pipelines are generated
/// from Minijinja templates and cached by [`crate::PipelineKey`]. A scratch pool reuses
/// intermediate textures across calls with the same dimensions.
///
/// # Thread safety
///
/// Prefer one `VTSampler` per thread, or wrap it in `Arc<Mutex<VTSampler>>` if shared.
/// Native bridge pools use internal locking.
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
    /// The wgpu device used for shaders, allocation, and processing.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The queue used when [`Self::process`] submits command buffers.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Allocates GPU textures suitable as conversion intermediates (`STORAGE_BINDING` / `COPY_DST`).
    pub fn allocate(&self, format: VTFormat, width: u32, height: u32) -> VTImageOwned {
        crate::pool::ScratchPool::allocate_owned(&self.device, format, width, height)
    }

    /// Converts / scales `input` → `output` and **submits** work to the GPU queue.
    ///
    /// To record into your own command buffer (e.g. combined with rendering), use [`Self::encode`].
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

    /// Records the conversion pass into `encoder` without submitting.
    ///
    /// Submit the encoder (with any other passes) via your own [`wgpu::Queue::submit`] call.
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
