use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use wgpu::Device;

use crate::{
    VTFormat, VTImageOwned,
    gpu::{create_plane_texture, plane_size},
};

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
struct PoolKey {
    format: VTFormat,
    width: u32,
    height: u32,
}

/// Internal scratch textures reused across process calls.
pub struct ScratchPool {
    device: Arc<Device>,
    entries: Mutex<HashMap<PoolKey, Arc<Mutex<ScratchTextures>>>>,
}

/// Plane storage used by scratch pool and [`crate::VTImageOwned`].
pub struct ScratchTextures {
    pub planes: Vec<wgpu::Texture>,
    pub format: VTFormat,
    pub width: u32,
    pub height: u32,
}

impl ScratchPool {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Returns a shared handle; multiple handles per frame are allowed (e.g. input + output scratch).
    pub fn acquire(
        &self,
        format: VTFormat,
        width: u32,
        height: u32,
    ) -> Arc<Mutex<ScratchTextures>> {
        let key = PoolKey {
            format,
            width,
            height,
        };
        let mut map = self.entries.lock().expect("scratch pool lock");
        map.entry(key)
            .or_insert_with(|| {
                Arc::new(Mutex::new(Self::create_planes(
                    &self.device,
                    format,
                    width,
                    height,
                )))
            })
            .clone()
    }

    fn create_planes(
        device: &Arc<Device>,
        format: VTFormat,
        width: u32,
        height: u32,
    ) -> ScratchTextures {
        let plane_count = format.plane_count();
        let mut planes = Vec::with_capacity(plane_count);
        for i in 0..plane_count {
            let (pw, ph) = plane_size(format, width, height, i);
            planes.push(create_plane_texture(
                device,
                pw,
                ph,
                format.plane_formats()[i],
                wgpu::TextureUsages::empty(),
            ));
        }
        ScratchTextures {
            planes,
            format,
            width,
            height,
        }
    }

    pub fn allocate_owned(
        device: &Arc<Device>,
        format: VTFormat,
        width: u32,
        height: u32,
    ) -> VTImageOwned {
        VTImageOwned::new(Self::create_planes(device, format, width, height))
    }
}
