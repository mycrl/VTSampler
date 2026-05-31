//! D3D11 shared textures ↔ wgpu (DX12 interop).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::{
    Direct3D11::{
        D3D11_BIND_SHADER_RESOURCE, D3D11_RESOURCE_MISC_SHARED, D3D11_TEXTURE2D_DESC,
        D3D11_USAGE_DEFAULT, ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
    },
    Direct3D12::ID3D12Resource,
    Dxgi::Common::{
        DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_NV12, DXGI_FORMAT_R8G8B8A8_UNORM,
    },
    Dxgi::IDXGIResource,
};

use wgpu::{
    Device, Extent3d, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    hal::api::Dx12,
};

use crate::{VTFormat, bridge::BridgeError};

/// D3D11 device + immediate context used for GPU copies into bridge textures.
pub struct VtD3d11Device<'a> {
    pub device: &'a ID3D11Device,
    pub context: &'a ID3D11DeviceContext,
}

impl<'a> VtD3d11Device<'a> {
    pub fn new(device: &'a ID3D11Device, context: &'a ID3D11DeviceContext) -> Self {
        Self { device, context }
    }
}

/// Owned D3D11 texture paired with a wgpu texture imported via shared handle.
pub struct VtD3d11Bridge {
    pub d3d11: ID3D11Texture2D,
    pub wgpu: Texture,
}

impl VtD3d11Bridge {
    pub fn new(
        d3d: &VtD3d11Device<'_>,
        wgpu_device: &Device,
        width: u32,
        height: u32,
        dxgi_format: DXGI_FORMAT,
        wgpu_format: TextureFormat,
        usage: TextureUsages,
    ) -> Result<Self, BridgeError> {
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        desc.Width = width;
        desc.Height = height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.SampleDesc.Count = 1;
        desc.Format = dxgi_format;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE.0 as u32;
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED.0 as u32;

        let mut texture = None;
        unsafe {
            d3d.device
                .CreateTexture2D(&desc, None, Some(&mut texture))?;
        }
        let d3d11 = texture.unwrap();

        let wgpu = import_d3d11_shared(wgpu_device, &d3d11, wgpu_format, width, height, usage)?;

        Ok(Self { d3d11, wgpu })
    }

    pub fn copy_from(
        &self,
        d3d: &VtD3d11Device<'_>,
        src: &ID3D11Texture2D,
        array_index: u32,
    ) -> Result<(), BridgeError> {
        unsafe {
            d3d.context.CopySubresourceRegion(
                &self.d3d11, 0, 0, 0, 0, src, array_index, None,
            );
        }
        Ok(())
    }

    pub fn copy_to(
        &self,
        d3d: &VtD3d11Device<'_>,
        dst: &ID3D11Texture2D,
        array_index: u32,
    ) -> Result<(), BridgeError> {
        unsafe {
            d3d.context.CopySubresourceRegion(
                dst, array_index, 0, 0, 0, &self.d3d11, 0, None,
            );
        }
        Ok(())
    }
}

/// Reused D3D11 bridges keyed by dimensions/format/usage.
pub struct VtD3d11Pool {
    entries: Mutex<HashMap<PoolKey, Arc<VtD3d11Bridge>>>,
}

#[derive(Hash, Eq, PartialEq)]
struct PoolKey {
    width: u32,
    height: u32,
    dxgi: i32,
    usage: u32,
}

impl VtD3d11Pool {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub fn acquire(
        &self,
        d3d: &VtD3d11Device<'_>,
        wgpu: &Device,
        width: u32,
        height: u32,
        format: VTFormat,
        usage: TextureUsages,
    ) -> Result<Arc<VtD3d11Bridge>, BridgeError> {
        let (dxgi, wgpu_format) = vt_format_to_dxgi_wgpu(format)?;
        let key = PoolKey {
            width,
            height,
            dxgi: dxgi.0,
            usage: usage.bits(),
        };
        let mut map = self.entries.lock().expect("d3d11 pool lock");
        if let Some(bridge) = map.get(&key) {
            return Ok(bridge.clone());
        }
        let bridge = Arc::new(VtD3d11Bridge::new(
            d3d, wgpu, width, height, dxgi, wgpu_format, usage,
        )?);
        map.insert(key, bridge.clone());
        Ok(bridge)
    }
}

pub fn vt_format_to_dxgi_wgpu(format: VTFormat) -> Result<(DXGI_FORMAT, TextureFormat), BridgeError> {
    match format {
        VTFormat::RGBA => Ok((DXGI_FORMAT_R8G8B8A8_UNORM, TextureFormat::Rgba8Unorm)),
        VTFormat::BGRA => Ok((DXGI_FORMAT_B8G8R8A8_UNORM, TextureFormat::Bgra8Unorm)),
        VTFormat::NV12 => Ok((DXGI_FORMAT_NV12, TextureFormat::NV12)),
        VTFormat::YUV420P => Err(BridgeError::UnsupportedFormat),
    }
}

pub fn dxgi_from_texture(texture: &ID3D11Texture2D) -> DXGI_FORMAT {
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    unsafe {
        texture.GetDesc(&mut desc);
    }
    desc.Format
}

fn shared_handle(texture: &ID3D11Texture2D) -> Result<HANDLE, BridgeError> {
    let handle = unsafe { texture.cast::<IDXGIResource>()?.GetSharedHandle()? };
    if handle.is_invalid() {
        return Err(BridgeError::InvalidSharedHandle);
    }
    Ok(handle)
}

fn import_d3d11_shared(
    wgpu_device: &Device,
    d3d11: &ID3D11Texture2D,
    wgpu_format: TextureFormat,
    width: u32,
    height: u32,
    usage: TextureUsages,
) -> Result<Texture, BridgeError> {
    let tex_desc = TextureDescriptor {
        label: Some("vtsampler_d3d11_bridge"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: wgpu_format,
        usage,
        view_formats: &[],
    };

    unsafe {
        Ok(wgpu_device.create_texture_from_hal::<Dx12>(
            <Dx12 as wgpu::hal::Api>::Device::texture_from_raw(
                wgpu_device
                    .as_hal::<Dx12, _, _>(|hdevice| {
                        let mut resource = None::<ID3D12Resource>;
                        hdevice
                            .ok_or(BridgeError::NotFoundDxBackend)?
                            .raw_device()
                            .OpenSharedHandle(shared_handle(d3d11)?, &mut resource)
                            .map(|_| resource.unwrap())
                            .map_err(BridgeError::Windows)
                    })
                    .ok_or(BridgeError::NotFoundDxBackend)??,
                wgpu_format,
                TextureDimension::D2,
                tex_desc.size,
                tex_desc.mip_level_count,
                tex_desc.sample_count,
            ),
            &tex_desc,
        ))
    }
}

/// Import an existing shared D3D11 texture into wgpu (no copy).
pub fn import_d3d11_texture(
    wgpu_device: &Device,
    d3d11: &ID3D11Texture2D,
    format: VTFormat,
    usage: TextureUsages,
) -> Result<Texture, BridgeError> {
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    unsafe {
        d3d11.GetDesc(&mut desc);
    }
    let (_, wgpu_format) = vt_format_to_dxgi_wgpu(format)?;
    import_d3d11_shared(
        wgpu_device,
        d3d11,
        wgpu_format,
        desc.Width,
        desc.Height,
        usage,
    )
}
