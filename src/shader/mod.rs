mod registry;

pub use registry::{PipelineKey, ShaderRegistry};

use minijinja::{Environment, context};
use wgpu::{Device, ShaderModule, ShaderModuleDescriptor, ShaderSource};

use crate::{
    VTFormat,
    format::{VTColorSpace, VTSampleError, VTScaleFilter},
};

const TEMPLATE_COMPUTE: &str = include_str!("templates/compute.wgsl.jinja");
const TEMPLATE_SAMPLE: &str = include_str!("templates/sample_input.wgsl.jinja");
const TEMPLATE_CONVERT: &str = include_str!("templates/color_convert.wgsl.jinja");
const TEMPLATE_STORE: &str = include_str!("templates/store_output.wgsl.jinja");

fn wgsl_storage_suffix(format: wgpu::TextureFormat) -> &'static str {
    match format {
        wgpu::TextureFormat::Rgba8Unorm => "rgba",
        wgpu::TextureFormat::Bgra8Unorm => "bgra",
        wgpu::TextureFormat::R8Unorm => "r",
        wgpu::TextureFormat::Rg8Unorm => "rg",
        _ => "rgba",
    }
}

struct MatrixConstants {
    mat_y_r: &'static str,
    mat_y_g: &'static str,
    mat_y_b: &'static str,
    mat_u_r: &'static str,
    mat_u_g: &'static str,
    mat_u_b: &'static str,
    mat_v_r: &'static str,
    mat_v_g: &'static str,
    mat_v_b: &'static str,
    mat_r_v: &'static str,
    mat_g_u: &'static str,
    mat_g_v: &'static str,
    mat_b_u: &'static str,
}

fn matrix_constants(color_space: VTColorSpace) -> MatrixConstants {
    // BT.601 limited (legacy) vs BT.709 limited (HD / display content).
    match color_space {
        VTColorSpace::Bt601Limited => MatrixConstants {
            mat_y_r: "0.299",
            mat_y_g: "0.587",
            mat_y_b: "0.114",
            mat_u_r: "-0.169",
            mat_u_g: "-0.331",
            mat_u_b: "0.5",
            mat_v_r: "0.5",
            mat_v_g: "-0.419",
            mat_v_b: "-0.081",
            mat_r_v: "1.5748",
            mat_g_u: "-0.187324",
            mat_g_v: "-0.468124",
            mat_b_u: "1.8556",
        },
        VTColorSpace::Bt709Limited => MatrixConstants {
            mat_y_r: "0.2126",
            mat_y_g: "0.7152",
            mat_y_b: "0.0722",
            mat_u_r: "-0.1146",
            mat_u_g: "-0.3854",
            mat_u_b: "0.5",
            mat_v_r: "0.5",
            mat_v_g: "-0.4542",
            mat_v_b: "-0.0458",
            mat_r_v: "1.5748",
            mat_g_u: "-0.187324",
            mat_g_v: "-0.468124",
            mat_b_u: "1.8556",
        },
    }
}

pub(crate) fn compile_wgsl(
    input: VTFormat,
    output: VTFormat,
    need_scale: bool,
    color_space: VTColorSpace,
    scale_filter: VTScaleFilter,
) -> Result<String, VTSampleError> {
    let mut env = Environment::new();
    env.add_template("compute.wgsl.jinja", TEMPLATE_COMPUTE)
        .map_err(|e| VTSampleError::Template(e.to_string()))?;
    env.add_template("sample_input.wgsl.jinja", TEMPLATE_SAMPLE)
        .map_err(|e| VTSampleError::Template(e.to_string()))?;
    env.add_template("color_convert.wgsl.jinja", TEMPLATE_CONVERT)
        .map_err(|e| VTSampleError::Template(e.to_string()))?;
    env.add_template("store_output.wgsl.jinja", TEMPLATE_STORE)
        .map_err(|e| VTSampleError::Template(e.to_string()))?;

    let output_storage_formats: Vec<_> = output
        .plane_formats()
        .iter()
        .map(|f| wgsl_storage_suffix(*f))
        .collect();

    let rgb_to_yuv = matches!(input, VTFormat::RGBA | VTFormat::BGRA)
        && matches!(output, VTFormat::NV12 | VTFormat::YUV420P);
    let yuv_to_rgb = matches!(input, VTFormat::NV12 | VTFormat::YUV420P)
        && matches!(output, VTFormat::RGBA | VTFormat::BGRA);

    let m = matrix_constants(color_space);

    let tmpl = env
        .get_template("compute.wgsl.jinja")
        .map_err(|e| VTSampleError::Template(e.to_string()))?;

    tmpl.render(context! {
        input_plane_count => input.plane_count(),
        output_plane_count => output.plane_count(),
        output_storage_formats,
        need_scale => need_scale,
        scale_filter => match scale_filter {
            VTScaleFilter::Nearest => "nearest",
            VTScaleFilter::Bilinear => "bilinear",
        },
        input_kind => input.shader_name(),
        output_kind => output.shader_name(),
        rgb_to_yuv => rgb_to_yuv,
        yuv_to_rgb => yuv_to_rgb,
        mat_y_r => m.mat_y_r,
        mat_y_g => m.mat_y_g,
        mat_y_b => m.mat_y_b,
        mat_u_r => m.mat_u_r,
        mat_u_g => m.mat_u_g,
        mat_u_b => m.mat_u_b,
        mat_v_r => m.mat_v_r,
        mat_v_g => m.mat_v_g,
        mat_v_b => m.mat_v_b,
        mat_r_v => m.mat_r_v,
        mat_g_u => m.mat_g_u,
        mat_g_v => m.mat_g_v,
        mat_b_u => m.mat_b_u,
    })
    .map_err(|e| VTSampleError::Template(e.to_string()))
}

pub(crate) fn create_shader_module(device: &Device, source: &str) -> ShaderModule {
    device.create_shader_module(ShaderModuleDescriptor {
        label: Some("vtsampler_compute"),
        source: ShaderSource::Wgsl(source.into()),
    })
}
