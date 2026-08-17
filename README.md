# VTSampler

Cross-platform **GPU video format conversion and scaling** on [wgpu](https://docs.rs/wgpu). One [`VTImage`](https://docs.rs/vtsampler/latest/vtsampler/struct.VTImage.html) in, one out — similar in spirit to D3D11 Video Processor, but portable.

## Features

- **Formats:** RGBA, BGRA, NV12, YUV420P (any → any)
- **Scaling** when input/output sizes differ
- **Color:** BT.601 / BT.709 limited range
- **Backings:** CPU, wgpu textures, D3D11 (Windows), `CVPixelBuffer` (macOS)

## Quick example

```rust
use vtsampler::{PixelData, VTFormat, VTImage, VTProcessOptions, VTSamplerBuilder};

async fn run() -> Result<(), vtsampler::VTSampleError> {
    let mut sampler = VTSamplerBuilder::default().build().await?;

    let input = VTImage::from_cpu(&pixel_data, 1920, 1080);
    let output = VTImage::from_render_target(&gpu_texture, VTFormat::BGRA);

    sampler.process(&input, &output, VTProcessOptions::default())?;
    Ok(())
}
```

Share your renderer's device with [`VTSamplerBuilder::with_arc_device`](https://docs.rs/vtsampler/latest/vtsampler/struct.VTSamplerBuilder.html#method.with_arc_device).

## Example program

```sh
cargo run --example simple
```

Requires `ffmpeg` on `PATH` (downloads a sample image).

## License

[MIT](./LICENSE) Copyright (c) 2025 mycrl.
