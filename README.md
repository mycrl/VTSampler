# VTSampler

A pure Rust implementation similar to ID3D11VideoProcessor, but cross-platform and cross-graphics API.

---

> Since it is still under development, it is not currently available on crates.io, but it is already usable.

This library currently supports conversion between RGBA, BGRA, NV12, and YUV420P, and also supports texture scaling.

However, there are currently some limitations, such as the color space being fixed (BT.601) and the scaling sampling mode being Nearest, which is the simplest sampling implementation. Additionally, it only supports copying from CPU to GPU and does not support writing back from GPU to CPU.

These limitations are related to the current simple implementation of the shader generator. Future plans include developing a more feature-rich shader generator.

Furthermore, future plans also include supporting native textures from certain graphics APIs, such as ID3D11Texture2D, MTLTexture, and CVPixelBuffer, allowing these texture types to be used directly without manual conversion to wgpu Texture.

## Example

```rust
use vtsample::{VTFormat, VTSamplerBuilder};

let sampler = VTSamplerBuilder::default().build().await?;

let input = sampler.create_pixel_buffer(VTFormat::NV12, 1920, 1080);
let output = sampler.create_pixel_buffer(VTFormat::RGBA, 800, 680);

let task = sampler.create_task(&input, &output);
task.run();
```

## License

[MIT](./LICENSE) Copyright (c) 2025 mycrl.
