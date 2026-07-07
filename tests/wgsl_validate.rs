//! Static validation of the WGSL compute shaders. wgpu only compiles them at
//! runtime (create_shader_module), so this parses + validates them with the same
//! naga version wgpu 29 uses — catching syntax/type/binding errors in `cargo test`
//! instead of at first render.

use naga::valid::{Capabilities, ValidationFlags, Validator};

fn validate(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("WGSL parse failed for {name}:\n{}", e.emit_to_string(src)));

    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("WGSL validation failed for {name}:\n{e:?}"));
}

#[test]
fn mandelbrot_fexp_is_valid() {
    validate(
        "mandelbrot_fexp.wgsl",
        include_str!("../src/gpu_pipeline/mandelbrot_fexp.wgsl"),
    );
}

#[test]
fn mandelbrot_f32_is_valid() {
    validate(
        "mandelbrot_f32.wgsl",
        include_str!("../src/gpu_pipeline/mandelbrot_f32.wgsl"),
    );
}

#[test]
fn mandelbrot_burningship_is_valid() {
    validate(
        "mandelbrot_burningship.wgsl",
        include_str!("../src/gpu_pipeline/mandelbrot_burningship.wgsl"),
    );
}

#[test]
fn mandelbrot_stateful_is_valid() {
    validate(
        "mandelbrot_stateful.wgsl",
        include_str!("../src/gpu_pipeline/mandelbrot_stateful.wgsl"),
    );
}
