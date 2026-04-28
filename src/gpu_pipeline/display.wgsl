// -------------------------------
// Uniforms
// -------------------------------
struct Uniforms {
    center_x:           f32,
    center_y:           f32,
    scale:              f32,
    max_iter:           u32,
    ref_orb_count:      u32,
    perturb_err_thresh: f32,
    grid_feedback_scale:f32,
    view_width:         f32,
    view_height:        f32,
    render_width:       u32,
    render_height:      u32,
    render_tex_width:   f32,
    render_tex_height:  f32,
};
@group(0) @binding(0) var<uniform> uni: Uniforms;

@group(0) @binding(1)
var render_tex: texture_2d<f32>;

@group(0) @binding(2)
var render_sampler: sampler;

struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
    var pos: vec2f;
    var uv: vec2f;

    switch (vid) {
        case 0u: {
            pos = vec2f(-1.0, -1.0);
            uv = vec2f(0.0, 0.0);
        }
        case 1u: {
            pos = vec2f(3.0, -1.0);
            uv = vec2f(2.0, 0.0);
        }
        case 2u: {
            pos = vec2f(-1.0, 3.0);
            uv = vec2f(0.0, 2.0);
        }
        default: {
            pos = vec2f(0.0);
            uv = vec2f(0.0);
        }
    }

    return VSOut(vec4f(pos, 0.0, 1.0), uv);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
    let render_size = vec2f(f32(uni.render_width), f32(uni.render_height));
    let texture_size = vec2f(uni.render_tex_width, uni.render_tex_height);

    let uv = in.uv * (render_size / texture_size);
    return textureSample(render_tex, render_sampler, uv);
}