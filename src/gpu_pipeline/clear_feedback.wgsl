@group(0) @binding(0)
var mandel_out_tex : texture_storage_2d<rgba32float, write>;

@group(0) @binding(1)
var refine_out_tex : texture_storage_2d<rgba32float, write>;

@group(0) @binding(2)
var render_tex : texture_storage_2d<rgba8unorm, write>;

// Aggregated screen-grid output (atomics)
struct GridFeedback {
    best_pixel_x:           atomic<i32>, 
    best_pixel_y:           atomic<i32>,
    best_pixel_flags:       atomic<u32>, 
    best_period:            atomic<u32>,
    best_contraction:       atomic<i32>,
    use_count:              atomic<u32>,
    max_iter_count:         atomic<u32>, 
    score:                  atomic<i32>,
};

@group(0) @binding(3)
var<storage, read_write> grid_feedback : array<GridFeedback>;

struct OrbitFeedback {
    min_iter_count:             atomic<u32>,
    max_iter_count:             atomic<u32>,
    use_count:                  atomic<u32>,
    escaped_count:              atomic<u32>,
    perurb_error_count:         atomic<u32>,
    max_iter_reached_count:     atomic<u32>,
};
@group(0) @binding(4)
var<storage, read_write> orbit_feedback : array<OrbitFeedback>;

const MAX_U32 : u32 = 0xFFFFFFFFu;
const MAX_P: u32 = 32u;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);

    // Clear per-pixel textures
    textureStore(mandel_out_tex, vec2i(x, y), vec4f(0.0, 0.0, 0.0, 0.0));
    textureStore(refine_out_tex, vec2i(x, y), vec4f(0.0, 0.0, 0.0, 0.0));
    textureStore(render_tex, vec2i(x, y), vec4f(0.0, 0.0, 0.0, 0.0));

    // Clear feedback (1D mapping)
    let grid_idx = gid.y * 65536u + gid.x;
    if (grid_idx < arrayLength(&grid_feedback)) {
        atomicStore(&grid_feedback[grid_idx].best_pixel_x, -1);
        atomicStore(&grid_feedback[grid_idx].best_pixel_y, -1);
        atomicStore(&grid_feedback[grid_idx].best_pixel_flags, 0u);
        atomicStore(&grid_feedback[grid_idx].best_period, MAX_P);
        atomicStore(&grid_feedback[grid_idx].best_contraction, 10000);
        atomicStore(&grid_feedback[grid_idx].use_count, 0u);
        atomicStore(&grid_feedback[grid_idx].max_iter_count, 0u);
        atomicStore(&grid_feedback[grid_idx].score, -2147483647);
    }
    
    let orbit_idx = gid.y * 65536u + gid.x;
    if (orbit_idx < arrayLength(&orbit_feedback)) {
        atomicStore(&orbit_feedback[orbit_idx].min_iter_count, MAX_U32);
        atomicStore(&orbit_feedback[orbit_idx].max_iter_count, 0u);
        atomicStore(&orbit_feedback[orbit_idx].use_count, 0u);
        atomicStore(&orbit_feedback[orbit_idx].escaped_count, 0u);
        atomicStore(&orbit_feedback[orbit_idx].perurb_error_count, 0u);
        atomicStore(&orbit_feedback[orbit_idx].max_iter_reached_count, 0u);
    }
}
