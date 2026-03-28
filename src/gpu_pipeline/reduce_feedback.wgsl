
// ---------------------------------------------
// Bind Group 0: Reduction inputs + outputs
// ---------------------------------------------
struct Uniforms {
    center_x:           f32,
    center_y:           f32,
    scale:              f32,
    max_iter:           u32,
    ref_orb_count:      u32,
    view_width:         f32,
    view_height:        f32,
    render_width:       u32,
    render_height:      u32,
    render_tex_width:   f32,
    render_tex_height:  f32,
    grid_size:          u32,
    grid_width:         u32,
};
@group(0) @binding(0) var<uniform> uni: Uniforms;

// Per-pixel inputs (read-only)
@group(0) @binding(1)
var calc_tex : texture_2d<f32>;

// Aggregated screen-grid output (atomics)
struct GridFeedback {
    best_pixel_x:           atomic<i32>, // Pixel location of deepest iteration in the sample grid
    best_pixel_y:           atomic<i32>,
    best_pixel_flags:       atomic<u32>, // Iteration feedback flags for the location
    max_iter_count:         atomic<u32>, // Also needed here to find best pixel
};
@group(0) @binding(2)
var<storage, read_write> grid_feedback : array<GridFeedback>;

struct OrbitFeedback {
    min_iter_count:             atomic<u32>,
    max_iter_count:             atomic<u32>,
    escaped_count:              atomic<u32>,
    perurb_error_inner_count:   atomic<u32>,
    perurb_error_outer_count:   atomic<u32>,
    max_iter_reached_count:     atomic<u32>,
};
@group(0) @binding(3)
var<storage, read_write> orbit_feedback : array<OrbitFeedback>;

const ESCAPED_BIT: u32              = 1u << 0u;
const PERTURB_BIT: u32              = 1u << 1u;
const PERTURB_ERR_INNER_BIT: u32    = 1u << 2u;
const PERTURB_ERR_OUTER_BIT: u32    = 1u << 3u;
const MAX_ITER_BIT: u32             = 1u << 4u;
const ORBIT_SHIFT: u32              = 20u;

// ---------------------------------------------
// Compute entry point
// ---------------------------------------------
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let pix = vec2i(i32(gid.x), i32(gid.y));

    // Bounds checking based on real screen size
    if (pix.x >= i32(uni.render_width) || pix.y >= i32(uni.render_height)) {
        return;
    }

    let flags = u32(textureLoad(calc_tex, pix, 0).w);
    let iter = u32(floor(textureLoad(calc_tex, pix, 0).x));

    let escaped = (flags & ESCAPED_BIT) != 0u;
    let used_perturb = (flags & PERTURB_BIT) != 0u;
    let perturb_err_inner = (flags & PERTURB_ERR_INNER_BIT) != 0u;
    let perturb_err_outer = (flags & PERTURB_ERR_OUTER_BIT) != 0u;
    let max_iter_reached = (flags & MAX_ITER_BIT) != 0u;
    let orbit_idx = flags >> ORBIT_SHIFT;

    // --- Screen-grid index ---
    let grid_x = u32(pix.x) / uni.grid_size;
    let grid_y = u32(pix.y) / uni.grid_size;
    let grid_idx = grid_y * uni.grid_width + grid_x;

    if (grid_idx >= arrayLength(&grid_feedback)) {
        return;
    }

    // Track max iters in GridFeedback to find best orbit location
    let prev_max = atomicMax(&grid_feedback[grid_idx].max_iter_count, iter);

    // If we won the max race, build complex 'c' for the pixel, from
    // the scene uniforms (i.e. same used in fragment shader).
    if (iter > prev_max) {
        atomicStore(&grid_feedback[grid_idx].best_pixel_x, pix.x);
        atomicStore(&grid_feedback[grid_idx].best_pixel_y, pix.y);
        atomicStore(&grid_feedback[grid_idx].best_pixel_flags, flags);
    }

    if (used_perturb) {
        atomicMin(&orbit_feedback[orbit_idx].min_iter_count, iter);
        atomicMax(&orbit_feedback[orbit_idx].max_iter_count, iter);

        if (escaped) {
            atomicAdd(&orbit_feedback[orbit_idx].escaped_count, 1u);
        }
        if (perturb_err_inner) {
            atomicAdd(&orbit_feedback[orbit_idx].perurb_error_inner_count, 1u);
        }
        if (perturb_err_outer) {
            atomicAdd(&orbit_feedback[orbit_idx].perurb_error_outer_count, 1u);
        }
        if (max_iter_reached) {
            atomicAdd(&orbit_feedback[orbit_idx].max_iter_reached_count, 1u);
        }
    }
}
