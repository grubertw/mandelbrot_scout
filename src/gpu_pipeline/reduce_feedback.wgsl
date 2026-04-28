fn safe_i32(x: f32) -> i32 {
    return i32(clamp(x, -1e6, 1e6));
}

// ---------------------------------------------
// Bind Group 0: Reduction inputs + outputs
// ---------------------------------------------
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
    grid_size:          u32,
    grid_width:         u32,
};
@group(0) @binding(0) var<uniform> uni: Uniforms;

// Per-pixel inputs (read-only)
@group(0) @binding(1)
var calc_tex : texture_2d<f32>;

@group(0) @binding(2)
var refine_tex : texture_2d<f32>;

// Aggregated screen-grid output (atomics)
struct GridFeedback {
    best_pixel_x:           atomic<i32>,
    best_pixel_y:           atomic<i32>,
    best_pixel_flags:       atomic<u32>,
    best_period:            atomic<u32>,
    best_contraction:       atomic<i32>,
    use_count:              atomic<u32>,
    max_iter_count:         atomic<u32>,
    score:                  atomic<i32>
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

const ESCAPED_BIT: u32              = 1u << 0u;
const PERTURB_BIT: u32              = 1u << 1u;
const PERTURB_ERR_BIT: u32          = 1u << 2u;
const MAX_ITER_BIT: u32             = 1u << 3u;
const ORBIT_SHIFT: u32              = 20u;

const MAX_U32: u32 = 0xFFFFFFFFu;
const MAX_P: u32 = 32u;

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
    let fi = textureLoad(calc_tex, pix, 0).x;
    let t = fi / f32(uni.max_iter);
    let iter = u32(floor(fi));
    
    let max_glitch_ratio = textureLoad(refine_tex, pix, 0).x;
    var best_period = u32(textureLoad(refine_tex, pix, 0).y);
    let best_period_error = textureLoad(refine_tex, pix, 0).z;
    var contraction = textureLoad(refine_tex, pix, 0).w;

    let escaped = (flags & ESCAPED_BIT) != 0u;
    let used_perturb = (flags & PERTURB_BIT) != 0u;
    let max_iter_reached = (flags & MAX_ITER_BIT) != 0u;
    let orbit_idx = flags >> ORBIT_SHIFT;

    // --- Screen-grid index ---
    let grid_x = u32(pix.x) / uni.grid_size;
    let grid_y = u32(pix.y) / uni.grid_size;
    let grid_idx = grid_y * uni.grid_width + grid_x;

    if (grid_idx >= arrayLength(&grid_feedback)) {
        return;
    }

    let feedback_scale = uni.grid_feedback_scale;
    let has_period = best_period < MAX_P;
    let period_score = select(0.0, 1.0 - f32(best_period) / f32(MAX_P), has_period);
    contraction = clamp(contraction, -100.0, 20.0);

    var score: i32 = 0;
    score  = safe_i32(t * feedback_scale * 0.5);
    score += safe_i32(period_score * feedback_scale * 2.0);
    score -= safe_i32(best_period_error * feedback_scale * 8.0);
    score -= safe_i32(contraction * feedback_scale * 0.1);
    score -= safe_i32(max_glitch_ratio * feedback_scale * 2.0);
    
    let prev_max_score = atomicMax(&grid_feedback[grid_idx].score, score);
    atomicMax(&grid_feedback[grid_idx].max_iter_count, iter);
    atomicMin(&grid_feedback[grid_idx].best_period, best_period);
    atomicMin(&grid_feedback[grid_idx].best_contraction,
        i32(contraction * 256.0));

    // If we won the max race, build complex 'c' for the pixel, from
    // the scene uniforms (i.e. same used in fragment shader).
    if (score > prev_max_score) {
        atomicStore(&grid_feedback[grid_idx].best_pixel_x, pix.x);
        atomicStore(&grid_feedback[grid_idx].best_pixel_y, pix.y);
        atomicStore(&grid_feedback[grid_idx].best_pixel_flags, flags);
    }

    if (used_perturb) {
        let perturb_err = (flags & PERTURB_ERR_BIT) != 0u;

        atomicMin(&orbit_feedback[orbit_idx].min_iter_count, iter);
        atomicMax(&orbit_feedback[orbit_idx].max_iter_count, iter);
        atomicAdd(&orbit_feedback[orbit_idx].use_count, 1u);

        if (escaped) {
            atomicAdd(&orbit_feedback[orbit_idx].escaped_count, 1u);
        }
        if (perturb_err) {
            atomicAdd(&orbit_feedback[orbit_idx].perurb_error_count, 1u);
        }
        if (max_iter_reached) {
            atomicAdd(&orbit_feedback[orbit_idx].max_iter_reached_count, 1u);
        }
    }
}
