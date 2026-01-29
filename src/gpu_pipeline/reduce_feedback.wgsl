// ---------------------------------------------
// Bind Group 0: Reduction inputs + outputs
// ---------------------------------------------

// Per-pixel inputs (read-only)
@group(0) @binding(0)
var last_valid_i_tex : texture_2d<u32>;

@group(0) @binding(1)
var orbit_idx_tex : texture_2d<u32>;

@group(0) @binding(2)
var flags_tex : texture_2d<u32>;

// Aggregated per-orbit output (atomics)
struct OrbitFeedback {
    min_last_valid_i : atomic<u32>,
    max_last_valid_i : atomic<u32>,
    perturb_attempted_count : atomic<u32>,
    perturb_valid_count : atomic<u32>,
    perturb_collapsed_count : atomic<u32>,
    perturb_escaped_count : atomic<u32>,
    max_iter_reached_count : atomic<u32>,
    absolute_fallback_count : atomic<u32>,
    absolute_escaped_count : atomic<u32>,
};

@group(0) @binding(3)
var<storage, read_write> orbit_feedback : array<OrbitFeedback>;

// Perturbance feedback flags
const PERTURB_ATTEMPTED = 1u << 0; // perturbation path taken
const PERTURB_VALID     = 1u << 1; // perturbation stayed valid to user max_iter
const PERTURB_COLLAPSED = 1u << 2; // |dz| exceeded validity radius
const PERTURB_ESCAPED   = 1u << 3; // escaped during perturb
const MAX_ITER_REACHED  = 1u << 4; // User max_iter reached
const ABSOLUTE_FALLBACK = 1u << 5; // required absolute continuation
const ABSOLUTE_ESCAPED  = 1u << 6; // escaped during absolute iteration

// ---------------------------------------------
// Compute entry point
// ---------------------------------------------
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let pix = vec2<i32>(i32(gid.x), i32(gid.y));

    // Bounds check against texture size
    let dims = textureDimensions(last_valid_i_tex);
    if (pix.x >= i32(dims.x) || pix.y >= i32(dims.y)) {
        return;
    }

    let last_valid_i = textureLoad(last_valid_i_tex, pix, 0).x;
    let orbit_idx    = textureLoad(orbit_idx_tex,    pix, 0).x;
    let flags        = textureLoad(flags_tex,        pix, 0).x;

    if (orbit_idx >= arrayLength(&orbit_feedback)) {
        return;
    }

    // Reduce last_valid_i
    atomicMin(&orbit_feedback[orbit_idx].min_last_valid_i, last_valid_i);
    atomicMax(&orbit_feedback[orbit_idx].max_last_valid_i, last_valid_i);

    // Aggregated perturbance flag counts
    if ((flags & PERTURB_ATTEMPTED) != 0u) {
        atomicAdd(&orbit_feedback[orbit_idx].perturb_attempted_count, 1u);
    }
    if ((flags & PERTURB_VALID) != 0u) {
        atomicAdd(&orbit_feedback[orbit_idx].perturb_valid_count, 1u);
    }
    if ((flags & PERTURB_COLLAPSED) != 0u) {
        atomicAdd(&orbit_feedback[orbit_idx].perturb_collapsed_count, 1u);
    }
    if ((flags & PERTURB_ESCAPED) != 0u) {
        atomicAdd(&orbit_feedback[orbit_idx].perturb_escaped_count, 1u);
    }
    if ((flags & MAX_ITER_REACHED) != 0u) {
        atomicAdd(&orbit_feedback[orbit_idx].max_iter_reached_count, 1u);
    }
    if ((flags & ABSOLUTE_FALLBACK) != 0u) {
        atomicAdd(&orbit_feedback[orbit_idx].absolute_fallback_count, 1u);
    }
    if ((flags & ABSOLUTE_ESCAPED) != 0u) {
        atomicAdd(&orbit_feedback[orbit_idx].absolute_escaped_count, 1u);
    }
}
