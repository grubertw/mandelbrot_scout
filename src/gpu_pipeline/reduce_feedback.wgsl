// ---------------------------------------------
// Bind Group 0: Reduction inputs + outputs
// ---------------------------------------------

// Per-pixel inputs (read-only)
@group(0) @binding(0)
var flags_tex : texture_2d<u32>;

// Aggregated per-orbit output (atomics)
struct TileFeedback {
    min_iter_count : atomic<u32>,
    max_iter_count : atomic<u32>,
    escaped_count : atomic<u32>,
    pertub_used_count : atomic<u32>,
    max_iter_reached_count: atomic<u32>,
};

@group(0) @binding(1)
var<storage, read_write> tile_feedback : array<TileFeedback>;

const ITER_MASK: u32 = 0x0000FFFFu;
const ESCAPED_BIT: u32 = 1u << 16u;
const PERTURB_BIT: u32 = 1u << 17u;
const MAX_ITER_BIT: u32 = 1u << 18u;
const TILE_SHIFT: u32 = 19u;

// ---------------------------------------------
// Compute entry point
// ---------------------------------------------
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let pix = vec2<i32>(i32(gid.x), i32(gid.y));

    // Bounds check against texture size
    let dims = textureDimensions(flags_tex);
    if (pix.x >= i32(dims.x) || pix.y >= i32(dims.y)) {
        return;
    }

    let flags = textureLoad(flags_tex, pix, 0).x;

    let iter = flags & ITER_MASK;
    let escaped = (flags & ESCAPED_BIT) != 0u;
    let used_perturb = (flags & PERTURB_BIT) != 0u;
    let max_iter_reached = (flags & MAX_ITER_BIT) != 0u;
    let tile_idx = flags >> TILE_SHIFT;

    if (tile_idx >= arrayLength(&tile_feedback)) {
        return;
    }

    // Track the minimum and maximum number of iterations performed
    // in the tile, against the anchor orbit. Along with the escaped
    // count, this gives a good indication of the complexity of the 
    // geometry within the tile, and may help ScoutEngine deside 
    // whether to split the tile.
    atomicMin(&tile_feedback[tile_idx].min_iter_count, iter);
    atomicMax(&tile_feedback[tile_idx].max_iter_count, iter);

    // Aggregated perturbance flag counts
    if (escaped) {
        atomicAdd(&tile_feedback[tile_idx].escaped_count, 1u);
    }
    if (used_perturb) {
        atomicAdd(&tile_feedback[tile_idx].pertub_used_count, 1u);
    }
    if (max_iter_reached) {
        atomicAdd(&tile_feedback[tile_idx].max_iter_reached_count, 1u);
    }
}
