// ---------------------------------------------
// Bind Group 0: Reduction inputs + outputs
// ---------------------------------------------

struct ReduceUniform {
    screen_width:  u32,
    screen_height: u32,
    grid_size:     u32,   // e.g. 64
    grid_width:    u32,   // screen_width / grid_size
};

@group(0) @binding(0)
var<uniform> reduce_uni : ReduceUniform;

// Per-pixel inputs (read-only)
@group(0) @binding(1)
var flags_tex : texture_2d<u32>;

// Aggregated screen-grid output (atomics)
struct GridFeedback {
    best_pixel_x:           atomic<i32>, // Pixel location of deepest iteration in the sample grid
    best_pixel_y:           atomic<i32>,
    best_pixel_flags:       atomic<u32>, // Iteration feedback flags for the location
    max_iter_count:         atomic<u32>, // Also needed here to find best pixel
};
@group(0) @binding(2)
var<storage, read_write> grid_feedback : array<GridFeedback>;

struct TileFeedback {
    min_iter_count:         atomic<u32>,
    max_iter_count:         atomic<u32>,
    escaped_count:          atomic<u32>,
    perurb_error_count:     atomic<u32>,
    max_iter_reached_count: atomic<u32>,
};
@group(0) @binding(3)
var<storage, read_write> tile_feedback : array<TileFeedback>;

const ITER_MASK: u32        = 0x0000FFFFu;
const ESCAPED_BIT: u32      = 1u << 16u;
const PERTURB_BIT: u32      = 1u << 17u;
const PERTURB_ERR_BIT: u32  = 1u << 18u;
const MAX_ITER_BIT: u32     = 1u << 19u;
const TILE_SHIFT: u32       = 20u;

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
    let perturb_err = (flags & PERTURB_ERR_BIT) != 0u;
    let max_iter_reached = (flags & MAX_ITER_BIT) != 0u;
    let tile_idx = flags >> TILE_SHIFT;

    // --- Screen-grid index ---
    let grid_x = u32(pix.x) / reduce_uni.grid_size;
    let grid_y = u32(pix.y) / reduce_uni.grid_size;
    let grid_idx = grid_y * reduce_uni.grid_width + grid_x;

    if (grid_idx >= arrayLength(&grid_feedback)) {
        return;
    }

    // Track max iters in GridFeedback to find best orbit location
    let prev_max = atomicMax(&grid_feedback[grid_idx].max_iter_count, iter);

    // If we won the max race, record pixel as best
    if (iter > prev_max) {
        atomicStore(&grid_feedback[grid_idx].best_pixel_x, pix.x);
        atomicStore(&grid_feedback[grid_idx].best_pixel_y, pix.y);
        atomicStore(&grid_feedback[grid_idx].best_pixel_flags, flags);
    }

    if (used_perturb && tile_idx < arrayLength(&tile_feedback)) {
        atomicMin(&tile_feedback[tile_idx].min_iter_count, iter);
        atomicMax(&tile_feedback[tile_idx].max_iter_count, iter);

        if (escaped) {
            atomicAdd(&tile_feedback[tile_idx].escaped_count, 1u);
        }
        if (perturb_err) {
            atomicAdd(&tile_feedback[tile_idx].perurb_error_count, 1u);
        }
        if (max_iter_reached) {
            atomicAdd(&tile_feedback[tile_idx].max_iter_reached_count, 1u);
        }
    }
}
