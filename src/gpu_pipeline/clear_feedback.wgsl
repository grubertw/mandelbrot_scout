@group(0) @binding(0)
var flags_tex : texture_storage_2d<r32uint, write>;

// Aggregated screen-grid output (atomics)
struct GridFeedback {
    best_pixel_x:           atomic<i32>, // Pixel location of deepest iteration in the sample grid
    best_pixel_y:           atomic<i32>,
    best_pixel_flags:       atomic<u32>, // Iteration feedback flags for the location
    max_iter_count:         atomic<u32>, // Also needed here to find best pixel
};

@group(0) @binding(1)
var<storage, read_write> grid_feedback : array<GridFeedback>;

struct TileFeedback {
    min_iter_count:         atomic<u32>,
    max_iter_count:         atomic<u32>,
    escaped_count:          atomic<u32>,
    perurb_error_count:     atomic<u32>,
    max_iter_reached_count: atomic<u32>,
};
@group(0) @binding(2)
var<storage, read_write> tile_feedback : array<TileFeedback>;

const MAX_U32 : u32 = 0xFFFFFFFFu;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);

    // Clear per-pixel textures
    textureStore(flags_tex, vec2<i32>(x, y), vec4<u32>(0u, 0u, 0u, 0u));

    // Clear feedback (1D mapping)
    let grid_idx = gid.y * 65536u + gid.x;
    if (grid_idx < arrayLength(&grid_feedback)) {
        atomicStore(&grid_feedback[grid_idx].best_pixel_x, -1);
        atomicStore(&grid_feedback[grid_idx].best_pixel_y, -1);
        atomicStore(&grid_feedback[grid_idx].best_pixel_flags, 0u);
        atomicStore(&grid_feedback[grid_idx].max_iter_count, 0u);
    }
    
    let tile_idx = gid.y * 65536u + gid.x;
    if (tile_idx < arrayLength(&tile_feedback)) {
        atomicStore(&tile_feedback[tile_idx].min_iter_count, MAX_U32);
        atomicStore(&tile_feedback[tile_idx].max_iter_count, 0u);
        atomicStore(&tile_feedback[tile_idx].escaped_count, 0u);
        atomicStore(&tile_feedback[tile_idx].perurb_error_count, 0u);
        atomicStore(&tile_feedback[tile_idx].max_iter_reached_count, 0u);
    }
}
