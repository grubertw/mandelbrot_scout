@group(0) @binding(0)
var flags_tex : texture_storage_2d<r32uint, write>;

struct TileFeedback {
    min_iter_count : atomic<u32>,
    max_iter_count : atomic<u32>,
    escaped_count : atomic<u32>,
    pertub_used_count : atomic<u32>,
    max_iter_reached_count: atomic<u32>,
};

@group(0) @binding(1)
var<storage, read_write> tile_feedback : array<TileFeedback>;

const MAX_U32 : u32 = 0xFFFFFFFFu;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);

    // Clear per-pixel textures
    textureStore(flags_tex, vec2<i32>(x, y), vec4<u32>(0u, 0u, 0u, 0u));

    // Clear per-orbit feedback (1D mapping)
    let tile_idx = gid.y * 65536u + gid.x;
    if (tile_idx < arrayLength(&tile_feedback)) {
        atomicStore(&tile_feedback[tile_idx].min_iter_count, MAX_U32);
        atomicStore(&tile_feedback[tile_idx].max_iter_count, 0u);
        atomicStore(&tile_feedback[tile_idx].escaped_count, 0u);
        atomicStore(&tile_feedback[tile_idx].pertub_used_count, 0u);
        atomicStore(&tile_feedback[tile_idx].max_iter_reached_count, 0u);
    }
}
