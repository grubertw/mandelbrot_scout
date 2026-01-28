
@group(0) @binding(0)
var last_valid_i_tex : texture_storage_2d<r32uint, write>;

@group(0) @binding(1)
var orbit_idx_tex : texture_storage_2d<r32uint, write>;

@group(0) @binding(2)
var flags_tex : texture_storage_2d<r32uint, write>;

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

const MAX_U32 : u32 = 0xFFFFFFFFu;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);

    // Clear per-pixel textures
    textureStore(last_valid_i_tex, vec2<i32>(x, y), vec4<u32>(MAX_U32, 0u, 0u, 0u));
    textureStore(orbit_idx_tex,    vec2<i32>(x, y), vec4<u32>(MAX_U32, 0u, 0u, 0u));
    textureStore(flags_tex,        vec2<i32>(x, y), vec4<u32>(0u,      0u, 0u, 0u));

    // Clear per-orbit feedback (1D mapping)
    let orbit_idx = gid.y * 65536u + gid.x;
    if (orbit_idx < arrayLength(&orbit_feedback)) {
        atomicStore(&orbit_feedback[orbit_idx].min_last_valid_i, MAX_U32);
        atomicStore(&orbit_feedback[orbit_idx].max_last_valid_i, 0u);
        atomicStore(&orbit_feedback[orbit_idx].perturb_attempted_count, 0u);
        atomicStore(&orbit_feedback[orbit_idx].perturb_valid_count, 0u);
        atomicStore(&orbit_feedback[orbit_idx].perturb_collapsed_count, 0u);
        atomicStore(&orbit_feedback[orbit_idx].perturb_escaped_count, 0u);
        atomicStore(&orbit_feedback[orbit_idx].max_iter_reached_count, 0u);
        atomicStore(&orbit_feedback[orbit_idx].absolute_fallback_count, 0u);
        atomicStore(&orbit_feedback[orbit_idx].absolute_escaped_count, 0u);
    }
}
