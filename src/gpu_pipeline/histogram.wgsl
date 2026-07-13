// -------------------------------------------------------------------------
// Histogram-equalization passes (adaptive coloring).
//
// Four @compute entry points share one bind group (see build_histogram_pipeline
// in builder.rs). They run between the reduce pass and the color pass:
//   clear  -> zero the bins and seed the min/max range
//   minmax -> atomic min/max of the included fi into hist_range
//   build  -> atomicAdd each included pixel into its bin
//   scan   -> single-thread box-blur + prefix-sum -> normalized CDF, temporal EMA
//
// The color pass then maps each pixel's fi -> CDF, so palette colors spread
// evenly over the actual on-screen escape-time distribution instead of being
// crammed by the raw linear `fi / max_iter`.
//
// Tier-3 dials (all consumed here): hist_bin_count (active bins <= HIST_BINS),
// hist_blur_radius (box blur over bins), hist_log_binning (bin in log space for
// heavy tails), hist_include_interior (also bin max-iter pixels).
//
// WGSL has no f32 atomics, but a reference/escape iteration count is >= 0, and
// the bit pattern of a non-negative f32 is monotonic in the value, so
// atomicMin/atomicMax on bitcast<u32>(fi) give a correct min/max.
// -------------------------------------------------------------------------

// Allocation / max bin count. The ACTIVE count is uni.hist_bin_count (<= this).
// Must match HIST_BINS in builder.rs and color.wgsl.
const HIST_BINS: u32 = 1024u;

// Full SceneUniform layout (kept in sync with color.wgsl / structs.rs). Only a
// few fields are read here, but declaring the whole struct keeps the tail fields
// at the correct offsets.
struct Uniforms {
    center_x:           f32,
    center_x_exp:       i32,
    center_y:           f32,
    center_y_exp:       i32,
    scale:              f32,
    scale_exp:          i32,
    max_iter:           u32,
    ref_orb_count:      u32,
    perturb_err_thresh: f32,
    view_width:         f32,
    view_height:        f32,
    render_width:       u32,
    render_height:      u32,
    render_tex_width:   f32,
    render_tex_height:  f32,
    grid_size:          u32,
    grid_width:         u32,
    sample_count:       u32,
    jitter_strength:    f32,
    sample_avg_bias:    f32,
    render_flags:       u32,
    stripe_trap_arg1:   f32,
    stripe_trap_arg2:   f32,
    stripe_trap_arg3:   f32,
    stripe_trap_arg4:   f32,
    trap_shape:          u32,
    trap_palette_cycles: f32,
    trap_iter_skip_frac: f32,
    formula_power:       u32,
    julia_c_re:          f32,
    julia_c_im:          f32,
    rot_cos:            f32,
    rot_sin:            f32,
    stateful_kind:      u32,
    color_scalar_mapping_mode:      u32,
    color_scaler_mapping_strength:  f32,
    palette_tex_width:  u32,
    palette_len:        u32,
    palette_cycles:     f32,
    palette_offset:     f32,
    palette_gamma:      f32,
    distance_multiplier:        f32,
    glow_intensity:             f32,
    neighbor_scale_multiplier:  f32,
    ambient_intensity:          f32,
    key_light_intensity:        f32,
    key_light_azimuth:          f32,
    key_light_elevation:        f32,
    fill_light_intensity:       f32,
    fill_light_azimuth:         f32,
    fill_light_elevation:       f32,
    specular_intensity:         f32,
    specular_power:             f32,
    ao_darkness:                f32,
    rim_intensity:              f32,
    rim_power:                  f32,
    hist_eq_amount:             f32,
    hist_black_pct:             f32,
    hist_white_pct:             f32,
    hist_temporal_alpha:        f32,
    hist_bin_count:             u32,
    hist_blur_radius:           u32,
    hist_log_binning:           u32,   // 0/1
    hist_include_interior:      u32,   // 0/1
    palette_interp_mode:        u32,
    _pad_palette0:              u32,
    _pad_palette1:              u32,
    _pad_palette2:              u32,
};
@group(0) @binding(0) var<uniform> uni: Uniforms;

// Per-pixel fractal results: .x = fi (fractional iters), .w = flags.
@group(0) @binding(1) var calc_tex: texture_2d<f32>;

// [0] = min fi bits, [1] = max fi bits (ordered u32 view of non-negative f32).
@group(0) @binding(2) var<storage, read_write> hist_range: array<atomic<u32>, 2>;

// Per-bin counts.
@group(0) @binding(3) var<storage, read_write> hist_bins: array<atomic<u32>, HIST_BINS>;

// Previous frame's CDF (for temporal EMA / freeze retention).
@group(0) @binding(4) var<storage, read> hist_cdf_prev: array<f32, HIST_BINS>;

// This frame's normalized CDF, consumed by the color pass.
@group(0) @binding(5) var<storage, read_write> hist_cdf: array<f32, HIST_BINS>;

const ESCAPED_BIT: u32 = 1u << 0u;
// +inf in ordered-u32 space: larger than any finite non-negative float, so it
// is the correct seed for an atomicMin over escape-time bit patterns.
const F32_POS_INF_BITS: u32 = 0x7F800000u;

// A pixel is included in the histogram if it escaped, or if interior pixels are
// being counted too (they carry fi = max_iter, so they pile into the top bin).
fn is_included(pix: vec2i) -> bool {
    let flags = u32(textureLoad(calc_tex, pix, 0).w);
    let escaped = (flags & ESCAPED_BIT) != 0u;
    return escaped || (uni.hist_include_interior != 0u);
}

fn in_bounds(pix: vec2i) -> bool {
    return pix.x < i32(uni.render_width) && pix.y < i32(uni.render_height);
}

// Position of fi within the observed [lo, hi] range, in [0,1]. Log-domain when
// enabled (concentrates bins over the heavy tail). MUST match color.wgsl.
fn bin_frac(fi: f32, lo: f32, hi: f32) -> f32 {
    if (uni.hist_log_binning != 0u) {
        let a   = log(1.0 + fi);
        let alo = log(1.0 + lo);
        let ahi = log(1.0 + hi);
        return clamp((a - alo) / max(ahi - alo, 1e-20), 0.0, 1.0);
    }
    return clamp((fi - lo) / max(hi - lo, 1e-20), 0.0, 1.0);
}

fn active_bins() -> u32 {
    return clamp(uni.hist_bin_count, 1u, HIST_BINS);
}

// -------------------------------------------------------------------------
// clear: zero all bins (up to the max) and seed the range. Dispatch enough
// 64-wide groups to cover HIST_BINS (thread 0 also seeds the min/max range).
// -------------------------------------------------------------------------
@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < HIST_BINS) {
        atomicStore(&hist_bins[i], 0u);
    }
    if (i == 0u) {
        atomicStore(&hist_range[0], F32_POS_INF_BITS);
        atomicStore(&hist_range[1], 0u);
    }
}

// -------------------------------------------------------------------------
// minmax: observed [fi_lo, fi_hi] over included pixels.
// -------------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn minmax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pix = vec2i(i32(gid.x), i32(gid.y));
    if (!in_bounds(pix) || !is_included(pix)) {
        return;
    }
    let bits = bitcast<u32>(textureLoad(calc_tex, pix, 0).x);
    atomicMin(&hist_range[0], bits);
    atomicMax(&hist_range[1], bits);
}

// -------------------------------------------------------------------------
// build: bin each included pixel into [fi_lo, fi_hi] (linear or log domain).
// -------------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pix = vec2i(i32(gid.x), i32(gid.y));
    if (!in_bounds(pix) || !is_included(pix)) {
        return;
    }
    let fi = textureLoad(calc_tex, pix, 0).x;
    let lo = bitcast<f32>(atomicLoad(&hist_range[0]));
    let hi = bitcast<f32>(atomicLoad(&hist_range[1]));
    if (hi <= lo) {
        return;
    }
    let n = active_bins();
    let bin = min(u32(bin_frac(fi, lo, hi) * f32(n)), n - 1u);
    atomicAdd(&hist_bins[bin], 1u);
}

// -------------------------------------------------------------------------
// scan: single-thread. Optional box blur over the bins (radius hist_blur_radius,
// via prefix-window sums), then a normalized prefix-sum -> CDF, blended with the
// previous frame by the temporal alpha. Runs once per frame over active bins.
// -------------------------------------------------------------------------
var<workgroup> ws_pre: array<f32, HIST_BINS>;  // prefix sums of raw bin counts

// Box-blurred count at bin i (radius r), from the prefix-sum window.
fn blurred_at(i: i32, r: i32, n: i32) -> f32 {
    let lo = max(i - r, 0);
    let hi = min(i + r, n - 1);
    var plo: f32 = 0.0;
    if (lo > 0) {
        plo = ws_pre[u32(lo) - 1u];
    }
    return (ws_pre[u32(hi)] - plo) / f32(hi - lo + 1);
}

@compute @workgroup_size(1)
fn scan(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    let n = active_bins();
    let ni = i32(n);
    let r = i32(min(uni.hist_blur_radius, HIST_BINS));
    let alpha = clamp(uni.hist_temporal_alpha, 0.0, 1.0);

    // Prefix sums of the raw counts (for O(n) box-blur windows).
    var run: f32 = 0.0;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        run = run + f32(atomicLoad(&hist_bins[i]));
        ws_pre[i] = run;
    }

    // Total of the (blurred) counts. r = 0 leaves counts unchanged.
    var total: f32 = 0.0;
    for (var i: i32 = 0; i < ni; i = i + 1) {
        total = total + blurred_at(i, r, ni);
    }

    // No included pixels: decay toward zero; the color pass falls back to linear.
    if (total <= 0.0) {
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            hist_cdf[i] = mix(hist_cdf_prev[i], 0.0, alpha);
        }
        return;
    }

    let inv_total = 1.0 / total;
    var cum: f32 = 0.0;
    for (var i: i32 = 0; i < ni; i = i + 1) {
        cum = cum + blurred_at(i, r, ni);
        hist_cdf[u32(i)] = mix(hist_cdf_prev[u32(i)], cum * inv_total, alpha);
    }
}
