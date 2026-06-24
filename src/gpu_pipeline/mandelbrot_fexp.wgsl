// Float-exp Mandelbrot shader.
// Each real value is stored as { m: f32, e: i32 } representing m * 2^e.
// Invariant (matching Fraktaler-3's floatexp): the mantissa is renormalized
// into [0.5,1) after EVERY operation — multiply, scale, and add — via frexp.
// A true zero is the one exception (m == 0), handled explicitly in fexp_add.
// Keeping a single normalized invariant is what makes the exponent-based
// comparisons and escape test correct.

// -------------------------------
// Render flags (must match render_flags bit assignments in structs.rs)
// -------------------------------
const DEBUG_COLORING: u32   = 1u << 0u;
const GLITCH_FIX: u32       = 1u << 1u;
const SMOOTH_COLORING: u32  = 1u << 2u;
const USE_DE: u32           = 1u << 3u;
const USE_STRIPES: u32      = 1u << 4u;
const USE_FEXP: u32         = 1u << 11u;
const USE_BLA: u32          = 1u << 12u;

// Off-center debug probe (fraction of render size). The scout's reference orbit
// usually sits near viewport center, so the dead-center pixel has a near-zero dz
// and exercises neither rebasing nor BLA. An off-center pixel is far more
// representative. The offset is a sub-view fraction, so the probe's c still
// matches the viewport center to well within the logged precision at deep zoom.
const PROBE_X_FRAC: f32 = 0.66;
const PROBE_Y_FRAC: f32 = 0.66;

const ESCAPED_BIT: u32      = 1u << 0u;
const PERTURB_BIT: u32      = 1u << 1u;
const PERTURB_ERR_BIT: u32  = 1u << 2u;
const MAX_ITER_BIT: u32     = 1u << 3u;
const ORBIT_SHIFT: u32      = 20u;

// -------------------------------
// Float-exp types
// -------------------------------
struct FExp {
    m: f32,
    e: i32,
}

struct ComplexFExp {
    re: FExp,
    im: FExp,
}

// One BLA table entry: dz_out = a*dz + b*dc, valid for `l` reference iterations.
// Layout MUST match BlaEntry in gpu_pipeline/structs.rs (a@0, b@16, l@32, 36 bytes).
struct BlaEntry {
    a: ComplexFExp,
    b: ComplexFExp,
    l: u32,
}

// -------------------------------
// Float-exp scalar arithmetic
// -------------------------------

// Multiply: normalize result so mantissa stays in [-1, 1).
fn fexp_mul(a: FExp, b: FExp) -> FExp {
    let n = frexp(a.m * b.m);
    return FExp(n.fract, a.e + b.e + n.exp);
}

// Scale an FExp by a plain f32 (e.g., a pixel offset) — normalize result.
fn fexp_scale(f: f32, a: FExp) -> FExp {
    let n = frexp(f * a.m);
    return FExp(n.fract, a.e + n.exp);
}

// Add: align to the larger exponent, sum, then RENORMALIZE the mantissa back
// into [0.5,1) via frexp — the same invariant Fraktaler-3 keeps after every op.
// Renormalizing is what makes the exponent-first comparisons (fexp_gt_pos and
// the escape test) valid; without it those compares see [0.5,2) mantissas and
// can rank values backwards, producing spurious rebases / radial artifacts.
fn fexp_add(a: FExp, b: FExp) -> FExp {
    // A true zero carries no meaningful exponent — treat it as -inf so the
    // other operand always dominates the alignment. Without this, dz == 0
    // (stored with e == 0) swallows a tiny delta_c at deep zoom: the alignment
    // does ldexp(delta_c, -130), which flushes to subnormal-zero on the GPU and
    // the perturbation is never seeded. That was the ~1e-39 failure.
    if (a.m == 0.0) { return b; }
    if (b.m == 0.0) { return a; }

    var sum: f32;
    var e: i32;
    if (a.e >= b.e) {
        sum = a.m + ldexp(b.m, b.e - a.e);
        e = a.e;
    } else {
        sum = ldexp(a.m, a.e - b.e) + b.m;
        e = b.e;
    }
    // frexp(0.0) -> (0.0, 0), so a fully-cancelling sum returns a clean zero
    // that the short-circuit above will absorb on the next add.
    let n = frexp(sum);
    return FExp(n.fract, e + n.exp);
}

fn fexp_sub(a: FExp, b: FExp) -> FExp {
    return fexp_add(a, FExp(-b.m, b.e));
}

// Reconstruct as f32. Only call when |value| is known to be within f32 range.
fn fexp_to_f32(a: FExp) -> f32 {
    return ldexp(a.m, a.e);
}

// Lift a plain f32 into a normalized FExp. frexp(0.0) -> (0.0, 0), giving our
// canonical zero (absorbed by the zero short-circuit in fexp_add).
fn fexp_from_f32(x: f32) -> FExp {
    let n = frexp(x);
    return FExp(n.fract, n.exp);
}

// -------------------------------
// Complex float-exp arithmetic
// -------------------------------

fn cfexp_add(a: ComplexFExp, b: ComplexFExp) -> ComplexFExp {
    return ComplexFExp(fexp_add(a.re, b.re), fexp_add(a.im, b.im));
}

fn cfexp_sub(a: ComplexFExp, b: ComplexFExp) -> ComplexFExp {
    return ComplexFExp(fexp_sub(a.re, b.re), fexp_sub(a.im, b.im));
}

// (ar + i*ai)(br + i*bi) = (ar*br - ai*bi) + i*(ar*bi + ai*br)
fn cfexp_mul(a: ComplexFExp, b: ComplexFExp) -> ComplexFExp {
    let rr = fexp_mul(a.re, b.re);
    let ii = fexp_mul(a.im, b.im);
    let ri = fexp_mul(a.re, b.im);
    let ir = fexp_mul(a.im, b.re);
    return ComplexFExp(fexp_sub(rr, ii), fexp_add(ri, ir));
}

// Squared magnitude as f32. Only safe when |z| is known to be in f32 normal range.
fn cfexp_mag2(z: ComplexFExp) -> f32 {
    let re = fexp_to_f32(z.re);
    let im = fexp_to_f32(z.im);
    return re * re + im * im;
}

// Squared magnitude as FExp. Correct at any scale — fexp_mul tracks the exponent
// without squaring through f32, so no underflow even when z.e << 0.
fn cfexp_mag2_fexp(z: ComplexFExp) -> FExp {
    let re2 = fexp_mul(z.re, z.re);
    let im2 = fexp_mul(z.im, z.im);
    return fexp_add(re2, im2);
}

// Greater-than for non-negative FExp values (mantissas >= 0).
fn fexp_gt_pos(a: FExp, b: FExp) -> bool {
    if (b.m == 0.0) { return a.m != 0.0; }
    if (a.m == 0.0) { return false; }
    if (a.e != b.e) { return a.e > b.e; }
    return a.m > b.m;
}

fn cfexp_to_vec2f(z: ComplexFExp) -> vec2f {
    return vec2f(fexp_to_f32(z.re), fexp_to_f32(z.im));
}

// -------------------------------
// Uniforms
// -------------------------------
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
    stripe_density:     f32,
    stripe_strength:    f32,
    stripe_gamma:       f32,
};
@group(0) @binding(0) var<uniform> uni: Uniforms;

@group(0) @binding(1) var noise_tex: texture_2d<f32>;

@group(0) @binding(2)
var calc_out_tex: texture_storage_2d<rgba32float, write>;

// -------------------------------
// Reference orbit bindings
// -------------------------------
struct GpuRefOrbitLocation {
    c_ref_re:               f32,
    c_ref_im:               f32,
    max_ref_iters:          u32,
    center_offset_re:       f32,
    center_offset_re_exp:   i32,
    center_offset_im:       f32,
    center_offset_im_exp:   i32,
};
@group(0) @binding(3) var<storage, read> orbit_location: array<GpuRefOrbitLocation>;

// Reference orbit points stored as f32 (vec2f, 8 bytes each) — the SAME buffers
// the f32 shader uses. |Z_n| is bounded by the escape radius, so f32 is plenty;
// each point is lifted to FExp on load. Only the deltas need extended range.
@group(0) @binding(4) var<storage, read> rank_one_orbit: array<vec2f>;
@group(0) @binding(5) var<storage, read> rank_two_orbit: array<vec2f>;

// -------------------------------
// BLA table (rank-1 orbit). a/b/l in bla_entries, validity r^2 in bla_radii
// (same flat order), and a small dims header that lets us index both:
//   bla_dims = [level_count, steps(M), off_0=0, off_1, ..., off_L = total]
// offset(level) = bla_dims[2 + level],  count(level) = bla_dims[3+level] - offset.
// All three may be zero-filled (no table yet) — bla_lookup then finds nothing.
// -------------------------------
@group(0) @binding(6) var<storage, read> bla_entries: array<BlaEntry>;
@group(0) @binding(7) var<storage, read> bla_radii:   array<FExp>;
@group(0) @binding(8) var<storage, read> bla_dims:    array<u32>;

// -------------------------------
// Jitter helpers (identical to f32 shader)
// -------------------------------
fn wrap(coord: vec2i, size: vec2i) -> vec2i {
    return vec2i(
        ((coord.x % size.x) + size.x) % size.x,
        ((coord.y % size.y) + size.y) % size.y,
    );
}

fn hash2(p: vec2i, s: u32) -> vec2i {
    let h = u32(p.x) * 1664525u + u32(p.y) * 1013904223u + s * 374761393u;
    return vec2i(i32((h >> 8u) & 127u), i32((h >> 16u) & 127u));
}

fn get_jitter(pix: vec2i, sample_idx: u32) -> vec2f {
    let tex_size = vec2i(textureDimensions(noise_tex));
    let coord = wrap(pix + hash2(pix, sample_idx), tex_size);
    let noise = textureLoad(noise_tex, coord, 0).xy;
    return noise * 2.0 - 1.0;
}

// -------------------------------
// Scene-coordinate helpers
// -------------------------------

// Pixel offset into scene space: (normalized pixel coord) * view_size * scale.
// cu * view_width is O(screen_px) — well within f32 — so fexp_scale is exact.
fn build_c_from_scene(pix: vec2i, jitter: vec2f) -> ComplexFExp {
    let jittered = vec2f(pix) + jitter * uni.jitter_strength;
    let cu = jittered.x / f32(uni.render_width)  - 0.5;
    let cv = jittered.y / f32(uni.render_height) - 0.5;

    let scale = FExp(uni.scale, uni.scale_exp);
    let off_x = fexp_scale(cu * uni.view_width,  scale);
    let off_y = fexp_scale(cv * uni.view_height, scale);

    return ComplexFExp(
        fexp_add(FExp(uni.center_x, uni.center_x_exp), off_x),
        fexp_add(FExp(uni.center_y, uni.center_y_exp), off_y),
    );
}

// Delta from a reference orbit's starting point to this pixel.
// center_offset is precomputed on the CPU as (scene_center - c_ref) in FExp.
fn build_delta_c_from_orbit_location(pix: vec2i, orbit_idx: u32, jitter: vec2f) -> ComplexFExp {
    let jittered = vec2f(pix) + jitter * uni.jitter_strength;
    let cu = jittered.x / f32(uni.render_width)  - 0.5;
    let cv = jittered.y / f32(uni.render_height) - 0.5;

    let scale  = FExp(uni.scale, uni.scale_exp);
    let off_x  = fexp_scale(cu * uni.view_width,  scale);
    let off_y  = fexp_scale(cv * uni.view_height, scale);
    let orbit  = orbit_location[orbit_idx];

    return ComplexFExp(
        fexp_add(FExp(orbit.center_offset_re, orbit.center_offset_re_exp), off_x),
        fexp_add(FExp(orbit.center_offset_im, orbit.center_offset_im_exp), off_y),
    );
}

fn load_ref_orbit(orbit_idx: u32, it: u32) -> ComplexFExp {
    var z = vec2f(0.0, 0.0);
    if (orbit_idx == 0u) { z = rank_one_orbit[it]; }
    else if (orbit_idx == 1u) { z = rank_two_orbit[it]; }
    return ComplexFExp(fexp_from_f32(z.x), fexp_from_f32(z.y));
}

// -------------------------------
// Direct Mandelbrot (no reference orbit available)
// -------------------------------
fn mandelbrot(c: ComplexFExp) -> vec4f {
    var z = ComplexFExp(FExp(0.0, 0), FExp(0.0, 0));
    var i: u32 = 0u;
    let max_i = uni.max_iter;
    var mag_z: f32       = 0.0;
    var escape_mag_z: f32 = 0.0;
    var dz_de = vec2f(0.0, 0.0); // d(z)/d(c) in f32 for DE, grows but only used for color
    var extra: u32       = 0u;
    var stripe_sum: f32  = 0.0;
    var stripe_count: f32 = 0.0;
    var flags: u32       = 0u;

    for (i = 0u; i < max_i; i++) {
        if ((uni.render_flags & USE_DE) != 0u) {
            let zf = cfexp_to_vec2f(z);
            dz_de = 2.0 * vec2f(zf.x * dz_de.x - zf.y * dz_de.y,
                                 zf.x * dz_de.y + zf.y * dz_de.x) + vec2f(1.0, 0.0);
        }

        z = cfexp_add(cfexp_mul(z, z), c);

        if ((uni.render_flags & USE_STRIPES) != 0u) {
            let zf = cfexp_to_vec2f(z);
            let angle = atan2(zf.y, zf.x);
            var stripe = 0.5 + 0.5 * sin(angle * uni.stripe_density);
            stripe = pow(stripe, uni.stripe_gamma);
            stripe_sum  += stripe;
            stripe_count += 1.0;
        }

        mag_z = cfexp_mag2(z);
        if (mag_z > 128.0) {
            flags |= ESCAPED_BIT;
            if (extra >= 2u) { break; }
            extra += 1u;
        }
        if ((flags & ESCAPED_BIT) == 0u) { escape_mag_z = mag_z; }
    }

    if (i == max_i) { flags |= MAX_ITER_BIT; }

    var fi = f32(i - extra);
    if ((uni.render_flags & SMOOTH_COLORING) != 0u) {
        let safe_mag = max(escape_mag_z, 1e-30);
        fi = clamp(fi + 1.0 - log2(log(safe_mag) * 0.5), 0.0, f32(max_i));
        if (i == max_i) { fi = f32(max_i); }
    }

    var distance: f32 = 0.0;
    if ((uni.render_flags & USE_DE) != 0u) {
        let r  = sqrt(mag_z);
        let dr = max(length(dz_de), 1e-30);
        distance = 0.5 * r * log(r) / dr;
    }

    var stripe_avg: f32 = 0.0;
    if ((uni.render_flags & USE_STRIPES) != 0u) {
        stripe_avg = stripe_sum / stripe_count;
    }

    return vec4f(fi, distance, stripe_avg, f32(flags));
}

// -------------------------------
// BLA lookup
// -------------------------------
struct BlaHit {
    found: bool,
    idx:   u32,   // flat index into bla_entries / bla_radii
    l:     u32,   // reference iterations this entry skips
}

// Largest valid BLA step starting at reference index m (1-based), given the
// current |dz|^2 = z2. Climbs levels while the entry stays aligned to m and z2
// is inside its radius, returning the highest such level (the biggest jump).
// Mirrors the CPU reference lookup() in scout_engine/bla.rs.
fn bla_lookup(m: u32, z2: FExp) -> BlaHit {
    var hit: BlaHit;
    hit.found = false;
    hit.idx   = 0u;
    hit.l     = 0u;

    let level_count = bla_dims[0];
    let steps       = bla_dims[1];
    if (m == 0u || m >= steps) { return hit; }

    var ix = m - 1u;
    for (var level = 0u; level < level_count; level = level + 1u) {
        let off   = bla_dims[2u + level];
        let count = bla_dims[3u + level] - off;
        if (ix >= count) { break; }
        let ixm = (ix << level) + 1u;
        // Aligned to m AND z2 within radius => record and keep climbing.
        if (m == ixm && fexp_gt_pos(bla_radii[off + ix], z2)) {
            hit.found = true;
            hit.idx   = off + ix;
            hit.l     = bla_entries[off + ix].l;
        } else {
            break;
        }
        ix = ix >> 1u;
    }
    return hit;
}

// -------------------------------
// Perturbed Mandelbrot
// dz_{n+1} = 2*Z_n*dz_n + dz_n^2 + delta_c
// -------------------------------
fn mandelbrot_perturb(delta_c: ComplexFExp,
                      rebase_count: ptr<function, u32>,
                      bla_max_step: ptr<function, u32>,
                      bla_step_count: ptr<function, u32>,
                      bla_iters_skipped: ptr<function, u32>) -> vec4f {
    var dz = ComplexFExp(FExp(0.0, 0), FExp(0.0, 0));
    var i: u32     = 0u;
    var ref_i: u32 = 0u;
    let max_i      = uni.max_iter;
    let max_ref_i  = orbit_location[0u].max_ref_iters;
    var mag_z: f32        = 0.0;
    var escape_mag_z: f32 = 0.0;
    var dzdc = vec2f(0.0, 0.0); // d(z)/d(c) in f32 for DE
    var z_f32 = vec2f(0.0, 0.0);
    var extra: u32        = 0u;
    var stripe_sum: f32   = 0.0;
    var stripe_count: f32 = 0.0;
    var flags: u32        = PERTURB_BIT;
    // Counts GLITCH rebases only (excludes benign end-of-reference wraps).
    // A growing count means the perturbed orbit keeps diverging from the
    // reference faster than the reference covers — the cheapest proxy we have
    // for precision/reference-match stress.
    *rebase_count = 0u;
    *bla_max_step = 0u;
    *bla_step_count = 0u;
    *bla_iters_skipped = 0u;

    // BLA accelerates only plain iteration/escape coloring: a jump skips l steps,
    // which would under-accumulate the per-iteration DE derivative and stripe
    // average. Gate it off when either is active — they then single-step (correct,
    // just unaccelerated). Uniform-constant, so hoisted out of the loop.
    let bla_enabled = ((uni.render_flags & USE_BLA) != 0u)
                   && ((uni.render_flags & USE_DE) == 0u)
                   && ((uni.render_flags & USE_STRIPES) == 0u);

    for (i = 0u; i < max_i; i++) {
        // Reconstruct the full value z = Z_n + dz with a VALID ref_i (always in
        // [0, max_ref_i-1]). Done before advancing, so we never read OOB and
        // never drop the Z_n term.
        let Z = load_ref_orbit(0u, ref_i);
        let z = cfexp_add(Z, dz);
        z_f32 = cfexp_to_vec2f(z);

        if ((uni.render_flags & USE_STRIPES) != 0u) {
            let angle = atan2(z_f32.y, z_f32.x);
            var stripe = 0.5 + 0.5 * sin(angle * uni.stripe_density);
            stripe = pow(stripe, uni.stripe_gamma);
            stripe_sum  += stripe;
            stripe_count += 1.0;
        }

        // Escape test in the FExp domain: |z|^2 >= 128  <=>  mag_z_fexp.e >= 8
        // (m in [0.5,1), so m*2^8 >= 128). The f32 reconstruction (mag_z) can
        // over/underflow at extreme exponents, so it must NOT drive escape — it
        // is only used for smooth coloring and distance estimation below.
        let mag_z_fexp  = cfexp_mag2_fexp(z);
        let mag_dz_fexp = cfexp_mag2_fexp(dz);
        mag_z = z_f32.x * z_f32.x + z_f32.y * z_f32.y;

        if ((mag_z_fexp.m != 0.0) && (mag_z_fexp.e >= 8)) {
            flags |= ESCAPED_BIT;
            if (extra >= 2u) { break; }
            extra += 1u;
        }
        if ((flags & ESCAPED_BIT) == 0u && mag_z < 1e30) { escape_mag_z = mag_z; }

        // Zhuoran rebasing, performed BEFORE advancing. Two independent triggers:
        //   1. End-of-reference (ref_i+1 >= max_ref_i): always on. Restarts the
        //      reference from orbit[0] (the critical point, = 0) with dz = z, so a
        //      reference shorter than max_iter stays valid.
        //   2. Glitch rebase (thresh*|dz|^2 > |z|^2): gated by GLITCH_FIX and tuned
        //      by perturb_err_thresh (= beta^2). A small beta only rebases once dz
        //      has genuinely overtaken z; the strict beta=1 over-rebases near a
        //      reference's zero crossings and produces false escapes.
        // Both set dz = z and ref_i = 0. Running with a valid ref_i keeps the Z_n
        // term and never reads OOB.
        let glitch_rebase = ((uni.render_flags & GLITCH_FIX) != 0u) &&
                            fexp_gt_pos(fexp_scale(uni.perturb_err_thresh, mag_dz_fexp), mag_z_fexp);
        var Z_adv = Z;
        if (glitch_rebase || (ref_i + 1u >= max_ref_i)) {
            dz    = z;
            ref_i = 0u;
            Z_adv = load_ref_orbit(0u, 0u);  // = 0 (critical point)
            flags |= PERTURB_ERR_BIT;
            if (glitch_rebase) { *rebase_count += 1u; }
        }

        // === BLA fast-forward (after rebase, before the single-step advance) ===
        // Replace a run of linear steps with a single dz = A*dz + B*delta_c. Only
        // fires while |dz|^2 is within the entry's validity radius, so it can never
        // skip an escape: a diverging pixel grows dz past every radius and falls
        // back to single stepping (where the escape test above catches it). After
        // a rebase ref_i == 0, so bla_lookup returns nothing — the rebased step is
        // taken as a normal single step. mag_dz_fexp is the pre-rebase |dz|^2,
        // which is exactly |dz|^2 when not rebased, and unused when rebased.
        if (bla_enabled) {
            let hit = bla_lookup(ref_i, mag_dz_fexp);
            if (hit.found && (i + hit.l) <= max_i) {
                let e = bla_entries[hit.idx];
                dz = cfexp_add(cfexp_mul(e.a, dz), cfexp_mul(e.b, delta_c));
                ref_i += hit.l;
                i     += hit.l - 1u;  // the loop's i++ adds the final step
                // Cheap, on-jump only (not per-iteration), so safe in the hot loop.
                *bla_max_step = max(*bla_max_step, hit.l);
                *bla_step_count += 1u;
                *bla_iters_skipped += hit.l;
                continue;
            }
        }

        if ((uni.render_flags & USE_DE) != 0u) {
            dzdc = 2.0 * vec2f(z_f32.x * dzdc.x - z_f32.y * dzdc.y,
                               z_f32.x * dzdc.y + z_f32.y * dzdc.x) + vec2f(1.0, 0.0);
        }

        // Advance: dz_{n+1} = 2*Z_adv*dz + dz^2 + delta_c
        let lambda = cfexp_add(Z_adv, Z_adv);
        dz = cfexp_add(
            cfexp_add(cfexp_mul(lambda, dz), cfexp_mul(dz, dz)),
            delta_c
        );
        ref_i += 1u;
    }

    if (i == max_i) { flags |= MAX_ITER_BIT; }

    var fi = f32(i - extra);
    if ((uni.render_flags & SMOOTH_COLORING) != 0u) {
        let safe_mag = max(escape_mag_z, 1e-30);
        fi = clamp(fi + 1.0 - log2(log(safe_mag) * 0.5), 0.0, f32(max_i));
        if (i == max_i) { fi = f32(max_i); }
    }

    var distance: f32 = 0.0;
    if ((uni.render_flags & USE_DE) != 0u) {
        let r  = sqrt(mag_z);
        let dr = max(length(dzdc), 1e-30);
        distance = 0.5 * r * log(r) / dr;
    }

    var stripe_avg: f32 = 0.0;
    if ((uni.render_flags & USE_STRIPES) != 0u) {
        stripe_avg = stripe_sum / stripe_count;
    }

    return vec4f(fi, distance, stripe_avg, f32(flags));
}

// -------------------------------
// Compute entry point
// -------------------------------
const OVERSAMPLE_GUARD = 50;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pix = vec2i(i32(gid.x), i32(gid.y));

    if (pix.x >= i32(uni.render_width + OVERSAMPLE_GUARD) ||
        pix.y >= i32(uni.render_height + OVERSAMPLE_GUARD)) {
        return;
    }

    var sample_idx: u32   = 0u;
    var max_iters: f32    = 0.0;
    var min_iters: f32    = 1e27;
    var accum_dist: f32   = 0.0;
    var accum_stripe: f32 = 0.0;
    var accum_flags: u32  = 0u;
    var c_for_log = ComplexFExp(FExp(0.0, 0), FExp(0.0, 0));
    var dbg_rebase: u32    = 0u;
    var dbg_bla_max: u32     = 0u;
    var dbg_bla_count: u32   = 0u;
    var dbg_bla_skipped: u32 = 0u;

    for (sample_idx = 0u; sample_idx < uni.sample_count; sample_idx++) {
        var results = vec4f(0.0, 0.0, 0.0, 0.0);
        let jitter  = get_jitter(pix, sample_idx);

        if (uni.ref_orb_count > 0u) {
            let delta_c = build_delta_c_from_orbit_location(pix, 0u, jitter);
            results     = mandelbrot_perturb(delta_c, &dbg_rebase, &dbg_bla_max, &dbg_bla_count, &dbg_bla_skipped);
            c_for_log   = delta_c;
        } else {
            let c     = build_c_from_scene(pix, jitter);
            results   = mandelbrot(c);
            c_for_log = c;
        }

        max_iters    = max(results.x, max_iters);
        min_iters    = min(results.x, min_iters);
        accum_dist   += results.y;
        accum_stripe += results.z;
        accum_flags  |= u32(results.w);
    }

    let sc          = f32(uni.sample_count);
    let accum_iters = mix(min_iters, max_iters, uni.sample_avg_bias);
    accum_dist   /= sc;
    accum_stripe /= sc;

    textureStore(calc_out_tex, pix,
        vec4f(accum_iters, accum_dist, accum_stripe, f32(accum_flags)));

    if (pix.x == i32(f32(uni.render_width)  * PROBE_X_FRAC) &&
        pix.y == i32(f32(uni.render_height) * PROBE_Y_FRAC)) {
        // Emit the raw FExp {mantissa, exponent} so the CPU can reconstruct the
        // true value — fexp_to_f32 would underflow to 0 past ~1e-38.
        debug_out.center_x      = c_for_log.re.m;
        debug_out.center_x_exp  = c_for_log.re.e;
        debug_out.center_y      = c_for_log.im.m;
        debug_out.center_y_exp  = c_for_log.im.e;
        debug_out.scale         = uni.scale;
        debug_out.scale_exp     = uni.scale_exp;
        debug_out.max_iters     = uni.max_iter;
        debug_out.fi            = accum_iters;
        debug_out.distance      = accum_dist;
        debug_out.stripe_avg    = accum_stripe;
        debug_out.flags         = accum_flags;
        debug_out.rebase_count  = dbg_rebase;
        debug_out.bla_max_step     = dbg_bla_max;
        debug_out.bla_step_count   = dbg_bla_count;
        debug_out.bla_iters_skipped = dbg_bla_skipped;
    }
}

struct DebugOut {
    center_x:       f32,
    center_x_exp:   i32,
    center_y:       f32,
    center_y_exp:   i32,
    scale:          f32,
    scale_exp:      i32,
    max_iters:      u32,
    fi:             f32,
    distance:       f32,
    stripe_avg:     f32,
    flags:          u32,
    rebase_count:   u32,
    bla_max_step:   u32,
    bla_step_count: u32,
    bla_iters_skipped: u32,
};
@group(1) @binding(0) var<storage, read_write> debug_out: DebugOut;
