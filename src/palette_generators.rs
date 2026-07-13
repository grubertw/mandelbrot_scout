//! Procedural palette generators, ported from Fractal-Zoomer's `randomPalette`
//! (`CustomPaletteEditorDialog.java`). See `docs/palette_editor.md`.
//!
//! Each generator returns a flat `Vec<[u8; 4]>` of exactly `params.colors`
//! colors (one count-1 stop each after RLE grouping). The master RNG is a
//! seedable `ChaCha8Rng` — the "Roll" button just picks a fresh seed, so
//! palettes are reproducible and tweaking a param re-runs with the same seed.
//!
//! Data-table generators (Material, ColorBrewer, Distance) are not here yet.

use noise::{NoiseFn, OpenSimplex, Perlin};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::f64::consts::TAU;
use strum_macros::{Display, EnumIter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter, Display)]
pub enum Generator {
    #[strum(to_string = "Golden Ratio")]
    GoldenRatio,
    #[strum(to_string = "Waves")]
    Waves,
    #[strum(to_string = "Distance")]
    Distance,
    #[strum(to_string = "Triad")]
    Triad,
    #[strum(to_string = "Tetrad")]
    Tetrad,
    #[strum(to_string = "Google Material")]
    GoogleMaterial,
    #[strum(to_string = "ColorBrewer 1")]
    ColorBrewer1,
    #[strum(to_string = "ColorBrewer 2")]
    ColorBrewer2,
    #[strum(to_string = "Google-ColorBrewer")]
    GoogleColorBrewer,
    #[strum(to_string = "Cubehelix")]
    Cubehelix,
    #[strum(to_string = "IQ Cosines")]
    IqCosines,
    #[strum(to_string = "Perlin")]
    Perlin,
    #[strum(to_string = "Simplex")]
    Simplex,
    #[strum(to_string = "Perlin + Simplex")]
    PerlinSimplex,
    #[strum(to_string = "Random Walk")]
    RandomWalk,
    #[strum(to_string = "Simple Random")]
    SimpleRandom,
}

impl Generator {
    /// Whether this generator exposes tunable params (drives the UI).
    pub fn has_cubehelix_params(self) -> bool {
        matches!(self, Generator::Cubehelix)
    }
    pub fn has_frequency_param(self) -> bool {
        matches!(self, Generator::IqCosines)
    }
    pub fn has_noise_param(self) -> bool {
        matches!(self, Generator::Perlin | Generator::Simplex | Generator::PerlinSimplex)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GenParams {
    pub colors: u32, // K: number of colors to produce
    pub seed: u64,
    // Cubehelix
    pub cube_start: f32,    // 0..9
    pub cube_rotation: f32, // -5..5
    pub cube_gamma: f32,    // 0.1..1.5
    // IQ Cosines
    pub iq_frequency: f32, // ~0.5..3
    // Perlin / Simplex noise
    pub noise_scale: f32, // ~0.3..4
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            colors: 12,
            seed: 0,
            cube_start: 0.5,
            cube_rotation: -1.5,
            cube_gamma: 1.0,
            iq_frequency: 1.0,
            noise_scale: 1.5,
        }
    }
}

/// Dispatch. Always returns at least one color.
pub fn generate(generator: Generator, p: &GenParams) -> Vec<[u8; 4]> {
    let n = (p.colors.max(1)) as usize;
    let mut rng = ChaCha8Rng::seed_from_u64(p.seed);
    match generator {
        Generator::GoldenRatio => golden_ratio(n, &mut rng),
        Generator::Waves => waves(n, &mut rng),
        Generator::Distance => distance(n, &mut rng),
        Generator::Triad => n_tad(n, &mut rng, 3),
        Generator::Tetrad => n_tad(n, &mut rng, 4),
        Generator::GoogleMaterial => category_select(MATERIAL_CATS, n, &mut rng),
        Generator::ColorBrewer1 => category_select(COLORBREWER_CATS, n, &mut rng),
        Generator::ColorBrewer2 => ramp_select(COLORBREWER_MIXED, n, &mut rng),
        Generator::GoogleColorBrewer => category_select(GOOGLE_CB_CATS, n, &mut rng),
        Generator::Cubehelix => cubehelix(n, &mut rng, p.cube_start, p.cube_rotation, p.cube_gamma),
        Generator::IqCosines => iq_cosines(n, &mut rng, p.iq_frequency),
        Generator::Perlin => noise_palette(n, &mut rng, p.noise_scale, NoiseKind::Perlin),
        Generator::Simplex => noise_palette(n, &mut rng, p.noise_scale, NoiseKind::Simplex),
        Generator::PerlinSimplex => noise_palette(n, &mut rng, p.noise_scale, NoiseKind::Both),
        Generator::RandomWalk => random_walk(n, &mut rng),
        Generator::SimpleRandom => simple_random(n, &mut rng),
    }
}

// ---------------------------------------------------------------------------
// Color-space helpers (shared with the picker in palette_window).
// ---------------------------------------------------------------------------

/// HSV (all 0..1) -> RGB (0..255, alpha 255).
pub fn hsv_to_rgb(hsv: [f32; 3]) -> [u8; 4] {
    let h = hsv[0].rem_euclid(1.0) * 6.0;
    let s = hsv[1].clamp(0.0, 1.0);
    let v = hsv[2].clamp(0.0, 1.0);
    let i = h.floor();
    let f = h - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match (i as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [to_u8(r), to_u8(g), to_u8(b), 255]
}

/// RGB (0..255) -> HSV, all components in 0..1. Hue is 0 when achromatic.
pub fn rgb_to_hsv(c: [u8; 4]) -> [f32; 3] {
    let r = c[0] as f32 / 255.0;
    let g = c[1] as f32 / 255.0;
    let b = c[2] as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let v = max;
    let s = if max <= 0.0 { 0.0 } else { d / max };
    let h = if d <= 0.0 {
        0.0
    } else if max == r {
        (((g - b) / d) % 6.0 + 6.0) % 6.0
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    [(h / 6.0).rem_euclid(1.0), s, v]
}

/// HSL (all 0..1) -> RGB (0..255, alpha 255).
pub fn hsl_to_rgb(hsl: [f32; 3]) -> [u8; 4] {
    let h = hsl[0].rem_euclid(1.0);
    let s = hsl[1].clamp(0.0, 1.0);
    let l = hsl[2].clamp(0.0, 1.0);
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = h * 6.0;
    let x = c * (1.0 - ((hp % 2.0) - 1.0).abs());
    let (r1, g1, b1) = match hp as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    [to_u8(r1 + m), to_u8(g1 + m), to_u8(b1 + m), 255]
}

fn to_u8(x: f32) -> u8 {
    (x * 255.0).round().clamp(0.0, 255.0) as u8
}

fn rand_unit(rng: &mut ChaCha8Rng) -> f64 {
    rng.gen_range(0.0..1.0)
}

/// Normalize a channel's raw values to 0..255 (used by the noise/IQ generators).
fn normalize(vals: &[f64]) -> Vec<u8> {
    let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let d = (max - min).max(1e-9);
    vals.iter()
        .map(|v| (255.0 * (v - min) / d).round().clamp(0.0, 255.0) as u8)
        .collect()
}

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

/// FZ alg 0: golden-ratio walk of brightness, random hue/saturation (HSB).
fn golden_ratio(n: usize, rng: &mut ChaCha8Rng) -> Vec<[u8; 4]> {
    const PHI: f64 = 0.618_033_988_749_895;
    let mut brightness = rand_unit(rng);
    (0..n)
        .map(|_| {
            brightness = (brightness + PHI) % 1.0;
            let h = rand_unit(rng) as f32;
            let s = rand_unit(rng) as f32;
            hsv_to_rgb([h, s, brightness as f32])
        })
        .collect()
}

/// FZ alg 1: per-channel sinusoids with random frequency and phase (RGB).
fn waves(n: usize, rng: &mut ChaCha8Rng) -> Vec<[u8; 4]> {
    let phase = [
        rand_unit(rng) * 1000.0,
        rand_unit(rng) * 1000.0,
        rand_unit(rng) * 1000.0,
    ];
    let coeff = [
        rng.gen_range(0..11) as f64 + rand_unit(rng) + 1.0,
        rng.gen_range(0..11) as f64 + rand_unit(rng) + 1.0,
        rng.gen_range(0..11) as f64 + rand_unit(rng) + 1.0,
    ];
    (0..n)
        .map(|m| {
            let mm = (m + 1) as f64;
            let mut c = [0u8; 4];
            for ch in 0..3 {
                let val =
                    127.5 * ((std::f64::consts::PI / coeff[ch] * mm + phase[ch]).sin() + 1.0) + 0.5;
                c[ch] = val.clamp(0.0, 255.0) as u8;
            }
            c[3] = 255;
            c
        })
        .collect()
}

/// FZ algs 3/4 generalized: `groups` hues evenly spaced around the wheel, each
/// spanning ~n/groups colors with a cyclic saturation/brightness ramp.
fn n_tad(n: usize, rng: &mut ChaCha8Rng, groups: usize) -> Vec<[u8; 4]> {
    let base_h = rand_unit(rng);
    let sat0 = rand_unit(rng);
    let bri0 = rand_unit(rng);
    let var_s = rng.gen_bool(0.5);
    let hue_step = 1.0 / groups as f64;

    let base = n / groups;
    let extra = n % groups;

    let mut out = Vec::with_capacity(n);
    for g in 0..groups {
        let len = base + if g < extra { 1 } else { 0 };
        let h = (base_h + g as f64 * hue_step) % 1.0;
        for l in 0..len {
            let f = if len > 1 { l as f64 / len as f64 } else { 0.0 };
            let s = if var_s { (sat0 + f) % 1.0 } else { sat0 };
            let v = (bri0 + f) % 1.0;
            out.push(hsv_to_rgb([h as f32, s as f32, v as f32]));
        }
    }
    out
}

/// FZ alg 9: D.A. Green (2011) cubehelix. `start`/`rotation`/`gamma` are exposed;
/// the saturation range is seed-driven (as in FZ).
fn cubehelix(n: usize, rng: &mut ChaCha8Rng, start: f32, rotation: f32, gamma: f32) -> Vec<[u8; 4]> {
    let (min_sat, max_sat) = if rng.gen_bool(0.5) {
        let a = rand_unit(rng) * 3.0;
        let b = rand_unit(rng) * 3.0;
        (a.min(b), a.max(b))
    } else {
        let s = rand_unit(rng) * 3.0;
        (s, s)
    };

    let start_r = (start as f64 * 120.0).to_radians();
    let rot_r = (rotation as f64 * 360.0).to_radians();
    let gamma = gamma as f64;
    let rot_matrix = [[-0.14861, 1.78277], [-0.29227, -0.90649], [1.97294, 0.0]];
    let denom = (n.max(2) - 1) as f64;

    (0..n)
        .map(|i| {
            let lambd = i as f64 / denom;
            let lg = lambd.powf(gamma);
            let phi = start_r + rot_r * lambd;
            let sat = min_sat + (max_sat - min_sat) / denom * i as f64;
            let amp = sat * lg * (1.0 - lg) / 2.0;
            let (cos, sin) = (phi.cos(), phi.sin());
            let mut c = [0u8; 4];
            for (j, m) in rot_matrix.iter().enumerate() {
                let dotp = m[0] * cos + m[1] * sin;
                let v = (lg + dotp * amp) * 255.0;
                c[j] = v.round().clamp(0.0, 255.0) as u8;
            }
            c[3] = 255;
            c
        })
        .collect()
}

/// FZ alg 10: Inigo Quilez cosine palette, `a + b*cos(2pi*(c*t + d) + g)` per
/// channel, normalized. `frequency` scales `c`.
fn iq_cosines(n: usize, rng: &mut ChaCha8Rng, frequency: f32) -> Vec<[u8; 4]> {
    // (a, b, c, d, g) per channel.
    let mut p = [[0.0f64; 5]; 3];
    for ch in 0..3 {
        let sum = (rand_unit(rng) + 0.4).min(1.0);
        let pct = rand_unit(rng);
        p[ch][0] = sum * pct; // a
        p[ch][1] = sum * (1.0 - pct); // b
        p[ch][2] = rand_unit(rng) * 2.0 * frequency as f64; // c
        p[ch][3] = rand_unit(rng) * 2.0; // d
        p[ch][4] = rand_unit(rng) * 2.0; // g
    }

    let mut chans: [Vec<f64>; 3] = [Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n)];
    for m in 0..n {
        let t = m as f64 / n as f64;
        for ch in 0..3 {
            let [a, b, c, d, g] = p[ch];
            chans[ch].push(a + b * (TAU * (c * t + d) + g).cos());
        }
    }

    assemble(n, [normalize(&chans[0]), normalize(&chans[1]), normalize(&chans[2])])
}

enum NoiseKind {
    Perlin,
    Simplex,
    Both,
}

/// FZ algs 11/12/13: sample coherent noise around a full 2pi circle (so the
/// palette loops), normalized per channel. `scale` sets the sampling radius.
fn noise_palette(n: usize, rng: &mut ChaCha8Rng, scale: f32, kind: NoiseKind) -> Vec<[u8; 4]> {
    let incr = TAU / n as f64;
    let nmax = scale.max(0.05) as f64;
    let mut chans: [Vec<f64>; 3] = [Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n)];

    for ch in 0..3 {
        let perlin = Perlin::new(rng.gen_range(0..u32::MAX));
        let simplex = OpenSimplex::new(rng.gen_range(0..u32::MAX));
        let perlin2 = Perlin::new(rng.gen_range(0..u32::MAX));
        let phase = rand_unit(rng) * 10.0;
        let phase2 = rand_unit(rng) * 10.0;

        for m in 0..n {
            let a = m as f64 * incr;
            let sample = |g: &dyn NoiseFn<f64, 2>, ph: f64| {
                let x = ((a + ph).cos() + 1.0) * nmax;
                let y = ((a + ph).sin() + 1.0) * nmax;
                g.get([x, y])
            };
            let v = match kind {
                NoiseKind::Perlin => sample(&perlin, phase),
                NoiseKind::Simplex => sample(&simplex, phase),
                NoiseKind::Both => (sample(&simplex, phase) + sample(&perlin2, phase2)) / 2.0,
            };
            chans[ch].push(v);
        }
    }

    assemble(n, [normalize(&chans[0]), normalize(&chans[1]), normalize(&chans[2])])
}

/// FZ alg 14: random walk per channel with reflecting bounds.
fn random_walk(n: usize, rng: &mut ChaCha8Rng) -> Vec<[u8; 4]> {
    const STEPS: [i32; 5] = [30, 15, 20, 10, 25];
    let mut c = [
        rng.gen_range(0..256) as i32,
        rng.gen_range(0..256) as i32,
        rng.gen_range(0..256) as i32,
    ];
    (0..n)
        .map(|_| {
            let val = STEPS[rng.gen_range(0..STEPS.len())];
            let mut out = [0u8; 4];
            for ch in 0..3 {
                c[ch] += if rng.gen_bool(0.5) { val } else { -val };
                if c[ch] < 0 {
                    c[ch] += 2 * val;
                } else if c[ch] > 255 {
                    c[ch] -= 2 * val;
                }
                out[ch] = c[ch].clamp(0, 255) as u8;
            }
            out[3] = 255;
            out
        })
        .collect()
}

/// FZ alg 15: fully random colors, drawn in one randomly chosen color space
/// (RGB / HSV / HSL) for the whole palette.
fn simple_random(n: usize, rng: &mut ChaCha8Rng) -> Vec<[u8; 4]> {
    let kind = rng.gen_range(0..3);
    (0..n)
        .map(|_| match kind {
            1 => hsv_to_rgb([rand_unit(rng) as f32, rand_unit(rng) as f32, rand_unit(rng) as f32]),
            2 => hsl_to_rgb([rand_unit(rng) as f32, rand_unit(rng) as f32, rand_unit(rng) as f32]),
            _ => [
                rng.gen_range(0..=255),
                rng.gen_range(0..=255),
                rng.gen_range(0..=255),
                255,
            ],
        })
        .collect()
}

/// Combine three normalized channels into `[u8; 4]` colors.
fn assemble(n: usize, chans: [Vec<u8>; 3]) -> Vec<[u8; 4]> {
    (0..n).map(|m| [chans[0][m], chans[1][m], chans[2][m], 255]).collect()
}

// ---------------------------------------------------------------------------
// Distance (FZ alg 2): farthest-point selection over an RYB-space grid.
// ---------------------------------------------------------------------------

/// FZ `ColorGenerator`: build an `n`-plus-skip pool of the most mutually-distant
/// colors, then take `n` of them. `skip` (seed-driven) offsets which run is used.
fn distance(n: usize, rng: &mut ChaCha8Rng) -> Vec<[u8; 4]> {
    let skip = rng.gen_range(0..n); // n >= 1
    let picks = farthest_point_picks(skip + n);
    picks[skip..skip + n]
        .iter()
        .map(|p| {
            let rgb = ryb_to_rgb(p[0], p[1], p[2]);
            [floor255(rgb[0]), floor255(rgb[1]), floor255(rgb[2]), 255]
        })
        .collect()
}

/// Greedy farthest-point sampling of a `numBase^3` unit-cube grid: first point is
/// the origin corner, then repeatedly take the point farthest from the running
/// centroid of already-picked points (a direct port of FZ's `Points.pick`).
fn farthest_point_picks(total: usize) -> Vec<[f64; 3]> {
    let num_base = ((total as f64).cbrt().ceil() as usize).max(2);
    let ceil = num_base.pow(3);
    let d = (num_base - 1) as f64;

    let mut points: Vec<[f64; 3]> = Vec::with_capacity(ceil);
    for i in 0..ceil {
        points.push([
            (i / (num_base * num_base)) as f64 / d,
            ((i / num_base) % num_base) as f64 / d,
            (i % num_base) as f64 / d,
        ]);
    }

    let sq_dist = |a: &[f64; 3], b: &[f64; 3]| {
        (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>()
    };

    let mut out = Vec::with_capacity(total);
    let mut centroid: Option<[f64; 3]> = None;
    let mut picked_count = 0.0f64;

    for _ in 0..total {
        if points.is_empty() {
            break;
        }
        let idx = match &centroid {
            None => 0,
            Some(c) => {
                let mut best_i = 0;
                let mut best_d = sq_dist(&points[0], c);
                for (i, p) in points.iter().enumerate() {
                    let dd = sq_dist(p, c);
                    if best_d < dd {
                        best_i = i;
                        best_d = dd;
                    }
                }
                best_i
            }
        };
        let p = points.remove(idx);
        match &mut centroid {
            None => {
                centroid = Some(p);
                picked_count = 1.0;
            }
            Some(c) => {
                for k in 0..3 {
                    c[k] = (picked_count * c[k] + p[k]) / (picked_count + 1.0);
                }
                picked_count += 1.0;
            }
        }
        out.push(p);
    }
    out
}

/// RYB -> RGB via trilinear blend of the eight RYB corner colors (FZ `RYB.ToRgb`).
fn ryb_to_rgb(r: f64, y: f64, b: f64) -> [f64; 3] {
    const WHITE: [f64; 3] = [1.0, 1.0, 1.0];
    const RED: [f64; 3] = [1.0, 0.0, 0.0];
    const YELLOW: [f64; 3] = [1.0, 1.0, 0.0];
    const BLUE: [f64; 3] = [0.163, 0.373, 0.6];
    const VIOLET: [f64; 3] = [0.5, 0.0, 0.5];
    const GREEN: [f64; 3] = [0.0, 0.66, 0.2];
    const ORANGE: [f64; 3] = [1.0, 0.5, 0.0];
    const BLACK: [f64; 3] = [0.2, 0.094, 0.0];

    let mut rgb = [0.0; 3];
    for i in 0..3 {
        rgb[i] = WHITE[i] * (1.0 - r) * (1.0 - b) * (1.0 - y)
            + RED[i] * r * (1.0 - b) * (1.0 - y)
            + BLUE[i] * (1.0 - r) * b * (1.0 - y)
            + VIOLET[i] * r * b * (1.0 - y)
            + YELLOW[i] * (1.0 - r) * (1.0 - b) * y
            + ORANGE[i] * r * (1.0 - b) * y
            + GREEN[i] * (1.0 - r) * b * y
            + BLACK[i] * r * b * y;
    }
    rgb
}

fn floor255(x: f64) -> u8 {
    (255.0 * x).floor().clamp(0.0, 255.0) as u8
}

// ---------------------------------------------------------------------------
// Table generators (FZ Google Material / ColorBrewer). Values are the exact FZ
// tables: Material as 0xRRGGBB, ColorBrewer as signed ARGB; RGB is extracted the
// same way for both (`new Color(int)` ignores alpha).
// ---------------------------------------------------------------------------

fn argb_to_rgb(v: i32) -> [u8; 4] {
    let u = v as u32;
    [((u >> 16) & 0xFF) as u8, ((u >> 8) & 0xFF) as u8, (u & 0xFF) as u8, 255]
}

/// FZ category selection (Material `generate`, ColorBrewer `generate2`, mixed):
/// repeatedly take a random category (once), a random shade ramp within it, and a
/// run of >= 2 shades (either direction). Categories are refilled if exhausted so
/// exactly `n` colors are produced.
fn category_select(categories: &[&[&[i32]]], n: usize, rng: &mut ChaCha8Rng) -> Vec<[u8; 4]> {
    let mut out = Vec::with_capacity(n);
    let mut pool: Vec<&[&[i32]]> = categories.to_vec();

    while out.len() < n {
        if pool.is_empty() {
            pool = categories.to_vec();
        }
        let category = pool.remove(rng.gen_range(0..pool.len()));
        let ramp = category[rng.gen_range(0..category.len())];

        if ramp.len() < 3 {
            for &v in ramp {
                if out.len() < n {
                    out.push(argb_to_rgb(v));
                }
            }
            continue;
        }

        let (mut from, mut to) = (0usize, 0usize);
        while (from as i32 - to as i32).abs() < 2 {
            from = rng.gen_range(0..ramp.len());
            to = rng.gen_range(0..ramp.len());
        }
        if from > to {
            let mut i = from as i32;
            while i >= to as i32 && out.len() < n {
                out.push(argb_to_rgb(ramp[i as usize]));
                i -= 1;
            }
        } else {
            for i in from..=to {
                if out.len() < n {
                    out.push(argb_to_rgb(ramp[i]));
                }
            }
        }
    }
    out
}

/// FZ ColorBrewer `generate`: append whole diverging schemes (ascending or
/// descending), one random scheme at a time, until `n` colors are produced.
fn ramp_select(schemes: &[&[i32]], n: usize, rng: &mut ChaCha8Rng) -> Vec<[u8; 4]> {
    let mut out = Vec::with_capacity(n);
    let mut pool: Vec<&[i32]> = schemes.to_vec();

    while out.len() < n {
        if pool.is_empty() {
            pool = schemes.to_vec();
        }
        let scheme = pool.remove(rng.gen_range(0..pool.len()));
        if rng.gen_bool(0.5) {
            for &v in scheme {
                if out.len() < n {
                    out.push(argb_to_rgb(v));
                }
            }
        } else {
            for &v in scheme.iter().rev() {
                if out.len() < n {
                    out.push(argb_to_rgb(v));
                }
            }
        }
    }
    out
}

// ---- Google Material (0xRRGGBB, plus signed BLACK) ----
const M_RED: &[i32] = &[0xFFEBEE, 0xFFCDD2, 0xEF9A9A, 0xE57373, 0xEF5350, 0xF44336, 0xE53935, 0xD32F2F, 0xC62828, 0xB71C1C];
const M_PINK: &[i32] = &[0xFCE4EC, 0xF8BBD0, 0xF48FB1, 0xF06292, 0xEC407A, 0xE91E63, 0xD81B60, 0xC2185B, 0xAD1457, 0x880E4F];
const M_PURPLE: &[i32] = &[0xF3E5F5, 0xE1BEE7, 0xCE93D8, 0xBA68C8, 0xAB47BC, 0x9C27B0, 0x8E24AA, 0x7B1FA2, 0x6A1B9A, 0x4A148C];
const M_DEEP_PURPLE: &[i32] = &[0xEDE7F6, 0xD1C4E9, 0xB39DDB, 0x9575CD, 0x7E57C2, 0x673AB7, 0x5E35B1, 0x512DA8, 0x4527A0, 0x311B92];
const M_INDIGO: &[i32] = &[0xE8EAF6, 0xC5CAE9, 0x9FA8DA, 0x7986CB, 0x5C6BC0, 0x3F51B5, 0x3949AB, 0x303F9F, 0x283593, 0x1A237E];
const M_BLUE: &[i32] = &[0xE3F2FD, 0xBBDEFB, 0x90CAF9, 0x64B5F6, 0x42A5F5, 0x2196F3, 0x1E88E5, 0x1976D2, 0x1565C0, 0x0D47A1];
const M_LIGHT_BLUE: &[i32] = &[0xE1F5FE, 0xB3E5FC, 0x81D4FA, 0x4FC3F7, 0x29B6F6, 0x03A9F4, 0x039BE5, 0x0288D1, 0x0277BD, 0x01579B];
const M_CYAN: &[i32] = &[0xE0F7FA, 0xB2EBF2, 0x80DEEA, 0x4DD0E1, 0x26C6DA, 0x00BCD4, 0x00ACC1, 0x0097A7, 0x00838F, 0x006064];
const M_TEAL: &[i32] = &[0xE0F2F1, 0xB2DFDB, 0x80CBC4, 0x4DB6AC, 0x26A69A, 0x009688, 0x00897B, 0x00796B, 0x00695C, 0x004D40];
const M_GREEN: &[i32] = &[0xE8F5E9, 0xC8E6C9, 0xA5D6A7, 0x81C784, 0x66BB6A, 0x4CAF50, 0x43A047, 0x388E3C, 0x2E7D32, 0x1B5E20];
const M_LIGHT_GREEN: &[i32] = &[0xF1F8E9, 0xDCEDC8, 0xC5E1A5, 0xAED581, 0x9CCC65, 0x8BC34A, 0x7CB342, 0x689F38, 0x558B2F, 0x33691E];
const M_LIME: &[i32] = &[0xF9FBE7, 0xF0F4C3, 0xE6EE9C, 0xDCE775, 0xD4E157, 0xCDDC39, 0xC0CA33, 0xAFB42B, 0x9E9D24, 0x827717];
const M_YELLOW: &[i32] = &[0xFFFDE7, 0xFFF9C4, 0xFFF59D, 0xFFF176, 0xFFEE58, 0xFFEB3B, 0xFDD835, 0xFBC02D, 0xF9A825, 0xF57F17];
const M_AMBER: &[i32] = &[0xFFF8E1, 0xFFECB3, 0xFFE082, 0xFFD54F, 0xFFCA28, 0xFFC107, 0xFFB300, 0xFFA000, 0xFF8F00, 0xFF6F00];
const M_ORANGE: &[i32] = &[0xFFF3E0, 0xFFE0B2, 0xFFCC80, 0xFFB74D, 0xFFA726, 0xFF9800, 0xFB8C00, 0xF57C00, 0xEF6C00, 0xE65100];
const M_DEEP_ORANGE: &[i32] = &[0xFBE9E7, 0xFFCCBC, 0xFFAB91, 0xFF8A65, 0xFF7043, 0xFF5722, 0xF4511E, 0xE64A19, 0xD84315, 0xBF360C];
const M_BROWN: &[i32] = &[0xEFEBE9, 0xD7CCC8, 0xBCAAA4, 0xA1887F, 0x8D6E63, 0x795548, 0x6D4C41, 0x5D4037, 0x4E342E, 0x3E2723];
const M_GREY: &[i32] = &[0xFAFAFA, 0xF5F5F5, 0xEEEEEE, 0xE0E0E0, 0xBDBDBD, 0x9E9E9E, 0x757575, 0x616161, 0x424242, 0x212121];
const M_BLUE_GREY: &[i32] = &[0xECEFF1, 0xCFD8DC, 0xB0BEC5, 0x90A4AE, 0x78909C, 0x607D8B, 0x546E7A, 0x455A64, 0x37474F, 0x263238];
const M_BLACK: &[i32] = &[-16777216, -14342875, -11382190, -9211021, -6908266, -4342339, -2500135, -986896, -1];

// ---- ColorBrewer (signed ARGB) ----
const CB_GREEN1: &[i32] = &[-16759511, -16750537, -14449597, -12473507, -8862087, -5382770, -2494301, -525127, -27];
const CB_GREEN2: &[i32] = &[-16759781, -16749268, -14447803, -12472714, -10042716, -6694711, -3347226, -1706503, -525059];
const CB_GREEN3: &[i32] = &[-16759781, -16749268, -14447803, -12473507, -9124746, -6170213, -3675712, -1706528, -525067];
const CB_BLUE1: &[i32] = &[-16245416, -14338924, -14524760, -14839360, -12470588, -8401477, -3675724, -1181519, -39];
const CB_BLUE2: &[i32] = &[-16236415, -16226132, -13923138, -11619373, -8663868, -5710411, -3347515, -2034725, -525072];
const CB_BLUE3: &[i32] = &[-16693706, -16683943, -16612982, -13201216, -9983537, -5849637, -3091994, -1252624, -2053];
const CB_BLUE4: &[i32] = &[-16631720, -16491891, -16420688, -13201216, -9131569, -5849637, -3091994, -1251342, -2053];
const CB_BLUE5: &[i32] = &[-16240533, -16232036, -14585419, -12414266, -9720106, -6370591, -3744785, -2167817, -525313];
const CB_PURPLE1: &[i32] = &[-11730869, -8319108, -7847523, -7574607, -7563578, -6374182, -4205594, -2036492, -525059];
const CB_PURPLE2: &[i32] = &[-11992982, -8781449, -5373570, -2280297, -563039, -352331, -211520, -139043, -2061];
const CB_PURPLE3: &[i32] = &[-10026977, -6815677, -3272106, -1627766, -2136656, -3566393, -2836006, -1580561, -527111];
const CB_PURPLE4: &[i32] = &[-12648323, -11262065, -9809501, -8356422, -6382904, -4407844, -2434325, -1053195, -197635];
const CB_RED1: &[i32] = &[-8454144, -5046272, -2674657, -1088184, -225959, -148604, -142178, -71480, -2068];
const CB_RED2: &[i32] = &[-8388570, -4390874, -1893860, -242134, -160452, -85428, -75402, -4704, -52];
const CB_RED3: &[i32] = &[-10026995, -5959915, -3467235, -1098964, -300470, -224654, -214111, -73518, -2576];
const CB_ORANGE1: &[i32] = &[-10083066, -6736892, -3388414, -1282028, -91863, -80817, -72815, -2116, -27];
const CB_ORANGE2: &[i32] = &[-8444156, -5884413, -2537471, -956141, -160452, -151957, -143198, -71986, -2581];
const CB_BLACK: &[i32] = &[-16777216, -14342875, -11382190, -9211021, -6908266, -4342339, -2500135, -986896, -1];
const CB_MIXED1: &[i32] = &[-13828021, -11262072, -8358996, -5067822, -2565397, -526345, -73546, -149405, -2063852, -5023738, -8439032];
const CB_MIXED2: &[i32] = &[-16761808, -16685474, -13265009, -8335935, -3675419, -657931, -595773, -2112899, -4226771, -7581430, -11259899];
const CB_MIXED3: &[i32] = &[-16759781, -14976969, -10834335, -5842016, -2494253, -526345, -1583896, -4020785, -6721365, -9033085, -12582837];
const CB_MIXED4: &[i32] = &[-14195687, -11693535, -8405951, -4660858, -1641008, -526345, -139025, -936230, -2197586, -3859587, -7470766];
const CB_MIXED5: &[i32] = &[-16437151, -14588244, -12348477, -7158306, -3021328, -526345, -140345, -744062, -2727859, -5105621, -10026977];
const CB_MIXED6: &[i32] = &[-15066598, -11711155, -7895161, -4539718, -2039584, -1, -140345, -744062, -2727859, -5105621, -10026977];
const CB_MIXED7: &[i32] = &[-13551979, -12225100, -9130543, -5514775, -2034696, -65, -73584, -151967, -758461, -2674649, -5963738];
const CB_MIXED8: &[i32] = &[-10596446, -13465411, -10042715, -5513820, -1641064, -65, -73589, -151967, -758461, -2802097, -6422206];
const CB_MIXED9: &[i32] = &[-16750537, -15034288, -10044061, -5842582, -2494581, -65, -73589, -151967, -758461, -2674649, -5963738];

// Category groupings (FZ order preserved so seeded output stays consistent).
const MATERIAL_CATS: &[&[&[i32]]] = &[
    &[M_TEAL, M_GREEN],
    &[M_LIGHT_GREEN, M_LIME],
    &[M_INDIGO, M_BLUE],
    &[M_LIGHT_BLUE, M_CYAN],
    &[M_PURPLE, M_DEEP_PURPLE],
    &[M_RED, M_PINK],
    &[M_ORANGE, M_DEEP_ORANGE],
    &[M_YELLOW, M_AMBER],
    &[M_BLACK],
    &[M_BROWN],
    &[M_GREY, M_BLUE_GREY],
];

const COLORBREWER_CATS: &[&[&[i32]]] = &[
    &[CB_GREEN1, CB_GREEN2, CB_GREEN3],
    &[CB_BLUE1, CB_BLUE2, CB_BLUE3, CB_BLUE4, CB_BLUE5],
    &[CB_PURPLE1, CB_PURPLE2, CB_PURPLE3, CB_PURPLE4],
    &[CB_RED1, CB_RED2, CB_RED3],
    &[CB_ORANGE1, CB_ORANGE2],
    &[CB_BLACK],
];

const COLORBREWER_MIXED: &[&[i32]] = &[
    CB_MIXED1, CB_MIXED2, CB_MIXED3, CB_MIXED4, CB_MIXED5, CB_MIXED6, CB_MIXED7, CB_MIXED8, CB_MIXED9,
];

const GOOGLE_CB_CATS: &[&[&[i32]]] = &[
    &[CB_GREEN1, CB_GREEN2, CB_GREEN3, M_TEAL, M_GREEN, M_LIGHT_GREEN, M_LIME],
    &[CB_BLUE1, CB_BLUE2, CB_BLUE3, CB_BLUE4, CB_BLUE5, M_INDIGO, M_BLUE, M_LIGHT_BLUE, M_CYAN],
    &[CB_PURPLE1, CB_PURPLE2, CB_PURPLE3, CB_PURPLE4, M_PURPLE, M_DEEP_PURPLE],
    &[CB_RED1, CB_RED2, CB_RED3, M_RED, M_PINK],
    &[CB_ORANGE1, CB_ORANGE2, M_ORANGE, M_DEEP_ORANGE],
    &[M_YELLOW, M_AMBER],
    &[CB_BLACK],
    &[M_GREY, M_BLUE_GREY],
    &[M_BROWN],
];

#[cfg(test)]
mod tests {
    use super::*;
    use strum::IntoEnumIterator;

    fn params(colors: u32, seed: u64) -> GenParams {
        GenParams { colors, seed, ..GenParams::default() }
    }

    #[test]
    fn every_generator_yields_requested_length() {
        for g in Generator::iter() {
            for &k in &[1u32, 3, 12, 32] {
                let out = generate(g, &params(k, 7));
                assert_eq!(out.len(), k as usize, "{g} produced wrong length for k={k}");
                assert!(out.iter().all(|c| c[3] == 255), "{g} alpha not opaque");
            }
        }
    }

    #[test]
    fn generators_are_deterministic_per_seed() {
        for g in Generator::iter() {
            let a = generate(g, &params(16, 42));
            let b = generate(g, &params(16, 42));
            assert_eq!(a, b, "{g} not deterministic");
        }
    }

    #[test]
    fn different_seeds_differ() {
        // Not guaranteed for every generator/seed, but the noise/arithmetic ones
        // should differ across these seeds.
        let a = generate(Generator::IqCosines, &params(16, 1));
        let b = generate(Generator::IqCosines, &params(16, 2));
        assert_ne!(a, b);
    }

    #[test]
    fn color_roundtrip_hsv() {
        for &c in &[[255, 0, 0, 255], [0, 128, 64, 255], [10, 20, 30, 255]] {
            let back = hsv_to_rgb(rgb_to_hsv(c));
            for ch in 0..3 {
                assert!((back[ch] as i32 - c[ch] as i32).abs() <= 1, "hsv roundtrip off");
            }
        }
    }
}
