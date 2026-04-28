use log::{trace};
use num_complex::{Complex32, Complex64};
use crate::signals::{CameraSnapshot, GpuGridSample};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rug::{Float, Complex};

// I.e. distance as a vector
pub fn complex_delta(a: &Complex, b: &Complex) -> Complex {
    // (a.real - b.real, a.imag - b.imag)
    let real = a.real().clone() - b.real();
    let imag = a.imag().clone() - b.imag();
    Complex::with_val(
        a.prec(), 
        (real, imag)
    )
}

// I.e. distance as a scalar 
pub fn complex_distance(a: &Complex, b: &Complex) -> Float {
    let delta = complex_delta(a, b);
    let sum = delta.real().clone() * delta.real() + delta.imag() * delta.imag();
    sum.sqrt()
}

pub fn norm_distance_error(c: &Complex, ref_c: &Complex, scale: &Float) -> f32 {
    let c_f32 = Complex32::new(c.real().to_f32(), c.imag().to_f32());
    let ref_c_f32 = Complex32::new(ref_c.real().to_f32(), ref_c.imag().to_f32());
    let delta_c_f32 = c_f32 - ref_c_f32;
    let dist_c_f32 = delta_c_f32.l1_norm();

    let true_dist = complex_distance(c, ref_c);
    let diff_err = (dist_c_f32 - true_dist.to_f32()).abs();

    (diff_err / scale.to_f32()).log10()
}

pub fn avg_sample_loc(samples: &[GpuGridSample]) -> Complex {
    let prec = samples[0].location.prec().0;
    let mut avg_loc_real = Float::with_val(prec, 0.0);
    let mut avg_loc_imag = Float::with_val(prec, 0.0);

    for s in samples {
        avg_loc_real += s.location.real();
        avg_loc_imag += s.location.imag();
    }

    avg_loc_real /= samples.len() as f64;
    avg_loc_imag /= samples.len() as f64;
    Complex::with_val(prec, (avg_loc_real, avg_loc_imag))
}

pub fn infer_direction(samples: &[GpuGridSample]) -> Complex64 {
    let mut dir = Complex64::new(0.0, 0.0);

    for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            let si = &samples[i];
            let sj = &samples[j];

            let di = si.iters_reached as f64;
            let dj = sj.iters_reached as f64;

            let delta_c = Complex64::new(
                sj.location.real().to_f64() - si.location.real().to_f64(),
                sj.location.imag().to_f64() - si.location.imag().to_f64(),
            );

            let delta_d = dj - di;

            // Only keep direction where depth increases
            if delta_d.abs() > 0.0 {
                dir += delta_c * delta_d;
            }
        }
    }

    normalize(dir)
}

pub fn walk_toward_basin(
    start: &Complex64,
    direction: Complex64,
    num_steps: u32,
    step_size: f64,
    max_iter: u32
) -> (Complex64, u32) {
    let mut best = start.clone();
    let mut best_depth = fast_depth_probe(start, max_iter);

    for i in 1..num_steps {
        let step = direction * (step_size * i as f64);
        let candidate = start + step;

        let depth = fast_depth_probe(&candidate, max_iter);

        if depth > best_depth {
            best = candidate;
            best_depth = depth;
        } else if depth + 2 >= best_depth {
            // allow slight noise tolerance
            best = candidate;
            best_depth = depth;
        } else {
            trace!("Walked {} steps of size {} for initial c={}. best={} depth={}",
                i, step, start, best, best_depth);
            break; // stop if we're not going deeper
        }
    }

    (best, best_depth)
}

pub fn fast_depth_probe(c: &Complex64, max_iter: u32) -> u32 {
    let mut z = Complex64::new(0.0, 0.0);

    for i in 0..max_iter {
        z = z * z + c;

        if z.norm_sqr() > 4.0 {
            return i; // escaped early → bad
        }
    }

    max_iter // deeper = better
}

pub fn normalize(z: Complex64) -> Complex64 {
    let n = z.norm();
    if n == 0.0 {
        Complex64::new(0.0, 0.0)
    } else {
        z / n
    }
}

fn rotate(z: Complex64, angle: f64) -> Complex64 {
    let (s, c) = angle.sin_cos();
    Complex64::new(
        z.re * c - z.im * s,
        z.re * s + z.im * c,
    )
}

pub fn generate_f64_seeds(
    num_seeds: u32,
    base: Complex64,
    direction: Complex64,
    frame_seed: u64,          // 👈 pass frame_id or similar
    step_max: f64,            // e.g. ~ 2–10 * pixel scale
    cone_angle: f64           // radians, e.g. 0.2 (~11 degrees)
) -> Vec<Complex64> {
    let mut seeds = Vec::with_capacity(num_seeds as usize);

    // Deterministic RNG per frame (VERY useful for debugging)
    let mut rng = ChaCha8Rng::seed_from_u64(frame_seed);

    // Normalize direction (CRITICAL)
    let dir_norm = normalize(direction);

    // Fallback if direction is garbage
    let dir = if dir_norm.norm_sqr() < 1e-30 {
        Complex64::new(1.0, 0.0)
    } else {
        dir_norm
    };

    for _ in 0..num_seeds {
        // Sample small angle inside cone
        let angle = rng.gen_range(-cone_angle..cone_angle);
        let rotated_dir = rotate(dir, angle);

        // Bias steps toward small values (better local exploration)
        let t: f64 = rng.r#gen::<f64>();
        let step = t * t * step_max; // quadratic bias

        let candidate = base + rotated_dir * step;

        seeds.push(candidate);
    }

    seeds
}

#[derive(Clone, Debug)]
pub struct SampleScore {
    pub depth: f64,
    pub dist: f64,
    pub escape_penalty: f64,

    pub total_score: f64,
    pub sample: GpuGridSample,
}

impl SampleScore {
    pub fn new(sample: &GpuGridSample, cam: &CameraSnapshot) -> Self {
        let cam_center = cam.center();
        let cam_half_extent = cam.half_extent();

        let mut sample_dist_from_cam_center = complex_distance(&sample.location, cam_center);
        sample_dist_from_cam_center = sample_dist_from_cam_center.abs();

        let depth = sample.iters_reached as f64 / sample.max_user_iters as f64;
        let dist = 1.0 - (sample_dist_from_cam_center.to_f64() / cam_half_extent.to_f64())
            .clamp(0.0, 1.0);
        let escape_penalty = if sample.escaped {1.0} else {0.0};

        let gpu_score = sample.score as f64;

        // normalize if needed
        let gpu_score = gpu_score.clamp(-10.0, 10.0);

        let total_score =
            gpu_score * 0.5 +     // weak influence
                depth * 3.0 +         // still important
                dist * 0.2 +          // mild bias
                SampleScore::contraction_bias(sample) +
                SampleScore::interior_bias(sample);

        Self {
            depth, dist, escape_penalty, total_score, sample: sample.clone()
        }
    }

    fn contraction_bias(sample: &GpuGridSample) -> f64 {
        if sample.contraction < 0.0 {
            // good (attracting)
            (-sample.contraction as f64).min(10.0)
        } else {
            // bad (repelling)
            -2.0
        }
    }

    fn interior_bias(sample: &GpuGridSample) -> f64 {
        if !sample.escaped {
            2.0
        } else {
            0.0
        }
    }
}
