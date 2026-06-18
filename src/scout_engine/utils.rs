use log::debug;
use num_complex::{Complex32};
use crate::signals::{CameraSnapshot, GpuGridSample};
use crate::numerics::{FixedComplex, FixedReal};

// I.e. distance as a vector
pub fn complex_delta(a: &FixedComplex, b: &FixedComplex) -> FixedComplex {
    // (a.real - b.real, a.imag - b.imag)
    let real = a.re.clone() - b.re.clone();
    let imag = a.im.clone() - b.im.clone();
    FixedComplex::new(
        real, imag
    )
}

// I.e. distance as a scalar 
pub fn complex_distance(a: &FixedComplex, b: &FixedComplex) -> FixedReal {
    let delta = complex_delta(a, b);
    let sum = delta.re.clone() * delta.re.clone() + delta.im.clone() * delta.im.clone();
    sum.sqrt()
}

pub fn norm_distance_error(c: &FixedComplex, ref_c: &FixedComplex, scale: &FixedReal) -> f32 {
    let c_f32 = Complex32::new(c.re().to_f32_lossy(), c.im().to_f32_lossy());
    let ref_c_f32 = Complex32::new(ref_c.re().to_f32_lossy(), ref_c.im().to_f32_lossy());
    let delta_c_f32 = c_f32 - ref_c_f32;
    let dist_c_f32 = delta_c_f32.l1_norm();

    let true_dist = complex_distance(c, ref_c);
    let diff_err = (dist_c_f32 - true_dist.to_f32_lossy()).abs();

    (diff_err / scale.to_f32_lossy()).log10()
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
        let mut sample = sample.clone();

        if sample.location.re.shift != cam.center().re.shift {
            let delta_shift = cam.center().re.shift as i32 - sample.location.re.shift as i32;
            debug!("Rescaling sample to match viewport. delta_shift={}", delta_shift);
            sample.location.rescale(delta_shift);
        }

        // Subtract FixedReal at same shift (exact), then convert to f64 to avoid
        // squaring underflow inside complex_distance at deep zoom.
        let dr = (sample.location.re().clone() - cam_center.re().clone()).to_f64_lossy();
        let di = (sample.location.im().clone() - cam_center.im().clone()).to_f64_lossy();
        let sample_dist_from_cam_center = (dr * dr + di * di).sqrt();

        let depth = sample.iters_reached as f64 / sample.max_user_iters as f64;
        let dist = 1.0 - (sample_dist_from_cam_center / cam_half_extent.to_f64_lossy())
            .clamp(0.0, 1.0);
        let escape_penalty = if sample.escaped {1.0} else {0.0};
        
        let total_score =
                depth * 3.0 +         
                dist * 0.2;

        Self {
            depth, dist, escape_penalty, total_score, sample
        }
    }
}
