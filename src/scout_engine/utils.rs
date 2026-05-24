use num_complex::{Complex32};
use crate::signals::{CameraSnapshot, GpuGridSample};
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
