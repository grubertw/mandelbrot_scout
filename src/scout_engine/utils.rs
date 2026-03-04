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

        let total_score =
            depth * 4.0 +
                dist * 0.1 +
                (-escape_penalty * 10.0);

        Self {
            depth, dist, escape_penalty, total_score, sample: sample.clone()
        }
    }
}
