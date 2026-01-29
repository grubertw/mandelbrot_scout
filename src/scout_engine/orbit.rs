use crate::numerics::ComplexDf;

use std::sync::{Arc, Mutex, Weak};
use std::time;

use rug::{Float, Complex};

pub type LiveOrbit      = Arc<Mutex<ReferenceOrbit>>;
pub type WeakOrbit      = Weak<Mutex<ReferenceOrbit>>;
pub type LivingOrbits   = Arc<Mutex<Vec<LiveOrbit>>>;
pub type OrbitId        = u64;
pub type OrbitIdFactory = Arc<Mutex<IdFactory>>;

pub struct IdFactory {
    next_id: OrbitId,
}

impl IdFactory {
    pub fn new() -> Self {Self {next_id: 0}}

    pub fn next_id(&mut self) -> OrbitId {
        self.next_id += 1;
        self.next_id
    }
}

#[derive(Clone, Debug)]
pub struct HeuristicWeights {
    pub w_dist:     f64, // Distance from camera center
    pub w_depth:    f64, // Escape Index
    pub w_age:      f64, // num framce since last use
}

impl HeuristicWeights {
    pub fn vectorize(&self) -> Vec<f64> {
        vec![self.w_dist, self.w_depth, self.w_age]
    }
}

#[derive(Clone, Debug)]
pub struct ReferenceOrbit {
    pub orbit_id: OrbitId,
    pub c_ref: Complex,
    pub c_ref_df: ComplexDf,
    pub orbit: Vec<Complex>,
    pub gpu_payload: OrbitGpuPayload,
    pub escape_index: Option<u32>,
    pub max_lambda: Float,
    pub max_valid_perturb_index: u32,
    pub weights: HeuristicWeights,
    pub current_score: i64,
    pub creation_time: time::Instant,
    pub creation_frame_id: u64,
}

impl PartialEq for ReferenceOrbit {
    fn eq(&self, other: &Self) -> bool {
        self.orbit_id == other.orbit_id
    }
}

impl Eq for ReferenceOrbit {}

#[derive(Debug)]
pub struct OrbitResult {
    pub orbit: Vec<Complex>,
    pub escape_index: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct OrbitGpuPayload {
    pub re_hi: Vec<f32>,
    pub re_lo: Vec<f32>,
    pub im_hi: Vec<f32>,
    pub im_lo: Vec<f32>,
}
