use crate::numerics::ComplexDf;
use crate::signals::{FrameStamp};

use std::sync::{Arc, Weak};

use rug::{Float, Complex};
use parking_lot::{Mutex, RwLock};

/// All ReferenceOrbit object are Arc+Mutex wrapped, to support
/// the massive concurrency needed when evaluating these orbits both
/// globally, and on a (complex)-tile-by-tile basis.
pub type LiveOrbit      = Arc<RwLock<ReferenceOrbit>>;
/// Tiles ONLY keep weak references to orbits, which means when they are 
/// deleted globally, they have effecivly been culled from ALL tiles
pub type WeakOrbit      = Weak<RwLock<ReferenceOrbit>>;
/// Our singular/globl pool of 'live' ref orbits on the system
/// Orbit scores are kept BESIDE rather than inside the orbit,
/// for more convient scoring & ranking
pub type LivingOrbits   = Arc<Mutex<Vec<(f64, LiveOrbit)>>>;
/// UniqueID for the ref orbit
pub type OrbitId        = u64;
/// Concurrently guarentees creation of unique orbit IDs.
pub type OrbitIdFactory = Arc<Mutex<IdFactory>>;

#[derive(Clone, Debug)]
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
pub struct ReferenceOrbit {
    /// Uniquely identifies the orbit, system-wide
    pub orbit_id: OrbitId,
    /// The starting co-ordinate of the orbit
    pub c_ref: Complex,
    /// Same as above, but for consumption on the GPU
    pub c_ref_df: ComplexDf,
    /// The complete orbit
    pub orbit: Vec<Complex>,
    /// orbit list is pre-transformed for GPU consuption
    pub gpu_payload: OrbitGpuPayload,
    pub escape_index: Option<u32>,
    /// Feedback from GPU pixels (sampled stat)
    pub max_lambda: Float,
    /// Feedback from GPU pixels (aggregated/reduced stat)
    pub min_valid_perturb_index: u32,
    pub max_valid_perturb_index: u32,
    pub created_at: FrameStamp,
    pub last_updated: FrameStamp,
}

impl ReferenceOrbit {
    pub fn new(
        id_fac: OrbitIdFactory, c_ref: Complex, orbit: Vec<Complex>,
        gpu_payload: OrbitGpuPayload, escape_index: Option<u32>,
        frame_st: FrameStamp
    ) -> Self {
        let orbit_id = id_fac.lock().next_id();
        let c_ref_df = ComplexDf::from_complex(&c_ref);
        let max_ref_len = orbit.len();
        let prec = c_ref.prec().0;
        
        Self {
            orbit_id, c_ref, c_ref_df,
            orbit, gpu_payload, escape_index,
            max_lambda: Float::with_val(prec, 0.0),
            min_valid_perturb_index: escape_index.unwrap_or(max_ref_len as u32),
            max_valid_perturb_index: 0,
            created_at: frame_st.clone(),
            last_updated: frame_st.clone()
        }
    }
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

impl OrbitGpuPayload {
    pub fn new() -> Self {
        Self {
            re_hi: Vec::<f32>::new(), 
            re_lo: Vec::<f32>::new(), 
            im_hi: Vec::<f32>::new(), 
            im_lo: Vec::<f32>::new()
        }
    }
}