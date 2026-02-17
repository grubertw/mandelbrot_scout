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
pub type LivingOrbits   = Arc<Mutex<Vec<LiveOrbit>>>;
/// UniqueID for the ref orbit
pub type OrbitId        = u64;
/// Concurrently guarentees creation of unique orbit IDs.
pub type OrbitIdFactory = Arc<Mutex<IdFactory>>;

/// Multipler against the derivitive computation of r_valid
/// Should be made adaptive in the future!
pub const ALPHA: f64 = 0.05;
/// Used during mandelbrot iteration to avoid log(0) and also
/// for period detection!
pub const NEAR_ZERO_THRESHOLD: f64 = 1e-30;
/// For period detection, wait for burn-in before starting
pub const BURN_IN: u32 = 64;
/// max size of the period (in iteration count)
pub const MAX_PERIOD_CHECK: u32 = 32;

/// Checked against contraction
const INTERIOR_THRESHOLD: f64 = -1e-3;
const BOUNDARY_THRESHOLD: f64 = 1e-4;
const STRONG_CONTRACTION: f64 = -0.01;
/// Checked against a_max 'stiffness'
const STIFFNESS_CHECK: f64 = 1e20;
/// Checked against z_min
const NEAR_CRITICAL: f64 = 1e-10;

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
    /// Escape, set interrior/exterrior distance, r_valid
    pub qualiy_metrics: OrbitQuality,
    /// orbit list is pre-transformed for GPU consuption
    pub gpu_payload: OrbitGpuPayload,
    /// Delta of the seed from tile-center
    /// Only set when this orbit is an anchor 
    pub delta_from_tile_center: Option<Complex>,
}

impl ReferenceOrbit {
    pub fn new(
        id_fac: OrbitIdFactory, c_ref: Complex, orbit: Vec<Complex>,
        qualiy_metrics: OrbitQuality, gpu_payload: OrbitGpuPayload, 
    ) -> Self {
        let orbit_id = id_fac.lock().next_id();
        let c_ref_df = ComplexDf::from_complex(&c_ref);
        
        Self {
            orbit_id, c_ref, c_ref_df,
            orbit, qualiy_metrics, 
            gpu_payload, delta_from_tile_center: None
        }
    }

    pub fn is_interior(&self) -> bool {
        self.qualiy_metrics.contraction < INTERIOR_THRESHOLD 
            && self.qualiy_metrics.escape_index.is_none()
    }

    pub fn is_strongly_interior(&self) -> bool {
        self.qualiy_metrics.contraction < STRONG_CONTRACTION
            && self.qualiy_metrics.escape_index.is_none()
    }

    pub fn is_boundary_like(&self) -> bool {
        self.qualiy_metrics.contraction < BOUNDARY_THRESHOLD
    }

    pub fn is_exterior(&self) -> bool {
        self.qualiy_metrics.escape_index.is_some()
    }

    pub fn is_stiff(&self) -> bool {
        self.qualiy_metrics.a_max > STIFFNESS_CHECK
    }

    pub fn is_near_critical(&self) -> bool {
        self.qualiy_metrics.z_min < NEAR_CRITICAL
    }

    pub fn c_ref(&self) -> &Complex {
        &self.c_ref
    }

    pub fn r_valid(&self) -> &Float {
        &self.qualiy_metrics.r_valid
    }

    pub fn contraction(&self) -> &Float {
        &self.qualiy_metrics.contraction
    }
}

impl PartialEq for ReferenceOrbit {
    fn eq(&self, other: &Self) -> bool {
        self.orbit_id == other.orbit_id
    }
}

impl Eq for ReferenceOrbit {}

// Helper struct for gathering initial orbit computation results
#[derive(Debug)]
pub struct OrbitResult {
    pub orbit: Vec<Complex>,
    pub escape_index: Option<u32>,
    pub r_valid: Float,
    pub contraction: Float,
    pub period: Option<u32>,
    pub z_min: Float,
    pub a_max: Float,
}

// Global orbit scoring/ranking metrics
#[derive(Clone, Debug)]
pub struct OrbitQuality {
    /// Is None of the orbit does not escape
    /// in this case, the len of the orbit vec can be checked
    pub escape_index: Option<u32>,
    /// The radious in the complex plane this orbit is valid
    /// for perturation.
    pub r_valid: Float,
    /// < 0 orbit is contracting, i.e. interrior
    /// ~ 0 near boundry
    /// > 0 expanding, i.e. exterior 
    pub contraction: Float,
    /// Orbit period - hyperbolic structure 
    /// strong indication of interrior
    pub period: Option<u32>,
    /// critical proximity
    pub z_min: Float,
    /// stiffness
    pub a_max: Float,
    /// Creation timestamp
    pub created_at: FrameStamp,
}

impl OrbitQuality {
    pub fn new(
        escape_index: Option<u32>, r_valid: Float, 
        contraction: Float, period: Option<u32>,
        z_min: Float, a_max: Float,
        created_at: FrameStamp
    ) -> Self {
        Self {
            escape_index, r_valid, 
            contraction, period, z_min, a_max,
            created_at
        }
    }
}

// Used for scoring, ranking/sorting!
#[derive(Clone, Debug)]
pub struct OrbitSnapshot {
    pub orbit: LiveOrbit,
    pub r_log: f64,
    pub contraction: f64,
    pub is_exterior: bool,
    pub age: f64,
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