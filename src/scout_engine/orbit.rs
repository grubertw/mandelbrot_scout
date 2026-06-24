use crate::numerics::{FixedComplex, FixedReal};
use crate::signals::{CameraSnapshot, FrameStamp};

use std::sync::{Arc, Weak};

use log::{warn};
use num_complex::{Complex32};
use parking_lot::{Mutex, RwLock};
use crate::scout_engine::{ScoutEngineConfig};

/// All ReferenceOrbit object are Arc+Mutex wrapped, to support
/// the massive concurrency needed when evaluating these orbits both
/// globally, and on a (complex)-tile-by-tile basis.
pub type LiveOrbit      = Arc<RwLock<ReferenceOrbit>>;
/// Tiles ONLY keep weak references to orbits, which means when they are 
/// deleted globally, they have effectively been culled from ALL tiles
pub type WeakOrbit      = Weak<RwLock<ReferenceOrbit>>;
/// Our singular/global pool of 'live' ref orbits on the system
pub type LivingOrbits   = Arc<Mutex<Vec<LiveOrbit>>>;
/// UniqueID for the ref orbit
pub type OrbitId        = u64;
/// Concurrently guarantees creation of unique orbit IDs.
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
    pub c_ref: FixedComplex,
    /// May not be complete until anchored
    pub orbit: Vec<FixedComplex>,
    /// Is None of the orbit does not escape
    /// in this case, the len of the orbit vec can be checked
    pub escape_index: Option<u32>,
    /// Creation timestamp
    pub created_at: FrameStamp,
    /// orbit list is pre-transformed for GPU consumption
    /// Only set when this orbit is anchored
    pub gpu_payload: OrbitGpuPayload,

    /// Private variables below mutate in-place during compute_to()
    curr_z: FixedComplex,
}

impl ReferenceOrbit {
    pub fn new(
        id_fac: OrbitIdFactory, c_ref: FixedComplex,
        frame_stamp: FrameStamp,
        // ONLY used to set the capacity
        max_ref_orbit_iters: u32
    ) -> Self {
        let orbit_id = id_fac.lock().next_id();
        let shift = *&c_ref.re.shift;
        let c_ref_32 = Complex32::new(c_ref.re().to_f32_lossy(), c_ref.im().to_f32_lossy());

        Self {
            orbit_id, c_ref,
            orbit: Vec::with_capacity(max_ref_orbit_iters as usize),
            escape_index: None,
            created_at: frame_stamp,
            gpu_payload: OrbitGpuPayload::new(c_ref_32),
            curr_z: FixedComplex::zero(shift),
        }
    }

    pub fn is_exterior(&self) -> bool {
        self.escape_index.is_some()
    }


    pub fn c_ref(&self) -> &FixedComplex {
        &self.c_ref
    }

    pub fn escape_index(&self) -> Option<u32> {
        self.escape_index
    }
    
    /// Try computing the orbit to max_iter, which should ALWAYS
    /// at least be max_user_iters. Going past bailout gets prohibitively
    /// expensive for FixedReal using a scale-controlled shift.
    pub fn compute_to(&mut self, max_iter: u32) {
        let curr_iter = self.orbit.len() as u32;
        if max_iter < curr_iter {
            warn!("For Orbit {}, curr_iter={} is already at (or past) max_iter={}", 
                self.orbit_id, curr_iter, max_iter);
            return;
        }

        let bailout = FixedReal::from_f64(4.0,  self.c_ref.re.shift);

        for i in curr_iter..max_iter {
            self.orbit.push(self.curr_z.clone());
            self.curr_z = self.curr_z.square();
            self.curr_z += &self.c_ref;
            
            if self.curr_z.norm_sqr() >= bailout && self.escape_index.is_none() {
                self.escape_index = Some(i);
                break;
            }
        }
    }
}

impl PartialEq for ReferenceOrbit {
    fn eq(&self, other: &Self) -> bool {
        self.orbit_id == other.orbit_id
    }
}

impl Eq for ReferenceOrbit {}

#[derive(Clone, Debug)]
pub struct OrbitGpuPayload {
    pub c_ref:      Complex32,
    pub c32_orbit:  Vec<Complex32>,
}

impl OrbitGpuPayload {
    pub fn new(c_ref: Complex32) -> Self {
        Self {
            c_ref,
            c32_orbit:  Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OrbitScore {
    pub depth: f64,
    pub dist: f64,

    pub total_score: f64,
    pub orbit: LiveOrbit,
}

impl OrbitScore {
    pub fn new(orbit: LiveOrbit, cam: &CameraSnapshot, cfg: &ScoutEngineConfig) -> Self {
        let mut orb_g = orbit.write();
        let cam_center = cam.center();
        let cam_half_extent = cam.half_extent().clone();

        // Orbits must be rescaled with the current viewport before being compared
        if orb_g.c_ref().re.shift != cam_center.re.shift {
            let delta_shift = cam_center.re.shift as i32 - orb_g.c_ref().re.shift as i32;
            orb_g.c_ref.rescale(delta_shift);
        }

        let curr_max_ref_iters = cfg.max_user_iters as f64 * cfg.ref_iters_multiplier;

        // FixedReal squaring underflows to zero at deep zoom (delta mantissa < 2^(shift/2)),
        // so compute the distance in f64 after subtracting — subtraction is exact at the same shift.
        let dr = (orb_g.c_ref().re().clone() - cam_center.re().clone()).to_f64_lossy();
        let di = (orb_g.c_ref().im().clone() - cam_center.im().clone()).to_f64_lossy();
        let sample_dist_f64 = (dr * dr + di * di).sqrt();
        let half_extent_f64 = cam_half_extent.to_f64_lossy();

        let depth = if orb_g.escape_index().is_none() {
            orb_g.orbit.len() as f64 / curr_max_ref_iters
        } else {
            orb_g.escape_index().unwrap() as f64 / curr_max_ref_iters
        };

        let dist = 1.0 - (sample_dist_f64 / half_extent_f64)
            .log2().clamp(0.0, 1000.0);

        let total_score =
            depth * cfg.depth_bonus +
                dist * cfg.distance_penalty;

        Self {
            depth, dist, total_score, orbit: orbit.clone()
        }
    }
}