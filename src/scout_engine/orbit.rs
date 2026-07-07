use crate::numerics::{FixedComplex, FixedReal};
use crate::scout_engine::formula::{Formula, Parameterization};
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
    /// Anchor point in the IMAGE plane (formerly "the c value"). For a
    /// Mandelbrot image this is a c-plane point; for a Julia image it is a
    /// z-plane point. Everything spatial (scoring, center_offset, BLA
    /// delta_c_max) treats it purely as a position, so it is parameterization-
    /// agnostic. The actual (z0, c) fed to the recurrence is derived from it.
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

    /// The iteration map this orbit was computed with.
    formula: Formula,
    /// The `c` fed to f(z, c): == c_ref for Mandelbrot, == the Julia constant
    /// for Julia. Resolved once at construction (like c_ref, it is only used at
    /// the shift the orbit was born at).
    c: FixedComplex,
    /// Private variables below mutate in-place during compute_to()
    curr_z: FixedComplex,
    /// Previous reference value Z_{n-1}, for second-order formulas (Manowar).
    /// Zero (unused) for simple formulas.
    curr_z_prev: FixedComplex,
}

impl ReferenceOrbit {
    pub fn new(
        id_fac: OrbitIdFactory, c_ref: FixedComplex,
        formula: Formula,
        param: &Parameterization,
        frame_stamp: FrameStamp,
        // ONLY used to set the capacity
        max_ref_orbit_iters: u32
    ) -> Self {
        let orbit_id = id_fac.lock().next_id();
        let c_ref_32 = Complex32::new(c_ref.re().to_f32_lossy(), c_ref.im().to_f32_lossy());

        // The one parameterization-aware step: anchor -> (z0, c).
        let (z0, c) = param.seed(&c_ref);

        // Formula-specific initial value + extra state. Manowar seeds z0 = c and
        // carries z_{-1} = z0 (FZ's default init); simple formulas ignore z_prev.
        let (curr_z, curr_z_prev) = match formula {
            // Manowar seeds z_{-1} = z_0 = c; Phoenix seeds z_0 = c, z_{-1} = 0.
            Formula::Manowar => (c.clone(), c.clone()),
            Formula::Phoenix => (c.clone(), FixedComplex::zero(c_ref.re.shift)),
            _ => (z0, FixedComplex::zero(c_ref.re.shift)),
        };

        Self {
            orbit_id, c_ref,
            orbit: Vec::with_capacity(max_ref_orbit_iters as usize),
            escape_index: None,
            created_at: frame_stamp,
            gpu_payload: OrbitGpuPayload::new(c_ref_32),
            formula,
            c,
            curr_z,
            curr_z_prev,
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

        let bailout = FixedReal::from_f64(4.0, self.c.re.shift);

        for i in curr_iter..max_iter {
            self.orbit.push(self.curr_z.clone());
            let z_next = self.formula.ref_step(&self.curr_z, &self.curr_z_prev, &self.c);
            // z_prev <- old z (Z_n), z <- z_next (Z_{n+1}).
            self.curr_z_prev = std::mem::replace(&mut self.curr_z, z_next);

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