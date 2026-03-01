use crate::numerics::ComplexDf;
use crate::signals::{FrameStamp};

use std::sync::{Arc, Weak};

use log::warn;
use rug::{Float, Complex};
use parking_lot::{Mutex, RwLock};
use crate::scout_engine::tile::TileGeometry;
use crate::scout_engine::utils::complex_distance;

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

/// Multiplier against the derivative computation of r_valid
/// Should be made adaptive in the future!
pub const ALPHA: f64 = 0.05;
/// Used during mandelbrot iteration to avoid log(0) and also
/// for period detection!
pub const NEAR_ZERO_THRESHOLD: f64 = 1e-30;
/// For period detection, wait for burn-in before starting
pub const BURN_IN: u32 = 64;
/// max size of the period (in iteration count)
pub const MAX_PERIOD_CHECK: u32 = 60;

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
    /// May not be complete until anchored
    pub orbit: Vec<Complex>,
    /// Escape, set interior/exterior distance, r_valid
    pub quality_metrics: OrbitQuality,
    /// Set to true when the orbit is anchored 
    pub is_anchored: bool,
    /// orbit list is pre-transformed for GPU consumption
    /// Only set when this orbit is anchored
    pub gpu_payload: OrbitGpuPayload,
    /// Used to finish the orbit, if it is used as an anchor
    pub max_ref_orbit_iters: u32,
    /// Private variables below mutate in-place during compute_to()
    curr_z: Complex,
    curr_a: Complex,
    curr_log_sum: Float,
    log_count: u32,
}

impl ReferenceOrbit {
    pub fn new(
        id_fac: OrbitIdFactory, c_ref: Complex, 
        frame_stamp: FrameStamp,
        // ONLY used to set the capacity
        max_ref_orbit_iters: u32
    ) -> Self {
        let orbit_id = id_fac.lock().next_id();
        let prec = *&c_ref.prec().0;
        let c_ref_df = ComplexDf::from_complex(&c_ref);
        
        Self {
            orbit_id, c_ref, 
            orbit: Vec::with_capacity(max_ref_orbit_iters as usize),
            quality_metrics: OrbitQuality{
                escape_index: None,
                r_valid: Float::with_val(prec, f64::INFINITY),
                contraction: Float::with_val(prec, 0),
                period: None,
                z_min: Float::with_val(prec, f64::INFINITY),
                a_max: Float::with_val(prec, 0),
                created_at: frame_stamp
            }, 
            is_anchored: false,
            gpu_payload: OrbitGpuPayload::new(c_ref_df), 
            max_ref_orbit_iters,
            // Private variables that mutate in-place during mandelbrot computation
            curr_z: Complex::with_val(prec, (0.0, 0.0)),
            curr_a: Complex::with_val(prec, (0.0, 0.0)),
            curr_log_sum: Float::with_val(prec, 0),
            log_count: 0,
        }
    }

    pub fn extend_orbit(&mut self, new_max_iters: u32) {
        self.max_ref_orbit_iters = new_max_iters;
        let size_to_increase = new_max_iters as usize - self.orbit.len();
        self.orbit.reserve(size_to_increase);
    }

    pub fn is_interior(&self) -> bool {
        self.quality_metrics.contraction < INTERIOR_THRESHOLD 
            && self.quality_metrics.escape_index.is_none()
    }

    pub fn is_strongly_interior(&self) -> bool {
        self.quality_metrics.contraction < STRONG_CONTRACTION
            && self.quality_metrics.escape_index.is_none()
    }

    pub fn is_boundary_like(&self) -> bool {
        self.quality_metrics.contraction < BOUNDARY_THRESHOLD
    }

    pub fn is_exterior(&self) -> bool {
        self.quality_metrics.escape_index.is_some()
    }

    pub fn is_stiff(&self) -> bool {
        self.quality_metrics.a_max > STIFFNESS_CHECK
    }

    pub fn is_near_critical(&self) -> bool {
        self.quality_metrics.z_min < NEAR_CRITICAL
    }

    pub fn c_ref(&self) -> &Complex {
        &self.c_ref
    }

    pub fn escape_index(&self) -> Option<u32> {
        self.quality_metrics.escape_index
    }

    pub fn r_valid(&self) -> &Float {
        &self.quality_metrics.r_valid
    }

    pub fn contraction(&self) -> &Float {
        &self.quality_metrics.contraction
    }
    
    /// Try computing the orbit to max_iter, which should ALWAYS be max_user_iter
    /// Internally, we will compute past this for perturbation
    /// Later, if this orbit is chosen as an anchor, but max_user_iter changes
    /// the orbit must be 'invalidated' (i.e. un-anchored), and then continued
    /// which will change the value of r_valid.
    pub fn compute_to(&mut self, max_iter: u32) {
        let curr_iter = self.orbit.len() as u32;
        if max_iter < curr_iter {
            warn!("For Orbit {}, curr_iter={} is already at (or past) max_iter={}", 
                self.orbit_id, curr_iter, max_iter);
            return;
        }
        
        // Loop constants
        let prec = self.c_ref.prec();
        let alpha = Float::with_val(prec.0, ALPHA);
        let two = Float::with_val(prec.0, 2);

        for i in curr_iter..max_iter {
            self.orbit.push(self.curr_z.clone());
            let two_z = self.curr_z.clone() * &two;

            // Compute ratio for r_valid, but only after first iteration
            // Also, stop r_valid computation once escape is reached.
            // Same logic applies for log-sum contraction
            if i > 0 && self.quality_metrics.escape_index.is_none() {
                let z_abs = self.curr_z.clone().abs().real().clone();
                let a_abs = self.curr_a.clone().abs().real().clone();
                let mag_two_z = two_z.clone().abs().real().clone();

                if a_abs > Float::with_val(prec.0, 0.0) {
                    let candidate = alpha.clone() * &z_abs / &a_abs;
                    if candidate < self.quality_metrics.r_valid {
                        // Final r_valid = min_n(alpha*(|Zn|/|An|))
                        self.quality_metrics.r_valid = candidate;
                    }
                }
                // Avoid log(0)
                if mag_two_z > NEAR_ZERO_THRESHOLD {
                    let log_val = mag_two_z.clone().ln();
                    self.curr_log_sum += &log_val;
                    self.log_count += 1;
                }
                // Grab min_z and max_a - they are super cheap and somewhat useful, 
                // so why not!
                if z_abs < self.quality_metrics.z_min {
                    self.quality_metrics.z_min = z_abs.clone();
                }
                if a_abs > self.quality_metrics.a_max {
                    self.quality_metrics.a_max = a_abs.clone();
                }
            }
            // Period detection logic
            if self.quality_metrics.period.is_none() && i > BURN_IN {
                for p in 1..=MAX_PERIOD_CHECK {
                    if i >= p {
                        let prev_z = &self.orbit[(i - p) as usize];
                        let diff = self.curr_z.clone() - prev_z;
                        let abs_diff = diff.abs().real().clone();

                        if abs_diff < NEAR_ZERO_THRESHOLD {
                            self.quality_metrics.period = Some(p);
                            break;
                        }
                    }
                }
            }

            // Core Mandelbrot recurrence, using rug::Complex arbitrary precision
            // Z_{n+1} = Z^2 * C
            self.curr_z = self.curr_z.clone() * &self.curr_z + self.c_ref();

            // Derivative recurrence for r_valid
            // A_{n+1} = 2 * z_n * A_n + 1
            self.curr_a = self.curr_a.clone() * &two_z + Complex::with_val(prec, (1.0, 0.0));
            
            if self.curr_z.clone().abs().real().to_f64() >= 2.0 && self.quality_metrics.escape_index.is_none() {
                self.quality_metrics.escape_index = Some(i);
            }
            
            // Early escape orbits are trash. Stop early and ensure r_valid check fails
            if self.quality_metrics.escape_index.is_some() && i < BURN_IN {
                self.quality_metrics.r_valid = Float::with_val(prec.0, 0);
                break;
            }
        }

        // Compute contraction metric taking the orbit's period into account.
        self.quality_metrics.contraction = if let Some(p) = self.quality_metrics.period {
            let mut sum = Float::with_val(prec.0, 0);
            let start = self.orbit.len() - p as usize;
            for k in start..self.orbit.len() {
                let two_z = self.orbit[k].clone() * &two;
                let mag_two_z = two_z.clone().abs().real().clone();
                if mag_two_z > NEAR_ZERO_THRESHOLD {
                    sum += mag_two_z.clone().ln();
                }
            }

            sum / Float::with_val(prec.0, p)
        } else {
            // fallback: global average
            self.curr_log_sum.clone() / Float::with_val(prec.0, self.log_count)
        };
    }
}

impl PartialEq for ReferenceOrbit {
    fn eq(&self, other: &Self) -> bool {
        self.orbit_id == other.orbit_id
    }
}

impl Eq for ReferenceOrbit {}

// Global orbit scoring/ranking metrics
#[derive(Clone, Debug)]
pub struct OrbitQuality {
    /// Is None of the orbit does not escape
    /// in this case, the len of the orbit vec can be checked
    pub escape_index: Option<u32>,
    /// The radios in the complex plane this orbit is valid
    /// for perturbation.
    pub r_valid: Float,
    /// < 0 orbit is contracting, i.e. interior
    /// ~ 0 near boundary
    /// > 0 expanding, i.e. exterior 
    pub contraction: Float,
    /// Orbit period - hyperbolic structure 
    /// strong indication of interior
    pub period: Option<u32>,
    /// critical proximity
    pub z_min: Float,
    /// stiffness
    pub a_max: Float,
    /// Creation timestamp
    pub created_at: FrameStamp,
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
    pub c_ref: ComplexDf,
    pub re_hi: Vec<f32>,
    pub re_lo: Vec<f32>,
    pub im_hi: Vec<f32>,
    pub im_lo: Vec<f32>,
}

impl OrbitGpuPayload {
    pub fn new(c_ref: ComplexDf) -> Self {
        Self {
            c_ref,
            re_hi: Vec::<f32>::new(), 
            re_lo: Vec::<f32>::new(), 
            im_hi: Vec::<f32>::new(), 
            im_lo: Vec::<f32>::new()
        }
    }
}

#[derive(Clone, Debug)]
pub struct OrbitScore {
    pub depth: f64,
    pub dist: f64,
    pub contraction: f64,

    pub total_score: f64,
    pub orbit: LiveOrbit,
}

impl OrbitScore {
    pub fn new(orbit: LiveOrbit, tile: &TileGeometry) -> Self {
        let orb_g = orbit.read();
        let tile_center = tile.center();
        let tile_radius = tile.radius();

        let sample_dist_from_tile_center = complex_distance(orb_g.c_ref(), tile_center);

        let depth = if orb_g.escape_index().is_none() {1.0} else {
            orb_g.escape_index().unwrap() as f64 / orb_g.max_ref_orbit_iters as f64
        };
        let contraction = (-orb_g.contraction().to_f64()).clamp(0.0, 2.0);
        let dist = 1.0 - (sample_dist_from_tile_center.to_f64() / tile_radius.to_f64())
            .clamp(0.0, 1.0);

        let total_score =
            depth * 2.0 +
                contraction * 200.0 +
                dist * 0.1;

        Self {
            depth, dist, contraction, total_score, orbit: orbit.clone()
        }
    }
}