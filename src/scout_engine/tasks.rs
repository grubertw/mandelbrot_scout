use crate::scout_engine::{ScoutEngineConfig};
use crate::scout_engine::orbit::*;

use crate::signals::{FrameStamp};

use std::sync::Arc;
use log::{debug, trace};
use num_complex::Complex32;
use parking_lot::RwLock;
use crate::numerics::FixedComplex;

pub async fn start_reference_orbit(
    c_ref: FixedComplex,
    id_factory: OrbitIdFactory,
    cfg: ScoutEngineConfig,
    frame_st: FrameStamp
) -> LiveOrbit {
    // Note that we will compute past max_user_iters, according to the multiplier
    let max_iters = (cfg.max_user_iters as f64 * cfg.ref_iters_multiplier) as u32;

    let mut orbit = ReferenceOrbit::new(
        id_factory, c_ref.clone(),
        frame_st, max_iters
    );
    trace!("Starting orbit calculation of seed {} to {} iterations for OrbitId={}.",
        c_ref, max_iters, orbit.orbit_id);
    orbit.compute_to(max_iters);
    trace!("Orbit calculation of seed {} for OrbitId={} complete! stopped at {} iterations",
        c_ref, orbit.orbit_id, orbit.orbit.len());

    let c32_orbit: Vec<Complex32> = orbit.orbit
        .iter()
        .map(|c| Complex32::new(c.re().to_f32_lossy(), c.im().to_f32_lossy()))
        .collect();

    for c32 in c32_orbit {
        orbit.gpu_payload.c32_orbit.push(c32);
    }
    Arc::new(RwLock::new(orbit))
}

pub async fn continue_reference_orbit(orbit: LiveOrbit, cfg: ScoutEngineConfig) {
    let mut orbit_g = orbit.write();
    orbit_g.is_anchored = false;
    let start_iter = orbit_g.orbit.len();

    // Increase the storage capacity of the orbit vector and (re)start compute from
    // where it last stopped.
    let max_iters = (cfg.max_user_iters as f64 * cfg.ref_iters_multiplier) as u32;
    orbit_g.extend_orbit(max_iters);
    orbit_g.compute_to(max_iters);

    debug!("Continued orbit {} to max_user_iters={}. escape={:?}",
        orbit_g.orbit_id, cfg.max_user_iters, orbit_g.quality_metrics.escape_index);

    for i in start_iter..=orbit_g.orbit.len() {
        let c = &orbit_g.orbit[i];
        let c32 = Complex32::new(c.re().to_f32_lossy(), c.im().to_f32_lossy());
        orbit_g.gpu_payload.c32_orbit.push(c32);
    }
}
