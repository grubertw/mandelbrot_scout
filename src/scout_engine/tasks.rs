use crate::scout_engine::{ScoutEngineConfig};
use crate::scout_engine::orbit::*;

use crate::signals::{FrameStamp};

use std::sync::Arc;
use log::{trace};
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