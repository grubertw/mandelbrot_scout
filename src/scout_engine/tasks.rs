use crate::scout_engine::{ScoutEngineConfig};
use crate::scout_engine::orbit::*;

use crate::signals::{FrameStamp};

use std::sync::Arc;
use log::{trace};
use num_complex::Complex32;
use parking_lot::RwLock;
use crate::numerics::{FixedComplex, ComplexFExp};

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

    for fc in &orbit.orbit {
        orbit.gpu_payload.c32_orbit.push(
            Complex32::new(fc.re().to_f32_lossy(), fc.im().to_f32_lossy())
        );
        orbit.gpu_payload.fexp_orbit.push(
            ComplexFExp::from_fixed(fc)
        );
    }
    Arc::new(RwLock::new(orbit))
}