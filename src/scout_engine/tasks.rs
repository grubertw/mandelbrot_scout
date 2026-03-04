use crate::scout_engine::{ScoutConfig};
use crate::scout_engine::orbit::*;

use crate::numerics::ComplexDf;
use crate::signals::{FrameStamp};

use std::sync::Arc;
use log::{debug};
use rug::{Complex};
use parking_lot::RwLock;

pub async fn start_reference_orbit(
    c_ref: Complex, 
    id_factory: OrbitIdFactory,
    config: ScoutConfig,
    frame_st: FrameStamp
) -> LiveOrbit {
    let (max_user_iters, multiplier) = {
        let config_g = config.lock();
        (
            config_g.max_user_iters,
            config_g.ref_iters_multiplier
        )
    };

    // Note that we will compute past max_user_iters, according to the multiplier
    let max_iters = (max_user_iters as f64 * multiplier) as u32;
    
    let mut orbit = ReferenceOrbit::new(
        id_factory, c_ref,
        frame_st, max_iters
    );
    orbit.compute_to(max_iters);

    let cdf_orbit: Vec<ComplexDf> = orbit.orbit
        .iter()
        .map(|c| ComplexDf::from_complex(c))
        .collect();

    for cdf in cdf_orbit {
        orbit.gpu_payload.re_hi.push(cdf.re.hi);
        orbit.gpu_payload.re_lo.push(cdf.re.lo);
        orbit.gpu_payload.im_hi.push(cdf.im.hi);
        orbit.gpu_payload.im_lo.push(cdf.im.lo);
    }
    Arc::new(RwLock::new(orbit))
}

pub async fn continue_reference_orbit(orbit: LiveOrbit, config: ScoutConfig) {
    let (max_user_iters, multiplier) = {
        let config_g = config.lock();
        (
            config_g.max_user_iters,
            config_g.ref_iters_multiplier
        )
    };

    let mut orbit_g = orbit.write();
    orbit_g.is_anchored = false;
    let start_iter = orbit_g.orbit.len();

    // Increase the storage capacity of the orbit vector and (re)start compute from
    // where it last stopped.
    let max_iters = (max_user_iters as f64 * multiplier) as u32;
    orbit_g.extend_orbit(max_iters);
    orbit_g.compute_to(max_iters);

    debug!("Continued orbit {} to max_user_iters={}. escape={:?}",
        orbit_g.orbit_id, max_user_iters, orbit_g.quality_metrics.escape_index);

    for i in start_iter..=orbit_g.orbit.len() {
        let cdf = ComplexDf::from_complex(&orbit_g.orbit[i]);

        orbit_g.gpu_payload.re_hi.push(cdf.re.hi);
        orbit_g.gpu_payload.re_lo.push(cdf.re.lo);
        orbit_g.gpu_payload.im_hi.push(cdf.im.hi);
        orbit_g.gpu_payload.im_lo.push(cdf.im.lo);
    }
}
