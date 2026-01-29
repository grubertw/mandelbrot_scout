use crate::scout_engine::{ScoutConfig};

use crate::scout_engine::orbit::*;
use crate::scout_engine::tile::*;
use crate::scout_engine::utils::*;

use crate::numerics::ComplexDf;
use crate::signals;

use std::sync::{Arc, Mutex};
use std::time;

use log::{debug, info};
use rug::{Float, Complex};

// Fast operation, can be awaited.
pub async fn create_new_orbit_seeds_from_camera_snapshot(
        snapshot: Arc<signals::CameraSnapshot>) -> Vec<Complex> {
    let mut seeds = Vec::<Complex>::new();
    let center = snapshot.center.clone();

    // Stay very simple now, with a 2x2 pattern now, i.e. top-bottom-left-right-middle of the viewport
    seeds.push(center.clone());

    seeds
}

pub async fn create_new_reference_orbit(c_ref: Complex, id_fac: OrbitIdFactory, 
    max_iter: u32, prec: u32, snap_frame: u64) 
        -> LiveOrbit {
    let orbit_id = id_fac.lock().unwrap().next_id();
    let orbit_result = compute_reference_orbit(&c_ref, max_iter, prec).await;
    let c_ref_df = ComplexDf::from_complex(&c_ref);
    let cdf_orbit: Vec<ComplexDf> = orbit_result.orbit.iter().map(|c| 
        ComplexDf::from_complex(c)).collect();
    let creation_time = time::Instant::now();
    
    let mut gpu_payload = OrbitGpuPayload{
        re_hi: Vec::<f32>::new(), 
        re_lo: Vec::<f32>::new(), 
        im_hi: Vec::<f32>::new(), 
        im_lo: Vec::<f32>::new()
    };

    for cdf in cdf_orbit {
        gpu_payload.re_hi.push(cdf.re.hi);
        gpu_payload.re_lo.push(cdf.re.lo);
        gpu_payload.im_hi.push(cdf.im.hi);
        gpu_payload.im_lo.push(cdf.im.lo);
    }

    info!("Creating new ReferenceOrbit orbit_id={} at time={:?} and snapshot frame_id={}", 
        orbit_id, creation_time, snap_frame);
    Arc::new(Mutex::new(ReferenceOrbit{ orbit_id,
        c_ref: c_ref.clone(), c_ref_df,
        orbit: orbit_result.orbit, gpu_payload,
        escape_index: orbit_result.escape_index,
        max_lambda: Float::with_val(prec, 0.0),
        max_valid_perturb_index: orbit_result.escape_index.unwrap_or(max_iter),
        weights: HeuristicWeights{ w_dist: 0.0, w_depth: 0.0, w_age: 0.0 },
        current_score: 0,
        creation_time,
        creation_frame_id: snap_frame
    }))
}

async fn compute_reference_orbit(c_ref: &Complex, max_iter: u32, prec: u32) 
        -> OrbitResult {
    let mut orbit = Vec::<Complex>::with_capacity(max_iter as usize);
    let mut escape_index: Option<u32> = None;

    let mut z = Complex::with_val(prec, (0.0, 0.0));

    for i in 0..max_iter {
        orbit.push(z.clone());
        z = z.clone() * &z + c_ref;

        if z.clone().abs().real().to_f64() >= 2.0 && escape_index == None {
            escape_index = Some(i);
        }
    }

    debug!("New Reference Orbit computed for c_ref={} orbit.len={} escape_index={:?}", 
        &c_ref, orbit.len(), escape_index);

    OrbitResult {orbit, escape_index}
}

// Calculate weights that contribute to the score
// Weight 'fairness' works on a decreasing scale
// Score scale is linear, so logs/exponents are used to linerize the number
// Negative weights are good, and large positives are bad
// Intrinsic vector ording likes to go from smallest to largest, which needs 
// to be kept in mind as the score will be used to rank/sort
pub async fn weigh_living_orbits(
        living_orbits: LivingOrbits, 
        snapshot: Option<Arc<signals::CameraSnapshot>>,
        _feedback: Option<Arc<signals::GpuFeedback>>) {
    let mut orb_pool_l = living_orbits.lock().unwrap();

    for s_orb in orb_pool_l.iter_mut() {
        if let Some(ref cam_snap) = snapshot {
            let mut orb = s_orb.lock().unwrap();
            let delta = orb.c_ref.clone() - &cam_snap.center;
            orb.weights.w_dist = delta.abs().real().to_f64() / &cam_snap.scale.to_f64();
            orb.weights.w_dist *= 100.0;

            orb.weights.w_depth = 
                if let Some(i) = orb.escape_index {i.into()} 
                else {orb.orbit.len() as f64};
            orb.weights.w_depth *= -0.1;

            // Frame id's should always be increasing, otherwize this breaks
            orb.weights.w_age = (cam_snap.frame_stamp.frame_id as f32 - 
                orb.creation_frame_id as f32) as f64;
            orb.weights.w_age /= 5.0;
        }
    };
}

// Simply adds all the weights in HeuristicWeights by first vectorizing the strcture,
// which is possible because all values are the same type, and taking a sum of all 
// elements in the vector, using an iterator.
pub async fn score_living_orbits(living_orbits: LivingOrbits) {
    let mut orb_pool_l = living_orbits.lock().unwrap();
    for s_orb in orb_pool_l.iter_mut() {
        let mut orb = s_orb.lock().unwrap();
        orb.current_score = orb.weights.vectorize().iter().sum::<f64>() as i64;
    };
}

pub async fn rank_living_orbits(living_orbits: LivingOrbits) {
    // convert the orbit pool into a vector for sorting
    let mut lom = living_orbits.lock().unwrap();

    // Sort the vector if orbit's by it's score.
    lom.sort_by_key(|orb| orb.lock().unwrap().current_score);
}

// Ultimatly, the TileRegistry is the best place to assess which ref-orbit will provide
// the best perturbation results for any given pixel in the viewport. It is critical to 
// understand that these tiles are not locked to the viewport itself, but instead they 
// are anchored to the complex plane. Once a Tile is created, it will remain within the 
// registry for the duration of the program, which is why the list of orbits found within
// TileOrbitView contains Weak references, as living_orbits is regularly trimmed.
//
// It is also important to understand that incomming orbits may not survive long insde the
// TileOrbitView lists, as these lists are meant to be kept very short, for the purposes 
// of per-pixel rebasing on the GPU. Maintining a healthy per-pixel perturbation is the 
// entire purpose for tiles, and also why these tiles divide complex geometry, and not 
// pixel-space. Maintining a low DeltaC from the reference orbit is the first step, but 
// because of the nature of Dynamical Systems, fractal geometry must be thought of in 
// (x,y,i) terms, hence the need for a list of orbits, with it's own ranking, and with
// feedback from per-pixel perturbation on the GPU, giving us insight into the behavior 
// of i, which can can encounter rapid and unexpected change. 
pub async fn insert_new_orbits_into_tile_registry(config: Arc<ScoutConfig>,
        new_orbits: &Vec<LiveOrbit>, tile_registry: TileRegistry) {
    let mut registry = tile_registry.lock().unwrap();
    for level in &config.tile_levels {
        let tile_size = level.tile_size.clone();
        let influence_radius = tile_size.clone() * level.influence_radius_factor;

        for orbit in new_orbits {
            let orbit_guard = orbit.lock().unwrap();
            let c_ref = orbit_guard.c_ref.clone();
            drop(orbit_guard);

            // Enumerate candidate tiles
            let candidates =
                tiles_influenced_by_orbit(&c_ref, &tile_size, &influence_radius);

            for tile_id in candidates {
                let view = registry.entry((level.level, tile_id.clone()))
                                   .or_insert_with(|| {
                    TileOrbitView::new(tile_id, &level)
                });

                if tile_overlaps_orbit(&view.geometry, &c_ref, &influence_radius) {
                    // Insert weak reference if not already present
                    let weak = Arc::downgrade(&orbit);

                    let already_present = view.weak_orbits
                        .iter()
                        .any(|w| w.ptr_eq(&weak));

                    if !already_present {
                        view.weak_orbits.push(weak);
                        view.last_updated = time::Instant::now();
                    }
                }
            }
        }
    }
}

// Rank current orbits within the Tile Registory. 
// This also includes trunkating each TileOrbitView orbit list to it's max allowed 
// orbits, depending on the tile's level. Also, if the weak reference to the orbit
// is no longer valid, it will be removed.
pub async fn rank_tile_registry_orbits(tile_registry: TileRegistry) {
    let mut reg = tile_registry.lock().unwrap();
    for ((_t_level, _tile_id), view) in reg.iter_mut() {
        let mut live_orbits = Vec::<LiveOrbit>::new();
        for weak_orb in view.weak_orbits.iter() {
            if let Some(orb) = weak_orb.upgrade() {
                live_orbits.push(orb);
            }
        }
        view.weak_orbits.clear();
        live_orbits.sort_by_key(|orb| score_tile_view_orbit(orb.clone()));
        live_orbits.truncate(view.max_orbits_per_tile);
        // Ensure only live orbits are kept, even if we revert them
        // back to weak references
        for orb in live_orbits {
            view.weak_orbits.push(Arc::downgrade(&orb));
        }
    }
}

pub async fn trim_living_orbits(max_live_orbits: u32, living_orbits: LivingOrbits) {
    let mut lom = living_orbits.lock().unwrap(); 

    lom.truncate(max_live_orbits as usize);
}