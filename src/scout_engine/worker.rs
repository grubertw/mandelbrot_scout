use std::cmp::Ordering;
use crate::scout_engine::{ScoutContext, CameraSnapshotReceiver, OrbitObservationsReceiver, ExploreSignalReceiver, ScoutSignal, ScoutEngineConfig};
use crate::scout_engine::orbit::*;
use crate::scout_engine::tasks::*;
use crate::scout_engine::utils::*;

use crate::signals::{CameraSnapshot};

use std::sync::Arc;
use log::{debug, info, error, trace};

use futures::task::SpawnExt;
use futures::StreamExt;
use futures::select;
use futures::future::{join_all, RemoteHandle};
use futures::executor::{ThreadPool};
use rug::Complex;

// ScoutEngine's internal work loop, a long-lived future that uses select to poll the
// the camera snaphot & gpu feedback channels.
pub async fn scout_worker(
    tp: ThreadPool,
    context: ScoutContext,
    camera_snapshot_rx: CameraSnapshotReceiver,
    orbit_observations_rx: OrbitObservationsReceiver,
    explore_rx :ExploreSignalReceiver,
) {
    let mut snap_rx = camera_snapshot_rx.fuse();
    let mut orbit_obvs_rx = orbit_observations_rx.fuse();
    let mut explore_rx = explore_rx.fuse();

    loop {
        select! {
            cs_res = snap_rx.next() => {
                match cs_res {
                    Some(snapshot) => {
                        handle_camera_snapshot(
                            tp.clone(), context.clone(), Arc::new(snapshot),
                        ).await;
                    }
                    None => {break;}
                }
            },
            explore_res = explore_rx.next() => {
                match explore_res {
                    Some(sig) => {
                        debug!("Scout Worker received explore signal! {:?}", sig);
                        handle_explore_signal(
                            tp.clone(), context.clone(), sig,
                        ).await;
                    }
                    None => {break;}
                }
            },
            obvs_res = orbit_obvs_rx.next() => {
                match obvs_res {
                    Some(feedback) => {
                        trace!("Scout Worker received {} OrbitObservation(s)", feedback.len());
                    }
                    None => {break;}
                }        
            },
        }
    }
}

async fn handle_camera_snapshot(
    tp: ThreadPool, context: ScoutContext,
    snapshot: Arc<CameraSnapshot>,
) {
    debug!("Scout Worker received camera snapshot. center={} scale={} extent={}",
        snapshot.center().to_string_radix(10, Some(14)),
        snapshot.scale().to_string_radix(10, Some(8)),
        snapshot.half_extent().to_string_radix(10, Some(6))
    );
    {
        let mut last_snap_g = context.last_camera_snapshot.lock();
        *last_snap_g = (*snapshot).clone();
    }

    let scale = snapshot.scale();
    let (starting_scale, auto_start, dist_err_thresh) = {
        let config_g = context.config.lock();
        (config_g.starting_scale, config_g.auto_start, config_g.distance_error_threshold)
    };

    if context.active() && scale.to_f64() > starting_scale {
        context.reset();
        return;
    }

    {
        let lo_g = context.living_orbits.lock();

        if let Some(curr_orb) = lo_g.first() {
            let orb_g = curr_orb.read();
            let dist_err = norm_distance_error(snapshot.center(), orb_g.c_ref(), scale);
            // Hold onto our interior reference orbit as long as possible
            if dist_err < dist_err_thresh && !orb_g.is_exterior() {
                info!("Scout detected {} normalized distance error from camera center and current non-escaping ref orb",
                dist_err);
                return;
            }
        }
    }

    if scale.to_f64() > starting_scale || !auto_start {
        return;
    }

    tp.spawn_ok(
        evaluate_orbits(tp.clone(), context.clone())
    );
}

async fn handle_explore_signal(
    tp: ThreadPool,
    context: ScoutContext,
    sig: ScoutSignal,
) {
    match sig {
        ScoutSignal::ExploreSignal(config) => {
            {// GUI sends configs through message to guarantee consistency 
                let mut config_g = context.config.lock();
                *config_g = config;
            }
            
            tp.spawn_ok(
                evaluate_orbits(tp.clone(), context.clone())
            );
        }
        ScoutSignal::ResetEngine => {
            context.reset();
        }
    }
}

async fn evaluate_orbits(
    tp: ThreadPool, context: ScoutContext,
) {
    let current_camera = {
        context.last_camera_snapshot.lock().clone()
    };
    info!("Evaluate orbits for screen center {} and scale {}", 
        current_camera.center().to_string_radix(10, Some(10)),
        current_camera.scale().to_string_radix(10, Some(6))
    );
    
    let prec = current_camera.scale().prec();

    // Snapshot GPU grid samples for eval
    let mut grid_samples = {
        context.grid_samples.lock().clone()
    };
    let mut num_interior_seeds: u32 = 0;

    // Perform preliminary scoring on GPU samples before spawning into orbits
    let mut sample_scores: Vec<SampleScore> = grid_samples
        .iter()
        .map(|sample| {
            if !sample.escaped {
                num_interior_seeds += 1;
            }
            SampleScore::new(sample, &current_camera)
        })
        .collect();
    
    // Rank GPU samples by depth and distance from cam center
    sample_scores
        .sort_by(|a, b|
            b.total_score.partial_cmp(&a.total_score).unwrap_or(Ordering::Equal));

    let mut best_sample_iters_reached: u32 = 0;
    let mut best_sample_escaped: bool = false;
    // Log the best/worst sample scores.
    if let Some(s) = sample_scores.first() {
        best_sample_iters_reached = s.sample.iters_reached;
        best_sample_escaped = s.sample.escaped;
    }

    debug!("{} GpuGridSamples ranked/sorted", sample_scores.len());
    let cfg = {
        context.config.lock().clone()
    };

    sample_scores.truncate(cfg.num_gpu_samples_to_eval as usize);
    grid_samples = sample_scores
        .iter()
        .map(|sample| sample.sample.clone())
        .collect();
    
    let gpu_seeds: Vec<Complex> = grid_samples
        .iter()
        .map(|sample| sample.location.clone())
        .collect();
    
    let mut gpu_orbit_scores = spawn_orbits_from_seeds(
        &gpu_seeds, tp.clone(), context.orbit_id_factory.clone(), cfg, &current_camera
    ).await;

    let total_orbits_spawned = gpu_orbit_scores.len(); // + f64_orbit_scores.len();
    let mut best_orbit_len: u32 = 0;
    let mut best_oribt_escape_index: Option<u32> = None;
    let mut best_orbit_contraction: f64 = 0.0;

    {
        // Re-score old orbits
        let mut pool_g = context.living_orbits.lock();
        let mut scored_pool_orbs: Vec<OrbitScore> = pool_g
            .iter()
            .map(|orb| OrbitScore::new(orb.clone(), &current_camera, &cfg))
            .collect();

        // Append new scored orbits
        scored_pool_orbs.append(&mut gpu_orbit_scores);
        //scored_pool_orbs.append(&mut f64_orbit_scores);

        // Perform global ranking
        scored_pool_orbs
            .sort_by(|a, b| b.total_score
                .partial_cmp(&a.total_score)
                .unwrap_or(Ordering::Equal));

        let mut trace_str = String::from(
            format!("Re-Scored LiveOrbits. len={}\n", scored_pool_orbs.len()).as_str());

        *pool_g = scored_pool_orbs
            .iter()
            .map(|s| {
                let orb_g = s.orbit.read();
                trace_str.push_str(format!("\tPool OrbitId={:<4} escape={:<7} len={:<6} contraction={:<8.4e}\t\
            \tSCORING: depth={:<6.4} dist={:<6.4} contraction={:<7.4} total={:<6.4}\n",
                                           orb_g.orbit_id, orb_g.escape_index().map_or(String::from("None"), |v| v.to_string()),
                                           orb_g.orbit.len(), orb_g.contraction(),
                                           s.depth, s.dist, s.contraction, s.total_score
                ).as_str());
                s.orbit.clone()
            })
            .collect();
        trace!("{}", trace_str);

        pool_g.truncate(cfg.max_live_orbits as usize);

        if let Some(s) = pool_g.first() {
            let orb_g = s.read();
            best_orbit_len = orb_g.orbit.len() as u32;
            best_oribt_escape_index = orb_g.escape_index();
            best_orbit_contraction = orb_g.contraction().to_f64();
        }
    }

    context.write_diagnostics(
            format!("Scout evaluated {} grid samples, fast (f64) probed {} orbits, and spawned {} orbits. {} interior samples found!\n\
        \tBest Sample Info: iters_reached={} escaped={}\n\
        \tBest Qualified Ref Orbit Info:\n\t\tprec={:<3} ref_len={}\n\t\tescape_index={}\n\t\tcontraction={:.5e}",
                    grid_samples.len(), 0, total_orbits_spawned,
                    num_interior_seeds,
                    best_sample_iters_reached, best_sample_escaped, prec,
                    best_orbit_len,
                    best_oribt_escape_index.map_or(String::from("None"), |v| v.to_string()),
                    best_orbit_contraction
        ));

    context.context_changed();

    // Send a signal to the winit window to wake up the render loop and redraw the viewport
    // This mechanism is largely in place so that the render loop need not run continually
    // and communications to/from the GPU can be tightly controlled.
    context.window.request_redraw();

    info!("Evaluate Orbits complete!");
}

async fn spawn_orbits_from_seeds(
    seeds: &[Complex],
    tp: ThreadPool,
    id_factory: OrbitIdFactory,
    cfg: ScoutEngineConfig,
    current_camera: &CameraSnapshot,
) -> Vec<OrbitScore> {

    let handles: Vec<RemoteHandle<LiveOrbit>> = seeds
        .iter()
        .filter_map(|seed| {
            let res = tp.spawn_with_handle(
                start_reference_orbit(seed.clone(), id_factory.clone(),
                                      cfg,
                                      current_camera.frame_stamp().clone())
            );
            if let Ok(h) = res { Some(h) } else {
                error!("Failed to spawn create_new_reference_orbit");
                None
            }
        })
        .collect();
    
    let live_orbits: Vec<LiveOrbit> = join_all(handles).await;

    let mut scored_orbits: Vec<OrbitScore> = live_orbits
        .iter()
        .map(|orb| OrbitScore::new(orb.clone(), current_camera, &cfg))
        .collect();
    scored_orbits
        .sort_by(|a, b| b.total_score
            .partial_cmp(&a.total_score)
            .unwrap_or(Ordering::Equal));

    for scored_orbit in &scored_orbits {
        let orb_g = scored_orbit.orbit.read();

        trace!("Scored orbit {} c_ref={} orbit.len={} escape={:?} contraction={} r_valid={} \
        period={:?} z_min={} a_max={} with precision={}\n\t\tScores: depth={} dist={} contraction={} total_score={}",
            orb_g.orbit_id, orb_g.c_ref().to_string_radix(10, Some(18)),
            orb_g.orbit.len(), orb_g.quality_metrics.escape_index,
            orb_g.contraction().to_string_radix(10, Some(4)),
            orb_g.r_valid().to_string_radix(10, Some(5)),
            orb_g.quality_metrics.period,
            orb_g.quality_metrics.z_min.to_string_radix(10, Some(6)),
            orb_g.quality_metrics.a_max.to_string_radix(10, Some(6)),
            orb_g.c_ref().prec().0,
            scored_orbit.depth, scored_orbit.dist,
            scored_orbit.contraction,
            scored_orbit.total_score
        );
    }
    
    scored_orbits
}
