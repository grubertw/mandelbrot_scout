use std::cmp::Ordering;
use crate::scout_engine::{ScoutContext, CameraSnapshotReceiver, OrbitObservationsReceiver, ExploreSignalReceiver, ScoutSignal};
use crate::scout_engine::orbit::*;
use crate::scout_engine::tasks::*;
use crate::scout_engine::utils::*;

use crate::signals::{CameraSnapshot, GpuGridSample};

use std::sync::Arc;
use log::{debug, info, error, trace};

use futures::task::SpawnExt;
use futures::StreamExt;
use futures::select;
use futures::future::{join_all, RemoteHandle};
use futures::executor::{ThreadPool};

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

    context.write_diagnostics("Scout Engine Ready!".to_string());
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
                        debug!("Scout Worker received {} OrbitObservation(s)", feedback.len());
                        trace!("OrbitObservations {:?}", feedback);
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

    // Scale gate check first, otherwise we have an explosion of tiles across too much
    // of the mandelbrot at once!
    let starting_scale = context.config.lock().starting_scale;
    let auto_start = context.config.lock().auto_start;
    if snapshot.scale().to_f64() > starting_scale || !auto_start {
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
            let mut living_orbits_g = context.living_orbits.lock();
            
            context.write_diagnostics(
        format!("Resetting ScoutEngine! cleaning up {} orbits!",
                    living_orbits_g.len())
            );

            living_orbits_g.clear();
            
            context.context_changed();
        }
    }
}

async fn evaluate_orbits(
    tp: ThreadPool, context: ScoutContext,
) {
    let current_camera = context.last_camera_snapshot.lock().clone();
    info!("Evaluate orbits for screen center {} and scale {}", 
        current_camera.center().to_string_radix(10, Some(10)),
        current_camera.scale().to_string_radix(10, Some(6))
    );

    // Snapshot GPU grid samples for eval
    let grid_samples = context.grid_samples.lock().clone();
    let mut num_interior_seeds: u32 = 0;

    // Perform preliminary scoring/ranking on the samples before spawning into orbits
    let mut sample_scores: Vec<SampleScore> = grid_samples
        .iter()
        .map(|sample| {
            if !sample.escaped {
                num_interior_seeds += 1;
            }
            SampleScore::new(sample, &current_camera)
        })
        .collect();
    
    // Sort samples in spawn order
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

    // Truncate by num seeds to spawn
    let num_seeds = context.config.lock().num_seeds_to_spawn_per_eval;
    sample_scores.truncate(num_seeds as usize);
    
    let seeds: Vec<Arc<GpuGridSample>> = sample_scores
        .iter()
        .map(|score| Arc::new(score.sample.clone()))
        .collect();

    // Spawn refence orbits from tile samples (candidate seeds)
    let mut orbit_scores = spawn_orbits_from_grid_samples(
        &seeds, tp.clone(), context.clone(), &current_camera
    ).await;

    let mut best_orbit_len: u32 = 0;
    let mut best_oribt_escape_index: Option<u32> = None;
    let mut best_orbit_contraction: f64 = 0.0;
    if let Some(s) = orbit_scores.first() {
        let orb_g = s.orbit.read();
        best_orbit_len = orb_g.orbit.len() as u32;
        best_oribt_escape_index = orb_g.escape_index();
        best_orbit_contraction = orb_g.contraction().to_f64();
    }
    
    // Re-score old orbits
    let mut pool_g = context.living_orbits.lock();
    let mut scored_pool_orbs: Vec<OrbitScore> =  pool_g
        .iter()
        .map(|orb| OrbitScore::new(orb.clone(), &current_camera, context.config.clone()))
        .collect();
    
    // Append new scored orbits
    scored_pool_orbs.append(&mut orbit_scores);
    
    // Sort the complete list
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

    context.write_diagnostics(
        format!("Scout evaluated {} grid samples and spawned {} orbits. {} interior samples found!\n\
        \tBest Sample Info: iters_reached={} escaped={}\n\
        \tBest Qualified Ref Orbit Info:\n\t\tref_len={}\n\t\tescape_index={}\n\t\tcontraction={:.5e}",
                grid_samples.len(), num_seeds, num_interior_seeds,
                best_sample_iters_reached, best_sample_escaped,
                best_orbit_len,
                best_oribt_escape_index.map_or(String::from("None"), |v| v.to_string()),
                best_orbit_contraction
        ));

    context.context_changed();
}

async fn spawn_orbits_from_grid_samples(
    samples: &[Arc<GpuGridSample>],
    tp: ThreadPool,
    context: ScoutContext,
    current_camera: &CameraSnapshot,
) -> Vec<OrbitScore> {

    let handles: Vec<RemoteHandle<LiveOrbit>> = samples
        .iter()
        .filter_map(|sample| {
            let res = tp.spawn_with_handle(
                start_reference_orbit(sample.location.clone(), context.orbit_id_factory.clone(),
                                      context.config.clone(),
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
        .map(|orb| OrbitScore::new(orb.clone(), current_camera, context.config.clone()))
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
