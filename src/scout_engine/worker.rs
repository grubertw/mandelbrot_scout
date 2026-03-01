use std::cmp::Ordering;
use crate::scout_engine::{ScoutContext, CameraSnapshotReceiver, TileObservationsReceiver, ExploreSignalReceiver, ScoutSignal};
use crate::scout_engine::orbit::*;
use crate::scout_engine::tile::*;
use crate::scout_engine::tasks::*;
use crate::scout_engine::utils::*;

use crate::signals::{CameraSnapshot};

use std::sync::Arc;
use log::{debug, info, error, trace};
use rug::{Complex};

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
    tile_observations_rx: TileObservationsReceiver,
    explore_rx :ExploreSignalReceiver,
) {
    let mut snap_rx = camera_snapshot_rx.fuse();
    let mut tile_obvs_rx = tile_observations_rx.fuse();
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
            obvs_res = tile_obvs_rx.next() => {
                match obvs_res {
                    Some(feedback) => {
                        debug!("Scout Worker received {} TileObservation(s)", feedback.len());
                    }
                    None => {break;}
                }        
            },
        };
    }
}

async fn handle_camera_snapshot(
    tp: ThreadPool, context: ScoutContext,
    snapshot: Arc<CameraSnapshot>,
) {
    info!("Scout Worker received camera snapshot {:?}", snapshot);
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
        evaluate_tiles(tp.clone(), context.clone())
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
                evaluate_tiles(tp.clone(), context.clone())
            );
        }
        ScoutSignal::ResetEngine => {
            let mut living_orbits_g = context.living_orbits.lock();
            let mut tile_reg_g = context.tile_registry.write();

            context.write_diagnostics(
        format!("Resetting ScoutEngine! cleaning up {} orbits and {} tiles!",
                    living_orbits_g.len(), tile_reg_g.len())
            );

            living_orbits_g.clear();
            tile_reg_g.clear();
            
            context.context_changed();
        }
    }
}

async fn evaluate_tiles(
    tp: ThreadPool, context: ScoutContext,
) {
    let current_camera = context.last_camera_snapshot.lock().clone();
    let tile_level = context.tile_level_for_snapshot(&current_camera);
    info!("Evaluating tiles at current camera center={} & scale={}, and \
    found tile level {} with tile_size={}",
        current_camera.center().to_string_radix(10, Some(10)),
        current_camera.scale().to_string_radix(10, Some(6)),
        tile_level.level, tile_level.tile_size.to_string_radix(10, Some(6))
    );

    let tile_ids = find_tile_ids_under_camera(&current_camera, &tile_level);
    let (tile_views, create_count) = fill_registry_with_missing_tiles(
        &tile_ids, context.tile_registry.clone(), &tile_level);

    let mut best_sample_score: f64 = f64::NEG_INFINITY;
    let mut worst_sample_score: f64 = f64::INFINITY;

    // Group current sample snapshot by tile
    // Snapshots are taken by 'hard grid' location, i.e. where pixels lie on the screen
    // for the frame, and then are captured by a computer shader that reduces/aggregates.
    let grid_samples = context.grid_samples.lock().clone();
    let tile_samples: Vec<Arc<TileSampleScores>> = tile_views
        .iter()
        .filter(|tile| tile.read().anchor_orbit.is_none())
        .map(|tile| {
            let tile_g = tile.read();
            let geom = &tile_g.geometry;
            // Map the samples taken by the GPU into complex tiles ScoutEngine uses to 'box' them
            let tile_samples = grid_samples_in_tile(tile.clone(), &grid_samples);
            trace!("{:?} found {} samples that fit within", tile_g.id,  tile_samples.len());
            // Score each taken sample, so they can be sorted in order to spawn.
            // This way, configuration can control not only how many reference orbits
            // are spawned from samples, but also based on quality.
            let mut scores: Vec<SampleScore> = tile_samples
                .iter()
                .map(|sample| SampleScore::new(sample, geom))
                .collect();
            // Now sort!
            scores
                .sort_by(|a, b|
                    b.total_score.partial_cmp(&a.total_score).unwrap_or(Ordering::Equal));
            trace!("{:?} sample scores: {:?}", tile_g.id, scores);

            // Log the best/worst sample scores.
            if let Some(s) = scores.first() {
                if s.total_score > best_sample_score {
                    best_sample_score = s.total_score;
                }
            }
            if let Some(s) = scores.last() {
                if s.total_score < worst_sample_score {
                    worst_sample_score = s.total_score;
                }
            }

            // We wrap in an Arc here because we pass each of these to a separate thread for long
            // reference orbit computation.
            Arc::new(TileSampleScores { tile: tile.clone(), scores })
        })
        .collect();

    // Spawn refence orbits from tile samples (candidate seeds)
    let orbit_scores = spawn_orbits_from_tile_samples(
        &tile_samples, tp.clone(), context.clone(), current_camera.clone()
    ).await;

    let mut best_orbit_score: f64 = f64::NEG_INFINITY;
    let mut worst_orbit_score: f64 = f64::INFINITY;

    let mut live_orbits: Vec<LiveOrbit> = orbit_scores
        .iter()
        .map(|s| {
            if s.total_score > best_orbit_score {
                best_orbit_score = s.total_score;
            }
            if s.total_score < worst_orbit_score {
                worst_orbit_score = s.total_score;
            }
            s.orbit.clone()
        })
        .collect();

    // Insert all created orbits into global pool for housekeeping, orbit culling.
    let mut pool_g = context.living_orbits.lock();
    pool_g.append(&mut live_orbits);

    context.write_diagnostics(
        format!("Scout evaluated {} tiles at level {} with tile_size={}. Created {} new tiles.\n\
        Found {} tiles that needed anchors. Spawned {} new orbits.\n\tBest & Worst grid sample scores: {} & {}\n\
        \tBest & Worst reference orbit scores: {} & {}",
            tile_views.len(), tile_level.level, tile_level.tile_size.to_string_radix(10, Some(6)),
            create_count, tile_samples.len(), orbit_scores.len(), best_sample_score, worst_sample_score,
            best_orbit_score, worst_orbit_score
    ));

    context.context_changed();
}

async fn spawn_orbits_from_tile_samples(
    tile_samples: &[Arc<TileSampleScores>],
    tp: ThreadPool,
    context: ScoutContext,
    current_camera: CameraSnapshot,
) -> Vec<OrbitScore> {
    let handles: Vec<RemoteHandle<Vec<OrbitScore>>> = tile_samples
        .iter()
        .filter_map(|tile_samples| {
            let res = tp.spawn_with_handle(
                spawn_orbits_for_tile(tile_samples.clone(), tp.clone(), context.clone(),
                                      current_camera.clone())
            );
            if let Ok(h) = res {Some(h)} else {
                error!("Failed to schedule spawn_orbits_for_tile");
                None
            }
        })
        .collect();

    let live_orbits: Vec<Vec<OrbitScore>> = join_all(handles).await;
    live_orbits.into_iter().flatten().collect()
}

async fn spawn_orbits_for_tile(
    tile_samples: Arc<TileSampleScores>,
    tp: ThreadPool,
    context: ScoutContext,
    current_camera: CameraSnapshot,
) -> Vec<OrbitScore> {
    let num_orbits_to_spawn = context.config.lock().num_seeds_to_spawn_per_tile_eval;
    let mut seeds: Vec<Complex> = Vec::new();

    for i in 0..num_orbits_to_spawn {
        if let Some(s) = tile_samples.scores.get(i as usize) {
            let tile_g = tile_samples.tile.read();
            trace!("{:?} will spawn seed {} with depth={}, dist={} escape_penalty={} total_score={}.\tTile center={}",
                tile_g.id, s.seed.to_string_radix(10, Some(10)),
                s.depth, s.dist, s.escape_penalty, s.total_score,
                tile_g.geometry.center().to_string_radix(10, Some(10))
            );
            seeds.push(s.seed.clone());
        }
    }
    
    let handles: Vec<RemoteHandle<LiveOrbit>> = seeds
        .iter()
        .filter_map(|seed| {
            let res = tp.spawn_with_handle(
                start_reference_orbit(seed.clone(), context.orbit_id_factory.clone(),
                                      context.config.clone(),
                                      current_camera.clone().frame_stamp().clone())
            );
            if let Ok(h) = res { Some(h) } else {
                error!("Failed to spawn create_new_reference_orbit");
                None
            }
        })
        .collect();

    let live_orbits: Vec<LiveOrbit> = join_all(handles).await;

    let mut tile_g = tile_samples.tile.write();
    let geom = &tile_g.geometry;

    let mut scored_orbits: Vec<OrbitScore> = live_orbits
        .iter()
        .map(|orb| OrbitScore::new(orb.clone(), geom))
        .collect();
    scored_orbits
        .sort_by(|a, b| b.total_score
            .partial_cmp(&a.total_score)
            .unwrap_or(Ordering::Equal));

    if let Some(scored_orbit) = scored_orbits.first() {
        tile_g.anchor_orbit = Some(scored_orbit.orbit.clone());
        let orb_g = scored_orbit.orbit.read();

        trace!("Anchored orbit {} c_ref={} to tile {:?} orbit.len={} escape={:?} contraction={} \
        period={:?} z_min={} a_max={} with precision={}\n\t\tScores: depth={} dist={} contraction={} total_score={}",
            orb_g.orbit_id, orb_g.c_ref().to_string_radix(10, Some(10)),
            tile_g.id, orb_g.orbit.len(), orb_g.quality_metrics.escape_index,
            orb_g.contraction().to_string_radix(10, Some(4)),
            orb_g.quality_metrics.period,
            orb_g.quality_metrics.z_min.to_string_radix(10, Some(6)),
            orb_g.quality_metrics.a_max.to_string_radix(10, Some(6)),
            orb_g.c_ref().prec().0,
            scored_orbit.depth, scored_orbit.dist, scored_orbit.contraction,
            scored_orbit.total_score
        );
    }

    scored_orbits
}
