use crate::scout_engine::{ScoutContext, CameraSnapshotReceiver, TileObservationsReceiver};
use crate::scout_engine::orbit::*;
use crate::scout_engine::tile::*;
use crate::scout_engine::tasks::*;
use crate::scout_engine::utils::*;

use crate::signals::{CameraSnapshot, FrameStamp};

use std::sync::Arc;

use log::{trace, debug, info, error};
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
    tile_observations_rx: TileObservationsReceiver
) {
    info!("Scout Worker started!");
    let mut snap_rx = camera_snapshot_rx.fuse();
    let mut tile_obvs_rx = tile_observations_rx.fuse();

    // Initialize the level-0 tile grid
    initialize_tile_grid(
        &context.last_camera_snapshot.lock(),
        context.tile_registry.clone(),
        context.config.num_orbits_to_spawn_per_tile,
        context.config.max_tile_anchor_failure_attempts,
        context.config.init_rug_precision,
    );    
    
    loop {
        select! {
            cs_res = snap_rx.next() => {
                match cs_res {
                    Some(snapshot) => {
                        handle_camera_snapshot(
                            tp.clone(), context.clone(), Arc::new(snapshot)
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
    snapshot: Arc<CameraSnapshot>
) {
    info!("Scout Worker received camera snapshot {:?}", snapshot); 
    if let Some(snap_inner) = Arc::into_inner(snapshot) {
        let mut last_snap_g = context.last_camera_snapshot.lock();
        *last_snap_g = snap_inner;
    }

    tp.spawn_ok(
        evaluate_tile_orbit_anchors(tp.clone(), context.clone())
    );
}

async fn evaluate_tile_orbit_anchors(
    tp: ThreadPool, context: ScoutContext,
) {
    info!("Evaluating Orbit Anchors of TileView(s)...");
    let current_camera = context.last_camera_snapshot.lock().clone();

    let tile_views = find_top_tiles_under_camera(&current_camera, context.tile_registry.clone());

    // Gate Tile evaluation/orbit spawning
    // While the tile-grid is created at the beginning, spawning too many orbits too early 
    // hinders usablility. The chances are low that an orbit will be found that fits a level-0
    // tile anyways.
    if tile_views.len() > context.config.level_zero_tile_constraint_before_eval as usize {
        debug!("L-Zero TileCount {} too large to begin tile evaluation. constraint={}", 
            tile_views.len(), context.config.level_zero_tile_constraint_before_eval);
        return
    }

    let mut tiles_to_seed: Vec<TileView> = Vec::new();
    find_tiles_needing_anchors(&tile_views, &current_camera, &mut tiles_to_seed);

    if try_to_anchor_tiles_from_pool(context.living_orbits.clone(), &tiles_to_seed) {
        tiles_to_seed.clear();
        find_tiles_needing_anchors(&tile_views, &current_camera, &mut tiles_to_seed);
        debug!("Anchors were found in global pool! Now will only seed {} tiles",
            tiles_to_seed.len());
    }

    info!("Found {} tiles that need anchors", tiles_to_seed.len());

    let spawned_orbits = spawn_orbits_for_tile_views(
        tiles_to_seed, tp.clone(), context.clone(), 
        current_camera.frame_stamp().clone()
    ).await;

    debug!("{} LiveOrbit(s) created for this spawn operation!", spawned_orbits.len());

    upkeep_orbit_pool(
        context.living_orbits.clone(), 
        spawned_orbits, 
        context.config.max_live_orbits,
        current_camera.frame_stamp()
    );

    // Flag the context has changed, which also wakes up the render loop!
    context.context_changed();

    info!("TileView(s) evaluated!");
}

async fn spawn_orbits_for_tile_views(
    tile_views: Vec<TileView>, 
    tp: ThreadPool, 
    context: ScoutContext,
    frame_stamp: FrameStamp
) -> Vec<LiveOrbit> {
    let handles: Vec<RemoteHandle<Vec<LiveOrbit>>> = tile_views
        .iter()
        .filter_map(|view| {
            let res = tp.spawn_with_handle(
                spawn_orbits_for_tile(view.clone(), tp.clone(), context.clone(), frame_stamp.clone())
            );
            if let Ok(h) = res {Some(h)} else {
                error!("Failed to schedule spawn_orbits_for_tile");
                None
            }
        })
        .collect();

    let live_orbits: Vec<Vec<LiveOrbit>> = join_all(handles).await;
    let flattened_orbits: Vec<LiveOrbit> = live_orbits.into_iter().flatten().collect();
    flattened_orbits
}

async fn spawn_orbits_for_tile(
    view: TileView, tp: ThreadPool, 
    context: ScoutContext,
    frame_stamp: FrameStamp
) -> Vec<LiveOrbit> {
    let mut seeds: Vec<Complex> = Vec::new();

    let tile_id = {    
        let view_g = view.read();
        let mut num_rand_orbits_to_spawn = view_g.num_orbits_per_spawn;
        let spawn_tile_center = view_g.failed_orbits.len() == 0;
        let geometry = view_g.geometry.clone();
        
        if spawn_tile_center {
            // If the tile center seed must be spawned, reduced the number of 
            // random seeds spawned
            num_rand_orbits_to_spawn -= 1;

            let mut center = generate_tile_candidate_seeds(
                &geometry, OrbitSeedStrategy::Center, 1, context.orbit_seed_rng.clone());
            seeds.append(&mut center);
        }
        if num_rand_orbits_to_spawn > 0 {
            seeds.append(
                &mut generate_tile_candidate_seeds(
                    &geometry, 
                    OrbitSeedStrategy::RandomInDisk,
                    num_rand_orbits_to_spawn as usize, context.orbit_seed_rng.clone()
                )
            );
        }

        view_g.id.clone()
    };
    
    let handles: Vec<RemoteHandle<LiveOrbit>> = seeds
        .iter()
        .filter_map(|seed| {
            let res = tp.spawn_with_handle(
                create_new_reference_orbit(seed.clone(), context.orbit_id_factory.clone(), 
                    context.config.max_orbit_iters,
                    frame_stamp.clone())
            );
            if let Ok(h) = res { Some(h) } else {
                error!("Failed to spawn create_new_reference_orbit");
                None
            }
        })
        .collect();

    let live_orbits: Vec<LiveOrbit> = join_all(handles).await;
    
    {
        let mut orbits_anchored_count = 0;
        let mut view_g = view.write();
        for orb in &live_orbits {
            if view_g.try_anchor_orbit(orb.clone(), true) {
                orbits_anchored_count += 1;
            }
            else if view_g.should_split() {
                view_g.split();
            }
        }

        trace!("{} LiveOrbit(s) created for {:>4?}. {} were successfully anchored!", 
            live_orbits.len(), &tile_id, orbits_anchored_count);
    }
    
    live_orbits
}
