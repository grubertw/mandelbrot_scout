use crate::scout_engine::{ScoutEngineContext, CameraSnapshotReceiver, GpuFeedbackReceiver, OrbitObservationsReceiver};
use crate::scout_engine::orbit::*;
use crate::scout_engine::tile::*;
use crate::scout_engine::tasks::*;
use crate::scout_engine::utils::*;

use crate::signals;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use log::{trace, debug, info, error};
use rug::{Complex};

use futures::task::SpawnExt;
use futures::StreamExt;
use futures::select;
use futures::future::{join_all, RemoteHandle};
use futures::executor::{ThreadPool};

use iced_winit::winit::window::Window;

// ScoutEngine's internal work loop, a long-lived future that uses select to poll the 
// the camera snaphot & gpu feedback channels.
pub async fn scout_worker(window: Arc<Window>, tp: ThreadPool, 
    context: Arc<ScoutEngineContext>, 
    camera_snapshot_rx: CameraSnapshotReceiver, 
    gpu_feedback_rx: GpuFeedbackReceiver,
    orbit_observations_rx: OrbitObservationsReceiver
) {
    info!("Scout Worker started!");
    let mut snap_rx = camera_snapshot_rx.fuse();
    let mut gpu_rx = gpu_feedback_rx.fuse();
    let mut orb_obs_rx = orbit_observations_rx.fuse();

    // Add the first set of tile levels before start of the main loop.
    add_tile_levels(
        context.tile_levels.clone(),
        context.config.init_rug_precision,
        context.config.tile_level_addition_increment,
        context.config.base_tile_size,
        context.config.ideal_tile_pix_width
    );

    loop {
        select! {
            cs_res = snap_rx.next() => {
                match cs_res {
                    Some(snapshot) => {
                        handle_camera_snapshot(
                            window.clone(), tp.clone(), context.clone(), 
                            Arc::new(snapshot)
                        ).await;
                    }
                    None => {break;}
                }
            },
            gpu_res = gpu_rx.next() => {
                match gpu_res {
                    Some(feedback) => {
                        debug!("Scout Worker received GpuFeedback={:?}", feedback);
                    }
                    None => {break;}
                }        
            },
            orb_obs_res = orb_obs_rx.next() => {
                match orb_obs_res {
                    Some(feedback) => {
                        handle_orbit_observations(
                            window.clone(), tp.clone(), context.clone(), 
                            Arc::new(feedback)
                        ).await;
                    }
                    None => {break;}
                }        
            },
        };
    }
}

async fn handle_camera_snapshot(
    window: Arc<Window>, tp: ThreadPool, context: Arc<ScoutEngineContext>,
    snapshot: Arc<signals::CameraSnapshot>
) {
    info!("Scout Worker received camera snapshot {:?}", snapshot);
    {
        let mut last_snap_g = context.last_camera_snapshot.lock(); 
        last_snap_g.frame_stamp = snapshot.frame_stamp.clone();
        last_snap_g.center = snapshot.center.clone();
        last_snap_g.scale = snapshot.scale.clone();
        last_snap_g.screen_extent_multiplier = snapshot.screen_extent_multiplier;
    }

    tp.spawn_ok(
        evaluate_tile_views_for_orbit_deficiencies(
            window.clone(), tp.clone(), context.clone(),
        )
    );
}

async fn handle_orbit_observations(
    window: Arc<Window>, tp: ThreadPool, context: Arc<ScoutEngineContext>,
    orbit_observations: Arc<Vec<signals::OrbitObservation>>
) {
    let mut obs_for_log: Vec<signals::OrbitObservation> = Vec::new();
    for &obs in orbit_observations.iter() {
        if obs.feedback.perturb_attempted_count > 0 {
            obs_for_log.push(obs.clone());
        }
    }
    info!("Scout Worker received {} Orbit Observations (with attempted perturbance)", 
        obs_for_log.len());

    tp.spawn_ok(
        update_context_with_orbit_observations(
            window.clone(), context.clone(),
            orbit_observations.clone()
        )
    );
}

// Checks the local scores of TileOrbitViews. If the score falls below a threshold,
// spawn a new orbit for that tile. Note that only tiles under the current camera 
// extent are checked. Also note that only the current camera's "ideal" tile level
// is checked, such that orbits are only ever spawned into one tile level - which 
// can be considered 'active' for the current scale.
async fn evaluate_tile_views_for_orbit_deficiencies(
    window: Arc<Window>, tp: ThreadPool, context: Arc<ScoutEngineContext>,
) {
    info!("Evaluating TileView(s)!");
    let last_snapshot = context.last_camera_snapshot.lock().clone();

    // First determine the active tile level for the current camera viewport
    let (active_tile_level, add_levels) = find_active_tile_level_for_scale(
        &context.tile_levels, &last_snapshot.scale); 
    
    if add_levels {
        // We are below the last level. Add more!
        add_tile_levels(context.tile_levels.clone(),
            last_snapshot.scale.prec(),
            context.config.tile_level_addition_increment,
            context.config.base_tile_size,
            context.config.ideal_tile_pix_width);
    }
    debug!("Found active tile level {:?}", active_tile_level);

    // Determine which TileIds to check, based on the active tile level
    // and which tiles are currently 'under' the camera/viewport.
    let tile_ids_under_cam = 
        find_active_tiles_under_camera(
            &active_tile_level, &last_snapshot
        );
    
    // Check next if any TileOrbitView(s) need to be created. Note that a 
    // local_score of 0.0 is used for these.
    let new_tiles = insert_tile_views_if_not_found(
        &tile_ids_under_cam,
        context.tile_registry.clone(),
        |tile_id| TileOrbitView::new(
                    tile_id, 
                    &active_tile_level.tile_size, 
                    context.config.num_orbits_to_spawn_per_tile,
                    context.config.initial_max_orbits_per_tile as usize,
                    last_snapshot.frame_stamp.clone())
    );
    debug!("Inserted {} new tiles!", new_tiles.len());

    // Find all tiles who's scores fall below the deficiency threshold
    let tiles_needing_orbits = 
        find_tile_views_with_deficient_scores(
            context.config.clone(), &tile_ids_under_cam, 
            context.tile_registry.clone());
    debug!("{} found that need better orbits!", tiles_needing_orbits.len());

    let spawned_orbits = spawn_orbits_for_tile_views(
        tiles_needing_orbits, tp.clone(), context.clone(), 
        last_snapshot.frame_stamp.clone()
    ).await;

    debug!("{} LiveOrbit(s) created for this spawn operation!", spawned_orbits.len());

    upkeep_orbit_pool(
        context.living_orbits.clone(), 
        spawned_orbits, 
        context.config.max_live_orbits,
        &last_snapshot.frame_stamp
    );

    context.context_changed.store(true, Ordering::Relaxed);

    // Send a signal to the winit window to wake up the render loop and redraw the viewport
    // This mechanism is largely in place so that the render loop need not run continually 
    // and communications to/from the GPU can be tightly controlled. 
    window.request_redraw();
    info!("TileView(s) evaluated!");
}

async fn spawn_orbits_for_tile_views(
    tile_views: Vec<TileView>, 
    tp: ThreadPool, 
    context: Arc<ScoutEngineContext>,
    frame_stamp: signals::FrameStamp
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
    context: Arc<ScoutEngineContext>,
    frame_stamp: signals::FrameStamp
) -> Vec<LiveOrbit> {
    let mut seeds: Vec<Complex> = Vec::new();

    let tile_id = {    
        let view_g = view.read();
        let mut num_rand_orbits_to_spawn = view_g.num_orbits_per_spawn;
        let spawn_tile_center = view_g.orbit_stats.len() == 0;
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

        view_g.tile.clone()
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
    
    trace!("{} LiveOrbit(s) created for {:?} --- from seeds: {:?}", 
        live_orbits.len(), &tile_id, seeds);

    {
        let mut view_g = view.write();
        for orb in &live_orbits {
            view_g.weak_orbits.push((STARTING_LOCAL_SCORE, Arc::downgrade(&orb)));

            // Flag these orbits as 'native' in stats.
            // Also init's the stat for further observations.
            let orb_g = orb.read();
            view_g.orbit_stats.insert(orb_g.orbit_id, 
                TileOrbitStats::new(orb_g.orbit_id, true, orb_g.min_valid_perturb_index,
                    frame_stamp.clone()));
        }
    }
    
    live_orbits
}

async fn update_context_with_orbit_observations(
    window: Arc<Window>, context: Arc<ScoutEngineContext>,
    orbit_observations: Arc<Vec<signals::OrbitObservation>>
) {
    let orbits_updated = update_tile_views_with_orbit_observations(
        context.config.clone(),
        context.tile_registry.clone(), 
        orbit_observations.clone()
    ).await;

    if orbits_updated {
        context.context_changed.store(true, Ordering::Relaxed);
        window.request_redraw();
    }
    info!("Update context with orbit observations complete!");
}
