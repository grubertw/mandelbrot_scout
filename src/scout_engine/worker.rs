use crate::scout_engine::{ScoutContext, CameraSnapshotReceiver, TileObservationsReceiver, ExploreSignalReceiver, ScoutSignal};
use crate::scout_engine::orbit::*;
use crate::scout_engine::tile::*;
use crate::scout_engine::tasks::*;
use crate::scout_engine::utils::*;

use crate::signals::{CameraSnapshot};

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
            obvs_res = tile_obvs_rx.next() => {
                match obvs_res {
                    Some(feedback) => {
                        debug!("Scout Worker received {} TileObservation(s)", feedback.len());
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
    if snapshot.scale().to_f64() > starting_scale && !auto_start {
        return;
    }

    tp.spawn_ok(
        evaluate_tile_orbit_anchors(tp.clone(), context.clone())
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
            let tile_reg_g = context.tile_registry.read();
            if tile_reg_g.len() == 0 {
                // Recalculate level zero tile size
                let current_camera = context.last_camera_snapshot.lock().clone();
                let starting_tile_pixel_span = context.config.lock().starting_tile_pixel_span;
                let mut tile_size = current_camera.scale().clone();
                tile_size *= starting_tile_pixel_span;

                let mut level_zero_g = context.level_zero.lock();
                *level_zero_g = TileLevel::new(tile_size);

                info!("Resetting level zero tile size: {:?}", level_zero_g);
            }

            tp.spawn_ok(
                evaluate_tile_orbit_anchors(tp.clone(), context.clone())
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


async fn evaluate_tile_orbit_anchors(
    tp: ThreadPool, context: ScoutContext,
) {
    info!("Evaluating Orbit Anchors of TileView(s)...");
    let current_camera = context.last_camera_snapshot.lock().clone();

    // Search for tiles is hierarchal, with the rule that only level-zero tiles
    // are stored in the tile registry. All tiles below level 0 are found within
    // those tiles, as children. This better respects fractal behavior
    let level_zero = context.level_zero.lock().clone();
    let top_tile_ids = find_tile_ids_under_camera(&current_camera, &level_zero);
    let (tile_views, create_count) = fill_registry_with_missing_tiles(
        &top_tile_ids, context.tile_registry.clone(), context.config.clone(), &level_zero);
    let mut tile_creation_count = create_count;

    debug!("Updating Tile Config {:?}", context.config.lock());
    let mut tiles_to_reset: Vec<TileView> = Vec::new();
    let max_user_iters = context.config.lock().max_user_iters;
    update_tile_config(&tile_views, context.config.clone(), &mut tiles_to_reset);

    for tile in &tiles_to_reset {
        let mut tile_g = tile.write();
        let orbits_to_continue = tile_g.reset();
        for orb in orbits_to_continue {
            tp.spawn_ok(continue_reference_orbit(orb, max_user_iters));
        }
    }

    info!("Level Zero tile_size={} and {} tiles found under viewport",
        level_zero.tile_size.to_string_radix(10, Some(6)),
        top_tile_ids.len());

    let mut tiles_to_anchor: Vec<TileView> = Vec::new();
    let mut total_spawned_orbits: Vec<LiveOrbit> = Vec::new();
    let mut tile_split_count = 0;
    let mut tile_anchor_count = 0;
    let mut exploration_count = 0;
    loop {
        find_tiles_needing_anchors(&tile_views, &current_camera, &mut tiles_to_anchor);

        let (spawned_orbits, tiles_split, tiles_anchored)
            = spawn_orbits_for_tile_views(
                &tiles_to_anchor, tp.clone(), context.clone(), current_camera.clone(),
        ).await;
        tile_split_count += tiles_split;
        tile_anchor_count += tiles_anchored;

        total_spawned_orbits.extend(spawned_orbits);

        tiles_to_anchor.clear();
        find_tiles_needing_anchors(&tile_views, &current_camera, &mut tiles_to_anchor);

        let min_size_count =
            count_tiles_that_reached_min_size_for_current_viewport_and_should_split(
                &tiles_to_anchor, &current_camera);
        
        if exploration_count > context.config.lock().exploration_budget {
            info!("Exploration count of {} exceeded for this cycle. Stopping Eval!", 
                exploration_count);
            break;
        }

        if tiles_to_anchor.len() == min_size_count {
            info!("{} Tiles need anchors but have reached minimum size for viewport. Stopping eval!",
                min_size_count);
            break;
        }
        
        exploration_count += 1;
    }

    // Try again to anchor from orbits in global pool
    tiles_to_anchor.clear();
    find_tiles_needing_anchors(&tile_views, &current_camera, &mut tiles_to_anchor);
    let num_orbits_anchored_from_pool = 
        try_to_anchor_tiles_from_pool(context.living_orbits.clone(), &tiles_to_anchor);

    let num_orbits_spawned = total_spawned_orbits.len();
    upkeep_orbit_pool(
        context.living_orbits.clone(), 
        total_spawned_orbits,
        context.config.lock().max_live_orbits,
        current_camera.frame_stamp()
    );

    //
    // Diagnostics code follows
    //
    tile_creation_count += tile_split_count;
    let mut smallest_tile_size = level_zero.tile_size.clone();
    find_smallest_tile_size(&tile_views, &mut smallest_tile_size);
    tiles_to_anchor.clear();
    find_tiles_needing_anchors(&tile_views, &current_camera, &mut tiles_to_anchor);

    let mut tiles_with_anchors: Vec<TileView> = Vec::new();
    find_anchor_orbits(&tile_views, &current_camera, &mut tiles_with_anchors);

    let largest_anchor_str = tiles_with_anchors.first().map(|first|{
        let anchor = tiles_with_anchors.iter().fold((0.0, first.clone()), |acc, tile| {
            let tile_g = tile.read();
            let orb_g = tile_g.anchor_orbit.as_ref().unwrap().read();
            if orb_g.r_valid().to_f64() > acc.0 {
                (orb_g.r_valid().to_f64(), tile.clone())
            }
            else {acc}
        });
        let tile_g = anchor.1.read();
        let orb_g = tile_g.anchor_orbit.as_ref().unwrap().read();

        format!("Anchored {:?} with size {} has largest r_valid={:.4e} and iter count {:?}",
            tile_g.id,
            tile_g.level.tile_size.to_string_radix(10, Some(6)),
            anchor.0,
            orb_g.escape_index()
        )
    });
    let smallest_anchor_str = tiles_with_anchors.first().map(|first|{
        let anchor = tiles_with_anchors.iter().fold((1.0, first.clone()), |acc, tile| {
            let tile_g = tile.read();
            let orb_g = tile_g.anchor_orbit.as_ref().unwrap().read();
            if orb_g.r_valid().to_f64() < acc.0 {
                (orb_g.r_valid().to_f64(), tile.clone())
            }
            else {acc}
        });
        let tile_g = anchor.1.read();
        let orb_g = tile_g.anchor_orbit.as_ref().unwrap().read();

        format!("Anchored {:?} with size {} has smallest r_valid={:.4e} and iter count {:?}",
                tile_g.id,
                tile_g.level.tile_size.to_string_radix(10, Some(6)),
                anchor.0,
                orb_g.escape_index()
        )
    });

    let best_tile_str = tiles_to_anchor.first().map(|best| {
        let mut best_pair = (0.0, best.clone());
        find_tile_with_best_coverage(&tiles_to_anchor, &mut best_pair);

        let best_g = best_pair.1.read();
        format!("Unanchored {:?} with size {} has best coverage {:.3e}",
                best_g.id,
                best_g.level.tile_size.to_string_radix(10, Some(6)),
                best_pair.0)
    });
    let worst_tile_str = tiles_to_anchor.first().map(|worst| {
        let mut worst_pair = (1.0, worst.clone());
        find_tile_with_worst_coverage(&tiles_to_anchor, &mut worst_pair);

        let worst_g = worst_pair.1.read();
        format!("Unanchored {:?} with size {} has worst coverage {:.3e}",
                worst_g.id,
                worst_g.level.tile_size.to_string_radix(10, Some(6)),
                worst_pair.0)
    });

    context.write_diagnostics(
        format!(
            "Tiles Created {}. Orbits Spawned {}. Largest tile size {} (in pixels {:.1}). \
            Smallest tile size {} (in pixels {:.1}).\nTiles Anchored this cycle {}. Total tiles anchored {}. Tiles that still need anchors {}. \
            Num Tile Splits {}. Num Tiles Reset {}. Tiles Anchored from Pool {}.\n\n{}\n{}\n\n{}\n{}",
            tile_creation_count, num_orbits_spawned, level_zero.tile_size.to_string_radix(10, Some(6)),
            level_zero.tile_size.to_f64() / current_camera.scale().to_f64(),
            smallest_tile_size.to_string_radix(10, Some(6)),
            smallest_tile_size.to_f64() / current_camera.scale().to_f64(),
            tile_anchor_count, tiles_with_anchors.len(), tiles_to_anchor.len(), tile_split_count, 
            tiles_to_reset.len(), num_orbits_anchored_from_pool,
            largest_anchor_str.unwrap_or("".to_string()), smallest_anchor_str.unwrap_or("".to_string()),
            best_tile_str.unwrap_or("".to_string()), worst_tile_str.unwrap_or("".to_string()),
        )
    );

    // Flag the context has changed, which also wakes up the render loop!
    context.context_changed();
}

async fn spawn_orbits_for_tile_views(
    tile_views: &[TileView],
    tp: ThreadPool, 
    context: ScoutContext,
    current_camera: CameraSnapshot,
) -> (Vec<LiveOrbit>, u32, u32) {
    let handles: Vec<RemoteHandle<(Vec<LiveOrbit>, u32, u32)>> = tile_views
        .iter()
        .filter_map(|view| {
            let res = tp.spawn_with_handle(
                spawn_orbits_for_tile(view.clone(), tp.clone(), context.clone(),
                                      current_camera.clone())
            );
            if let Ok(h) = res {Some(h)} else {
                error!("Failed to schedule spawn_orbits_for_tile");
                None
            }
        })
        .collect();

    let live_orbits_with_counts: Vec<(Vec<LiveOrbit>, u32, u32)> = join_all(handles).await;
    let mut num_tiles_split: u32 = 0;
    let mut num_tiles_anchored: u32 = 0;

    let live_orbits: Vec<Vec<LiveOrbit>> = live_orbits_with_counts
        .iter()
        .map(|vvc| {
            num_tiles_split += vvc.1;
            num_tiles_anchored += vvc.2;
            vvc.0.clone()
        })
        .collect();

    let flattened_orbits: Vec<LiveOrbit> = live_orbits.into_iter().flatten().collect();
    (flattened_orbits, num_tiles_split, num_tiles_anchored)
}

async fn spawn_orbits_for_tile(
    view: TileView, tp: ThreadPool, 
    context: ScoutContext,
    current_camera: CameraSnapshot,
) -> (Vec<LiveOrbit>, u32, u32) {
    let mut seeds: Vec<Complex> = Vec::new();

    let tile_id = {    
        let view_g = view.read();
        let spawn_tile_center = view_g.candidate_orbits.len() == 0;
        let spawn_corners = view_g.candidate_orbits.len() == 1;
        let spawn_direction = view_g.candidate_orbits.len() > 2;
        let geometry = view_g.geometry.clone();
        
        if spawn_tile_center {
            let mut center = generate_tile_candidate_seeds(
                &geometry, OrbitSeedStrategy::Center, None, None, None
            );
            trace!("{:?} Spawn Tile Center: {}",
                view_g.id, center[0].to_string_radix(10, Some(10))
            );
            seeds.append(&mut center);
        }
        if spawn_corners {
            let mut corners = generate_tile_candidate_seeds(
                &geometry,
                OrbitSeedStrategy::Corners,
                None, None, None
            );
            trace!("{:?} Spawn Tile Corners: {} {} {} {}",
                view_g.id,
                corners[0].to_string_radix(10, Some(10)),
                corners[1].to_string_radix(10, Some(10)),
                corners[2].to_string_radix(10, Some(10)),
                corners[3].to_string_radix(10, Some(10))
            );
            seeds.append(&mut corners);
        }
        if spawn_direction {
            let candidates: Vec<(f64, Complex)> = view_g.candidate_orbits
                .iter()
                .filter_map(|tc| {
                    if let Some(o) = tc.orbit.upgrade() {
                        Some((tc.score.coverage, o.read().c_ref().clone()))
                    }
                    else {
                        None
                    }
                })
                .collect();
            let mut weighted_dir = generate_tile_candidate_seeds(
                &geometry, OrbitSeedStrategy::WeightedDirection,
                None, None, Some(&candidates)
            );
            trace!("{:?} Spawn weighted direction {}",
                view_g.id,
                weighted_dir[0].to_string_radix(10, Some(10))
            );
            seeds.append(&mut weighted_dir);
        }

        view_g.id.clone()
    };
    
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
    let mut num_tiles_split: u32 = 0;
    let mut num_tiles_anchored: u32 = 0;
    {
        let mut orbits_anchored_count = 0;
        let mut view_g = view.write();
        for orb in &live_orbits {
            if view_g.try_anchor_orbit(orb.clone(), true) {
                orbits_anchored_count += 1;
            }
            else if view_g.should_split() {
                orbits_anchored_count += view_g.split();
                num_tiles_split += 4;
            }
        }

        let mut tile_pix_ratio = view_g.level.tile_size.clone();
        tile_pix_ratio /= current_camera.scale();
        num_tiles_anchored += orbits_anchored_count;

        trace!("{} LiveOrbit(s) created for {:>4?} tile_size={}. {} were successfully anchored! num_candidates={}. curr_tile_pixels={}",
            live_orbits.len(), &tile_id, view_g.level.tile_size.to_string_radix(10, Some(6)),
            orbits_anchored_count, view_g.candidate_orbits.len(),
            tile_pix_ratio.to_string_radix(10, Some(4))
        );
    }

    (live_orbits, num_tiles_split, num_tiles_anchored)
}
