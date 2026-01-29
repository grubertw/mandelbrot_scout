use crate::scout_engine::{ScoutConfig, CameraSnapshotReceiver, GpuFeedbackReceiver};
use crate::scout_engine::orbit::*;
use crate::scout_engine::tile::*;
use crate::scout_engine::tasks::*;

use crate::signals;

use std::sync::{Arc, Mutex};

use log::{debug, info};

use futures::task::SpawnExt;
use futures::StreamExt;
use futures::select;
use futures::future::{join_all, RemoteHandle};
use futures::executor::{ThreadPool};

use iced_winit::winit::window::Window;

// ScoutEngine's internal work loop, a long-lived future that uses select to poll the 
// the camera snaphot & gpu feedback channels.
pub async fn scout_worker(window: Arc<Window>, config: Arc<ScoutConfig>, 
        tp: ThreadPool, living_orbits: LivingOrbits, tile_registry: TileRegistry,
        cameara_snapshot_rx: CameraSnapshotReceiver, 
        gpu_feedback_rx: GpuFeedbackReceiver) {
    
    let id_fac = Arc::new(Mutex::new(IdFactory::new()));

    let mut snap_rx = cameara_snapshot_rx.fuse();
    let mut gpu_rx = gpu_feedback_rx.fuse();

    info!("Scout Worker started!");

    loop {
        select! {
            cs_res = snap_rx.next() => {
                match cs_res {
                    Some(snapshot) => {
                        info!("Scout Worker received camera snapshot {:?}", snapshot);
                        // Create a new set of reference orbits asynchronously.
                        // This is a computationally heavy operation, so it is best to 
                        // invoke on the threadpool without waiting.
                        tp.spawn_ok(
                            create_new_reference_orbits_from_camera_snapshot(
                            window.clone(), config.clone(), tp.clone(), id_fac.clone(),
                            living_orbits.clone(), tile_registry.clone(), 
                            Arc::new(snapshot))
                        );
                    }
                    None => {
                        break;
                    }
                }
            },
            gpu_res = gpu_rx.next() => {
                match gpu_res {
                    Some(feedback) => {
                        debug!("Scout Worker received GpuFeedback={:?}", feedback);
                    }
                    None => {
                        break;
                    }
                }
                
            },
        };
    }
}

async fn create_new_reference_orbits_from_camera_snapshot(
        window: Arc<Window>, config: Arc<ScoutConfig>, 
        tp: ThreadPool, id_fac: OrbitIdFactory, 
        living_orbits: LivingOrbits, tile_registry: TileRegistry, 
        snapshot: Arc<signals::CameraSnapshot>) {
    let seeds = create_new_orbit_seeds_from_camera_snapshot(snapshot.clone()).await;
    debug!("Created orbit seeds from snapshot: {:?}", seeds);

    // Schedule thread-pool execution for the creation of reference orbits from seeds.
    let handles = seeds.iter().fold(Vec::<RemoteHandle<LiveOrbit>>::new(), |mut acc, seed| {
        let res = tp.spawn_with_handle(
            create_new_reference_orbit(seed.clone(), id_fac.clone(), 
                config.max_orbit_iters, config.rug_precision,
                snapshot.frame_stamp.frame_id)
        );
        if let Ok(h) = res {
            acc.push(h);
        };
        acc
    });

    // Wait for all the orbit computations to complete on the thread-pool
    let results: Vec<LiveOrbit> = join_all(handles).await;
    let r_len = &results.len();

    // Insert new orbits into living orbits
    for orb in &results {
        let mut lo_mut = living_orbits.lock().unwrap();
        lo_mut.push(orb.clone());
    }
    debug!("New reference orbit results collected! results.len={} living_orbits.len()={}", 
        r_len, living_orbits.lock().unwrap().len());

    // Recalculate weights for all entires in the pool after addition
    // And with the snapshot provided on creation of these new orbits
    weigh_living_orbits(living_orbits.clone(), Some(snapshot), None).await;

    // Score the orbits
    score_living_orbits(living_orbits.clone()).await;

    // Rank/order the orbits
    rank_living_orbits(living_orbits.clone()).await;

    // Insert into our complex-anchored tiles.
    insert_new_orbits_into_tile_registry(config.clone(), &results, tile_registry.clone()).await;

    // Rank orbits in each TileOrbitView of the TileRegistry
    rank_tile_registry_orbits(tile_registry.clone()).await;

    // Trim the pool
    trim_living_orbits(config.max_live_orbits, living_orbits.clone()).await;

    let orb_pool_g = living_orbits.lock().unwrap();
    for live_orb in orb_pool_g.iter() {
        let orb = live_orb.lock().unwrap();
        debug!("Orbit {} has score {}\tFrom weights: w_dist={} w_depth={} w_age={}\tAt c_ref_df={:?}", 
            orb.orbit_id, orb.current_score, orb.weights.w_dist, orb.weights.w_depth, orb.weights.w_age, orb.c_ref_df);
    };
    drop(orb_pool_g);

    // Send a signal to the winit window to wake up the render loop and redraw the viewport
    // This mechinism is largely in place so that the render loop need not run continually 
    // and communications to/from the GPU can be tightly controlled. 
    window.request_redraw();
}
