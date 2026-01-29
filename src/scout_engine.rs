pub mod orbit;
pub mod tile;
mod worker;
mod tasks;
mod utils;

use crate::scout_engine::orbit::*;
use crate::scout_engine::tile::*;
use crate::scout_engine::worker::*;

use super::signals;

use iced_winit::winit::window::Window;

use futures::channel;
use futures::executor::{ThreadPool};

use rug::{Complex};
use log::{trace, debug};

use std::collections::HashMap;
use std::time;
use std::sync::{Arc, Mutex};

type CameraSnapshotSender   = channel::mpsc::Sender<signals::CameraSnapshot>;
type CameraSnapshotReceiver = channel::mpsc::Receiver<signals::CameraSnapshot>;
type GpuFeedbackSender      = channel::mpsc::Sender<signals::GpuFeedback>;
type GpuFeedbackReceiver    = channel::mpsc::Receiver<signals::GpuFeedback>;

#[derive(Clone, Debug)]
pub struct HeuristicConfig {
    pub weight_1: f64, // Just a placeholder
}

#[derive(Clone, Debug)]
pub struct ScoutConfig {
    pub max_live_orbits: u32,
    pub max_orbit_iters: u32,
    pub rug_precision: u32,
    pub heuristic_config: HeuristicConfig,
    pub tile_levels: Vec<TileLevel>,
    pub exploration_budget: f64,
}

#[derive(Debug)]
pub struct ScoutEngine {
    // Our winit window 
    window: Arc<Window>,
    // Our startup configuration
    config: Arc<ScoutConfig>,
    // Internal thread-pool, used both for the primary event loop, and internal async tasks
    thread_pool: ThreadPool,
    // Our working pool of reference orbits
    living_orbits: LivingOrbits,
    // Where our (logical) tiles live
    tile_registry: TileRegistry,
    // Async channels
    cameara_snapshot_tx: CameraSnapshotSender,
    gpu_feedback_tx: GpuFeedbackSender,
}

impl ScoutEngine {
    pub fn new(window: Arc<Window>, config: ScoutConfig) -> Self {
        let config = Arc::new(config);
        let thread_pool = ThreadPool::new().expect("Failed to build ThreadPool for ScoutEngine");
        let living_orbits = Arc::new(Mutex::new(Vec::new()));
        let tile_registry = Arc::new(Mutex::new(HashMap::new()));

        // Create asynch channels for communicating between the render loop/thread and 
        // the long-lived ScoutEngine async task
        let (cameara_snapshot_tx, cameara_snapshot_rx) 
            = channel::mpsc::channel::<signals::CameraSnapshot>(10);
        let (gpu_feedback_tx, gpu_feedback_rx) 
            = channel::mpsc::channel::<signals::GpuFeedback>(10);

        // Launch ScoutEngine's long-lived task, which 'blocks' on the above async channels
        thread_pool.spawn_ok(
            scout_worker(window.clone(), config.clone(), thread_pool.clone(),
            living_orbits.clone(), tile_registry.clone(), cameara_snapshot_rx, gpu_feedback_rx)
        );

        Self {
            window, config, thread_pool, living_orbits, tile_registry,
            cameara_snapshot_tx, gpu_feedback_tx,
        } 
    }

    pub fn submit_camera_snapshot(&mut self, snapshot: signals::CameraSnapshot) {
        debug!("Camera Snapshot received: {:?}", snapshot);
        self.cameara_snapshot_tx.try_send(snapshot).ok();
    }

    pub fn submit_gpu_feedback(&mut self, feedback: signals::GpuFeedback) {
        debug!("Gpu Feedback received: {:?}", feedback);
        self.gpu_feedback_tx.try_send(feedback).ok();
    }

    // Determine which complex tiles lie within a bounding box, based on a complex
    // top left point and complex bottom right point. Bounding itself is driven by
    // the integer lattice calculation of TileId(s), based on tile_size, and is the 
    // same mechinizm for determining the base tile for any complex C.
    // 
    // It is very important to remember here that ScoutEngine is ONLY aware of 
    // complex-space tiles, and NOT screen-space tiles, whos geometry is derived 
    // from a rug::Float tile_size, rather than an integer pixel count (i.e. 16 or 32)
    // It is up to the scene to map these complex tiles to corresponding screen tiles.
    //
    // It is also important to remember that TileOrbitView(s) hold WEAK reference to 
    // orbits, which means there is a possibility that no LiveOrbit(s) are found - 
    // and hence no converted ReferenceOrbitDf(s). Concerning this reference orbit 
    // conversion, the cost is relativly cheap, as the LiveOrbit(s) themselves hold
    // the GPU-packed arrays (i.e. x4 per orbit)
    pub fn query_tiles_in_bounding_box(&self, 
        top_left: &Complex, bot_rght: &Complex
    )-> Vec<signals::TileOrbitViewDf> {
        //trace!("Query tiles in bounding box [{:?} {:?}]", top_left, bot_rght);
        let mut tl_tiles = Vec::<TileId>::new();
        let mut br_tiles = Vec::<TileId>::new();

        for level in self.config.tile_levels.iter() {
            let t_tl = TileId::from_point(top_left, &level.tile_size);
            let t_br = TileId::from_point(bot_rght, &level.tile_size);

            //trace!("Query Tile BBox at level {} is: [{:?} {:?}]", level.level, &t_tl, &t_br);
            tl_tiles.push(t_tl);
            br_tiles.push(t_br);
        }

        let mut num_tiles_in_bounds: u32 = 0;
        let mut num_orbits_found: u32 = 0;

        let mut df_tiles = Vec::<signals::TileOrbitViewDf>::new();
        let reg = self.tile_registry.lock().unwrap();
        for ((t_level, tile_id), view) in reg.iter() {
            let t_lev = *t_level as usize;
            let t_tl_x = tl_tiles[t_lev].tx;
            let t_tl_y = tl_tiles[t_lev].ty; 
            let t_br_x = br_tiles[t_lev].tx;
            let t_br_y = br_tiles[t_lev].ty;

            if tile_id.tx >= t_tl_x && tile_id.tx <= t_br_x &&
               tile_id.ty >= t_tl_y && tile_id.ty <= t_br_y {
                let mut live_orbits = Vec::<LiveOrbit>::new();
                for weak_orb in &view.weak_orbits {
                    if let Some(orb) = weak_orb.upgrade() {
                        live_orbits.push(orb);
                    }
                }
                let df_orbits: Vec<signals::ReferenceOrbitDf> = live_orbits.iter().map(|orb_g| {
                    let orb = orb_g.lock().unwrap();
                    num_orbits_found += 1;
                    signals::ReferenceOrbitDf {
                        orbit_id: orb.orbit_id,
                        c_ref: orb.c_ref_df,
                        orbit_re_hi: orb.gpu_payload.re_hi.clone(),
                        orbit_re_lo: orb.gpu_payload.re_lo.clone(),
                        orbit_im_hi: orb.gpu_payload.im_hi.clone(),
                        orbit_im_lo: orb.gpu_payload.im_lo.clone(),
                        escape_index: orb.escape_index,
                        max_valid_perturb_index: orb.max_valid_perturb_index,
                        creation_time: orb.creation_time,
                    }
                }).collect();

                let orb_id_log: Vec<u64> = df_orbits.iter().map(|orb| orb.orbit_id).collect();
                if orb_id_log.len() > 0 {
                    trace!("Query Tiles in BBox found {:?} with {} orbits.\tIDs: {:?}",
                        &tile_id, &df_orbits.len(), orb_id_log);
                }

                df_tiles.push(signals::TileOrbitViewDf {
                    tile: tile_id.clone(),
                    geometry: view.geometry.clone(),
                    orbits: df_orbits,
                });
                num_tiles_in_bounds += 1;
            }
        }
        if num_orbits_found > 0 {
            trace!("Total number of tiles found in-bounds: {}. Total number of orbits found: {}", 
                num_tiles_in_bounds, num_orbits_found);
        }
        df_tiles
    }

    pub fn diagnostics(&self) -> signals::ScoutDiagnostics {
        signals::ScoutDiagnostics {
            timestamp: time::SystemTime::now(),
            message: String::new()
        }
    }
}
