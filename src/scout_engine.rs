pub mod orbit;
pub mod tile;
mod worker;
mod tasks;
mod utils;

use crate::scout_engine::orbit::*;
use crate::scout_engine::tile::*;
use crate::scout_engine::worker::*;
use crate::scout_engine::utils::*;

use super::signals;

use iced_winit::winit::window::Window;

use futures::channel;
use futures::executor::{ThreadPool};

use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use rug::{Complex};
use parking_lot::{Mutex, RwLock};
use log::{trace, debug, info};

use std::collections::HashMap;
use std::time;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

type OrbitSeedRng               = Arc<Mutex<ChaCha8Rng>>;
type CameraSnapshotSender       = channel::mpsc::Sender<signals::CameraSnapshot>;
type CameraSnapshotReceiver     = channel::mpsc::Receiver<signals::CameraSnapshot>;
type GpuFeedbackSender          = channel::mpsc::Sender<signals::GpuFeedback>;
type GpuFeedbackReceiver        = channel::mpsc::Receiver<signals::GpuFeedback>;
type OrbitObservationsSender    = channel::mpsc::Sender<Vec<signals::OrbitObservation>>;
type OrbitObservationsReceiver  = channel::mpsc::Receiver<Vec<signals::OrbitObservation>>;

#[derive(Clone, Debug)]
pub struct HeuristicConfig {
    /// Change in number of frames when stats are considered decayed
    /// Primarily used to apply a freshness score to stats, i.e.
    /// (1.0 - (age/frame_decay_increment)).clamp(1.0, 0.0)
    pub frame_decay_increment: f64, // 30-100
    /// If the local_score of a TileOrbitView falls beneath this value, 
    /// then it is considered 'deficient', which will cause ScoutEngine
    /// to spawn new orbits for that tile. The number of orbits spawned
    /// for that tile is bounded by the exploration budget.
    pub tile_deficiency_threshold: f64,
}

#[derive(Clone, Debug)]
pub struct ScoutConfig {
    pub max_live_orbits: u32,
    pub max_orbit_iters: u32,
    pub heuristic_config: HeuristicConfig,
    // (Psudo) random number generator seed to use for creating orbit 'seeds'
    // for complex tiles. Random Number Generator is seeded ONCE, at start of 
    // the program, so randomness is deterministic for repeated program executions.
    // (i.e. good for testing!)
    pub orbit_rng_seed: u64,
    pub init_rug_precision: u32,
    /// Tile levels are create on a 'power-of-2' descending latter
    /// i.e. base_tile_size / 2^level
    pub base_tile_size: f64, // 1e-2 or 1e-3 is good
    /// How many TileLevel(s) to add when none are found for the current scale
    /// NOTE: because tile_size can be derived by base size + level,
    /// more tile levels can easily be added as the user zooms further 
    /// into the fractal
    pub tile_level_addition_increment: u32, 
    /// Ideal number of pixels that would fit inside a complex tile
    pub ideal_tile_pix_width: f64, // 32.0 - 64.0
    /// For any orbit spawn operation, spawn this number of orbits per tile
    pub num_orbits_to_spawn_per_tile: u32,
    /// Initial max/cap orbits per tile.
    /// This value can grow more orbits are needed for neighbor tiles
    /// to settile on a better local score for the tile view.
    pub initial_max_orbits_per_tile: u32,
    /// Numer of 'convergence' cycles that scout engine is allowed to
    /// repeat after a camera pan/zoom. Each cycle may result in new
    /// orbits being spawned for the tile, depending on if the tile
    /// is found with a deficient orbit score. 
    pub exploration_budget: f64,
}

#[derive(Clone, Debug)]
pub struct ScoutEngineContext {
    /// Our startup configuration
    pub config: Arc<ScoutConfig>,
    /// Our working pool of reference orbits
    pub living_orbits: LivingOrbits,
    /// Our deterministic set of grid boundies that bound reference orbit applicablity
    /// Level 0 starts with a known base size, and subsequent (descending) levels are 
    /// created as needed with depth of zoom.  
    pub tile_levels: TileLevels,
    /// Where our (logical) tiles live
    pub tile_registry: TileRegistry,
    /// Creates unique orbit_id's
    pub orbit_id_factory: OrbitIdFactory,
    /// (Psudo) Random Number Generator used to create/spawn orbit seed (Complex) 
    /// values for ReferenceOrbit(s). C value is usually bounded by the geometry
    /// described by TileGeometry in OrbitTileView
    pub orbit_seed_rng: OrbitSeedRng,
    /// the most recent camera snapshot, essential for knowning when the user 
    /// last changed the viewport. Only scout_worker should update this.
    pub last_camera_snapshot: Arc<Mutex<signals::CameraSnapshot>>,
    /// Flag that signifies that scout_worker has changed the context since 
    /// last viewed from without (i.e. by the Scene)
    pub context_changed: Arc<AtomicBool>
}

#[derive(Debug)]
pub struct ScoutEngine {
    // Our winit window 
    window: Arc<Window>,
    // Internal thread-pool, used both for the primary event loop, and internal async tasks
    thread_pool: ThreadPool,
    // ScoutEngine data/memory context
    context: Arc<ScoutEngineContext>,
    // Async channels
    cameara_snapshot_tx: CameraSnapshotSender,
    gpu_feedback_tx: GpuFeedbackSender,
    orbit_observations_tx: OrbitObservationsSender,
}

impl ScoutEngine {
    pub fn new(window: Arc<Window>, config: ScoutConfig) -> Self {
        let orb_rng_seed = config.orbit_rng_seed;
        let config = Arc::new(config);
        let thread_pool = ThreadPool::new().expect("Failed to build ThreadPool for ScoutEngine");
        let context = Arc::new(ScoutEngineContext {
            config,
            living_orbits: Arc::new(Mutex::new(Vec::new())),
            tile_levels: Arc::new(RwLock::new(Vec::new())),
            tile_registry: Arc::new(RwLock::new(HashMap::new())),
            orbit_id_factory: Arc::new(Mutex::new(IdFactory::new())),
            orbit_seed_rng: Arc::new(Mutex::new(ChaCha8Rng::seed_from_u64(orb_rng_seed))),
            last_camera_snapshot: Arc::new(Mutex::new(signals::CameraSnapshot::new())),
            context_changed: Arc::new(AtomicBool::new(false))
        });

        // Create asynch channels for communicating between the render loop/thread and 
        // the long-lived ScoutEngine async task
        let (cameara_snapshot_tx, cameara_snapshot_rx) 
            = channel::mpsc::channel::<signals::CameraSnapshot>(10);
        let (gpu_feedback_tx, gpu_feedback_rx) 
            = channel::mpsc::channel::<signals::GpuFeedback>(10);
        let (orbit_observations_tx, orbit_observations_rx)
            = channel::mpsc::channel::<Vec<signals::OrbitObservation>>(10);

        // Launch ScoutEngine's long-lived task, which 'blocks' on the above async channels
        thread_pool.spawn_ok(
            scout_worker(window.clone(), thread_pool.clone(),
            context.clone(),
            cameara_snapshot_rx, gpu_feedback_rx, orbit_observations_rx)
        );

        Self {
            window, thread_pool, context,
            cameara_snapshot_tx, gpu_feedback_tx, orbit_observations_tx
        } 
    }

    pub fn context_changed(&mut self) -> bool {
        self.context.context_changed.swap(false, Ordering::Relaxed)
    }

    pub fn submit_camera_snapshot(&mut self, snapshot: signals::CameraSnapshot) {
        info!("Camera Snapshot received: {:?}", snapshot);
        self.cameara_snapshot_tx.try_send(snapshot).ok();
    }

    pub fn submit_gpu_feedback(&mut self, feedback: signals::GpuFeedback) {
        debug!("Gpu Feedback received: {:?}", feedback);
        self.gpu_feedback_tx.try_send(feedback).ok();
    }

    pub fn submit_orbit_observations(&mut self, feedback: Vec<signals::OrbitObservation>) {
        info!("Orbit Observations received. len={}", &feedback.len());
        self.orbit_observations_tx.try_send(feedback).ok();
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
    // the GPU-packed arrays (i.e. x4 per orbit).
    pub fn query_tiles_in_bounding_box(&self, 
        top_left: &Complex, bot_rght: &Complex,
    )-> Vec<signals::TileOrbitViewDf> {
        let mut appl_tile_lvls = find_applicable_tile_levels_for_scale(
            &self.context.tile_levels, 
            &self.context.last_camera_snapshot.lock().scale);
        // Ensure lower tile level is retured first into the vector
        // so orbit slots of lower tile level get higher priority.
        appl_tile_lvls.reverse();

        let mut num_tiles_in_bounds: u32 = 0;
        let mut num_orbits_found: u32 = 0;
        let mut df_tiles = Vec::<signals::TileOrbitViewDf>::new();

        let mut tile_orbits_log: Vec<(TileId, Vec<u64>)> = Vec::new();

        for level in appl_tile_lvls.iter() {
            let t_tl = TileId::from_point(top_left, &level);
            let t_br = TileId::from_point(bot_rght, &level);

            let reg_g = self.context.tile_registry.read();
        
            for (tile_id, view) in reg_g.iter() {
                if tile_id.tx >= t_tl.tx && tile_id.tx <= t_br.tx &&
                   tile_id.ty >= t_tl.ty && tile_id.ty <= t_br.ty {
                    let view_g = view.read();
                    let live_orbits = upgrade_orbit_list(&view_g.weak_orbits);

                    let df_orbits: Vec<signals::ReferenceOrbitDf> = live_orbits.iter().map(|(_, orb)| {
                        let orb_g = orb.read();
                        num_orbits_found += 1;
                        signals::ReferenceOrbitDf {
                            orbit_id: orb_g.orbit_id,
                            c_ref: orb_g.c_ref_df,
                            orbit_re_hi: orb_g.gpu_payload.re_hi.clone(),
                            orbit_re_lo: orb_g.gpu_payload.re_lo.clone(),
                            orbit_im_hi: orb_g.gpu_payload.im_hi.clone(),
                            orbit_im_lo: orb_g.gpu_payload.im_lo.clone(),
                            escape_index: orb_g.escape_index,
                            min_valid_perturb_index: orb_g.min_valid_perturb_index,
                            max_valid_perturb_index: orb_g.max_valid_perturb_index,
                            created_at: orb_g.created_at,
                        }
                    }).collect();

                    let orb_id_log: Vec<u64> = df_orbits.iter().map(|orb| orb.orbit_id).collect();
                    tile_orbits_log.push((tile_id.clone(), orb_id_log));

                    df_tiles.push(signals::TileOrbitViewDf {
                        tile: tile_id.clone(),
                        geometry: view_g.geometry.clone(),
                        orbits: df_orbits,
                    });
                    num_tiles_in_bounds += 1;
                }
            }
        }
        if num_orbits_found > 0 {
            let mut trace_str = String::from(
                format!("BBox query found {} tiles in-bounds with a total of {} orbits...\n\t\t",
                    num_tiles_in_bounds, num_orbits_found).as_str());
            let mut curr_row = 0;
            for (tile, orbs) in tile_orbits_log {
                if curr_row > 5 {
                    trace_str.push_str("\n\t\t");
                    curr_row = 0;
                }
                trace_str.push_str(format!("(t x, y, l: {:>3}, {:>3}, {:>3}) {:>3?}\t", 
                    tile.tx, tile.ty, tile.level, orbs ).as_str());
                curr_row += 1;
            }
            trace!("{}", trace_str);
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
