pub mod orbit;
pub mod tile;
mod worker;
mod tasks;
mod utils;

use crate::scout_engine::orbit::*;
use crate::scout_engine::tile::*;
use crate::scout_engine::worker::*;
use crate::scout_engine::utils::*;

use crate::signals::*;
use crate::numerics::*;

use iced_winit::winit::window::Window;

use futures::channel;
use futures::executor::{ThreadPool};

use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use parking_lot::{Mutex, RwLock};
use log::{trace, info};

use std::collections::HashMap;
use std::time;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

type OrbitSeedRng               = Arc<Mutex<ChaCha8Rng>>;
type CameraSnapshotSender       = channel::mpsc::Sender<CameraSnapshot>;
type CameraSnapshotReceiver     = channel::mpsc::Receiver<CameraSnapshot>;
type TileObservationsSender     = channel::mpsc::Sender<Vec<TileObservation>>;
type TileObservationsReceiver   = channel::mpsc::Receiver<Vec<TileObservation>>;
type ScoutContext               = Arc<ScoutEngineContext>;

#[derive(Clone, Debug)]
pub struct ScoutConfig {
    pub max_live_orbits: u32,
    pub max_orbit_iters: u32,
    // (Psudo) random number generator seed to use for creating orbit 'seeds'
    // for complex tiles. Random Number Generator is seeded ONCE, at start of 
    // the program, so randomness is deterministic for repeated program executions.
    // (i.e. good for testing!)
    pub orbit_rng_seed: u64,
    /// For any orbit spawn operation, spawn this number of orbits per tile
    pub num_orbits_to_spawn_per_tile: u32,
    /// Maximum number of times a 'ligitimate' attempt was made to anchor
    /// a tile and resulted in failure
    pub max_tile_anchor_failure_attempts: u32,
    /// Used to initialize the level-0 tile grid
    pub init_rug_precision: u32,
    /// Do not evaluate level-zero tiles untill user is zommed enough
    /// that the number of tiles is limited.
    pub level_zero_tile_constraint_before_eval: u32,
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
    /// Where our live complex tiles are stored
    pub tile_registry: TileRegistry,
    /// Creates unique orbit_id's
    pub orbit_id_factory: OrbitIdFactory,
    /// (Psudo) Random Number Generator used to create/spawn orbit seed (Complex) 
    /// values for ReferenceOrbit(s). C value is usually bounded by the geometry
    /// described by TileGeometry in OrbitTileView
    pub orbit_seed_rng: OrbitSeedRng,
    /// the most recent camera snapshot, essential for knowning when the user 
    /// last changed the viewport. Only scout_worker should update this.
    pub last_camera_snapshot: Arc<Mutex<CameraSnapshot>>,
    /// Flag that signifies that scout_worker has changed the context since 
    /// last viewed from without (i.e. by the Scene)
    context_changed: Arc<AtomicBool>,
    /// Our winit window 
    window: Arc<Window>,
}

impl ScoutEngineContext {
    pub fn context_changed(&self) {
        self.context_changed.store(true, Ordering::Relaxed);

        // Send a signal to the winit window to wake up the render loop and redraw the viewport
        // This mechanism is largely in place so that the render loop need not run continually 
        // and communications to/from the GPU can be tightly controlled. 
        self.window.request_redraw();
    }
}

#[derive(Debug)]
pub struct ScoutEngine {
    // Internal thread-pool, used both for the primary event loop, and internal async tasks
    thread_pool: ThreadPool,
    // ScoutEngine data/memory context
    context: ScoutContext,
    // Async channels
    cameara_snapshot_tx: CameraSnapshotSender,
    tile_observations_tx: TileObservationsSender,
}

impl ScoutEngine {
    pub fn new(window: Arc<Window>, config: ScoutConfig, snapshot: CameraSnapshot) -> Self {
        let orb_rng_seed = config.orbit_rng_seed;
        let config = Arc::new(config);
        let thread_pool = ThreadPool::new().expect("Failed to build ThreadPool for ScoutEngine");
        let context = Arc::new(ScoutEngineContext {
            config,
            living_orbits: Arc::new(Mutex::new(Vec::new())),
            tile_registry: Arc::new(RwLock::new(HashMap::new())),
            orbit_id_factory: Arc::new(Mutex::new(IdFactory::new())),
            orbit_seed_rng: Arc::new(Mutex::new(ChaCha8Rng::seed_from_u64(orb_rng_seed))),
            last_camera_snapshot: Arc::new(Mutex::new(snapshot)),
            context_changed: Arc::new(AtomicBool::new(false)),
            window,
        });

        // Create asynch channels for communicating between the render loop/thread and 
        // the long-lived ScoutEngine async task
        let (cameara_snapshot_tx, cameara_snapshot_rx) 
            = channel::mpsc::channel::<CameraSnapshot>(10);
        let (tile_observations_tx, tile_observations_rx) 
            = channel::mpsc::channel::<Vec<TileObservation>>(10);

        // Launch ScoutEngine's long-lived task, which 'blocks' on the above async channels
        thread_pool.spawn_ok(
            scout_worker(thread_pool.clone(), context.clone(),
                cameara_snapshot_rx, tile_observations_rx)
        );

        Self {
            thread_pool, context,
            cameara_snapshot_tx, tile_observations_tx
        } 
    }

    pub fn context_changed(&mut self) -> bool {
        self.context.context_changed.swap(false, Ordering::Relaxed)
    }

    pub fn submit_camera_snapshot(&mut self, snapshot: CameraSnapshot) {
        info!("Camera Snapshot received: {:?}", snapshot);
        self.cameara_snapshot_tx.try_send(snapshot).ok();
    }

    pub fn submit_tile_observations(&mut self, feedback: Vec<TileObservation>) {
        info!("Tile Feedback received");
        self.tile_observations_tx.try_send(feedback).ok();
    }

    pub fn query_tiles_for_orbits(&self,
        snapshot: &CameraSnapshot
    ) -> Vec<TileOrbitViewDf> {
        // First find the top-level tiles that fall within the camera bounds
        let tile_views = find_top_tiles_under_camera(
            snapshot, self.context.tile_registry.clone());

        let mut collected_tiles: Vec<(TileId, TileGeometry, LiveOrbit)> = Vec::new();

        for tile in &tile_views {
            find_anchor_orbits_in_tile_tree(tile.clone(), snapshot, &mut collected_tiles);
        }

        let df_tiles: Vec<TileOrbitViewDf> = collected_tiles
            .iter()
            .map(|(tile, geometry, orb)| {
                // Delta from viewport center to tile center must be computed per query
                // This is kept is a rug Complex because it is directly translated to pixel
                // co-ordinates, and this ensures greatest accuracy.
                let delta_from_center = 
                    complex_delta(geometry.center(), snapshot.center());
                let orb_g = orb.read();
                // Distance from tile center to the anchor orbit is cached inside the OrbRef
                let delta_from_anchor =
                    ComplexDf::from_complex(&orb_g.delta_from_tile_center.clone().unwrap());

                TileOrbitViewDf{
                    id: *tile,
                    geometry: geometry.clone(), 
                    delta_from_center, delta_from_anchor,
                    orbit: ReferenceOrbitDf {
                        orbit_id: orb_g.orbit_id,
                        c_ref: orb_g.c_ref_df,
                        orbit_re_hi: orb_g.gpu_payload.re_hi.clone(),
                        orbit_re_lo: orb_g.gpu_payload.re_lo.clone(),
                        orbit_im_hi: orb_g.gpu_payload.im_hi.clone(),
                        orbit_im_lo: orb_g.gpu_payload.im_lo.clone(),
                        escape_index: orb_g.qualiy_metrics.escape_index,
                        r_valid: Df::from_float(orb_g.r_valid()),
                        contraction: Df::from_float(orb_g.contraction()),
                        created_at: orb_g.qualiy_metrics.created_at,
                    }
                }
            }).collect();
        
        let mut trace_str = String::from(
            format!("Query tiles for orbits found {} tiles with anchor orbits...\n", 
            df_tiles.len()).as_str());
        for tile in &df_tiles {
            trace_str.push_str(format!("{:>3?}\torb_id={:>4} delta_from_sc={:?} delta_from_a={:?}\n", 
                tile.id, tile.orbit.orbit_id, tile.delta_from_center, tile.delta_from_anchor).as_str());
        }
        trace!("{}", trace_str);

        df_tiles
    }

    pub fn diagnostics(&self) -> ScoutDiagnostics {
        ScoutDiagnostics {
            timestamp: time::SystemTime::now(),
            message: String::new()
        }
    }
}
