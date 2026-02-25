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

use iced_winit::winit::window::Window;

use futures::{channel};
use futures::executor::{ThreadPool};

use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use parking_lot::{Mutex, RwLock};
use rug::Float;
use log::{trace, info};

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::numerics::Df;

pub type ScoutConfig                = Arc<Mutex<ScoutEngineConfig>>;
type OrbitSeedRng               = Arc<Mutex<ChaCha8Rng>>;
type CameraSnapshotSender       = channel::mpsc::Sender<CameraSnapshot>;
type CameraSnapshotReceiver     = channel::mpsc::Receiver<CameraSnapshot>;
type TileObservationsSender     = channel::mpsc::Sender<Vec<TileObservation>>;
type TileObservationsReceiver   = channel::mpsc::Receiver<Vec<TileObservation>>;
type ExploreSignalSender        = channel::mpsc::Sender<ScoutSignal>;
type ExploreSignalReceiver      = channel::mpsc::Receiver<ScoutSignal>;
type ScoutContext               = Arc<ScoutEngineContext>;

#[derive(Copy, Clone, Debug)]
pub enum ScoutSignal {
    ResetEngine,
    ExploreSignal(ScoutEngineConfig)
}

#[derive(Copy, Clone, Debug)]
pub struct ScoutEngineConfig {
    /// Total number of orbits ScoutEngine will track/keep in memory
    pub max_live_orbits: u32,
    /// User-defined max_iters
    /// The primary slider that the user uses to control mandelbrot iteration,
    /// at program start. This same value is fed to the Scene Uniform and GPU for
    /// absolute iteration. In ScoutEngine, this is used to stop reference orbit
    /// computation 'early' - i.e. until an anchor orbit is found, at which point
    /// the remainder of the orbit will be computed
    pub max_user_iters: u32,
    /// The 'true' maximum ScoutEngine uses to compute reference orbits.
    /// For perturbation to succeed, reference orbit iterations must go past
    /// it's own escape. 'How much past' can only be determined empirically,
    /// with per-pixel orbit observations from the GPU.
    pub max_ref_orbit_iters: u32,
    /// Start when the scale threshold is crossed
    pub auto_start: bool,
    /// Starting scale for tile creation, orbit spawning, and everything scout-engine
    /// does! Changes to this value should be tightly bounded, according to the rules
    /// of perturbation math.
    pub starting_scale: f64,
    /// Once the starting scale threshold has been crossed, start tiles with this pixel span
    pub starting_tile_pixel_span: f64,
    /// The smallest tile-size scout-engine will try to create, in pixels
    pub smallest_tile_pixel_span: f64,
    /// The required coverage percentage to anchor a tile
    /// Scale is between 0 and 1, with 0 being no coverage, and 1 being full coverage.
    /// More specifically, this is a ratio between the tile_size and a candidate orbit's
    /// r_valid, where a coverage of 1 means that the tile fully fits the orbit's radius
    /// for valid perturbation.
    pub coverage_to_anchor: f64,
    /// (Psudo) random number generator seed to use for creating orbit 'seeds'
    /// for complex tiles. Random Number Generator is seeded ONCE, at start of
    /// the program, so randomness is deterministic for repeated program executions.
    /// (i.e. good for testing!)
    pub orbit_rng_seed: u64,
    /// For any orbit spawn operation, spawn this number of orbits per tile
    pub num_orbits_to_spawn_per_tile: u32,
    /// Maximum number of times a 'legitimate' attempt was made to anchor
    /// a tile and resulted in failure
    pub max_tile_anchor_failure_attempts: u32,
    /// After a set number of seeds that are spawned for a tile, check for
    /// poor coverage, and if the average falls under a threshold, split the tile
    pub split_tile_on_poor_coverage_check: u32,
    /// Used to initialize the level-0 tile grid
    pub rug_precision: u32,
    /// Numer of 'convergence' cycles that scout engine is allowed to
    /// repeat after a camera pan/zoom. Each cycle may result in new
    /// orbits being spawned for tiles, depending on if the tile
    /// is found with a deficient orbit score. 
    pub exploration_budget: i32,
}

#[derive(Clone, Debug)]
pub struct ScoutEngineContext {
    /// Our startup configuration
    pub config: ScoutConfig,
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
    /// Tile Level zero
    /// If this is unset, it means ScoutEngine has never been started.
    pub level_zero: Arc<Mutex<TileLevel>>,
    /// Flag that signifies that scout_worker has changed the context since 
    /// last viewed from without (i.e. by the Scene)
    context_changed: Arc<AtomicBool>,
    /// Our winit window 
    window: Arc<Window>,
    /// For sending diagnostics messages back to the renderer/GUI
    diagnostics: Arc<Mutex<ScoutDiagnostics>>,
}

impl ScoutEngineContext {
    pub fn context_changed(&self) {
        self.context_changed.store(true, Ordering::Relaxed);

        // Send a signal to the winit window to wake up the render loop and redraw the viewport
        // This mechanism is largely in place so that the render loop need not run continually 
        // and communications to/from the GPU can be tightly controlled. 
        self.window.request_redraw();
    }

    pub fn write_diagnostics(&self, diag_msg: String) {
        info!("{}", diag_msg);
        let mut diag_g = self.diagnostics.lock();
        *diag_g = ScoutDiagnostics::new(diag_msg);
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
    explore_tx: ExploreSignalSender,
}

impl ScoutEngine {
    pub fn new(window: Arc<Window>, config: ScoutConfig, snapshot: CameraSnapshot) -> Self {
        // Initial level-zero tile size. Note, it could change when the scout button is clicked.
        let prec = config.lock().rug_precision;
        let starting_scale = config.lock().starting_scale;
        let starting_tile_pixel_span = config.lock().starting_tile_pixel_span;
        let tile_size = Float::with_val(
            prec, starting_scale * starting_tile_pixel_span);

        let orb_rng_seed = config.lock().orbit_rng_seed;
        let thread_pool = ThreadPool::new().expect("Failed to build ThreadPool for ScoutEngine");

        // Create asynch channels for communicating between the render loop/thread and
        // the long-lived ScoutEngine async task
        let (cameara_snapshot_tx, cameara_snapshot_rx)
            = channel::mpsc::channel::<CameraSnapshot>(10);
        let (tile_observations_tx, tile_observations_rx)
            = channel::mpsc::channel::<Vec<TileObservation>>(10);
        let (explore_tx, explore_rx) = channel::mpsc::channel::<ScoutSignal>(10);

        let context = Arc::new(ScoutEngineContext {
            config,
            living_orbits: Arc::new(Mutex::new(Vec::new())),
            tile_registry: Arc::new(RwLock::new(HashMap::new())),
            orbit_id_factory: Arc::new(Mutex::new(IdFactory::new())),
            orbit_seed_rng: Arc::new(Mutex::new(ChaCha8Rng::seed_from_u64(orb_rng_seed))),
            last_camera_snapshot: Arc::new(Mutex::new(snapshot)),
            context_changed: Arc::new(AtomicBool::new(false)),
            level_zero: Arc::new(Mutex::new(TileLevel::new(tile_size))),
            window,
            diagnostics: Arc::new(Mutex::new(ScoutDiagnostics::new(
                "Scout Engine initializing...".to_string()))
            ),
        });

        // Launch ScoutEngine's long-lived task, which 'blocks' on the above async channels
        thread_pool.spawn_ok(
            scout_worker(thread_pool.clone(), context.clone(),
                cameara_snapshot_rx, tile_observations_rx, explore_rx)
        );

        Self {
            thread_pool, context,
            cameara_snapshot_tx, tile_observations_tx, explore_tx,
        } 
    }

    pub fn config(&self) -> ScoutConfig {
        self.context.config.clone()
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
    
    pub fn submit_scout_signal(&mut self, signal: ScoutSignal) {
        info!("Scout signal received: {:?}", signal);
        self.explore_tx.try_send(signal).ok();
    }

    pub fn read_diagnostics(&self) -> Arc<Mutex<ScoutDiagnostics>> {
        self.context.diagnostics.clone()
    }

    pub fn set_max_user_iterations(&mut self, max_iters: u32) {
        let mut config_g = self.context.config.lock();
        info!("Set max_user_iters={}", max_iters);
        config_g.max_user_iters = max_iters;
    }

    pub fn query_tiles_for_orbits(&self,
        snapshot: &CameraSnapshot
    ) -> Vec<TileOrbitViewDf> {

        // First find the top-level tiles that fall within the camera bounds
        let level_zero_g = self.context.level_zero.lock();
        let top_tile_ids = find_tile_ids_under_camera(snapshot, &level_zero_g);
        let tile_views = find_tiles_in_registry(
            self.context.tile_registry.clone(), &top_tile_ids);

        let mut collected_tiles: Vec<TileView> = Vec::new();
        find_anchor_orbits(&tile_views, snapshot, &mut collected_tiles);

        let df_tiles: Vec<TileOrbitViewDf> = collected_tiles
            .iter()
            .map(|tile| {
                let tile_g = tile.read();
                let orb_g = tile_g.anchor_orbit.as_ref().unwrap().read();
                // Delta from viewport center to tile center must be computed per query
                // This is kept is a rug Complex because it is directly translated to pixel
                // co-ordinates, and this ensures greatest accuracy.
                let delta_from_center =
                    complex_delta(tile_g.geometry.center(), snapshot.center());
                TileOrbitViewDf {
                    id: tile_g.id.clone(),
                    geometry: tile_g.geometry.clone(),
                    delta_from_center,
                    orbit: ReferenceOrbitDf {
                        orbit_id: orb_g.orbit_id,
                        c_ref: orb_g.gpu_payload.c_ref,
                        orbit_re_hi: orb_g.gpu_payload.re_hi.clone(),
                        orbit_re_lo: orb_g.gpu_payload.re_lo.clone(),
                        orbit_im_hi: orb_g.gpu_payload.im_hi.clone(),
                        orbit_im_lo: orb_g.gpu_payload.im_lo.clone(),
                        escape_index: orb_g.quality_metrics.escape_index,
                        r_valid: Df::from_float(orb_g.r_valid()),
                        contraction: Df::from_float(orb_g.contraction()),
                        created_at: orb_g.quality_metrics.created_at,
                    }
                }
            }).collect();
        
        let mut trace_str = String::from(
            format!("Query tiles for orbits found {} tiles with anchor orbits...\n", 
            df_tiles.len()).as_str());
        for tile in &df_tiles {
            trace_str.push_str(format!("{:>3?}\torb_id={:>4} delta_from_sc={} \tr_valid={:.6e} contraction={:.2} escape={:?} len={}\n",
               tile.id, tile.orbit.orbit_id,
               tile.delta_from_center.to_string_radix(10, Some(6)),
               tile.orbit.r_valid.hi,
               tile.orbit.contraction.hi,
               tile.orbit.escape_index,
                tile.orbit.orbit_re_hi.len()
            ).as_str());
        }
        trace!("{}", trace_str);

        df_tiles
    }
}
