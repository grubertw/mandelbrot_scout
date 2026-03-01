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
use futures::executor::ThreadPool;

use parking_lot::{Mutex, RwLock};
use log::{trace, info};

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::numerics::{ComplexDf, Df};
use crate::scout_engine::tasks::create_tile_levels;

pub type ScoutConfig                = Arc<Mutex<ScoutEngineConfig>>;
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
    /// does! 
    pub starting_scale: f64,
    /// Maximum number of tile levels. While it makes sense to increase this at runtime,
    /// there is no reason to decrease it.
    pub max_tile_levels: u32,
    /// Ideal size of a tile in the level hierarchy
    /// Scout will use current camera scale and this value to extrapolate which
    /// tile level to use
    pub ideal_tile_size: f64,
    /// Number of extra iterations Scout will perform on the ref orbit
    /// Ref orbits should always at least have max_user_iters,
    /// unless they escape early, and in which case they are not worth extending as far
    pub ref_iters_multiplier: f64,
    /// Number of Reference Orbits ScoutEngine will create from sampled seed candidates
    /// Note, the seed candidate list is taken from a snapshot of GpuGridSample(s), which
    /// are reduced within a compute shader running on the GPU.
    pub num_seeds_to_spawn_per_tile_eval: u32,
    /// Orbit ranking/selection criteria
    pub contraction_threshold: f64,
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
    /// Deterministic descending power-of-two tile hierarchy 
    pub tile_levels: Vec<TileLevel>,
    /// Our working pool of reference orbits
    pub living_orbits: LivingOrbits,
    /// Where our live complex tiles are stored
    pub tile_registry: TileRegistry,
    /// Creates unique orbit_id's
    pub orbit_id_factory: OrbitIdFactory,
    /// the most recent camera snapshot, essential for knowning when the user 
    /// last changed the viewport. Only scout_worker should update this.
    pub last_camera_snapshot: Arc<Mutex<CameraSnapshot>>,
    /// Most recent set of GPU Grid Samples
    /// Polled by Scout worker during tile evaluation
    pub grid_samples: Arc<Mutex<Vec<GpuGridSample>>>,
    /// Flag that signifies that scout_worker has changed the context since 
    /// last viewed from without (i.e. by the Scene)
    context_changed: Arc<AtomicBool>,
    /// Our winit window 
    window: Arc<Window>,
    /// For sending diagnostics messages back to the renderer/GUI
    diagnostics: Arc<Mutex<ScoutDiagnostics>>,
}

impl ScoutEngineContext {
    pub fn new(
        config: ScoutEngineConfig, 
        window: Arc<Window>, 
        snapshot: CameraSnapshot
    ) -> Self {
        info!("Creating ScoutEngineContext with {} TileLevels and level zero tile_size={}",
            config.max_tile_levels, config.max_tile_levels);
        
        let tile_levels = create_tile_levels(
            config.rug_precision, config.max_tile_levels);
        
        Self {
            config: Arc::new(Mutex::new(config)),
            tile_levels,
            living_orbits: Arc::new(Mutex::new(Vec::new())),
            tile_registry: Arc::new(RwLock::new(HashMap::new())),
            orbit_id_factory: Arc::new(Mutex::new(IdFactory::new())),
            last_camera_snapshot: Arc::new(Mutex::new(snapshot)),
            grid_samples: Arc::new(Mutex::new(Vec::new())),
            context_changed: Arc::new(AtomicBool::new(false)),
            window,
            diagnostics: Arc::new(Mutex::new(ScoutDiagnostics::new(
                "Scout Engine initializing...".to_string()))
            ),
        }
    }
    
    pub fn tile_level_for_snapshot(&self, snapshot: &CameraSnapshot) -> TileLevel{
        let ideal_tile_size_px = self.config.lock().ideal_tile_size;
        let mut ideal_tile_size = snapshot.scale().clone();
        ideal_tile_size *= ideal_tile_size_px;
        
        self.tile_levels
            .iter()
            .rfind(|lvl| lvl.tile_size > ideal_tile_size)
            .cloned()
            .unwrap()
    }
    
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
    pub fn new(config: ScoutEngineConfig, window: Arc<Window>, snapshot: CameraSnapshot) -> Self {
        let context = 
            Arc::new(ScoutEngineContext::new(config, window, snapshot));
        
        let thread_pool = ThreadPool::new().expect("Failed to build ThreadPool for ScoutEngine");

        // Create asynch channels for communicating between the render loop/thread and
        // the long-lived ScoutEngine async task
        let (cameara_snapshot_tx, cameara_snapshot_rx)
            = channel::mpsc::channel::<CameraSnapshot>(10);
        let (tile_observations_tx, tile_observations_rx)
            = channel::mpsc::channel::<Vec<TileObservation>>(10);
        let (explore_tx, explore_rx) = channel::mpsc::channel::<ScoutSignal>(10);
        
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

    pub fn set_grid_samples(&self, grid_samples: Vec<GpuGridSample>) {
        let mut cxt_grid_samples = self.context.grid_samples.lock();
        //trace!("Scout Engine set grid {} samples: {:?}", grid_samples.len(), grid_samples);
        *cxt_grid_samples = grid_samples;
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
        let tile_level = self.context.tile_level_for_snapshot(snapshot);
        let tile_ids = find_tile_ids_under_camera(snapshot, &tile_level);
        let tile_views = find_tiles_in_registry(
            self.context.tile_registry.clone(), &tile_ids);
        
        let df_tiles: Vec<TileOrbitViewDf> = tile_views
            .iter()
            .filter(|tile| tile.read().anchor_orbit.is_some())
            .map(|tile| {
                let tile_g = tile.read();
                let orb_g = tile_g.anchor_orbit.as_ref().unwrap().read();
                // Delta from viewport center to tile center must be computed per query
                // This is kept is a rug Complex because it is directly translated to pixel
                // co-ordinates, and this ensures greatest accuracy.
                let delta_from_center =
                    complex_delta(tile_g.geometry.center(), snapshot.center());
                let delta_from_center_to_anchor =
                    complex_delta(orb_g.c_ref(), snapshot.center());
                TileOrbitViewDf {
                    id: tile_g.id.clone(),
                    geometry: tile_g.geometry.clone(),
                    delta_from_center,
                    delta_from_center_to_anchor: ComplexDf::from_complex(&delta_from_center_to_anchor),
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
                }})
            .collect();
        
        let mut trace_str = String::from(
            format!("Query tiles at level {} for orbits. Found {} tiles with anchor orbits...\n",
                tile_level.level, df_tiles.len()).as_str());
        for tile in &df_tiles {
            trace_str.push_str(format!("{:>3?}\torb_id={:>4} delta_from_sc_to_tc={} delta_from_sc_to_a={:?} \tescape={:?} len={}\n",
               tile.id, tile.orbit.orbit_id,
               tile.delta_from_center.to_string_radix(10, Some(10)),
                tile.delta_from_center_to_anchor,
               tile.orbit.escape_index,
                tile.orbit.orbit_re_hi.len()
            ).as_str());
        }
        trace!("{}", trace_str);

        df_tiles
    }
}
