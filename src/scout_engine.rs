pub mod orbit;
pub mod tile;
mod worker;
mod tasks;
pub(crate) mod utils;

use crate::scout_engine::orbit::*;
use crate::scout_engine::worker::*;

use crate::signals::*;

use iced_winit::winit::window::Window;

use futures::{channel};
use futures::executor::ThreadPool;

use parking_lot::{Mutex};
use log::{trace, info};

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::numerics::{Df};

pub type ScoutConfig            = Arc<Mutex<ScoutEngineConfig>>;
type CameraSnapshotSender       = channel::mpsc::Sender<CameraSnapshot>;
type CameraSnapshotReceiver     = channel::mpsc::Receiver<CameraSnapshot>;
type OrbitObservationsSender    = channel::mpsc::Sender<Vec<OrbitObservation>>;
type OrbitObservationsReceiver  = channel::mpsc::Receiver<Vec<OrbitObservation>>;
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
    /// Multiply max_user_iters by this value for computing the length of a reference orbit
    pub ref_iters_multiplier: f64,
    /// Number of Reference Orbits ScoutEngine will create from sampled seed candidates
    /// Note, the seed candidate list is taken from a snapshot of GpuGridSample(s), which
    /// are reduced within a compute shader running on the GPU.
    pub num_seeds_to_spawn_per_eval: u32,
    /// Number of reference orbits to qualify (or keep qualified), per cycle.
    /// With orbits of sufficient quality, the GPU should not have need to rebase
    /// any more than 2 or three times.
    pub num_qualified_orbits: u32,
    pub rug_precision: u32,
    /// Numer of 'convergence' cycles that scout engine is allowed to
    /// repeat after a camera pan/zoom. Most of the time, only one should be needed,
    /// but if perturbance is behaving poorly, more cycles may be needed to find a better
    /// orbit (or set of orbits).
    pub exploration_budget: i32,
}

#[derive(Clone, Debug)]
pub struct ScoutEngineContext {
    /// Our startup configuration
    pub config: ScoutConfig,
    /// Our working pool of reference orbits
    pub living_orbits: LivingOrbits,
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
        Self {
            config: Arc::new(Mutex::new(config)),
            living_orbits: Arc::new(Mutex::new(Vec::new())),
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
    tile_observations_tx: OrbitObservationsSender,
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
            = channel::mpsc::channel::<Vec<OrbitObservation>>(10);
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
        self.cameara_snapshot_tx.try_send(snapshot).ok();
    }

    pub fn submit_orbit_observations(&mut self, feedback: Vec<OrbitObservation>) {
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

    pub fn query_qualified_orbits(&self) -> Vec<QualifiedOrbit> {
        let num_orbits_to_qualify = self.config().lock().num_qualified_orbits as usize;
        let pool_g = self.context.living_orbits.lock();

        let df_orbits: Vec<QualifiedOrbit> = pool_g
            .iter()
            .take(num_orbits_to_qualify)
            .enumerate()
            .map(|(i, orb)| {
                let orb_g = orb.read();

                QualifiedOrbit {
                    rank: i as u32,
                    orbit_id: orb_g.orbit_id,
                    c_ref: orb_g.c_ref().clone(),
                    c_ref_df: orb_g.gpu_payload.c_ref,
                    orbit: orb_g.gpu_payload.cdf_orbit.clone(),
                    escape_index: orb_g.quality_metrics.escape_index,
                    r_valid: Df::from_float(orb_g.r_valid()),
                    contraction: Df::from_float(orb_g.contraction()),
                    created_at: orb_g.quality_metrics.created_at,
                    }
                })
            .collect();
        
        let mut trace_str = String::from(
            format!("Query Living Orbit Pool for qualified orbits. Found {} total orbits, but only qualifying {}\n",
                pool_g.len(), num_orbits_to_qualify ).as_str());
        for q_orb in &df_orbits {
            trace_str.push_str(format!("Rank #{:<2}\torb_id={:<4}\tescape={:?} len={} contraction={:.4e}\n",
               q_orb.rank, q_orb.orbit_id,
                q_orb.escape_index,
                q_orb.orbit.len(),
                q_orb.contraction.hi
            ).as_str());
        }
        trace!("{}", trace_str);

        df_orbits
    }
}
