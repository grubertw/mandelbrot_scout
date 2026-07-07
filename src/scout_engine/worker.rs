use std::cmp::Ordering;
use crate::scout_engine::{ScoutContext, CameraSnapshotReceiver, OrbitObservationsReceiver, ExploreSignalReceiver, ScoutSignal, ScoutEngineConfig};
use crate::scout_engine::orbit::*;
use crate::scout_engine::formula::Parameterization;
use crate::scout_engine::tasks::*;
use crate::scout_engine::utils::*;
use crate::scout_engine::bla::{self, QualifiedOrbitBLAInfo};

use crate::signals::{CameraSnapshot};

use std::sync::Arc;
use log::{debug, error, trace};
use num_complex::Complex32;

use futures::task::SpawnExt;
use futures::StreamExt;
use futures::select;
use futures::future::{join_all, RemoteHandle};
use futures::executor::{ThreadPool};
use crate::numerics::{ComplexFExp, FixedComplex, FixedReal};

// ScoutEngine's internal work loop, a long-lived future that uses select to poll the
// the camera snaphot & gpu feedback channels.
pub async fn scout_worker(
    tp: ThreadPool,
    context: ScoutContext,
    camera_snapshot_rx: CameraSnapshotReceiver,
    orbit_observations_rx: OrbitObservationsReceiver,
    explore_rx :ExploreSignalReceiver,
) {
    let mut snap_rx = camera_snapshot_rx.fuse();
    let mut orbit_obvs_rx = orbit_observations_rx.fuse();
    let mut explore_rx = explore_rx.fuse();

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
            obvs_res = orbit_obvs_rx.next() => {
                match obvs_res {
                    Some(feedback) => {
                        trace!("Scout Worker received {} OrbitObservation(s)", feedback.len());
                    }
                    None => {break;}
                }        
            },
        }
    }
}

async fn handle_camera_snapshot(
    tp: ThreadPool, context: ScoutContext,
    snapshot: Arc<CameraSnapshot>,
) {
    debug!("Scout Worker received camera snapshot. center={} scale={} extent={}",
        snapshot.center().to_ui_string(10),
        snapshot.scale().to_ui_string(5),
        snapshot.half_extent().to_ui_string(4)
    );
    {
        let mut last_snap_g = context.last_camera_snapshot.lock();
        *last_snap_g = (*snapshot).clone();
    }

    let cfg = { context.config.lock().clone() };
    let scale = snapshot.scale();

    if !scout_gate_open(&cfg, scale) {
        // Gate closed. If Auto Start is on but we've zoomed out past the
        // starting scale while active, tear the pool down (perturbation isn't
        // needed out here, and the orbits would be stale on the way back in).
        // With Auto Start off we simply idle — manual GoScout still works.
        trace!("Scout gate closed for scale {} (starting_scale {}, auto_start {}).",
            scale.to_f64_lossy(), cfg.starting_scale, cfg.auto_start);
        if cfg.auto_start && context.active() {
            trace!("Context was active; resetting");
            context.reset();
        }
        return;
    }

    tp.spawn_ok(
        evaluate_orbits(tp.clone(), context.clone())
    );
}

/// The auto_start + starting_scale gate: whether the engine should be actively
/// scouting at the given camera scale. Shared by the camera-snapshot path and
/// the fractal-definition-change (ReScout) path so both honor "Auto Start" and
/// the starting-scale threshold identically. Manual GoScout (ExploreSignal)
/// deliberately bypasses this so users can force perturbation on/off and compare
/// image quality with/without a good reference orbit.
fn scout_gate_open(cfg: &ScoutEngineConfig, scale: &FixedReal) -> bool {
    cfg.auto_start && scale.to_f64_lossy() <= cfg.starting_scale
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
            
            tp.spawn_ok(
                evaluate_orbits(tp.clone(), context.clone())
            );
        }
        ScoutSignal::ResetEngine => {
            context.reset();
        }
        ScoutSignal::ReScout => {
            // A fractal-definition change (Julia toggle, formula/power) already
            // reset the pool synchronously. Re-evaluate ONLY if the same gate a
            // camera snapshot would face is open — so switching Julia while
            // zoomed out (or with Auto Start off) leaves the pool empty and the
            // GPU on the naive path, rather than force-starting perturbation.
            let (cfg, scale) = {
                let cfg = context.config.lock().clone();
                let scale = context.last_camera_snapshot.lock().scale().clone();
                (cfg, scale)
            };
            if scout_gate_open(&cfg, &scale) {
                tp.spawn_ok(
                    evaluate_orbits(tp.clone(), context.clone())
                );
            }
        }
    }
}

async fn evaluate_orbits(
    tp: ThreadPool, context: ScoutContext,
) {
    // Naive-only formulas (e.g. Manowar) have no reference-orbit perturbation
    // path yet, and their stateless ref_step would be wrong — don't build any.
    if !context.config.lock().formula.supports_perturbation() {
        trace!("Formula has no perturbation path; skipping orbit evaluation.");
        return;
    }

    let current_camera = {
        context.last_camera_snapshot.lock().clone()
    };
    trace!("Evaluate orbits for screen center {} and scale {} and half extent {}",
        current_camera.center(), current_camera.scale(), current_camera.half_extent()
    );
    
    let shift = current_camera.scale().shift;

    // Snapshot GPU grid samples for eval
    let mut grid_samples = {
        context.grid_samples.lock().clone()
    };
    let mut num_interior_seeds: u32 = 0;

    // Perform preliminary scoring on GPU samples before spawning into orbits
    let mut sample_scores: Vec<SampleScore> = grid_samples
        .iter()
        .map(|sample| {
            if !sample.escaped {
                num_interior_seeds += 1;
            }
            SampleScore::new(sample, &current_camera)
        })
        .collect();
    
    // Rank GPU samples by depth and distance from cam center
    sample_scores
        .sort_by(|a, b|
            b.total_score.partial_cmp(&a.total_score).unwrap_or(Ordering::Equal));

    let mut best_sample_iters_reached: u32 = 0;
    let mut best_sample_escaped: bool = false;
    // Log the best/worst sample scores.
    if let Some(s) = sample_scores.first() {
        best_sample_iters_reached = s.sample.iters_reached;
        best_sample_escaped = s.sample.escaped;
    }

    trace!("{} GpuGridSamples ranked/sorted", sample_scores.len());
    let cfg = {
        context.config.lock().clone()
    };

    sample_scores.truncate(cfg.num_gpu_samples_to_eval as usize);
    grid_samples = sample_scores
        .iter()
        .map(|sample| sample.sample.clone())
        .collect();
    
    let gpu_seeds: Vec<FixedComplex> = grid_samples
        .iter()
        .map(|sample| sample.location.clone())
        .collect();
    trace!("{} GPU seeds will be used to calculate reference orbits", gpu_seeds.len());

    // Snapshot the parameterization rescaled to this cycle's shift, so a Julia
    // constant is aligned with the seeds spawned below (no-op for Mandelbrot).
    let param = context.parameterization_at(shift);

    debug!("evaluate_orbits computing reference orbits with formula={:?}, parameterization={:?}",
        cfg.formula, param);

    let mut gpu_orbit_scores = spawn_orbits_from_seeds(
        &gpu_seeds, tp.clone(), context.orbit_id_factory.clone(), cfg, &param, &current_camera
    ).await;

    let total_orbits_spawned = gpu_orbit_scores.len(); // + f64_orbit_scores.len();
    let mut best_orbit_id: u64 = 0;
    let mut best_orbit_len: u32 = 0;
    let mut best_oribt_escape_index: Option<u32> = None;
    // Inputs for a BLA (re)build, captured under the pool/orbit locks below and
    // consumed after they're released: (orbit_id, ref_len, weak orbit, truncated
    // f32 orbit, c_ref at camera shift).
    let mut bla_build_inputs:
        Option<(OrbitId, u32, WeakOrbit, Vec<Complex32>)> = None;

    {
        // Re-score old orbits
        let mut pool_g = context.living_orbits.lock();
        let mut scored_pool_orbs: Vec<OrbitScore> = pool_g
            .iter()
            .map(|orb| OrbitScore::new(orb.clone(), &current_camera, &cfg))
            .collect();

        // Append new scored orbits
        scored_pool_orbs.append(&mut gpu_orbit_scores);
        //scored_pool_orbs.append(&mut f64_orbit_scores);

        // Perform global ranking
        scored_pool_orbs
            .sort_by(|a, b| b.total_score
                .partial_cmp(&a.total_score)
                .unwrap_or(Ordering::Equal));

        let mut trace_str = String::from(
            format!("Re-Scored LiveOrbits. len={}\n", scored_pool_orbs.len()).as_str());
        let mut print_orb_count = 0u32;

        *pool_g = scored_pool_orbs
            .iter()
            .map(|s| {
                let orb_g = s.orbit.read();
                if print_orb_count < 5 {
                    trace_str.push_str(
                    format!("\tPool OrbitId={:<4} escape={:<7} len={:<6}\t\
            \tSCORING: depth={:<6.4} dist={:<8.4} total={:<8.4} loc={}\n",
                           orb_g.orbit_id,
                           orb_g.escape_index().map_or(String::from("None"), |v| v.to_string()),
                           orb_g.orbit.len(),
                           s.depth, s.dist, s.total_score,
                           orb_g.c_ref
                    ).as_str());
                }
                print_orb_count += 1;
                s.orbit.clone()
            })
            .collect();
        trace!("{}", trace_str);

        pool_g.truncate(cfg.max_live_orbits as usize);

        if let Some(s) = pool_g.first() {
            let orb_g = s.read();
            best_orbit_id = orb_g.orbit_id;
            best_orbit_len = orb_g.orbit.len() as u32;
            best_oribt_escape_index = orb_g.escape_index();

            // (Re)build the BLA table when the anchored orbit changes — or when
            // the same orbit grows (future: extending on a raised iteration
            // count). The staleness check avoids upgrading the weak ref.
            //
            // BLA is only implemented for power-2 Mandelbrot so far. The leaf
            // A = f'(Z) and validity-radius formula are power-2-specific, and the
            // merge/radii assume a Z[0]=0 (Mandelbrot) reference — Julia's
            // viewport-centered reference breaks that. Power-of-X and Julia
            // perturbation still render correctly, just without BLA speedup; a
            // formula/parameterization switch already cleared any old table via
            // reset().
            let needs_bla = cfg.formula.bla_supported()
                && param.is_mandelbrot()
                && match context.rank1_bla.lock().as_ref() {
                    Some(info) => info.orbit_id != best_orbit_id
                               || info.built_len != best_orbit_len,
                    None => true,
                };
            if needs_bla {
                bla_build_inputs = Some((
                    best_orbit_id, best_orbit_len,
                    Arc::downgrade(s),
                    orb_g.gpu_payload.c32_orbit.clone(),
                ));
            }
        }
    }

    // Spawn the BLA build off the eval loop. The renderer falls back to the plain
    // perturbation path until the table lands in context.rank1_bla.
    if let Some((oid, olen, weak, c32)) = bla_build_inputs {
        let orbit_fexp: Vec<ComplexFExp> = c32
            .iter()
            .map(|z| ComplexFExp::from_f64_pair(z.re as f64, z.im as f64))
            .collect();
        let ctx = context.clone();
        let pool = tp.clone();
        // Power for the leaf A = p*Z^(p-1). BLA only builds for Mandelbrot Power
        // formulae (gated above), so this is the real power.
        let power = cfg.formula.power();
        tp.spawn_ok(async move {
            // Build only the view-independent table; the renderer computes radii.
            let table = bla::build_bla(Arc::new(orbit_fexp), power, pool).await;
            let bla_info = Some(QualifiedOrbitBLAInfo {
                orbit_id: oid, built_len: olen, power,
                orbit_ref: weak, table: Arc::new(table),
            });
            trace!("BLA table built for rank-1 orbit {:?}", bla_info);
            *ctx.rank1_bla.lock() = bla_info;
            ctx.context_changed();
            // The build finishes after evaluate_orbits' own redraw request, so
            // ask for another frame: without it the new table sits unused until
            // the next pan/zoom. (No-op for pixels until the GPU consumes it.)
            ctx.window.request_redraw();
        });
    }

    context.write_diagnostics(
            format!("Scout evaluated {} grid samples, fast (f64) probed {} orbits, and spawned {} orbits. {} interior samples found!\n\
        \tBest Sample Info: iters_reached={} escaped={}\n\
        \tBest Qualified Ref Orbit Info:\n\t\tOrbitId={} shift={:<3} ref_len={}\n\t\tescape_index={}",
                    grid_samples.len(), 0, total_orbits_spawned,
                    num_interior_seeds,
                    best_sample_iters_reached, best_sample_escaped,
                    best_orbit_id, shift,
                    best_orbit_len,
                    best_oribt_escape_index.map_or(String::from("None"), |v| v.to_string()),
        ));

    context.context_changed();

    // Send a signal to the winit window to wake up the render loop and redraw the viewport
    // This mechanism is largely in place so that the render loop need not run continually
    // and communications to/from the GPU can be tightly controlled.
    context.window.request_redraw();

    trace!("Evaluate Orbits complete!");
}

async fn spawn_orbits_from_seeds(
    seeds: &[FixedComplex],
    tp: ThreadPool,
    id_factory: OrbitIdFactory,
    cfg: ScoutEngineConfig,
    param: &Parameterization,
    current_camera: &CameraSnapshot,
) -> Vec<OrbitScore> {

    let handles: Vec<RemoteHandle<LiveOrbit>> = seeds
        .iter()
        .filter_map(|seed| {
            let res = tp.spawn_with_handle(
                start_reference_orbit(seed.clone(), id_factory.clone(),
                                      cfg, param.clone(),
                                      current_camera.frame_stamp().clone())
            );
            if let Ok(h) = res { Some(h) } else {
                error!("Failed to spawn create_new_reference_orbit");
                None
            }
        })
        .collect();
    
    let live_orbits: Vec<LiveOrbit> = join_all(handles).await;

    let mut scored_orbits: Vec<OrbitScore> = live_orbits
        .iter()
        .map(|orb| OrbitScore::new(orb.clone(), current_camera, &cfg))
        .collect();
    scored_orbits
        .sort_by(|a, b| b.total_score
            .partial_cmp(&a.total_score)
            .unwrap_or(Ordering::Equal));

    for scored_orbit in &scored_orbits {
        let orb_g = scored_orbit.orbit.read();

        trace!("Scored orbit {} c_ref={} orbit.len={} escape={:?} \
        with shift={}\n\t\tScores: depth={} dist={} total_score={}",
            orb_g.orbit_id, orb_g.c_ref(),
            orb_g.orbit.len(), orb_g.escape_index,
            orb_g.c_ref().re.shift,
            scored_orbit.depth, scored_orbit.dist,
            scored_orbit.total_score
        );
    }
    
    scored_orbits
}
