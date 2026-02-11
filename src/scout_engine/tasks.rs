use crate::scout_engine::{ScoutConfig, OrbitSeedRng};
use crate::scout_engine::orbit::*;
use crate::scout_engine::tile::*;
use crate::scout_engine::utils::*;

use crate::numerics::ComplexDf;
use crate::signals;

use std::collections::HashMap;
use std::sync::Arc;

use log::{trace, debug};
use rand_chacha::ChaCha8Rng;
use rand::Rng;
use rug::{Float, Complex};
use parking_lot::RwLock;

pub async fn create_new_reference_orbit(
    c_ref: Complex, id_fac: OrbitIdFactory, 
    max_iter: u32, frame_st: signals::FrameStamp
) -> LiveOrbit {
    let orbit_result = compute_reference_orbit(&c_ref, max_iter).await;
    let cdf_orbit: Vec<ComplexDf> = orbit_result.orbit
        .iter()
        .map(|c| ComplexDf::from_complex(c))
        .collect();
    
    let mut gpu_payload = OrbitGpuPayload::new();

    for cdf in cdf_orbit {
        gpu_payload.re_hi.push(cdf.re.hi);
        gpu_payload.re_lo.push(cdf.re.lo);
        gpu_payload.im_hi.push(cdf.im.hi);
        gpu_payload.im_lo.push(cdf.im.lo);
    }

    Arc::new(RwLock::new(ReferenceOrbit::new(
        id_fac, c_ref, orbit_result.orbit, gpu_payload, 
        orbit_result.escape_index, frame_st
    )))
}

async fn compute_reference_orbit(
    c_ref: &Complex, max_iter: u32, 
) -> OrbitResult {
    let mut orbit = Vec::<Complex>::with_capacity(max_iter as usize);
    let mut escape_index: Option<u32> = None;

    let mut z = Complex::with_val(c_ref.prec(), (0.0, 0.0));

    for i in 0..max_iter {
        orbit.push(z.clone());
        z = z.clone() * &z + c_ref;

        if z.clone().abs().real().to_f64() >= 2.0 && escape_index == None {
            escape_index = Some(i);
        }
    }

    OrbitResult {orbit, escape_index}
}

pub fn insert_tile_views_if_not_found(
    tile_ids: &Vec<TileId>,
    tile_registry: TileRegistry,
    default_view_factory: impl Fn(&TileId) -> TileOrbitView,
) -> Vec<TileView> {
    let mut registry_g = tile_registry.write();
    tile_ids
        .iter()
        .filter_map(|tile_id| {
            if registry_g.contains_key(tile_id) {
                None
            } else {
                let tile_view = Arc::new(RwLock::new(default_view_factory(tile_id)));
                registry_g.insert(tile_id.clone(), tile_view.clone());
                Some(tile_view)
            }
        })
        .collect()
}

pub fn find_tile_views_with_deficient_scores(
    config: Arc<ScoutConfig>,
    tile_ids: &Vec<TileId>,
    tile_registry: TileRegistry,
) -> Vec<TileView> {
    let threshold = config.heuristic_config.tile_deficiency_threshold;
    let registry_g = tile_registry.read();
/*
    let tile_scores_log: Vec<(TileId, f64)> = tile_ids
        .iter()
        .filter_map(|tile_id| {
            registry_g.get(&tile_id).and_then(|view| {
                let view_g = view.read();
                Some((view_g.tile.clone(), view_g.local_score))
            })
        })
        .collect();

    trace!("Evaluating tiles under camera for deficient scores {:?}", 
        tile_scores_log);
*/
    tile_ids.iter()
        .filter_map(|tile_id| {
            registry_g.get(&tile_id).and_then(|view| {
                if view.read().local_score > threshold {
                    Some(view.clone())
                } else {
                    None
                }
            })
        })
        .collect()
}

/// Generate points within a tile according to the given strategy.
/// - Grid: evenly spaced including corners
/// - Corners: the four corner points of the tile square (center ± radius in Re/Im)
/// - RandomInDisk: `count` points, uniform in disk (probabilistically robust)
/// - Center: just the center
pub fn generate_tile_candidate_seeds(
    tile: &TileGeometry,
    strategy: OrbitSeedStrategy,
    count: usize, // Used for RandomInDisk, ignored for others
    rng: OrbitSeedRng,
) -> Vec<Complex> {
    let (prec, _) = tile.center.prec(); // maintain precision

    match strategy {
        OrbitSeedStrategy::Center => {
            vec![tile.center.clone()]
        }
        OrbitSeedStrategy::RandomInDisk => {
            let mut rng_g = rng.lock();
            let rng: &mut ChaCha8Rng = &mut *rng_g;
            (0..count).map(|_| {
                // Random point uniformly in disk (normalized)
                let theta = rng.gen_range(0.0..(2.0*std::f64::consts::PI));
                let sqrt_r = rng.r#gen::<f64>().sqrt();

                // Now scale by the tile's radius
                let r = Float::with_val(prec, sqrt_r) * tile.radius.clone();
                let cos = Float::with_val(prec, theta).cos();
                let sin = Float::with_val(prec, theta).sin();

                let offset = Complex::with_val(prec, (&r * &cos, &r * &sin));
                //trace!("RandomInDisk offset={:?} r={:?} theta={} sqrt_r={} tile.radius={:?}", 
                //    &offset, &r, theta, sqrt_r, &tile.radius);
                tile.center.clone() + offset
            }).collect()
        }
        OrbitSeedStrategy::Corners => {
            // Generate all four corners of the square around center, using ±radius
            let mut seeds = Vec::with_capacity(4);
            let r = &tile.radius;
            let corners = [
                (Float::with_val(prec, 1), Float::with_val(prec, 1)),
                (Float::with_val(prec, 1), Float::with_val(prec, -1)),
                (Float::with_val(prec, -1), Float::with_val(prec, 1)),
                (Float::with_val(prec, -1), Float::with_val(prec, -1)),
            ];
            for (rx, ry) in corners.iter() {
                let dx = r.clone() * rx;
                let dy = r.clone() * ry;
                let offset = Complex::with_val(prec, (dx.clone(), dy.clone()));
                seeds.push(tile.center.clone() + offset);
            }
            seeds
        }
        OrbitSeedStrategy::Grid(nx, ny) => {
            // Place points on the (nx x ny) grid, including corners
            // Even if nx/ny==1, will place in center
            let nx = nx.max(1) as usize;
            let ny = ny.max(1) as usize;

            let mut seeds = Vec::with_capacity(nx * ny);

            let two = Float::with_val(prec, 2);
            for ix in 0..nx {
                // Grid fraction in [-1, 1]
                let fx = if nx == 1 {
                    Float::with_val(prec, 0)
                } else {
                    Float::with_val(prec, ix as f64) / Float::with_val(prec, (nx-1) as f64) * two.clone() - Float::with_val(prec, 1)
                };
                for iy in 0..ny {
                    let fy = if ny == 1 {
                        Float::with_val(prec, 0)
                    } else {
                        Float::with_val(prec, iy as f64) / Float::with_val(prec, (ny-1) as f64) * two.clone() - Float::with_val(prec, 1)
                    };
                    let dx = tile.radius.clone() * fx.clone();
                    let dy = tile.radius.clone() * fy.clone();
                    let offset = Complex::with_val(prec, (dx, dy));
                    seeds.push(tile.center.clone() + offset);
                }
            }
            seeds
        }
    }
}

pub async fn update_tile_views_with_orbit_observations(
    config: Arc<ScoutConfig>,
    tile_registry: TileRegistry,
    orbit_observations: Arc<Vec<signals::OrbitObservation>>
) -> bool {
    let mut orbit_rank_updated = false;
    let mut reg_g = tile_registry.write();
    let mut views_modified: HashMap<TileId, TileView> = HashMap::new();
    let mut frame_st = signals::FrameStamp::new();

    // OrbitObservations are structured in such a way that inherently carries
    // screen-space tile information. As such, orbits (and c-tiles) get re-used.
    // Here is the best place to compile observations accross these seporated 
    // indecies.
    for obs in orbit_observations.iter() {
        frame_st = obs.frame_stamp;

        if let Some(view) = reg_g.get_mut(&obs.tile_id) {
            let mut view_modified = false;
            let mut view_g = view.write();

            // Get the TileOrbitStats for this corresponding orbit, creating
            // new stats if none were found.
            view_g.orbit_stats.entry(obs.orbit_id)
                .and_modify(|stats| {
                    stats.min_last_valid_i =
                        stats.min_last_valid_i.min(obs.feedback.min_last_valid_i);

                    stats.max_last_valid_i =
                        stats.max_last_valid_i.max(obs.feedback.max_last_valid_i);

                    stats.perturb_attempted += obs.feedback.perturb_attempted_count as u64;
                    stats.perturb_valid     += obs.feedback.perturb_valid_count as u64;
                    stats.perturb_collapsed += obs.feedback.perturb_collapsed_count as u64;
                    stats.perturb_escaped   += obs.feedback.perturb_escaped_count as u64;
                    stats.absolute_fallback += obs.feedback.absolute_fallback_count as u64;
                    stats.absolute_escaped  += obs.feedback.absolute_escaped_count as u64;
                    view_modified = true;
                });

            // Update the (global) live orbit
            let live_orbits = upgrade_orbit_list(&view_g.weak_orbits);
        
            live_orbits
                .iter()
                .find(|(_, orb)| orb.read().orbit_id == obs.orbit_id)
                .and_then(|(s, orb)| {
                    update_orbit_with_observation(orb.clone(), obs);
                    Some((s, orb))
                });

            if view_modified {
                views_modified.insert(obs.tile_id.clone(), view.clone());
            }       
        }
    }
    debug!("{} TileView(s) modified with orbit observations", views_modified.len());

    let mut view_scores_log: Vec<(TileId, f64, f64, f64)> = Vec::new();

    for (tile_id, view) in views_modified.iter() {
        let mut view_g = view.write();
        let mut live_orbits = upgrade_orbit_list(&view_g.weak_orbits);

        // Now score the (live) orbits in the tile view.
        live_orbits = score_tile_orbits(config.clone(), 
            &live_orbits, &mut view_g.orbit_stats, frame_st);

        // With the orbits now scored, also compute the local score for the view
        let view_score = score_tile_view(&live_orbits);
        view_g.local_score = view_score;

        // Now rank the (live) orbits in the tile view.
        let ranking_changed = rank_tile_orbits(
                &mut live_orbits, view_g.curr_max_orbits);
        view_g.weak_orbits.clear();

        let mut orb_id_list: Vec<(f64, OrbitId)> = Vec::new();
        for (s, o) in &live_orbits {
            orb_id_list.push((*s, o.read().orbit_id));
            view_g.weak_orbits.push((*s, Arc::downgrade(&o)));
        }

        if ranking_changed {
            orbit_rank_updated = true;
            let first_score = if let Some((s, _)) = live_orbits.first() {*s} else {0.0};
            let last_score  = if let Some((s, _)) = live_orbits.last()  {*s} else {0.0};

            view_scores_log.push((tile_id.clone(), view_score, first_score, last_score));
        }
    }

    if orbit_rank_updated {
        let mut trace_str = String::from(
            format!("{} Views scored after orbit observations...\n\t\t",
                view_scores_log.len()).as_str());
        let mut curr_row = 0;
        for (tile, score, hi, lo) in &view_scores_log {
            if curr_row > 2 {
                trace_str.push_str("\n\t\t");
                curr_row = 0;
            }
            trace_str.push_str(format!("(tile x, y, l: {:>3}, {:3>}, {:3>}) (score avg, hi, lo: {:>7.2}, {:>7.2}, {:>7.2})\t", 
                tile.tx, tile.ty, tile.level, score, hi, lo).as_str());
            curr_row += 1;
        }
        trace!("{}", trace_str);
    }
    
    debug!("{} TileView(s) scores were updated!", view_scores_log.len());
    orbit_rank_updated
}

fn update_orbit_with_observation(
    live_orbit: LiveOrbit, obs: &signals::OrbitObservation
) {
    let mut orb_g = live_orbit.write();
    orb_g.min_valid_perturb_index = 
        orb_g.min_valid_perturb_index.min(obs.feedback.min_last_valid_i);
    orb_g.max_valid_perturb_index = 
        orb_g.max_valid_perturb_index.max(obs.feedback.max_last_valid_i);
    orb_g.last_updated = obs.frame_stamp;
}

// Score Orbits within this TileOrbitView
// Checks first if orbits are live, and only scores these orbits.
// Primarily uses TileOrbitStats, which correspond to the Orbit via a hashmap
// Timestamps are only updated after scoring
fn score_tile_orbits(
    config: Arc<ScoutConfig>, 
    live_orbits: &Vec<(f64, LiveOrbit)>,
    orbit_stats: &mut HashMap<OrbitId, TileOrbitStats>,
    frame_st: signals::FrameStamp,
) -> Vec<(f64, LiveOrbit)> {
    let frame_decay_increment = config.heuristic_config.frame_decay_increment;
    let mut scored_orbits: Vec<(f64, LiveOrbit)> = Vec::new();
    //let tile = view_g.tile.clone();

    // Score only orbits that are currently live.
    for (_, orb) in live_orbits {
        let orb_g = orb.read();
        let score = if let Some(stats) = orbit_stats.get_mut(&orb_g.orbit_id) {
            let attempted = stats.perturb_attempted.max(1) as f64;

            let valid_ratio =
                stats.perturb_valid as f64 / attempted;

            let collapse_ratio =
                stats.perturb_collapsed as f64 / attempted;

            let escape_ratio =
                stats.perturb_escaped as f64 / attempted;

            let stability =
                (stats.min_last_valid_i + stats.max_last_valid_i) as f64 / 2.0;

            let age_frames =
                frame_st.frame_id.saturating_sub(stats.last_updated.frame_id) as f64;

            let age_factor = age_frames / frame_decay_increment;

            // Compute score from above ratios
            // Lower scores are better, as ranking sorts from lowest to highest
            stats.cached_score = 
                  (2.0 * -stability) 
                + (500.0 * valid_ratio) 
                + (300.0 * collapse_ratio)
                + (200.0 * escape_ratio)
                + age_factor;

            //trace!("Local Scoring for {:?} orbit_id={} stability={} valid_ratio={} collapse_ratio={} escape_ratio={} age_factor={}",
            //    &tile, orb_g.orbit_id, stability, valid_ratio, collapse_ratio, escape_ratio, age_factor);
            
            // Treat scoring as the final update to the timestamp in stats
            stats.last_updated = frame_st;
            stats.cached_score
        } else { 0.0 };

        scored_orbits.push((score, orb.clone()));
    }

    scored_orbits
}

/// Compute and update the local score for a tile view, based on the scores kept beside the orbits
pub fn score_tile_view(
    live_orbits: &Vec<(f64, LiveOrbit)>,
) -> f64 {
    if live_orbits.is_empty() {
        return 0.0;
    }
    // Sum the cached scores of live orbits; if not present, treat as 0.0
    let sum: f64 = live_orbits.iter().map(|(score, _)| score).sum();
    let score = sum / (live_orbits.len() as f64);
    
    score 
}

// Sorts the Orbits within the tile view
// Note that the orbits are also truncated here, to fit the view's configured
// max_orbits_per_tile. Returns true ONLY if the orbit ranking has changed.
fn rank_tile_orbits(
    live_orbits: &mut Vec<(f64, LiveOrbit)>,
    max_orbits: usize,
) -> bool {
    // Create a simple hash of the orbit id's, based on their position
    // within the list.
    let pre_hash = compute_orbit_list_hash(&live_orbits);

    live_orbits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Remove low-ranked orbits
    live_orbits.truncate(max_orbits);

    let post_hash = compute_orbit_list_hash(&live_orbits);

    pre_hash != post_hash
}

pub fn upkeep_orbit_pool(
    living_orbits: LivingOrbits, 
    orbits_to_add: Vec<LiveOrbit>,
    max_live_orbits: u32,
    frame_stamp: &signals::FrameStamp
) {
    let mut orb_pool_g = living_orbits.lock();
    for o in orbits_to_add {
        orb_pool_g.push((0.0, o));
    }

    let mut scored_orbits: Vec<(f64, LiveOrbit)> = orb_pool_g
        .iter()
        .map(|(_, orb)| (
            score_live_orbit(orb.clone(), frame_stamp), orb.clone()
        ))
        .collect();

    scored_orbits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    scored_orbits.truncate(max_live_orbits as usize);
    debug!("Living Orbits now has {} elements", scored_orbits.len());

    if let Some(first_25) = scored_orbits.first_chunk::<25>() {
        let first_25_orbs: Vec<(u64, f64)> = first_25
            .iter()
            .map(|(score, orb)| {
                let orb_g = orb.read();
                (orb_g.orbit_id, *score)
            })
            .collect();
        trace!("First 25 Live Orbits are now: {:?}", first_25_orbs);
    }
    orb_pool_g.clear();
    orb_pool_g.append(&mut scored_orbits);
}

fn score_live_orbit(live_orbit: LiveOrbit, frame_stamp: &signals::FrameStamp) -> f64 {
    let orb_g = live_orbit.read();

    let stability = orb_g.max_valid_perturb_index as f64; 
    let age_frames = 
        frame_stamp.frame_id.saturating_sub(orb_g.created_at.frame_id) as f64 * 0.01;

    // Lower scores are better, as ranking sorts lowest to highest
    let score = 
        -stability 
        + age_frames;

    //trace!("Global Scoring for orbit_id={} stability={} age_frames={}", 
    //    orb_g.orbit_id, stability, age_frames);
    score
}
