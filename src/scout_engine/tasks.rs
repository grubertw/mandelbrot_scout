use crate::scout_engine::{OrbitSeedRng, ScoutConfig};
use crate::scout_engine::orbit::*;
use crate::scout_engine::tile::*;

use crate::numerics::ComplexDf;
use crate::signals::FrameStamp;

use std::sync::Arc;
use log::{trace, debug};
use rand_chacha::ChaCha8Rng;
use rand::Rng;
use rug::{Float, Complex};
use parking_lot::RwLock;

pub async fn start_reference_orbit(
    c_ref: Complex, 
    id_factory: OrbitIdFactory,
    config: ScoutConfig,
    frame_st: FrameStamp
) -> LiveOrbit {
    let (max_ref_orbit_iters, max_user_iters) = {
        let config_g = config.lock();
        (
            config_g.max_ref_orbit_iters,
            config_g.max_user_iters
        )
    };
    
    let mut orbit = ReferenceOrbit::new(
        id_factory, c_ref,
        frame_st, max_ref_orbit_iters,
    );

    // Note that we will compute 'a bit' past user_iter, for perturbation validity
    orbit.compute_to(max_user_iters);

    let cdf_orbit: Vec<ComplexDf> = orbit.orbit
        .iter()
        .map(|c| ComplexDf::from_complex(c))
        .collect();

    for cdf in cdf_orbit {
        orbit.gpu_payload.re_hi.push(cdf.re.hi);
        orbit.gpu_payload.re_lo.push(cdf.re.lo);
        orbit.gpu_payload.im_hi.push(cdf.im.hi);
        orbit.gpu_payload.im_lo.push(cdf.im.lo);
    }
    Arc::new(RwLock::new(orbit))
}

pub async fn continue_reference_orbit(orbit: LiveOrbit, max_user_iters: u32) {
    let mut orbit_g = orbit.write();
    orbit_g.is_anchored = false;

    let start_iter = orbit_g.orbit.len();
    orbit_g.compute_to(max_user_iters);

    debug!("Continued orbit {} to max_user_iters={}. escape={:?}",
        orbit_g.orbit_id, max_user_iters, orbit_g.quality_metrics.escape_index);

    for i in start_iter..=orbit_g.orbit.len() {
        let cdf = ComplexDf::from_complex(&orbit_g.orbit[i]);

        orbit_g.gpu_payload.re_hi.push(cdf.re.hi);
        orbit_g.gpu_payload.re_lo.push(cdf.re.lo);
        orbit_g.gpu_payload.im_hi.push(cdf.im.hi);
        orbit_g.gpu_payload.im_lo.push(cdf.im.lo);
    }
}

pub  fn try_to_anchor_tiles_from_pool(
    living_orbits: LivingOrbits,
    tile_views: &Vec<TileView>,
) -> u32 {
    let mut orbits_anchored_count = 0;
    let orb_pool_g = living_orbits.lock();

    for tile in tile_views {
        let mut tile_g = tile.write();

        for orb in orb_pool_g.iter() {
            if tile_g.try_anchor_orbit(orb.clone(), false) {
                orbits_anchored_count += 1;
            }
        }
    }

    debug!("From {} pool orbits, {} were anchored!",
        orb_pool_g.len(), orbits_anchored_count);

    orbits_anchored_count
}

pub fn fill_registry_with_missing_tiles(
    tile_ids: &[TileId],
    tile_registry: TileRegistry,
    config: ScoutConfig,
    level: &TileLevel,
) -> (Vec<TileView>, u32) {
    let mut tile_reg_g = tile_registry.write();
    let (num_orbits_to_spawn_per_tile,
        max_tile_anchor_failure_attempts,
        split_on_poor_coverage_check,
        smallest_tile_pixel_span,
        coverage_to_anchor,
        max_user_iter) = {
        let config_g = config.lock();
        (
            config_g.num_orbits_to_spawn_per_tile,
            config_g.max_tile_anchor_failure_attempts,
            config_g.split_tile_on_poor_coverage_check,
            config_g.smallest_tile_pixel_span,
            config_g.coverage_to_anchor,
            config_g.max_user_iters
        )
    };
    debug!("Fill registry with missing tiles!");

    let mut creation_count = 0;
    let tiles: Vec<TileView> = tile_ids
        .iter()
        .map(|tile_id| {
            if tile_reg_g.contains_key(&tile_id) == false {
               creation_count += 1; 
            }
            tile_reg_g.entry(tile_id.clone())
                .or_insert(Arc::new(RwLock::new(TileOrbitView::new(
                    tile_id, level,
                    num_orbits_to_spawn_per_tile,
                    max_tile_anchor_failure_attempts,
                    split_on_poor_coverage_check,
                    smallest_tile_pixel_span,
                    coverage_to_anchor,
                    max_user_iter
                )))).clone()
        })
        .collect();
    (tiles, creation_count)
}

// Update the tiles config params
// If the user_max_iter changed, not only do we need to update it's internal value,
// but we also might need to reset anchor tiles, as orbit r_valid increases
pub  fn update_tile_config(tiles: &[TileView], config: ScoutConfig, tiles_to_reset: &mut Vec<TileView>) {
    let (num_orbits_to_spawn_per_tile,
        max_tile_anchor_failure_attempts,
        split_on_poor_coverage_check,
        smallest_tile_pixel_span,
        coverage_to_anchor,
        max_user_iter) = {
        let config_g = config.lock();
        (
            config_g.num_orbits_to_spawn_per_tile,
            config_g.max_tile_anchor_failure_attempts,
            config_g.split_tile_on_poor_coverage_check,
            config_g.smallest_tile_pixel_span,
            config_g.coverage_to_anchor,
            config_g.max_user_iters
        )
    };
    
    for tile in tiles {
        let mut tile_g = tile.write();
        if max_user_iter > tile_g.tile_iter && tile_g.anchor_orbit.is_some() {
            tiles_to_reset.push(tile.clone());
        }
        tile_g.num_orbits_per_spawn = num_orbits_to_spawn_per_tile;
        tile_g.max_tile_anchor_failure_attempts = max_tile_anchor_failure_attempts;
        tile_g.split_on_poor_coverage_check = split_on_poor_coverage_check;
        tile_g.smallest_tile_pixel_span = smallest_tile_pixel_span;
        tile_g.coverage_to_anchor = coverage_to_anchor;
        tile_g.tile_iter = max_user_iter;

        update_tile_config(&tile_g.children, config.clone(), tiles_to_reset);
    }
}

/// Generate points within a tile according to the given strategy.
/// - Grid: evenly spaced including corners
/// - Corners: the four corner points of the tile square (center ± radius in Re/Im)
/// - RandomInDisk: `count` points, uniform in disk (probabilistically robust)
/// - Center: just the center
/// - WeightedDirection: where the real magic happens!
pub fn generate_tile_candidate_seeds(
    tile: &TileGeometry,
    strategy: OrbitSeedStrategy,
    count: Option<usize>,
    rng: Option<OrbitSeedRng>,
    candidates: Option<&[(f64, Complex)]>
) -> Vec<Complex> {
    let (prec, _) = tile.center().prec(); // maintain precision

    match strategy {
        OrbitSeedStrategy::Center => {
            vec![tile.center().clone()]
        }
        OrbitSeedStrategy::RandomInDisk => {
            if count.is_none() || rng.is_none() {
                vec![]
            }
            else {
                let orbit_rng = rng.unwrap();
                let mut rng_g = orbit_rng.lock();
                let rng: &mut ChaCha8Rng = &mut *rng_g;
                (0..count.unwrap()).map(|_| {
                    // Random point uniformly in disk (normalized)
                    let theta = rng.gen_range(0.0..(2.0 * std::f64::consts::PI));
                    let sqrt_r = rng.r#gen::<f64>().sqrt();

                    // Now scale by the tile's radius
                    let r = Float::with_val(prec, sqrt_r) * tile.radius().clone();
                    let cos = Float::with_val(prec, theta).cos();
                    let sin = Float::with_val(prec, theta).sin();

                    let offset = Complex::with_val(prec, (&r * &cos, &r * &sin));
                    //trace!("RandomInDisk offset={:?} r={:?} theta={} sqrt_r={} tile.radius={:?}",
                    //    &offset, &r, theta, sqrt_r, &tile.radius);
                    tile.center().clone() + offset
                }).collect()
            }
        }
        OrbitSeedStrategy::Corners => {
            // Generate all four corners of the square around center, using ±radius
            let mut seeds = Vec::with_capacity(4);
            let r = tile.radius();
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
                seeds.push(tile.center().clone() + offset);
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
                    let dx = tile.radius().clone() * fx.clone();
                    let dy = tile.radius().clone() * fy.clone();
                    let offset = Complex::with_val(prec, (dx, dy));
                    seeds.push(tile.center().clone() + offset);
                }
            }
            seeds
        }
        OrbitSeedStrategy::WeightedDirection => {
            if candidates.is_none() {
                vec![]
            }
            else {
                let candidates = candidates.unwrap();
                let mut accum_dir = Complex::with_val(prec, (0, 0));
                let mut weight_sum = Float::with_val(prec, 0);
                let mut best_dir = Complex::with_val(prec, (0, 0));
                let mut best_coverage = 0.0;

                for (coverage, cand_c_ref) in candidates.iter() {
                    let delta = cand_c_ref.clone() - tile.center();
                    let mag = delta.clone().abs();

                    if mag.real().to_f64() > tile.dir_epsilon {
                        let dir = delta / mag;

                        let weight = Float::with_val(prec, *coverage);
                        let dir_weight = dir.clone() * &weight;
                        accum_dir += &dir_weight;
                        weight_sum += &weight;

                        if *coverage > best_coverage {
                            best_coverage = *coverage;
                            best_dir = dir;
                        }
                    }
                }

                let avg_dir = if weight_sum > tile.dir_epsilon {
                    let avg = accum_dir / &weight_sum;
                    let mag = avg.clone().abs();
                    if mag.real().to_f64() > tile.dir_epsilon {
                        avg / mag
                    } else {
                        best_dir.clone()
                    }
                } else {
                    best_dir.clone()
                };

                // Take a combination, which best avoids barycentric averaging
                let mut final_dir = avg_dir.clone() * Float::with_val(prec, 0.7);
                let bias_best_dir = best_dir.clone() * Float::with_val(prec, 0.3);
                final_dir += &bias_best_dir;

                let mag = final_dir.clone().abs();
                if mag.real().to_f64() > tile.dir_epsilon {
                    final_dir /= mag;
                }

                let mut step = tile.half_diagonal().clone();
                step *= GRADIENT_ASCENT_STEP_SIZE;
                let step_dir = final_dir.clone() * &step;
                let new_seed = tile.center().clone() + &step_dir;
                vec![new_seed]
            }
        }
    }
}

pub fn upkeep_orbit_pool(
    living_orbits: LivingOrbits, 
    orbits_to_add: Vec<LiveOrbit>,
    max_live_orbits: u32,
    frame_stamp: &FrameStamp
) {
    let mut orb_pool_g = living_orbits.lock();
    for o in orbits_to_add {
        orb_pool_g.push(o);
    }

    let mut snapshots: Vec<OrbitSnapshot> = orb_pool_g
        .iter()
        .map(|orb| score_live_orbit(orb.clone(), frame_stamp))
        .collect();

    snapshots.sort_by(|a, b| {
        use std::cmp::Ordering;

        // 1️⃣ Larger r_valid first
        match b.r_log.partial_cmp(&a.r_log).unwrap_or(Ordering::Equal) {
            Ordering::Equal => {}
            ord => return ord,
        }

        // 2️⃣ Prefer non-exterior
        match a.is_exterior.cmp(&b.is_exterior) {
            Ordering::Equal => {}
            ord => return ord,
        }

        // 3️⃣ More negative contraction is better
        match a.contraction.partial_cmp(&b.contraction).unwrap_or(Ordering::Equal) {
            Ordering::Equal => {}
            ord => return ord,
        }

        // 4️⃣ Younger preferred
        a.age.partial_cmp(&b.age).unwrap_or(Ordering::Equal)
    });

    *orb_pool_g = snapshots
        .into_iter()
        .map(|snap| snap.orbit)
        .collect();


    orb_pool_g.truncate(max_live_orbits as usize);
    debug!("Living Orbits now has {} elements", orb_pool_g.len());

    if let Some(first_25) = orb_pool_g.first_chunk::<25>() {
        let first_25_orbs: Vec<u64> = first_25
            .iter()
            .map(|orb| {
                let orb_g = orb.read();
                orb_g.orbit_id
            })
            .collect();
        trace!("First 25 Live Orbits are now: {:?}", first_25_orbs);
    }
}

fn score_live_orbit(
    live_orbit: LiveOrbit, 
    frame_stamp: &FrameStamp
) -> OrbitSnapshot {
    let orb_g = live_orbit.read();

    let r_log = orb_g.r_valid()
        .to_f64()
        .max(1e-300)
        .abs()
        .log10();

    let contraction = orb_g.contraction().to_f64();
    let is_exterior = orb_g.is_exterior();

    let age = (frame_stamp.frame_id.saturating_sub(
                orb_g.quality_metrics.created_at.frame_id)) as f64 * 0.01;

    let snap = OrbitSnapshot {
        orbit: live_orbit.clone(),
        r_log,
        contraction,
        is_exterior,
        age,
    };

    //trace!("Global Scoring for orbit {} snapshot r_log={:?} contraction={:?} exterior={:?} age={}", 
    //    orb_g.orbit_id, r_log, contraction, is_exterior, age);
    snap
}
