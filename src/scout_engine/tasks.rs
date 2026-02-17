use crate::scout_engine::OrbitSeedRng;
use crate::scout_engine::orbit::*;
use crate::scout_engine::tile::*;
use crate::scout_engine::utils::*;

use crate::numerics::ComplexDf;
use crate::signals::{CameraSnapshot, FrameStamp};

use std::sync::Arc;

use log::{trace, debug};
use rand_chacha::ChaCha8Rng;
use rand::Rng;
use rug::{Float, Complex};
use parking_lot::RwLock;

pub async fn create_new_reference_orbit(
    c_ref: Complex, id_fac: OrbitIdFactory, 
    max_iter: u32, frame_st: FrameStamp
) -> LiveOrbit {
    let orbit_result = compute_reference_orbit(&c_ref, max_iter).await;
    let cdf_orbit: Vec<ComplexDf> = orbit_result.orbit
        .iter()
        .map(|c| ComplexDf::from_complex(c))
        .collect();
    
    let qualiy_metrics = OrbitQuality::new(
        orbit_result.escape_index, orbit_result.r_valid, 
        orbit_result.contraction, orbit_result.period, 
        orbit_result.z_min, orbit_result.a_max, frame_st);

    let mut gpu_payload = OrbitGpuPayload::new();
    for cdf in cdf_orbit {
        gpu_payload.re_hi.push(cdf.re.hi);
        gpu_payload.re_lo.push(cdf.re.lo);
        gpu_payload.im_hi.push(cdf.im.hi);
        gpu_payload.im_lo.push(cdf.im.lo);
    }

    Arc::new(RwLock::new(ReferenceOrbit::new(
        id_fac, c_ref, orbit_result.orbit, qualiy_metrics, gpu_payload,
    )))
}

async fn compute_reference_orbit(
    c_ref: &Complex, max_iter: u32, 
) -> OrbitResult {
    // Loop constants
    let prec = c_ref.prec();
    let alpha = Float::with_val(prec.0, ALPHA);
    let two = Float::with_val(prec.0, 2);

    // Loop output
    let mut orbit = Vec::<Complex>::with_capacity(max_iter as usize);
    let mut escape_index: Option<u32> = None;
    let mut r_valid = Float::with_val(prec.0, f64::INFINITY);

    // Mutates in-loop
    let mut z = Complex::with_val(prec, (0.0, 0.0));
    let mut a = Complex::with_val(prec, (0.0, 0.0)); // our Talor 'A' term.
    let mut z_min = Float::with_val(prec.0, f64::INFINITY);
    let mut a_max = Float::with_val(prec.0, 0);
    let mut detected_period: Option<u32> = None;
    let mut log_sum = Float::with_val(prec.0, 0); // for contraction
    let mut log_count: u32 = 0;

    for i in 0..max_iter {
        orbit.push(z.clone());
        let two_z = z.clone() * &two;

        // Compute ratio for r_valid, but only after first iteration
        // Also, stop r_valid computation once escape is reached.
        // Same logic applies for log-sum contraction
        if i > 0 && escape_index == None {
            let z_abs = z.clone().abs().real().clone();
            let a_abs = a.clone().abs().real().clone();
            let mag_two_z = two_z.clone().abs().real().clone();

            if a_abs > Float::with_val(prec.0, 0.0) {
                let candidate = alpha.clone() * &z_abs / &a_abs;
                if candidate < r_valid {
                    // Final r_valid = min_n(alpha*(|Zn|/|An|))
                    r_valid = candidate;
                }
            }
            // Avoid log(0)
            if mag_two_z > NEAR_ZERO_THRESHOLD {
                let log_val = mag_two_z.clone().ln();
                log_sum += &log_val;
                log_count += 1;
            }
            // Grab min_z and max_a - they are super cheap and somewhat useful, 
            // so why not!
            if z_abs < z_min {
                z_min = z_abs.clone();
            }
            if a_abs > a_max {
                a_max = a_abs.clone();
            }
        }
        // Period detection logic
        if detected_period.is_none() && i > BURN_IN {
            for p in 1..=MAX_PERIOD_CHECK {
                if i >= p {
                    let prev_z = &orbit[(i - p) as usize];
                    let diff = z.clone() - prev_z;
                    let abs_diff = diff.abs().real().clone();

                    if abs_diff < NEAR_ZERO_THRESHOLD {
                        detected_period = Some(p);
                        break;
                    }
                }
            }
        }

        // Core Mandelbrot recurrence, using rug::Complex arbitrary precicision
        // Z_{n+1} = Z^2 * C
        z = z.clone() * &z + c_ref;

        // Derivative recurrence for r_valid
        // A_{n+1} = 2 * z_n * A_n + 1
        a = a.clone() * &two_z + Complex::with_val(prec, (1.0, 0.0));

        // Escape index tracking. Note that we do not bailout here!
        // Ref orbits must go past bailout for perturbance!
        if z.clone().abs().real().to_f64() >= 2.0 && escape_index == None {
            escape_index = Some(i);
        }
    }

    // Compute contraction metric taking the orbit's period into account.
    let contraction = if let Some(p) = detected_period {
        let mut sum = Float::with_val(prec.0, 0);
        let start = orbit.len() - p as usize;

        for k in start..orbit.len() {
            let two_z = orbit[k].clone() * &two;
            let mag_two_z = two_z.clone().abs().real().clone();
            if mag_two_z > NEAR_ZERO_THRESHOLD {
                sum += mag_two_z.clone().ln();
            }
        }

        sum / Float::with_val(prec.0, p)
    } else {
        // fallback: global average
        log_sum / Float::with_val(prec.0, log_count)
    };

    OrbitResult {orbit, escape_index, r_valid, 
        contraction, period: detected_period, z_min, a_max}
}

 // Called at start of the scout_worker 
pub fn initialize_tile_grid(
    current_camera: &CameraSnapshot,
    tile_registry: TileRegistry,
    num_orbits_to_spawn_per_tile: u32,
    max_tile_anchor_failure_attempts: u32,
    rug_precision: u32,
) {
    let level_zero = TileLevel::new(0, rug_precision);

    let tile_ids = find_tile_ids_under_camera(current_camera);

    debug!("Initializing top-level layout with {} tiles", tile_ids.len());

    let new_tiles: Vec<TileOrbitView> = tile_ids
        .iter()
        .map(|tile_id| TileOrbitView::new(
            tile_id, &level_zero, 
            num_orbits_to_spawn_per_tile, 
            max_tile_anchor_failure_attempts
        ))
        .collect();

    let mut tile_reg_g = tile_registry.write();
    for tile in new_tiles {
        tile_reg_g.insert(tile.id.clone(), Arc::new(RwLock::new(tile)));
    }
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
    let (prec, _) = tile.center().prec(); // maintain precision

    match strategy {
        OrbitSeedStrategy::Center => {
            vec![tile.center().clone()]
        }
        OrbitSeedStrategy::RandomInDisk => {
            let mut rng_g = rng.lock();
            let rng: &mut ChaCha8Rng = &mut *rng_g;
            (0..count).map(|_| {
                // Random point uniformly in disk (normalized)
                let theta = rng.gen_range(0.0..(2.0*std::f64::consts::PI));
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
    }
}

pub fn try_to_anchor_tiles_from_pool(
    living_orbits: LivingOrbits,
    tile_views: &Vec<TileView>,
) -> bool {
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

    orbits_anchored_count > 0
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
                orb_g.qualiy_metrics.created_at.frame_id)) as f64 * 0.01;

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
