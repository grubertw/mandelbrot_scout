use std::cmp::Ordering;
use crate::scout_engine::orbit::*;
use crate::scout_engine::utils::*;

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

use rug::{Float, Complex};
use log::{trace, debug, info};
use crate::signals::CameraSnapshot;

pub type TileLevels = Arc<RwLock<Vec<TileLevel>>>;
pub type TileView = Arc<RwLock<TileOrbitView>>;
pub type TileRegistry = Arc<RwLock<HashMap<TileId, TileView>>>;

pub const POOR_COVERAGE_THRESHOLD: f64 = 1e-3;
pub const GRADIENT_ASCENT_STEP_SIZE: f64 = 0.4;

const EPSILON: f64 = 1e-6;

#[derive(Clone, Debug)]
pub struct TileLevel {
    /// Logical level index (0 = coarse, 1 = fine, etc.)
    pub level: u32,
    /// Size of one tile in complex-plane units
    /// (e.g. 0.25, 2.5e-6, ...)
    pub tile_size: Float,
}

impl TileLevel {
    pub fn new(tile_size: Float) -> Self {
        Self {
            level: 0, tile_size,
        }
    }

    pub fn from_parent(level: &TileLevel) -> Self {
        Self {
            level: level.level + 1,
            tile_size: level.tile_size.clone() / 2.0,
        }
    }
}

// TileId's are our way to create a deterministic 'integer lattice' atop
// the complex co-orditate space. Think graph paper. And like graph paper, 
// it remains stationary, and the viewport camera behaves like a 'lens' or
// magnifying glass. 
// Integers should always be derived from a 'known' tile_size, i.e. 
// From a TileLevel assigned to a TileView, or the 'master list' inside the 
// TileFactory
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct TileId {
    pub tx: i64, // floor(re / tile_size)
    pub ty: i64, // floor(im / tile_size)
}

impl TileId {
    // By taking C and dividing by tile_size (and flooring), we have an
    // easy way to determine C's inclusion into that tile. It is also
    // very handy for determining bounding boxes, when given two diagnal
    // C values, as they will intrinsiclly include extra area, so as not 
    // to neglect tiles (and their included orbits) for consideration.
    pub fn from_point(c: &Complex, level: &TileLevel) -> Self {
        let mut re = c.real().clone();
        let mut im = c.imag().clone();
        re /= level.tile_size.clone();
        im /= level.tile_size.clone();

        let tx = re.floor().to_f64() as i64;
        let ty = im.floor().to_f64() as i64;
        Self { tx, ty }
    }
}



#[derive(Clone, Debug)]
pub struct TileGeometry {
    /// Center of the tile in the complex plane
    center: Complex,
    // C-Tiles are always square 
    radius: Float,
    // Cache half-diagnal, as it's used heavily
    half_diagonal: Float,
    /// Used during weighted direction seed computation
    pub dir_epsilon: f64
}

impl TileGeometry {
    pub fn from_id(tile_id: &TileId, tile_size: &Float) -> Self {
        let mut center_re = tile_size.clone() * tile_id.tx;
        let mut center_im = tile_size.clone() * tile_id.ty;

        let mut radius = tile_size.clone();
        radius /= 2;

        center_re += radius.clone();
        center_im += radius.clone();

        let mut diag = radius.clone();
        diag *= &radius;
        diag *= 2;
        let half_diagonal = Float::sqrt(diag);
        let dir_epsilon = half_diagonal.to_f64() * EPSILON;

        Self {
            center: Complex::with_val(
                tile_size.prec(),
                (center_re, center_im)
            ),
            radius, half_diagonal,
            dir_epsilon
        }
    }

    pub fn center(&self) -> &Complex {
        &self.center
    }

    pub fn radius(&self) -> &Float {
        &self.radius
    }

    pub fn half_diagonal(&self) -> &Float {
        &self.half_diagonal
    }
}

#[derive(Debug, Clone, Copy)]
pub enum OrbitSeedStrategy {
    Center,
    RandomInDisk,
    Corners,
    Grid(u8, u8), // (nx, ny)
    WeightedDirection,
}

#[derive(Clone, Debug)]
pub struct TileOrbitScore {
    pub coverage: f64,      // r_valid / needed
    pub depth_score: f64,
    pub stability_score: f64,
    pub total_score: f64,
}

#[derive(Clone, Debug)]
pub struct TileOrbitCandidate {
    pub score: TileOrbitScore,
    pub orbit: WeakOrbit,
}

#[derive(Clone, Debug)]
pub struct TileOrbitView {
    /// Tile identity (stable)
    pub id: TileId,
    /// Tile's level on the descending 1/2^lv heirarchy 
    pub level: TileLevel,
    /// Geometry of this tile
    pub geometry: TileGeometry,
    /// Our orbit anchor, who's r_valid fills the square bounds
    /// i.e. the half-diagnal of the tile's geometry 
    pub anchor_orbit: Option<LiveOrbit>,
    /// Direct children the level below this tile view
    /// Should never exceed 4.
    pub children: Vec<TileView>,
    /// A collection of weak refs to orbits that fall within the tile
    /// but have an insufficient r_valid to cover the tile.
    /// NOTE: These are the first orbits that should be tested when 
    /// the tile is split, and are worth keeping as references (weakly)
    /// in the parent tile.
    pub candidate_orbits: Vec<TileOrbitCandidate>,
    /// Number of orbits to create per spawn operation
    pub num_orbits_per_spawn: u32,
    /// Maximum number of times a 'ligitimate' attempt was made to anchor
    /// a tile and resulted in failure
    pub max_tile_anchor_failure_attempts: u32,
    /// If the orbits have bad coverage, split the tile after this many attempts
    pub split_on_poor_coverage_check: u32,
    /// Checked before a tile is split, and is our safeguard that prevents 
    /// too small tiles from being created
    pub smallest_tile_pixel_span: f64,
    /// Required coverage to anchor a tile
    pub coverage_to_anchor: f64,
    /// Tile max_iter maps to user max iter
    pub tile_iter: u32,
    // Measure the rate coverage is increasing, per orbit spawn iteration
    curr_best_coverage: f64,
    plateau_counter: u32,
}

impl TileOrbitView {
    pub fn new(
        tile_id: &TileId, level: &TileLevel, 
        num_orbits_per_spawn: u32,
        max_tile_anchor_failure_attempts: u32,
        split_on_poor_coverage_check: u32,
        smallest_tile_pixel_span: f64,
        coverage_to_anchor: f64,
        tile_iter: u32,
    ) -> Self {
        Self {
            id: tile_id.clone(), 
            level: level.clone(),
            geometry: TileGeometry::from_id(
                tile_id, &level.tile_size),
            anchor_orbit: None,
            children: Vec::new(),
            candidate_orbits: Vec::new(),
            num_orbits_per_spawn,
            max_tile_anchor_failure_attempts,
            split_on_poor_coverage_check,
            smallest_tile_pixel_span,
            coverage_to_anchor,
            tile_iter,
            curr_best_coverage: 0.0,
            plateau_counter: 0
        }
    }

    // Test if the orbit can be used as an anchor. 
    // if not, then place the orbit in the failed orbit
    // list (as a weak ref).
    // When the orbit is anchored, also calculate it's distance
    // from the tile's center. This is needed by the renderer and
    // is performant to calculate here, only once.
    pub fn try_anchor_orbit(
        &mut self, orbit: LiveOrbit,
        seeded_from_tile: bool,
    ) -> bool {
        if seeded_from_tile {
            let orb_g = orbit.read();
            trace!("{:?} spawned orbit {} c_ref={}. escape_index={:?} len={} r_valid={} contraction={} period={:?} z_min={} a_max={}",
                self.id, orb_g.orbit_id, orb_g.c_ref().to_string_radix(10, Some(6)),
                orb_g.quality_metrics.escape_index,
                orb_g.orbit.len(),
                orb_g.quality_metrics.r_valid.to_string_radix(10, Some(5)),
                orb_g.quality_metrics.contraction.to_string_radix(10, Some(3)),
                orb_g.quality_metrics.period,
                orb_g.quality_metrics.z_min.to_string_radix(10, Some(8)),
                orb_g.quality_metrics.a_max.to_string_radix(10, Some(8)),
            );
        }
        // Evaluate the orbit for potential fit as anchor
        let score = self.evaluate_orbit(orbit.clone());

        // if Orbit has coverage, use right away anchor!
        if score.coverage >= self.coverage_to_anchor {
            let mut orb_g = orbit.write();
            self.anchor_orbit = Some(orbit.clone());
            orb_g.is_anchored = true;

            info!("{:?} promoting orbit {}! with coverage above {:.4e} score={:?}",
                self.id, orb_g.orbit_id, self.coverage_to_anchor, score);
        }
        else if seeded_from_tile {
            let candidate = TileOrbitCandidate {
                score, orbit: Arc::downgrade(&orbit)
            };
            self.candidate_orbits.push(candidate);
            self.candidate_orbits
                .sort_by(|a, b| b.score.total_score
                    .partial_cmp(&a.score.total_score).unwrap_or(Ordering::Equal));
        }

        self.anchor_orbit.is_some()
    }

    pub fn should_split(&mut self) -> bool {
        if self.children.is_empty() && self.anchor_orbit.is_none() {
            let attempts = self.candidate_orbits.len();
            if attempts < self.split_on_poor_coverage_check as usize {
                return false;
            }

            let best_coverage = self.candidate_orbits
                .iter()
                .map(|tc| tc.score.coverage)
                .fold(0.0, f64::max);

            // Split the tile quickly if starting coverage is bad
            if best_coverage < POOR_COVERAGE_THRESHOLD {
                debug!("{:?} Tile will split because coverage was poor!", self.id);
                return true;
            }

            // Plateau detection (relative improvement)
            let improvement = best_coverage - self.curr_best_coverage;
            if best_coverage > 0.0 {
                let relative_improvement = improvement / best_coverage;

                if relative_improvement < 0.01 { // <1% improvement
                    self.plateau_counter += 1;
                } else {
                    self.plateau_counter = 0;
                }

                if self.plateau_counter >= 3 {
                    debug!("{:?} Tile will split because best coverage has reached a plateau! best_coverage={} relative_improvement={}",
                        self.id, best_coverage, relative_improvement);
                    return true;
                }
            }
            self.curr_best_coverage = best_coverage;

            // Hard cap
            if attempts >= self.max_tile_anchor_failure_attempts as usize {
                debug!("{:?} Tile will split because max anchor failure attempts has been reached!",
                    self.id);
                return true;
            }
        }
        false
    }

    pub fn split(&mut self) -> u32 {
        let scores: Vec<TileOrbitScore> = self.candidate_orbits
            .iter()
            .map(|tc| tc.score.clone())
            .collect();

        trace!("{:?} has {} scores: {:>2.3?}",
                self.id, scores.len(), scores);

        let child_level = TileLevel::from_parent(&self.level);
        // Deterministic creation of TileId(s) from parent!
        let child_tx = self.id.tx * 2;
        let child_ty = self.id.ty * 2;
        let child_ids = vec![
            TileId { tx: child_tx,     ty: child_ty     },
            TileId { tx: child_tx + 1, ty: child_ty     },
            TileId { tx: child_tx,     ty: child_ty + 1 },
            TileId { tx: child_tx + 1, ty: child_ty + 1 },
        ];

        debug!("Splitting TILE {:?} into {} Children {:?}\t@ {:?}",
            self.id, child_ids.len(), child_ids, child_level);

        let new_tiles: Vec<TileOrbitView> = child_ids
            .iter()
            .map(|tile_id| TileOrbitView::new(
                tile_id, &child_level, 
                self.num_orbits_per_spawn, 
                self.max_tile_anchor_failure_attempts,
                self.split_on_poor_coverage_check,
                self.smallest_tile_pixel_span,
                self.coverage_to_anchor,
                self.tile_iter
            ))
            .collect();
        
        let mut orbits_anchored_count = 0;

        for mut child in new_tiles {
            // Check if any failed orbits of parent fit around children
            let parent_orbits: Vec<LiveOrbit> = self.candidate_orbits
                .iter()
                .filter_map(|w| w.orbit.upgrade())
                .collect();
            for orb in parent_orbits {
                if child.try_anchor_orbit(orb, false) {
                    orbits_anchored_count += 1;
                }
            }

            self.children.push(Arc::new(RwLock::new(child)));
        }
        debug!("Child tiles promoted {} orbits from parent {:?} as anchors after split!", 
            orbits_anchored_count, self.id);

        // Parent to stop spawning orbits, in favor of children!
        self.num_orbits_per_spawn = 0;

        orbits_anchored_count
    }

    // If the tile's anchor has insufficient iterations
    // Invoked when the user increases user_max_iters
    pub fn reset(&mut self) -> Vec<LiveOrbit>  {
        let mut orbits_to_continue: Vec<LiveOrbit> = Vec::new();
        let old_anchor = self.anchor_orbit.clone().unwrap().clone();

        orbits_to_continue.push(old_anchor);
        self.anchor_orbit = None;

        // Try continuing a few of the old candidates as well
        self.candidate_orbits.truncate(3);
        for can in &self.candidate_orbits {
            if let Some(orb) = can.orbit.upgrade() {
                orbits_to_continue.push(orb);
            }
        }
        orbits_to_continue
    }

    pub fn min_size_reached_for_current_viewport(&self, current_camera: &CameraSnapshot) -> bool {
        // Don't split further if the child's tile_size gets too close to current camera scale
        let mut next_child_tile_size_constraint = current_camera.scale().clone();
        next_child_tile_size_constraint *= self.smallest_tile_pixel_span;
        if self.geometry.radius() < &next_child_tile_size_constraint {
             true
        } else { false }
    }

    fn evaluate_orbit(&self, orbit: LiveOrbit) -> TileOrbitScore {
        let orb_g = orbit.read();

        let distance = complex_distance(orb_g.c_ref(), self.geometry.center());
        let needed = distance.clone() + self.geometry.half_diagonal();
        let coverage = orb_g.r_valid().clone().to_f64() / needed.to_f64();

        let depth_score = match orb_g.escape_index() {
            Some(i) => (i as f64) / self.tile_iter as f64,
            None => 1.0, // interior
        };

        let contraction =  orb_g.contraction().clone();
        let stability_score = 1.0 / (1.0 + contraction.abs().to_f64());

        let total_score =
            50.0 * coverage +
                1.0 * depth_score +
                0.5 * stability_score;

        TileOrbitScore {
            coverage, depth_score, stability_score, total_score
        }
    }
}
