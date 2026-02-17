use crate::scout_engine::orbit::*;
use crate::scout_engine::utils::*;

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

use rug::{Float, Complex};
use log::{trace, debug, info};

pub type TileLevels = Arc<RwLock<Vec<TileLevel>>>;
pub type TileView = Arc<RwLock<TileOrbitView>>;
pub type TileRegistry = Arc<RwLock<HashMap<TileId, TileView>>>;

pub const BASE_TILE_SIZE: f64 = 1e-1;

const TILE_LEVEL_ADDITION_INCREMENT: u32 = 9;

#[derive(Clone, Debug)]
pub struct TileLevel {
    /// Logical level index (0 = coarse, 1 = fine, etc.)
    pub level: u32,
    /// Size of one tile in complex-plane units
    /// (e.g. 0.25, 2.5e-6, ...)
    pub tile_size: Float,
}

impl TileLevel {
    pub fn new(
        level: u32, rug_precision: u32, 
    ) -> Self {
        let base_size = Float::with_val(rug_precision, BASE_TILE_SIZE);
        let tile_size = base_size.clone() / (2.0_f64).powi(level as i32);

        Self {
            level, tile_size, 
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

        Self {
            center: Complex::with_val(
                tile_size.prec(),
                (center_re, center_im)
            ),
            radius, half_diagonal
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
    pub failed_orbits: Vec<WeakOrbit>,
    /// Number of orbits to create per spawn operation
    pub num_orbits_per_spawn: u32,
    /// Maximum number of times a 'ligitimate' attempt was made to anchor
    /// a tile and resulted in failure
    pub max_tile_anchor_failure_attempts: u32,
}

impl TileOrbitView {
    pub fn new(
        tile_id: &TileId, level: &TileLevel, 
        num_orbits_per_spawn: u32,
        max_tile_anchor_failure_attempts: u32,
    ) -> Self {
        Self {
            id: tile_id.clone(), 
            level: level.clone(),
            geometry: TileGeometry::from_id(
                tile_id, &level.tile_size),
            anchor_orbit: None,
            children: Vec::new(),
            failed_orbits: Vec::new(),
            num_orbits_per_spawn,
            max_tile_anchor_failure_attempts,
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
        let mut orb_g = orbit.write();
        if seeded_from_tile {
            trace!("{:>4?} spawned orbit {:>4} c_ref={:?} with quality metrics: \n\t{:?}",
                self.id, orb_g.orbit_id, orb_g.c_ref(), orb_g.qualiy_metrics);
        }
        if orb_g.is_interior() {
            let distance = complex_distance(orb_g.c_ref(), self.geometry.center());
            let needed = distance.clone() + self.geometry.half_diagonal();
            if needed <= *orb_g.r_valid() {
                self.anchor_orbit = Some(orbit.clone());
                orb_g.delta_from_tile_center = Some(
                    complex_delta(orb_g.c_ref(), self.geometry.center())
                );

                info!("{:?} {:?} promoting orbit {} c_ref={:?} with r_valid={:?} as an anchor! tile.center={:?} delta_from_tile_center={:?}", 
                    self.id, self.level, orb_g.orbit_id, orb_g.c_ref(), orb_g.r_valid(), 
                    self.geometry.center(), orb_g.delta_from_tile_center.clone().unwrap());
            } else if seeded_from_tile {
                self.failed_orbits.push(Arc::downgrade(&orbit));

                debug!("{:?} {:?} failed orbit {} with r_valid={:?}. Needed={:?} failed_orbits_len={}",
                    self.id, self.level, orb_g.orbit_id, orb_g.r_valid(), needed, self.failed_orbits.len());
            }
        }

        self.anchor_orbit.is_some()
    }

    pub fn should_split(&self) -> bool {
           self.failed_orbits.len() >= self.max_tile_anchor_failure_attempts as usize
        && self.children.len() == 0 && self.anchor_orbit.is_none()
    }

    pub fn split(&mut self) {
        let child_level = TileLevel::new(
            self.level.level + 1, self.geometry.radius().prec()
        );
        // Deterministic creation of TileId(s) from parent!
        let child_tx = self.id.tx * 2;
        let child_ty = self.id.ty * 2;
        let child_ids = vec![
            TileId { tx: child_tx,     ty: child_ty     },
            TileId { tx: child_tx + 1, ty: child_ty     },
            TileId { tx: child_tx,     ty: child_ty + 1 },
            TileId { tx: child_tx + 1, ty: child_ty + 1 },
        ];

        debug!("Splitting {:?} into {:?} @ level={:?}", 
            self.id, child_ids, child_level);

        let new_tiles: Vec<TileOrbitView> = child_ids
            .iter()
            .map(|tile_id| TileOrbitView::new(
                tile_id, &child_level, 
                self.num_orbits_per_spawn, 
                self.max_tile_anchor_failure_attempts
            ))
            .collect();
        
        let mut orbits_anchored_count = 0;

        for mut child in new_tiles {
            // Check if any failed orbits of parent fit around children
            let parent_orbits: Vec<LiveOrbit> = self.failed_orbits
                .iter()
                .filter_map(|w| w.upgrade())
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
    }
}
