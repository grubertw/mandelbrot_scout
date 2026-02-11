use crate::scout_engine::orbit::*;
use crate::signals::{FrameStamp};

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

use rug::{Float, Complex};

pub type TileLevels = Arc<RwLock<Vec<TileLevel>>>;
pub type TileView = Arc<RwLock<TileOrbitView>>;
pub type TileRegistry = Arc<RwLock<HashMap<TileId, TileView>>>;

const INIT_INFLUENCE_RADIUS_FAC: f64 = 1.0;
const INIT_MAX_ORBITS_PER_TILE: usize = 3;
pub const STARTING_LOCAL_SCORE: f64 = 100.0; // A 'bad' score, but not the worst!

#[derive(Clone, Debug)]
pub struct TileLevel {
    /// Logical level index (0 = coarse, 1 = fine, etc.)
    pub level: u32,
    /// Size of one tile in complex-plane units
    /// (e.g. 0.25, 2.5e-6, ...)
    pub tile_size: Float,
    /// Influence that an orbit in the "resident" tile has on surrounding
    /// tiles in the grid. 1=no influence, 1.2=includes the cross, 
    /// 1.5=inclueds all 8 surrounding tiles
    pub influence_radius_factor: f64, // between 1 & 2 (usually 1.5)
    /// The ideal scale the camera should be to fit the ideal #'of pixels
    /// for a tile level.
    pub ideal_scale: Float,
    /// Preferred minimum scale threshold
    pub preferred_scale_min: Float,
    /// preferred maximum scale threshold
    pub preferred_scale_max: Float,
}

impl TileLevel {
    // Good starting base is 1e-1, with ideal pixel width of 32.
    // This maps approximately to the starting conditions for level 0 
    // to begin at a viewport scale that contains the entire mandelbrot
    // from -2 to 1. (i.e. vieport scale ~3). Scout engine deals with pixel
    // widths however, and hence why the tile_size is devided by pixels 
    // rather than multiplied.
    pub fn new(
        level: u32, base_tile_size: &Float, 
        ideal_tile_pix_width: &Float,
    ) -> Self {
        let tile_size = base_tile_size.clone() / (2.0_f64).powi(level as i32);
        let ideal_scale = tile_size.clone() / ideal_tile_pix_width;
        let preferred_scale_min = ideal_scale.clone() / (10.0_f64).sqrt();
        let preferred_scale_max = ideal_scale.clone() * (10.0_f64).sqrt();

        Self {
            level, tile_size, 
            influence_radius_factor: INIT_INFLUENCE_RADIUS_FAC,
            ideal_scale,
            preferred_scale_min, preferred_scale_max,
        }
    }
}

// TileId's are our way to create a deterministic 'integer lattice' atop
// the complex co-orditate space. Think graph paper. And like graph paper, 
// it remains stationary, and the viewport camera behaves like a 'lens' or
// magnifying glass. Tile levels come into play as we zoom, offering a 
// new set of grid-lines for us to define boundires as scale changes - mainly
// with an incrementing level number as we decrease our scale and zoom into
// the fractal. 
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct TileId {
    pub tx: i64, // floor(re / tile_size)
    pub ty: i64, // floor(im / tile_size)
    pub level: u32 // level of the integer lattice constraints above
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
        Self { tx, ty, level: level.level }
    }
}

#[derive(Clone, Debug)]
pub struct TileGeometry {
    /// Center of the tile in the complex plane
    pub center: Complex,
    // C-Tiles are always square 
    pub radius: Float,
}

impl TileGeometry {
    pub fn from_id(tile_id: &TileId, tile_size: &Float) -> Self {
        let mut center_re = tile_size.clone() * tile_id.tx;
        let mut center_im = tile_size.clone() * tile_id.ty;

        center_re += tile_size.clone() / 2;
        center_im += tile_size.clone() / 2;

        Self {
            center: Complex::with_val(
                tile_size.prec(),
                (center_re, center_im)
            ),
            radius: tile_size.clone() / 2,
        }
    }

    pub fn center(&self) -> &Complex {
        &self.center
    }

    pub fn half_diagonal(&self) -> Float {
        let mut rad = self.radius.clone();
        rad *= &self.radius;
        rad *= 2;
        Float::sqrt(rad)
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
    pub tile: TileId,
    /// Geometry of this tile
    pub geometry: TileGeometry,
    /// Per-orbit local statistics
    pub orbit_stats: HashMap<OrbitId, TileOrbitStats>,
    /// Weak refs into the global LivingOrbits pool
    pub weak_orbits: Vec<(f64, WeakOrbit)>,
    /// Tile-level aggregate signal
    pub local_score: f64,
    /// Number of orbits to create per spawn operation
    pub num_orbits_per_spawn: u32,
    /// Adjustible capicity of orbits for this tile
    pub curr_max_orbits: usize,
    /// Timestamp for decay / aging heuristics
    pub last_updated: FrameStamp
}

impl TileOrbitView {
    pub fn new(tile_id: &TileId, tile_size: &Float, 
               num_orbits_per_spawn: u32, curr_max_orbits: usize,
               frame_st: FrameStamp) -> Self {
        Self {
            tile: tile_id.clone(), 
            geometry: TileGeometry::from_id(
                tile_id, tile_size),
            orbit_stats: HashMap::new(),
            weak_orbits: Vec::new(),
            local_score: STARTING_LOCAL_SCORE,
            num_orbits_per_spawn, curr_max_orbits,
            last_updated: frame_st,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TileOrbitStats {
    pub orbit_id: OrbitId,

    // Orbit is native/falls within the geometry of this tile.
    pub is_native: bool, 

    // --- Stability envelope ---
    pub min_last_valid_i: u32,
    pub max_last_valid_i: u32,

    // --- Running counters (decayed or windowed) ---
    pub perturb_attempted: u64,
    pub perturb_valid: u64,
    pub perturb_collapsed: u64,
    pub perturb_escaped: u64,

    pub absolute_fallback: u64,
    pub absolute_escaped: u64,

    // --- derived ---
    pub cached_score: f64,

    // --- Temporal ---
    pub last_updated: FrameStamp,
}

impl TileOrbitStats {
    pub fn new(
        orbit_id: OrbitId, is_native: bool, min_last_valid_i: u32,
        frame_st: FrameStamp
    ) -> Self {
        Self {
            orbit_id, is_native, min_last_valid_i,
            max_last_valid_i: 0,
            perturb_attempted: 0,
            perturb_valid: 0,
            perturb_collapsed: 0,
            perturb_escaped: 0,
            absolute_fallback: 0,
            absolute_escaped: 0,
            cached_score: STARTING_LOCAL_SCORE,
            last_updated: frame_st
        }
    }
}