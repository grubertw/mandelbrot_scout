use crate::scout_engine::orbit::*;

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

use rug::{Float, Complex};

type TileView = Arc<RwLock<TileOrbitView>>;
type TileRegistry = Arc<RwLock<HashMap<TileId, TileView>>>;

// Keep it super simple!
const BASE_TILE_SIZE: f64 = 1.0;

#[derive(Clone, Debug)]
pub struct TileLevel {
    /// Logical level index (0 = coarse, 1 = fine, etc.)
    pub level: u32,
    /// Size of one tile in complex-plane units
    /// (e.g. 0.25, 2.5e-6, ...)
    pub tile_size: Float,
}

impl TileLevel {
    pub fn new(level: u32, rug_precision: u32) -> Self {
        let base_size = Float::with_val(rug_precision, BASE_TILE_SIZE);
        let tile_size = base_size.clone() / (2.0_f64).powi(level as i32);

        Self { level, tile_size }
    }
}

// TileId's are our way to create a deterministic 'integer lattice' atop
// the complex co-ordinate space. Think graph paper. And like graph paper,
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
    // very handy for determining bounding boxes, when given two diagonal
    // C values, as they will intrinsically include extra area, so as not
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
    // Cache half-diagonal, as it's used heavily
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
            radius, half_diagonal,
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


#[derive(Clone, Debug)]
pub struct TileOrbitView {
    /// Tile identity (stable)
    pub id: TileId,
    /// Tile's level on the descending 1/2^lv heirarchy 
    pub level: TileLevel,
    /// Geometry of this tile
    pub geometry: TileGeometry,
    /// Our orbit anchor, who's r_valid fills the square bounds
    /// i.e. the half-diagonal of the tile's geometry
    pub anchor_orbit: Option<LiveOrbit>,
}

impl TileOrbitView {
    pub fn new(
        tile_id: &TileId, level: &TileLevel,
    ) -> Self {
        Self {
            id: tile_id.clone(), 
            level: level.clone(),
            geometry: TileGeometry::from_id(
                tile_id, &level.tile_size),
            anchor_orbit: None,
        }
    }
}