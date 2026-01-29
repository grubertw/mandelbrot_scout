use crate::scout_engine::orbit::*;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time;

use rug::{Float, Complex};

pub type TileRegistry = Arc<Mutex<HashMap<(u32, TileId), TileOrbitView>>>;

#[derive(Clone, Debug)]
pub struct TileLevel {
    /// Logical level index (0 = coarse, 1 = fine, etc.)
    pub level: u32,
    /// Size of one tile in complex-plane units
    /// (e.g. 0.25, 2.5e-6, ...)
    pub tile_size: Float,
    pub influence_radius_factor: f64, // between 1 & 2 
    /// Optional soft limit on how many orbits a tile should keep
    pub max_orbits_per_tile: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct TileId {
    pub tx: i64, // floor(re / tile_size)
    pub ty: i64, // floor(im / tile_size)
}

impl TileId {
    pub fn from_point(c: &Complex, tile_size: &Float) -> Self {
        let mut re = c.real().clone();
        let mut im = c.imag().clone();
        re /= tile_size;
        im /= tile_size;

        let tx = re.floor().to_f64() as i64;
        let ty = im.floor().to_f64() as i64;
        Self { tx, ty }
    }
}

#[derive(Clone, Debug)]
pub struct TileGeometry {
    /// Center of the tile in the complex plane
    pub center: Complex,
    pub radius: Float,
    pub level: u32,
}

impl TileGeometry {
    pub fn from_id(tile_id: &TileId, tile_size: &Float, level: u32) -> Self {
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
            level,
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

#[derive(Clone, Debug)]
pub struct TileOrbitView {
    /// Tile identity (stable)
    pub tile: TileId,
    /// Geometry of this tile
    pub geometry: TileGeometry,
    /// Weak refs into the global LivingOrbits pool
    pub weak_orbits: Vec<WeakOrbit>,
    pub local_score: f64,
    pub max_orbits_per_tile: usize,
    /// Timestamp for decay / aging heuristics
    pub last_updated: time::Instant
}

impl TileOrbitView {
    pub fn new(tile_id: TileId, level: &TileLevel) -> Self {
        Self {
            tile: tile_id.clone(), 
            geometry: TileGeometry::from_id(&tile_id, &level.tile_size, level.level),
            weak_orbits: Vec::<WeakOrbit>::new(),
            local_score: 0.0,
            max_orbits_per_tile: level.max_orbits_per_tile,
            last_updated: time::Instant::now(),
        }
    }
}
