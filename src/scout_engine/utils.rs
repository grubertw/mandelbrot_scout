use crate::scout_engine::orbit::*;
use crate::scout_engine::tile::*;

use rug::{Float, Complex};

pub fn tiles_influenced_by_orbit(
    c_ref: &Complex,
    tile_size: &Float,
    influence_radius: &Float,
) -> Vec<TileId> {
    let base = TileId::from_point(c_ref, tile_size);

    let mut inf_per_rad = influence_radius.clone();
    inf_per_rad /= tile_size;
    let tiles_per_radius = inf_per_rad.ceil().to_f64() as i64;

    let mut out = Vec::<TileId>::new();

    for dx in -tiles_per_radius..=tiles_per_radius {
        for dy in -tiles_per_radius..=tiles_per_radius {
            out.push(TileId {
                tx: base.tx + dx, 
                ty: base.ty + dy,
            });
        }
    }
    //debug!("Tiles Influnced by Orbit c_ref={:?} and tile_size={:?} and influence_radius={:?} tiles_per_radius={} len={}\n\t\t\t----> {:?}",
    //    c_ref, tile_size, influence_radius, tiles_per_radius, out.len(), out);
    out
}

pub fn tile_overlaps_orbit(
    tile_geom: &TileGeometry,
    c_ref: &Complex,
    influence_radius: &Float,
) -> bool {
    let dx = tile_geom.center().real().clone() - c_ref.real();
    let dy = tile_geom.center().imag().clone() - c_ref.imag();

    let dx2 = dx.clone() * &dx;
    let dy2 = dy.clone() * &dy;

    let dist = Float::sqrt(dx2 + dy2);
    let threshold = influence_radius + tile_geom.half_diagonal();

    dist <= threshold
}

pub fn score_tile_view_orbit(view_orbit: LiveOrbit) -> i64 {
    let orb = view_orbit.lock().unwrap();
    (  orb.weights.w_depth 
     + orb.weights.w_age) as i64
}
