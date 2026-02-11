use crate::scout_engine::orbit::*;
use crate::scout_engine::tile::*;

use crate::signals::CameraSnapshot;

use log::debug;
use rug::{Float, Complex};

pub fn add_tile_levels(
    tile_levels: TileLevels,
    rug_prec: u32,
    add_levels: u32,
    base_tile_size: f64,
    ideal_tile_pix_width: f64,
) {
    let base_size = Float::with_val(rug_prec, base_tile_size);
    let ideal_tile_pix_width = Float::with_val(rug_prec, ideal_tile_pix_width);

    debug!("Adding {} new Tile Levels. base_size={} ideal_tile_pix_width={}", 
        add_levels, base_tile_size, ideal_tile_pix_width);

    let mut tlvs_g = tile_levels.write();
    let (start_level, stop_level) = if let Some(last) = tlvs_g.last() {
        (last.level + 1, last.level + add_levels + 1)
    } else {
        (0, add_levels)
    };

    for level in start_level..stop_level {
        let tile_level = TileLevel::new(level, &base_size, &ideal_tile_pix_width);
        debug!("*** {:?}", &tile_level);
        tlvs_g.push(tile_level);
    }
}

pub fn find_applicable_tile_levels_for_scale(
    tile_levels: &TileLevels, scale: &Float
) -> Vec<TileLevel> {
    let tlvs_g = tile_levels.read();
    let mut appl_tls = Vec::<TileLevel>::new();
    
    for tl in tlvs_g.iter() {
        if *scale > tl.preferred_scale_min && 
           *scale < tl.preferred_scale_max {
            appl_tls.push(tl.clone());
        }
    }

    // If no levels were found, we've either at the start of the program, or have
    // zoomed past the mimumum scale of the lowest level.
    if appl_tls.is_empty() {
        if let Some(lev_0) = tlvs_g.first() {
            if *scale > lev_0.preferred_scale_max {
                appl_tls.push(lev_0.clone());
            }
        }
    }

    appl_tls
}

// Does a reverse search to find the tile level where the current camera
// scale is just under the level's ideal. 
// IMPORTANT - ensure add_tile_levels() is called at least once at the 
// start of the program.
// Also returns a flag that indicates this is the last/deepest tile level
// which can be used to create new tile levels.
pub fn find_active_tile_level_for_scale(
    tile_levels: &TileLevels, scale: &Float
) -> (TileLevel, bool) {
    let tlvs_g = tile_levels.read();
    let last = tlvs_g.last().unwrap();

    let tl = tlvs_g.iter()
        .rfind(|tl| tl.ideal_scale > *scale)
        .cloned()
        // Should never panic
        .unwrap_or(tlvs_g.first().unwrap().clone());

    if tl.level == last.level {
        (tl, true)
    } else {
        (tl, false)
    }
}

pub fn find_active_tiles_under_camera(
    tile_level: &TileLevel, current_camera: &CameraSnapshot
) -> Vec<TileId> {
    let center = current_camera.center.clone();

    let half_extent = current_camera.scale.clone() * current_camera.screen_extent_multiplier;
    let real_diff = center.real().clone() * &half_extent;
    let imag_diff = center.imag().clone() * &half_extent;

    let mut c_tl = center.clone();
    let (c_tl_real, c_tl_imag) = c_tl.as_mut_real_imag();
    *c_tl_real -= &real_diff;
    *c_tl_imag += &imag_diff;

    let mut c_br = center.clone();
    let (c_br_real, c_br_imag) = c_br.as_mut_real_imag();
    *c_br_real += &real_diff;
    *c_br_imag -= &imag_diff;

    let t_tl = TileId::from_point(&c_tl, tile_level);
    let t_br = TileId::from_point(&c_br, tile_level);

    let (min_x, max_x) = if t_tl.tx <= t_br.tx { (t_tl.tx, t_br.tx) } else { (t_br.tx, t_tl.tx) };
    let (min_y, max_y) = if t_tl.ty <= t_br.ty { (t_tl.ty, t_br.ty) } else { (t_br.ty, t_tl.ty) };

    (min_x..=max_x)
        .flat_map(|tx| (min_y..=max_y)
            .map(move |ty| TileId { tx, ty, level: tile_level.level })
        )
        .collect()
}

pub fn tiles_influenced_by_orbit(
    c_ref: &Complex,
    tile_level: &TileLevel,
    influence_radius: &Float,
) -> Vec<TileId> {
    let base = TileId::from_point(c_ref, tile_level);

    let mut inf_per_rad = influence_radius.clone();
    inf_per_rad /= tile_level.tile_size.clone();
    let tiles_per_radius = inf_per_rad.ceil().to_f64() as i64;

    let num = (2*tiles_per_radius+1).pow(2) as usize;
    let mut out = Vec::with_capacity(num);

    for dx in -tiles_per_radius..=tiles_per_radius {
        for dy in -tiles_per_radius..=tiles_per_radius {
            out.push(TileId {
                tx: base.tx + dx, 
                ty: base.ty + dy,
                level: tile_level.level
            });
        }
    }
    out
}

// Checks if the tile’s bounding disk overlaps the orbit’s influence disk.
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

pub fn upgrade_orbit_list(weak_orbits: &Vec<(f64, WeakOrbit)>) -> Vec<(f64, LiveOrbit)> {
    weak_orbits.iter()
      .filter_map(|(score, orb)| orb.upgrade().map(|o| (*score, o)))
      .collect()
}

pub fn compute_orbit_list_hash(live_orbits: &Vec<(f64, LiveOrbit)>) -> u64 {
    live_orbits
        .iter()
        .enumerate()
        .fold(0u64, |acc, (i, (_, orb))| {
            let id = orb.read().orbit_id as u64;
            acc.wrapping_mul(1315423911).wrapping_add(id ^ (i as u64))
        })
}