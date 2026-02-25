use crate::scout_engine::tile::*;

use crate::signals::CameraSnapshot;

use rug::{Float, Complex, Assign};

// I.e. distance as a vector
pub fn complex_delta(a: &Complex, b: &Complex) -> Complex {
    // (a.real - b.real, a.imag - b.imag)
    let real = a.real().clone() - b.real();
    let imag = a.imag().clone() - b.imag();
    Complex::with_val(
        a.prec(), 
        (real, imag)
    )
}

// I.e. distance as a scalar 
pub fn complex_distance(a: &Complex, b: &Complex) -> Float {
    let delta = complex_delta(a, b);
    let sum = delta.real().clone() * delta.real() + delta.imag() * delta.imag();
    sum.sqrt()
}

pub fn find_tile_ids_under_camera(
    current_camera: &CameraSnapshot,
    level: &TileLevel,
) -> Vec<TileId> {
    let center = current_camera.center().clone();
    let half_extent = current_camera.half_extent();

    let mut c_tl = center.clone();
    let (c_tl_real, c_tl_imag) = c_tl.as_mut_real_imag();
    *c_tl_real -= half_extent;
    *c_tl_imag += half_extent;

    let mut c_br = center.clone();
    let (c_br_real, c_br_imag) = c_br.as_mut_real_imag();
    *c_br_real += half_extent;
    *c_br_imag -= half_extent;

    let t_tl = TileId::from_point(&c_tl, level);
    let t_br = TileId::from_point(&c_br, level);

    let (min_x, max_x) = if t_tl.tx <= t_br.tx { (t_tl.tx, t_br.tx) } else { (t_br.tx, t_tl.tx) };
    let (min_y, max_y) = if t_tl.ty <= t_br.ty { (t_tl.ty, t_br.ty) } else { (t_br.ty, t_tl.ty) };

    let mut num_tiles = half_extent.clone();
    num_tiles *= 2.0;
    num_tiles /= &level.tile_size;
    
    (min_x..=max_x)
        .flat_map(|tx| (min_y..=max_y)
            .map(move |ty| TileId { tx, ty })
        )
        .collect()
}

pub fn find_tiles_in_registry(
    tile_registry: TileRegistry,
    tile_ids: &[TileId],
) -> Vec<TileView> {
    let tile_reg_g = tile_registry.read();

    tile_ids
        .iter()
        .filter_map(|tile_id| tile_reg_g.get(tile_id))
        .cloned()
        .collect()
}

/// Use real complex geometry to test for tile/viewport overlap
/// Note, this implementation only works when both the camera and tile are square
/// Also note, the user window itself does not need to be a square, as the Scene
/// takes MAX of width/height
fn tile_overlaps_camera(tile: &TileGeometry, cam: &CameraSnapshot) -> bool {
    let tile_c = tile.center();
    let cam_c = cam.center();

    let tile_hw = tile.radius(); // half-width, not half-diagonal!

    // x and y projections (real and imag parts)
    let mut dx = tile_c.real().clone() - cam_c.real();
    dx = dx.clone().abs();    

    let mut dy = tile_c.imag().clone() - cam_c.imag();
    dy = dy.clone().abs();

    let sum_hw_x = tile_hw.clone() + cam.half_extent();
    let sum_hw_y = tile_hw.clone() + cam.half_extent();

    // Overlap occurs if on both axes, centers are less than or equal to sum of half-widths
    dx <= sum_hw_x && dy <= sum_hw_y
}

pub fn find_anchor_orbits(
    tiles: &[TileView],
    current_camera: &CameraSnapshot,
    collected_orbits: &mut Vec<TileView>
) {
    for tile in tiles {
        let tile_g = tile.read();
        // Only collect anchors if the tile overlaps the camera
        let tile_in_viewport = tile_overlaps_camera(&tile_g.geometry, current_camera);

        if tile_g.anchor_orbit.is_some() && tile_in_viewport {
            collected_orbits.push(tile.clone());
        }

        find_anchor_orbits(&tile_g.children, current_camera, collected_orbits);
    }
}

pub fn find_tiles_needing_anchors(
    tiles: &[TileView], 
    current_camera: &CameraSnapshot,
    tiles_to_seed: &mut Vec<TileView>
) {
    for tile in tiles {
        let tile_g = tile.read();
        // Only spawn anchors for tiles that have a center within viewport
        let tile_in_viewport = tile_overlaps_camera(&tile_g.geometry, current_camera);

        if tile_g.anchor_orbit.is_none() && tile_g.num_orbits_per_spawn > 0 && tile_in_viewport {
            tiles_to_seed.push(tile.clone());
        }

        find_tiles_needing_anchors(&tile_g.children, current_camera, tiles_to_seed);
    }
}

pub fn find_smallest_tile_size(
    tiles: &[TileView],
    smallest_tile_size: &mut Float,
) {
    for tile in tiles {
        let tile_g = tile.read();
        if tile_g.level.tile_size.to_f64() < smallest_tile_size.to_f64() {
            smallest_tile_size.assign(&tile_g.level.tile_size);
        }

        find_smallest_tile_size(&tile_g.children, smallest_tile_size);
    }
}

pub fn find_tile_with_best_coverage(
    tiles: &[TileView],
    best_tile: &mut (f64, TileView),
) {
    for tile in tiles {
        let tile_g = tile.read();
        if tile_g.anchor_orbit.is_none()
            && let Some(can) = tile_g.candidate_orbits.first() {

             if can.score.coverage > best_tile.0 {
                 *best_tile = (can.score.coverage, tile.clone());
             }
        }

        find_tile_with_best_coverage(&tile_g.children, best_tile);
    }
}

pub fn find_tile_with_worst_coverage(
    tiles: &[TileView],
    worst_tile: &mut (f64, TileView),
) {
    for tile in tiles {
        let tile_g = tile.read();
        if tile_g.anchor_orbit.is_none()
            && let Some(can) = tile_g.candidate_orbits.first() {

            if can.score.coverage < worst_tile.0 {
                *worst_tile = (can.score.coverage, tile.clone());
            }
        }

        find_tile_with_worst_coverage(&tile_g.children, worst_tile);
    }
}

pub fn count_tiles_that_reached_min_size_for_current_viewport_and_should_split(
    tiles: &[TileView],
    current_camera: &CameraSnapshot,
) -> usize {
    let mut min_size_count = 0;
    for tile in tiles {
        let mut tile_g = tile.write();
        if tile_g.min_size_reached_for_current_viewport(&current_camera) && tile_g.should_split() {
            min_size_count += 1;
        }
        
        min_size_count += count_tiles_that_reached_min_size_for_current_viewport_and_should_split(
            &tile_g.children, current_camera);
    }
    min_size_count
}
