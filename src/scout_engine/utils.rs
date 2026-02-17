use crate::scout_engine::orbit::*;
use crate::scout_engine::tile::*;

use crate::signals::CameraSnapshot;

use rug::{Float, Complex};

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
) -> Vec<TileId> {
    // Search for tile views always starts at level-0, where the grid is 'even'
    // i.e. there are always a uniform number of l-0 tiles accorss the complex plane.
    let level_zero = TileLevel::new(0, current_camera.scale().prec());

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

    let t_tl = TileId::from_point(&c_tl, &level_zero);
    let t_br = TileId::from_point(&c_br, &level_zero);

    let (min_x, max_x) = if t_tl.tx <= t_br.tx { (t_tl.tx, t_br.tx) } else { (t_br.tx, t_tl.tx) };
    let (min_y, max_y) = if t_tl.ty <= t_br.ty { (t_tl.ty, t_br.ty) } else { (t_br.ty, t_tl.ty) };

    (min_x..=max_x)
        .flat_map(|tx| (min_y..=max_y)
            .map(move |ty| TileId { tx, ty })
        )
        .collect()
}

pub fn find_top_tiles_under_camera(
    current_camera: &CameraSnapshot,
    tile_registry: TileRegistry,
) -> Vec<TileView> {
    let tile_ids = find_tile_ids_under_camera(current_camera);

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

pub fn find_anchor_orbits_in_tile_tree(
    tile: TileView, 
    current_camera: &CameraSnapshot,
    collected_orbits: &mut Vec<(TileId, TileGeometry, LiveOrbit)>
) {
    let tile_g = tile.read();
    // Only collect anchors if the tile overlaps the camera
    let tile_in_viewport = tile_overlaps_camera(&tile_g.geometry, current_camera);

    if let Some(orb) = &tile_g.anchor_orbit && tile_in_viewport {
        collected_orbits.push(
            (tile_g.id.clone(), tile_g.geometry.clone(), orb.clone())
        );
    }

    for child in &tile_g.children {
        find_anchor_orbits_in_tile_tree(child.clone(), current_camera, collected_orbits);
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