use crate::scout_engine::tile::*;

use crate::signals::{CameraSnapshot, GpuGridSample};

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

pub fn grid_samples_in_tile(
    tile: TileView,
    samples: &[GpuGridSample]
) -> Vec<GpuGridSample> {
    let tile_g = tile.read();
    let tile_c_re = tile_g.geometry.center().real();
    let tile_c_imag = tile_g.geometry.center().imag();
    let tile_r = tile_g.geometry.radius();

    samples
        .iter()
        .filter_map(|sample| {
            let mut dx = tile_c_re.clone() - sample.best_sample.real();
            dx = dx.clone().abs();

            let mut dy = tile_c_imag.clone() - sample.best_sample.imag();
            dy = dy.clone().abs();

            if dx <= *tile_r && dy <= *tile_r {
                Some(sample)
            }
            else {None}
        })
        .cloned()
        .collect()
}
