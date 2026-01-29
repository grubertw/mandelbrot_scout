use crate::gpu_pipeline::structs::{GpuOrbitSlot, GpuOrbitMeta, OrbitMetaMask, OrbitMetaFlags};
use crate::signals::TileOrbitViewDf;

pub const MAX_ORBITS_PER_FRAME: u32 = 64;
pub const MAX_LIVE_ORBITS: u32 = 500;
pub const MAX_REF_ORBIT: u32 = 8192;
pub const ROWS_PER_ORBIT: u32 = 4;
pub const INIT_RUG_PRECISION: u32 = 128;
pub const K: f32 = 0.25; // Multiplied by scale, and often used as a radius of validity test
pub const PERTURB_THRESHOLD: f32 = 1e-5;
pub const DEEP_ZOOM_THRESHOLD: f64 = 1e-8;
pub const SHORT_THRESHOLD: u32 = 32;
pub const MIN_PERTURB_ITERS: u32 = 32; // or 64 later
pub const SCREEN_TILE_SIZE: f64 = 128.0;
pub const MAX_SCREEN_WIDTH:  u32 = 3840; // Support for a 4k display
pub const MAX_SCREEN_HEIGHT: u32 = 2160; // Support for a 4k display
pub const MAX_SCREEN_TILES_X: u32 = 256; // Support for a 4k display
pub const MAX_SCREEN_TILES_Y: u32 = 160; // Support for a 4k display
pub const NO_ORBIT: u32 = 0xFFFF_FFFF;   // Sentinel: means "no perturbation"

use rug::{Float, Complex};
use log::debug;

#[derive(Debug, Clone)]
pub struct PixelToComplexMapper {
    pub width: f64,
    pub height: f64,
    pub center: Complex,
    pub pix_dx: Float,
    pub pix_dy: Float,
}

impl PixelToComplexMapper {
    pub fn new(width: f64, height: f64, center: Complex, scale: Float) -> Self {
        Self {
            width, height, center: center.clone(),
            pix_dx: scale.clone() / width,
            pix_dy: scale.clone() / height
        }
    }

    pub fn pixel_to_complex(&self, px: f64, py: f64) -> Complex {
        let off_x = self.pix_dx.clone() * (px - (self.width / 2.0));
        let off_y = self.pix_dy.clone() * (py - (self.height / 2.0));

        let mut c = self.center.clone();
        let (c_re, c_im) = c.as_mut_real_imag();
        *c_re += off_x;
        *c_im += off_y;

        c
    }
}

pub fn select_best_orbit_slot(mapper: &PixelToComplexMapper, 
    orbit_slots: &Vec<GpuOrbitSlot>, complex_tiles: Vec<TileOrbitViewDf>,
    px0: f64, py0: f64, px1: f64, py1: f64,
) -> u32 {
    if orbit_slots.is_empty() || complex_tiles.is_empty() {
        return NO_ORBIT;
    }

    // Screen-tile center in complex space
    let cx = (px0 + px1) * 0.5;
    let cy = (py0 + py1) * 0.5;
    let c_center = mapper.pixel_to_complex(cx, cy);

    let mut best_score = f64::INFINITY;
    let mut best_slot = NO_ORBIT;

    for tile in complex_tiles {
        for orb in tile.orbits {
            // Find GPU slot
            let slot_idx = if let Some(i) = find_slot_index(orbit_slots, orb.orbit_id) {
                i
            } else { 
                continue;
            };

            // Policy: must be usable
            let meta = &orbit_slots[slot_idx as usize].meta;
            if !meta.is_usable() {
                continue;
            }

            // Distance from screen-tile center
            let dr = (orb.c_ref.re.hi as f64) - c_center.real().to_f64();
            let di = (orb.c_ref.im.hi as f64) - c_center.imag().to_f64();
            let dist2 = dr * dr + di * di;

            // Prefer deeper perturbation validity
            let depth_bonus = -(orb.max_valid_perturb_index as f64);
            
            // Simple weighted score
            let score = dist2 * 1.0 + depth_bonus * 0.01;

            if score < best_score {
                best_score = score;
                best_slot = slot_idx;
            }
        }
    }
    
    best_slot
}

pub fn build_gpu_orbit_slots_from_tiles(tiles: Vec<TileOrbitViewDf>, scale: f64) -> Vec<GpuOrbitSlot> {
    let max_orb_count = MAX_ORBITS_PER_FRAME as usize;
    let mut gpu_orbits = Vec::<GpuOrbitSlot>::new();
    let mut slot_num: u32 = 0;

    for tile in tiles {
        for orb in &tile.orbits {
            let mut flags = OrbitMetaMask::none();

            match orb.escape_index {
                Some(esc_idx) => {
                    flags.set(OrbitMetaFlags::OrbitEscapes);

                    if esc_idx < SHORT_THRESHOLD {
                        flags.set(OrbitMetaFlags::OrbitShort);
                    } 
                }
                None => {
                    flags.set(OrbitMetaFlags::OrbitInterior);
                }
            }
            // Orbit stability is primarily driven by how well perturbation iteration goes,
            // and for ScoutEngine, it starts from a maximum (i.e. MAX_REF_ORBIT), but may
            // be reduced as the reference orbit gets re-used and min_last_valid_i is taken
            // into account.
            if orb.max_valid_perturb_index >= MIN_PERTURB_ITERS {
                flags.set(OrbitMetaFlags::OrbitUsable);
            }

            let meta = GpuOrbitMeta::new(
                orb.max_valid_perturb_index,
                orb.escape_index,
                flags
            );
            debug!("Built GpuOrbitSlot {} for {:?}\twith orbit_id={}\tand meta={:?}", 
                slot_num, &tile.tile, &orb.orbit_id, &meta);

            gpu_orbits.push(GpuOrbitSlot {
                tile: tile.tile.clone(), 
                orbit: orb.clone(),
                meta
            });
            
            if gpu_orbits.len() == max_orb_count {
                break;
            }
            slot_num += 1;
        }
        if gpu_orbits.len() == max_orb_count {
            break;
        }
    }
    gpu_orbits
}

fn find_slot_index(
    orbit_slots: &Vec<GpuOrbitSlot>,
    orbit_id: u64,
) -> Option<u32> {
    orbit_slots
        .iter()
        .position(|s| s.orbit.orbit_id == orbit_id)
        .map(|i| i as u32)
}


