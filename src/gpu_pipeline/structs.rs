#![allow(unexpected_cfgs)]

use crate::signals::ReferenceOrbitDf;
use crate::scout_engine::tile::TileId;

use bytemuck;
use bitmask::bitmask;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniform {
    pub center_x_hi:        f32,
    pub center_x_lo:        f32,
    pub center_y_hi:        f32,
    pub center_y_lo:        f32,
    pub scale_hi:           f32,
    pub scale_lo:           f32,
    pub pix_dx_hi:          f32, 
    pub pix_dx_lo:          f32,
    pub pix_dy_hi:          f32, 
    pub pix_dy_lo:          f32,
    pub screen_width:       f32,
    pub screen_height:      f32,
    pub screen_tile_size:   f32,
    pub max_iter:           u32,
    pub ref_orb_len:        u32,
    pub ref_orb_count:      u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FrameFeedbackOut {
    pub max_lambda_re_hi:   f32,
    pub max_lambda_re_lo:   f32,
    pub max_lambda_im_hi:   f32,
    pub max_lambda_im_lo:   f32,
    pub max_delta_z_re_hi:  f32,
    pub max_delta_z_re_lo:  f32,
    pub max_delta_z_im_hi:  f32,
    pub max_delta_z_im_lo:  f32,
    pub escape_ratio:       f32,
}
unsafe impl bytemuck::Pod for FrameFeedbackOut {}
unsafe impl bytemuck::Zeroable for FrameFeedbackOut {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OrbitFeedbackOut {
    // --- Perturbation stability ---
    pub min_last_valid_i: u32,         // Worst-case perturbation validity
    pub max_last_valid_i: u32,         // Best-case     
    
    // --- Perturbation Flag counts ---
    pub perturb_attempted_count: u32,
    pub perturb_valid_count: u32,
    pub perturb_collapsed_count: u32,
    pub perturb_escaped_count: u32,
    pub max_iter_reached_count: u32,
    pub absolute_fallback_count: u32,
    pub absolute_escaped_count: u32,
}
unsafe impl bytemuck::Pod for OrbitFeedbackOut {}
unsafe impl bytemuck::Zeroable for OrbitFeedbackOut {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DebugOut {
    pub c_ref_re_hi: f32,
    pub c_ref_re_lo: f32,
    pub c_ref_im_hi: f32,
    pub c_ref_im_lo: f32,
    pub delta_c_re_hi: f32,
    pub delta_c_re_lo: f32,
    pub delta_c_im_hi: f32,
    pub delta_c_im_lo: f32,
    pub orbit_idx:     u32,
    pub orbit_meta_ref_len: u32,
    pub perturb_escape_seq: u32,
    pub last_valid_i: u32,
    pub abs_i: u32,
    pub last_valid_z_re_hi: f32,
    pub last_valid_z_re_lo: f32,
    pub last_valid_z_im_hi: f32,
    pub last_valid_z_im_lo: f32,
}
unsafe impl bytemuck::Pod for DebugOut {}
unsafe impl bytemuck::Zeroable for DebugOut {}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuOrbitMeta {
    pub ref_len: u32,
    pub escape_index: u32,
    pub flags: u32,
    pub pad: u32,
}

bitmask! {
    pub mask OrbitMetaMask: u32 where flags OrbitMetaFlags {
        OrbitEscapes    = 0b00000001,
        OrbitInterior   = 0b00000010,
        OrbitShort      = 0b00000100,
        OrbitUnstable   = 0b00001000,
        OrbitUsable     = 0b00010000
    }
}

impl GpuOrbitMeta {
    pub fn new(ref_len: u32, escape_index: Option<u32>, orbit_meta_mask: OrbitMetaMask) -> Self {
        Self {
            ref_len,
            escape_index: escape_index.unwrap_or(u32::MAX),
            flags: *orbit_meta_mask,
            pad: 0
        }
    }

    pub fn is_usable(&self) -> bool {
        self.flags & *OrbitMetaFlags::OrbitUsable > 0
    }
}

#[derive(Clone, Debug)]
pub struct GpuOrbitSlot {
    pub tile: TileId,
    pub orbit: ReferenceOrbitDf,
    pub meta: GpuOrbitMeta,
}