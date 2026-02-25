use bytemuck;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniform {
    pub center_x_hi:        f32,
    pub center_x_lo:        f32,
    pub center_y_hi:        f32,
    pub center_y_lo:        f32,
    pub scale_hi:           f32,
    pub scale_lo:           f32,
    pub screen_width:       f32,
    pub screen_height:      f32,
    pub max_iter:           u32,
    pub tile_count:         u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TileFeedbackOut {
    pub min_iter_count:     u32,    // Min iters per pixel, per tile
    pub max_iter_count:     u32,    // Max iters per pixel, per tile 
    
    // --- Flag counts ---
    pub escaped_count:          u32,
    pub pertub_used_count:      u32,
    pub max_iter_reached_count: u32,
}
unsafe impl bytemuck::Pod for TileFeedbackOut {}
unsafe impl bytemuck::Zeroable for TileFeedbackOut {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DebugOut {
    pub center_x_hi:        f32,
    pub center_x_lo:        f32,
    pub center_y_hi:        f32,
    pub center_y_lo:        f32,
    pub scale_hi:           f32,
    pub scale_lo:           f32,
    pub screen_width:       f32,
    pub screen_height:      f32,
    pub tile_count:         u32,
    pub tile_idx:           i32,
}
unsafe impl bytemuck::Pod for DebugOut {}
unsafe impl bytemuck::Zeroable for DebugOut {}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTileGeometry {
    pub anchor_c_ref_re_hi: f32,
    pub anchor_c_ref_re_lo: f32,
    pub anchor_c_ref_im_hi: f32,
    pub anchor_c_ref_im_lo: f32,
    pub r_valid_hi:         f32,
    pub r_valid_lo:         f32,
    pub tile_screen_min_x:  f32,
    pub tile_screen_min_y:  f32,
    pub tile_screen_max_x:  f32,
    pub tile_screen_max_y:  f32,
}
