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
    pub max_iter:           u32,
    pub ref_orb_count:      u32,
    pub screen_width:       u32,
    pub screen_height:      u32,
    pub grid_size:          u32,
    pub grid_width:         u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridFeedbackOut {
    pub best_pixel_x:           i32, // Pixel location of deepest iteration in the sample grid
    pub best_pixel_y:           i32,
    pub best_pixel_flags:       u32, // Iteration feedback flags for the location
    pub max_iter_count:         u32,
}

pub const ORBIT_ITERS_MASK: u32         = 0x0000FFFF;
pub const ORBIT_ESCAPED: u32            = 1 << 16;
pub const ORBIT_PERTURBED: u32          = 1 << 17;
pub const ORBIT_PERTURB_ERR: u32        = 1 << 18;
pub const ORBIT_MAX_ITER_REACHED: u32   = 1 << 19;
pub const ORBIT_SHIFT: u32              = 20;

impl GridFeedbackOut {
    pub fn iter(&self) -> u32 {
        self.best_pixel_flags & ORBIT_ITERS_MASK
    }

    pub fn escaped(&self) -> bool {
        (self.best_pixel_flags & ORBIT_ESCAPED) != 0
    }

    pub fn perturbed(&self) -> bool {
        (self.best_pixel_flags & ORBIT_PERTURBED) != 0
    }

    pub fn perturb_err(&self) -> bool {
        (self.best_pixel_flags & ORBIT_PERTURB_ERR) != 0
    }

    pub fn max_iters_reached(&self) -> bool {
        (self.best_pixel_flags & ORBIT_MAX_ITER_REACHED) != 0
    }

    pub fn orbit_idx(&self) -> u32 {
        self.best_pixel_flags >> ORBIT_SHIFT
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OrbitFeedbackOut {
    pub min_iter_count:     u32,    // Min iters per pixel, per ref_orb
    pub max_iter_count:     u32,    // Max iters per pixel, per ref_orb

    // --- Flag counts ---
    pub escaped_count:          u32,
    pub perurb_error_count:     u32,
    pub max_iter_reached_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DebugOut {
    pub center_x_hi:        f32,
    pub center_x_lo:        f32,
    pub center_y_hi:        f32,
    pub center_y_lo:        f32,
    pub scale_hi:           f32,
    pub scale_lo:           f32,
    pub screen_width:       u32,
    pub screen_height:      u32,
    pub ref_orb_count:      u32,
    pub orbit_idx:          u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRefOrbitLocation {
    pub c_ref_re_hi:            f32,
    pub c_ref_re_lo:            f32,
    pub c_ref_im_hi:            f32,
    pub c_ref_im_lo:            f32,
    pub center_offset_re_hi:    f32,
    pub center_offset_re_lo:    f32,
    pub center_offset_im_hi:    f32,
    pub center_offset_im_lo:    f32,
}
