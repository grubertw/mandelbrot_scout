use bytemuck;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniform {
    pub center_x: f32,
    pub center_y: f32,
    pub scale: f32,
    pub max_iter: u32,
    pub ref_orb_count: u32,
    pub view_width: f32,
    pub view_height: f32,
    pub render_width: u32,
    pub render_height: u32,
    pub render_tex_width: f32,
    pub render_tex_height: f32,
    pub grid_size: u32,
    pub grid_width: u32,
    pub sample_count: u32,
    pub jitter_strength: f32,
    pub sample_avg_bias: f32,
    pub render_flags: u32,
    pub stripe_density: f32,
    pub stripe_strength: f32,
    pub stripe_gamma: f32,
    pub color_scalar_mapping_mode: u32,
    pub color_scaler_mapping_strength: f32,
    pub palette_tex_width: u32,
    pub palette_len: u32,
    pub palette_cycles: f32,
    pub palette_offset: f32,
    pub palette_gamma: f32,
    pub distance_multiplier: f32,
    pub glow_intensity: f32,
    pub neighbor_scale_multiplier: f32,
    pub ambient_intensity: f32,
    pub key_light_intensity: f32,
    pub key_light_azimuth: f32,
    pub key_light_elevation: f32,
    pub fill_light_intensity: f32,
    pub fill_light_azimuth: f32,
    pub fill_light_elevation: f32,
    pub specular_intensity: f32,
    pub specular_power: f32,
    pub ao_darkness: f32,
    pub rim_intensity: f32,
    pub rim_power: f32,
}

impl SceneUniform {
    pub fn set_debug_coloring(&mut self, debug_coloring: bool) {
        if debug_coloring { self.render_flags |= 1 << 0; } else { self.render_flags &= !(1 << 0) }
    }

    pub fn set_glitch_fix(&mut self, glitch_fix: bool) {
        if glitch_fix { self.render_flags |= 1 << 1; } else { self.render_flags &= !(1 << 1) }
    }
    
    pub fn set_smooth_coloring(&mut self, smooth_coloring: bool) {
        if smooth_coloring { self.render_flags |= 1 << 2; } else { self.render_flags &= !(1 << 2) }
    }
    
    pub fn set_use_de(&mut self, use_de: bool) {
        if use_de { self.render_flags |= 1 << 3; } else { self.render_flags &= !(1 << 3) }
    }
    
    pub fn set_use_stripes(&mut self, use_stripes: bool) {
        if use_stripes { self.render_flags |= 1 << 4; } else { self.render_flags &= !(1 << 4) }
    }
    
    pub fn set_enable_glow(&mut self, enable_glow: bool) {
        if enable_glow { self.render_flags |= 1 << 5; } else { self.render_flags &= !(1 << 5) }
    }
    
    pub fn set_enable_key_light(&mut self, enable_key_light: bool) {
        if enable_key_light { self.render_flags |= 1 << 6; } else { self.render_flags &= !(1 << 6) }
    }
    
    pub fn set_enable_fill_light(&mut self, enable_fill_light: bool) {
        if enable_fill_light { self.render_flags |= 1 << 7; } else { self.render_flags &= !(1 << 7) }
    }
    
    pub fn set_enable_specular(&mut self, enable_specular: bool) {
        if enable_specular { self.render_flags |= 1 << 8; } else { self.render_flags &= !(1 << 8) }
    }
    
    pub fn set_enable_ao(&mut self, enable_ao: bool) {
        if enable_ao { self.render_flags |= 1 << 9; } else { self.render_flags &= !(1 << 9) }
    }
    
    pub fn set_enable_rim(&mut self, enable_rim: bool) {
        if enable_rim { self.render_flags |= 1 << 10; } else { self.render_flags &= !(1 << 10) }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridFeedbackOut {
    pub best_pixel_x:           i32, // Pixel location of deepest iteration in the sample grid
    pub best_pixel_y:           i32,
    pub best_pixel_flags:       u32, // Iteration feedback flags for the location
    pub max_iter_count:         u32,
}

pub const ORBIT_ESCAPED: u32            = 1 << 0;
pub const ORBIT_PERTURBED: u32          = 1 << 1;
//pub const PERTURB_ERR_INNER: u32        = 1 << 2;
//pub const PERTURB_ERR_OUTER: u32        = 1 << 3;
pub const ORBIT_MAX_ITER_REACHED: u32   = 1 << 4;
pub const ORBIT_SHIFT: u32              = 20;

impl GridFeedbackOut {
    pub fn iter(&self) -> u32 {
        self.max_iter_count
    }

    pub fn escaped(&self) -> bool {
        (self.best_pixel_flags & ORBIT_ESCAPED) != 0
    }

    pub fn perturbed(&self) -> bool {
        (self.best_pixel_flags & ORBIT_PERTURBED) != 0
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
    pub escaped_count:              u32,
    pub perurb_error_inner_count:   u32,
    pub perurb_error_outer_count:   u32,
    pub max_iter_reached_count:     u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DebugOut {
    pub center_x:           f32,
    pub center_y:           f32,
    pub max_iters:          u32,
    pub fi:                 f32,
    pub distance:           f32,
    pub stripe_avg:         f32,
    pub flags:              u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRefOrbitLocation {
    pub c_ref_re:           f32,
    pub c_ref_im:           f32,
    pub r_valid:            f32,
    pub max_ref_iters:      u32,
    pub center_offset_re:   f32,
    pub center_offset_im:   f32,
}

