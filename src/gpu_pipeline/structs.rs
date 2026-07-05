use bytemuck;
use serde::{Deserialize, Serialize};

use crate::numerics::ComplexFExp;

fn default_formula_power() -> u32 { 2 }

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize)]
pub struct SceneUniform {
    pub center_x: f32,
    pub center_x_exp: i32,
    pub center_y: f32,
    pub center_y_exp: i32,
    pub scale: f32,
    pub scale_exp: i32,
    pub max_iter: u32,
    pub ref_orb_count: u32,
    pub perturb_err_threshold: f32,
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
    // Shared args for stripe averaging and orbit traps (only one active at a time;
    // traps override stripes when USE_TRAPS is set). Stripe meanings kept as comments.
    pub stripe_trap_arg1: f32,  // stripes: density  | traps: radius / size
    pub stripe_trap_arg2: f32,  // stripes: strength  | traps: sides (ngon) / spiral arms
    pub stripe_trap_arg3: f32,  // stripes: gamma     | traps: extra shape param
    pub stripe_trap_arg4: f32,  // stripes: (unused)  | traps: trap blend weight
    pub trap_shape: u32,
    pub trap_palette_cycles: f32,
    pub trap_iter_skip_frac: f32,
    // --- Fractal function (naive-path Julia + Power-of-X) ---
    // Placed here (mid-struct) so the calc shaders, which declare only the
    // prefix up to this point, can see them without pulling in the color/light
    // tail. color.wgsl pads for them to keep its later offsets aligned.
    // serde defaults keep older PNG metadata (which lacks these) loadable.
    #[serde(default = "default_formula_power")]
    pub formula_power: u32,     // z^power for the naive path (2 = classic)
    #[serde(default)]
    pub julia_c_re: f32,        // fixed Julia constant (world coords, f32 ok at
    #[serde(default)]
    pub julia_c_im: f32,        // naive/shallow scales); active via USE_JULIA bit
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
    
    pub fn set_glitch_fix(&mut self, fix: bool) {
        if fix { self.render_flags |= 1 << 1; } else { self.render_flags &= !(1 << 1) }
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

    pub fn set_use_fexp(&mut self, use_fexp: bool) {
        if use_fexp { self.render_flags |= 1 << 11; } else { self.render_flags &= !(1 << 11) }
    }

    pub fn set_use_bla(&mut self, use_bla: bool) {
        if use_bla { self.render_flags |= 1 << 12; } else { self.render_flags &= !(1 << 12) }
    }

    pub fn set_use_traps(&mut self, use_traps: bool) {
        if use_traps { self.render_flags |= 1 << 13; } else { self.render_flags &= !(1 << 13) }
    }

    pub fn set_use_trap_interior(&mut self, use_trap_interior: bool) {
        if use_trap_interior { self.render_flags |= 1 << 14; } else { self.render_flags &= !(1 << 14) }
    }

    pub fn set_use_julia(&mut self, use_julia: bool) {
        if use_julia { self.render_flags |= 1 << 15; } else { self.render_flags &= !(1 << 15) }
    }

    pub fn set_formula_power(&mut self, power: u32) {
        self.formula_power = power;
    }

    /// Set the fixed Julia constant `c` (world coordinates). Only consulted by
    /// the shader when the USE_JULIA flag is set.
    pub fn set_julia_c(&mut self, c_re: f32, c_im: f32) {
        self.julia_c_re = c_re;
        self.julia_c_im = c_im;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridFeedbackOut {
    pub best_pixel_x:           i32, // Pixel location of deepest iteration in the sample grid
    pub best_pixel_y:           i32,
    pub best_pixel_flags:       u32, // Iteration feedback flags for the location
    pub use_count:              u32,
    pub max_iter_count:         u32,
}

pub const ORBIT_ESCAPED: u32            = 1 << 0;
pub const ORBIT_PERTURBED: u32          = 1 << 1;
//pub const PERTURB_ERR: u32        = 1 << 2;
pub const ORBIT_MAX_ITER_REACHED: u32   = 1 << 3;
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
    pub use_count:                  u32,
    pub escaped_count:              u32,
    pub perurb_error_count:         u32,
    pub max_iter_reached_count:     u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DebugOut {
    // center/scale carried as FExp {mantissa, exponent} so they stay accurate
    // past 1e-38, where a plain-f32 reconstruction underflows to 0. In the f32
    // shader the exponents are 0 (mantissa holds the plain value).
    pub center_x:           f32,
    pub center_x_exp:       i32,
    pub center_y:           f32,
    pub center_y_exp:       i32,
    pub scale:              f32,
    pub scale_exp:          i32,
    pub max_iters:          u32,
    pub fi:                 f32,
    pub distance:           f32,
    pub stripe_avg:         f32,
    pub flags:              u32,
    // Cheap precision/health metric for the probe pixel (FExp shader only):
    // count of GLITCH rebases (excludes benign end-of-reference wraps). A high
    // count means the perturbed orbit keeps diverging from the reference =>
    // reference poorly matched / precision-stressed.
    pub rebase_count:       u32,
    // BLA probe stats (FExp shader only; 0 on the f32 path). Collected at the
    // off-center probe pixel — more representative than dead-center, where the
    // scout's reference orbit usually sits and dz stays trivially small.
    pub bla_max_step:       u32,   // largest single BLA jump (l) taken
    pub bla_step_count:     u32,   // number of BLA jumps taken
    pub bla_iters_skipped:  u32,   // total iterations covered by BLA jumps (sum of l)
}

/// One BLA (Bivariate Linear Approximation) table entry as uploaded to the GPU:
/// `dz_out = a*dz + b*dc`, valid for `l` reference iterations. Mandelbrot is
/// holomorphic, so `a`/`b` are complex scalars (`ComplexFExp`), not 2x2 matrices.
/// The validity radius (r^2) is uploaded separately in the `bla_radii` buffer
/// because it depends on the view (delta_c_max), while a/b/l do not.
///
/// repr(C): a@0, b@16, l@32, size 36, align 4 — matches the std430 `BlaEntry`
/// declared in mandelbrot_fexp.wgsl. Built by scout_engine::bla, which imports
/// this type so there is a single source of truth for the layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlaEntry {
    pub a: ComplexFExp,
    pub b: ComplexFExp,
    pub l: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRefOrbitLocation {
    pub c_ref_re:               f32,
    pub c_ref_im:               f32,
    pub max_ref_iters:          u32,
    pub center_offset_re:       f32,
    pub center_offset_re_exp:   i32,
    pub center_offset_im:       f32,
    pub center_offset_im_exp:   i32,
}

