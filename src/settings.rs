use std::env;

use config::{Config, ConfigError, File, Map};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Palette {
    pub name: String,
    pub array: Vec<u8>,
    pub frequency: f32,
    pub frequency_range: (f32, f32),
}

#[derive(Debug, Deserialize)]
pub struct Settings {
    // Maps to ScoutEngineConfig
    pub init_rug_precision: u32,
    pub max_live_orbits: u32,
    pub auto_start: bool,
    pub starting_scale: f64,
    pub ref_iters_multiplier: f64,
    pub num_seeds_to_spawn_per_eval: u32,
    pub num_qualified_orbits: u32,
    pub exploration_budget: i32,

    // Initial Scene config
    pub center: (f64, f64),
    pub complex_span: f64,
    pub max_user_iter: u32,

    // Controls WGPU texture allocation
    pub max_orbits_per_frame: u32,
    pub max_ref_orbit: u32,
    pub max_screen_width: u32,
    pub max_screen_height: u32,
    pub screen_grid_size: u32,
    pub max_palette_colors: u32,
    
    // Color scene uniform settings
    pub distance_multiplier: f32,
    pub distance_multiplier_range: (f32, f32),
    pub glow_intensity:     f32,
    pub neighbor_scale_multiplier: f32,
    pub neighbor_scale_range: (f32, f32),
    pub ambient_intensity: f32,
    pub key_light_intensity: f32,
    pub key_light_azimuth: f32,
    pub key_light_elevation: f32,
    pub fill_light_intensity: f32,
    pub fill_light_azimuth: f32,
    pub fill_light_elevation: f32,
    pub specular_intensity: f32,
    pub specular_power: f32,
    pub specular_power_range: (f32, f32),
    pub ao_darkness: f32,
    pub stripe_density: f32,
    pub stripe_density_range: (f32, f32),
    pub stripe_strength: f32,
    pub stripe_strength_range: (f32, f32),
    pub stripe_gamma: f32,
    pub stripe_gamma_range: (f32, f32),
    pub rim_intensity: f32,
    pub rim_power: f32,
    pub rim_power_range: (f32, f32),
    
    pub palettes: Map<String, Palette>,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        let settings_dir_prefix =
            env::var("SETTINGS_DIR").unwrap_or(String::from("settings"));

        let s = Config::builder()
            .add_source(File::with_name(format!("{}/settings.toml", settings_dir_prefix).as_str()))
            .build()?;

        s.try_deserialize()
    }
}