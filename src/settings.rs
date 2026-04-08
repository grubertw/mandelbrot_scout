use std::{env, io};
use std::collections::HashMap;
use config::{Config, ConfigBuilder, ConfigError, File, Map, Value};
use config::builder::DefaultState;
use log::{trace, warn};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct RgbPalette {
    pub name: String,
    pub array: Vec<u8>,
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
    pub render_res_factor: f64,
    pub render_res_factor_during_pan: f64,
    pub default_export_directory: String,
    pub default_export_filename: String,

    // Controls WGPU texture allocation
    pub max_orbits_per_frame: u32,
    pub max_ref_orbit: u32,
    pub render_tex_width: u32,
    pub render_tex_height: u32,
    pub screen_grid_size: u32,
    pub max_palette_colors: u32,
    
    // Color scene uniform settings
    pub scalar_mapping_power_range: (f32, f32),
    pub scalar_mapping_log_range: (f32, f32),
    pub scalar_mapping_atan_range: (f32, f32),
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
    
    pub palettes: Map<String, RgbPalette>,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        // Optional env override for config dir
        let settings_dir = env::var("SETTINGS_DIR").unwrap_or_else(|_| String::from("settings"));
        let primary_path = format!("{}/settings.toml", settings_dir);
        let fallback_path = "settings.toml";

        // Try loading primary config first
        match add_default_settings(Config::builder())?
            .add_source(File::with_name(&primary_path))
            .build()
        {
            Ok(config) => {
                trace!("build config Ok, attempting to parse.");
                config.try_deserialize()
            },
            Err(e) => {
                if is_not_found(&e) {
                    warn!("Did not find settings in SETTINGS_DIR {}, trying fallback", settings_dir);
                    // Try fallback path
                    match add_default_settings(Config::builder())?
                        .add_source(File::with_name(fallback_path))
                        .build() {
                        Ok(c) => c.try_deserialize(),
                        Err(e2) => {
                            if is_not_found(&e2) {
                                warn!("Could not find any settings.toml! Using program defaults, and a single default color palette!");
                                return add_default_settings(Config::builder())?.build()?.try_deserialize();
                            }
                            Err(e2)
                        }
                    }
                } else {
                    Err(e)
                }
            }
        }
    }
}

fn is_not_found(e: &ConfigError) -> bool {
    match e {
        ConfigError::NotFound(_) => true,
        ConfigError::Foreign(f) => {
            // Check the root cause downcast, or check the message
            if let Some(io_err) = f.downcast_ref::<io::Error>() {
                io_err.kind() == io::ErrorKind::NotFound
            } else {
                // Sometimes, Config wraps the io::Error further or uses Messages
                let msg = f.to_string();
                msg.contains("not found")
            }
        }
        _ => false
    }
}

fn add_default_settings(builder: ConfigBuilder<DefaultState>) -> Result<ConfigBuilder<DefaultState>, ConfigError> {
    // Provide a default color palette if settings.toml cannot be found!
    let mut default_palette = HashMap::new();
    default_palette.insert("name".to_string(), Value::from("Default"));
    default_palette.insert("array".to_string(), Value::from(vec![255, 0, 0, 0, 255, 0, 0, 0, 255]));
    let mut palettes_map = HashMap::new();
    palettes_map.insert("default".to_string(), Value::from(default_palette));

    builder
        .set_default("init_rug_precision", 128)?
        .set_default("max_live_orbits", 100)?
        .set_default("auto_start", false)?
        .set_default("starting_scale", 1e-6)?
        .set_default("ref_iters_multiplier", 1.25)?
        .set_default("num_seeds_to_spawn_per_eval", 4)?
        .set_default("num_qualified_orbits", 1)?
        .set_default("exploration_budget", 2)?
        .set_default("center", vec![-0.75, 0.0])?
        .set_default("complex_span", 3.0)?
        .set_default("max_user_iter", 500)?
        .set_default("render_res_factor", 1.0)?
        .set_default("render_res_factor_during_pan", 0.85)?
        .set_default("default_export_directory", "/Pictures")?
        .set_default("default_export_filename", "fractal")?
        .set_default("max_orbits_per_frame", 4)?
        .set_default("max_ref_orbit", 65535)?
        .set_default("render_tex_width", 8000)?
        .set_default("render_tex_height", 8000)?
        .set_default("screen_grid_size", 64)?
        .set_default("max_palette_colors", 1024)?
        .set_default("scalar_mapping_power_range", vec![0.2, 5.0])?
        .set_default("scalar_mapping_log_range", vec![1.0, 100.0])?
        .set_default("scalar_mapping_atan_range", vec![1.0, 50.0])?
        .set_default("distance_multiplier", 1.0)?
        .set_default("distance_multiplier_range", vec![0.1, 50.0])?
        .set_default("glow_intensity", 0.3)?
        .set_default("neighbor_scale_multiplier", 1.0)?
        .set_default("neighbor_scale_range", vec![1.0, 10.0])?
        .set_default("ambient_intensity", 0.75)?
        .set_default("key_light_intensity", 0.85)?
        .set_default("key_light_azimuth", 0.0)?
        .set_default("key_light_elevation", 35.0)?
        .set_default("fill_light_intensity", 0.35)?
        .set_default("fill_light_azimuth", 180.0)?
        .set_default("fill_light_elevation", 15.0)?
        .set_default("specular_intensity", 0.3)?
        .set_default("specular_power", 12.0)?
        .set_default("specular_power_range", vec![0.8, 128.0])?
        .set_default("ao_darkness", 0.3)?
        .set_default("stripe_density", 12.0)?
        .set_default("stripe_density_range", vec![0.1, 32.0])?
        .set_default("stripe_strength", 0.3)?
        .set_default("stripe_strength_range", vec![0.1, 8.0])?
        .set_default("stripe_gamma", 0.8)?
        .set_default("stripe_gamma_range", vec![0.1, 2.0])?
        .set_default("rim_intensity", 0.3)?
        .set_default("rim_power", 1.0)?
        .set_default("rim_power_range", vec![0.01, 4.0])?
        .set_default("palettes", palettes_map)
}