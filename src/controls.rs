use super::scene::{Scene};

use iced_wgpu::Renderer;
use iced_widget::{text_input, column, row, text, button, space, Row, slider, pick_list, checkbox, container, Column, radio};
use iced_widget::core::{Alignment, Color, Element, Theme, Length, border, color, Font};

use log::{debug, info, trace, warn};

use std::rc::Rc;
use std::cell::RefCell;
use std::env;
use std::fs::File;
use chrono::Local;
use iced_wgpu::core::font;
use iced_wgpu::core::text::Wrapping;
use png::Compression;
use rfd::FileDialog;
use rug::{Complex, Float};
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter};

use crate::export::{render_scene_to_jpeg, render_scene_to_png};
use crate::scene::import::{load_metadata_from_png, ExtPalette};
use crate::scout_engine::ScoutSignal;
use crate::settings::Settings;

pub mod built_info {
    // The file has been placed there by the build script.
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PaletteSelection {
    key: String,
    name: String
}

impl PaletteSelection {
    fn new(key: String, name: String) -> Self {
        Self {key, name}
    }
}

impl std::fmt::Display for PaletteSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter, Display )]
pub enum ColorScalarMappingMode {
    #[strum(to_string="Linear")]
    Linear,
    #[strum(to_string="Power")]
    Power,
    #[strum(to_string="Log")]
    Log,
    #[strum(to_string="Atan")]
    Atan,
}

impl From<ColorScalarMappingMode> for u32 {
    fn from(m: ColorScalarMappingMode) -> Self {
        match m {
            ColorScalarMappingMode::Linear => 0,
            ColorScalarMappingMode::Power => 1,
            ColorScalarMappingMode::Log => 2,
            ColorScalarMappingMode::Atan => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq )]
pub enum ExportImgFormat {
    Png,
    Jpeg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PngCompression {
    None,
    Fast,
    Default,
    Best,
}
impl From<PngCompression> for Compression {
    fn from(c: PngCompression) -> Self {
        match c {
            PngCompression::None => Compression::NoCompression,
            PngCompression::Fast => Compression::Fast,
            PngCompression::Default => Compression::Balanced,
            PngCompression::Best => Compression::High,
        }
    }
}

#[derive(Clone)]
pub struct Controls {
    // Bools that show/hide rows
    editing_iters: bool,
    editing_location: bool,
    editing_resolution: bool,
    editing_color: bool,
    editing_scout: bool,
    editing_export: bool,

    // Iterations config
    iter_step: u32,
    iter_range_min: u32,
    iter_range_max: u32,
    max_iterations: u32,

    // Location config
    center_x: String,
    center_y: String,
    scale: String,

    // Render resolution settings
    render_res_factor: f64,
    render_res_factor_during_pan: f64,
    render_sample_count: u32,
    render_jitter_strength: f32,
    render_sample_avg_bias: f32,

    // Color config
    palettes: Vec<PaletteSelection>,
    selected_palette: Option<PaletteSelection>,
    cycles: f32,
    cycles_max: f32,
    offset: f32,
    gamma: f32,
    color_scalar_mapping_mode: Option<ColorScalarMappingMode>,
    color_scaler_mapping_strength: f32,
    scalar_mapping_power_range: (f32, f32),
    scalar_mapping_log_range: (f32, f32),
    scalar_mapping_atan_range: (f32, f32),
    smooth_coloring: bool,
    use_de: bool,
    use_stripes: bool,
    debug_coloring: bool,
    // DE controls
    enable_glow: bool,
    distance_multiplier: f32,
    distance_multiplier_range: (f32, f32),
    glow_intensity: f32,
    neighbor_scale_multiplier: f32,
    neighbor_scale_range: (f32, f32),
    ambient_intensity: f32,
    enable_key_light: bool,
    key_light_intensity: f32,
    key_light_azimuth: f32,
    key_light_elevation: f32,
    enable_fill_light: bool,
    fill_light_intensity: f32,
    fill_light_azimuth: f32,
    fill_light_elevation: f32,
    enable_specular: bool,
    specular_intensity: f32,
    specular_power: f32,
    specular_power_range: (f32, f32),
    enable_ao: bool,
    ao_darkness: f32,
    stripe_density: f32,
    stripe_density_range: (f32, f32),
    stripe_strength: f32,
    stripe_strength_range: (f32, f32),
    stripe_gamma: f32,
    stripe_gamma_range: (f32, f32),
    enable_rim: bool,
    rim_intensity: f32,
    rim_power: f32,
    rim_power_range: (f32, f32),

    // Scout config
    auto_start: bool,
    ref_iters_multiplier: f64,
    num_samples_to_infer_dir: u32,
    distance_error_threshold: f32,

    // Export/save config
    export_img_format: Option<ExportImgFormat>,
    export_img_dir: String,
    export_img_file_name: String,
    export_img_png_compression: Option<PngCompression>,
    export_img_jpeg_quality: u8,
    use_alternate_dimensions: bool,
    append_timestamp_to_filename: bool,
    export_img_width: String,
    export_img_height: String,

    debug_msg: String,
    info_text: String,
    scene: Rc<RefCell<Scene>>
}

#[derive(Debug, Clone)]
pub enum Message {
    DisplayInfo,
    EditingItersChanged(bool),
    EditingLocationChanged(bool),
    EditingResolutionChanged(bool),
    EditingColorChanged(bool),
    EditingScoutConfigChanged(bool),
    EditingExportChanged(bool),

    IterStepChanged(String),
    IterRangeMinChanged(String),
    IterRangeMaxChanged(String),
    IterValueChanged(u32),

    CenterXChanged(String),
    CenterYChanged(String),
    ScaleChanged(String),
    PollFromScene,
    ApplyCenterScale,
    RestoreFromPng,

    RenderResFactorChanged(f64),
    RenderResFactorDuringPanChanged(f64),
    RenderSampleCountChanged(u32),
    RenderJitterStrengthChanged(f32),
    RenderSampleAvgBiasChanged(f32),

    SelectedPaletteChanged(PaletteSelection),
    ImportPalette,
    CyclesChanged(f32),
    CyclesMaxChanged(String),
    OffsetChanged(f32),
    GammaChanged(f32),
    ColorScalarMappingModeChanged(ColorScalarMappingMode),
    ColorScalerMappingStrengthChanged(f32),
    SmoothColoringChanged(bool),
    UseDEChanged(bool),
    UseStripesChanged(bool),
    EnableGlowChanged(bool),
    DebugColoringChanged(bool),
    DistanceMultiplierChanged(f32),
    GlowIntensityChanged(f32),
    NeighborScaleChanged(f32),
    AmbientIntensityChanged(f32),
    EnableKeyLightChanged(bool),
    KeyLightIntensityChanged(f32),
    KeyLightAzimuthChanged(f32),
    KeyLightElevationChanged(f32),
    EnableFillLightChanged(bool),
    FillLightIntensityChanged(f32),
    FillLightAzimuthChanged(f32),
    FillLightElevationChanged(f32),
    EnableSpecularChanged(bool),
    SpecularIntensityChanged(f32),
    SpecularPowerChanged(f32),
    EnableAoChanged(bool),
    AoDarknessChanged(f32),
    StripeDensityChanged(f32),
    StripeStrengthChanged(f32),
    StripeGammaChanged(f32),
    EnableRimChanged(bool),
    RimIntensityChanged(f32),
    RimPowerChanged(f32),

    ResetScoutEngine,
    GoScout,
    AutoStartChanged(bool),
    RefItersMultiplierChanged(f64),
    SamplesToInferDirChanged(u32),
    DistanceErrorThresholdChanged(f32),

    ExportImgFormatChanged(ExportImgFormat),
    ExportImgDirChanged(String),
    ExportImgFileNameChanged(String),
    ExportImgPngCompressionChanged(PngCompression),
    ExportImgJpegQualityChanged(u8),
    UseAlternateDimensionsChanged(bool),
    AppendTimestampToFileNameChanged(bool),
    ExportImgWidthChanged(String),
    ExportImgHeightChanged(String),
    ExportImage,

    UpdateDebugText(String),
}

impl Controls {
    pub fn new(settings: &Settings, scene: Rc<RefCell<Scene>>) -> Controls {
        let scene_b = scene.borrow();
        let center = scene_b.center().clone();
        let center_x = center.real().to_string_radix(10, Some(10));
        let center_y = center.imag().to_string_radix(10, Some(10));
        let scale = scene_b.scale().to_string_radix(10, Some(6));

        let mut palettes: Vec<PaletteSelection> = scene_b.get_palette_list()
            .iter()
            .map(|( key, name)|
                PaletteSelection::new(key.clone(), name.clone())
            ).collect();
        palettes.sort_by(|a, b| a.key.cmp(&b.key));

        let palette_selection =
            PaletteSelection::new("default".to_string(), "Default".to_string());

        let info_text = format!("pkg_name={}\n\tpkg_version={}\n\tauthors={}\n\trepository={}\n\n\
        rustc_version={}",
            built_info::PKG_NAME,
            built_info::PKG_VERSION,
            built_info::PKG_AUTHORS,
            built_info::PKG_REPOSITORY,
            built_info::RUSTC_VERSION,
        );

        let default_export_directory = match env::var("HOME") {
            Ok(val) => format!("{}{}", val, settings.default_export_directory),
            Err(_) => "".to_string(),
        };

        let debug_msg = if settings.palettes.len() == 1 {
            "settings.toml was not found. Starting with only one color palette!"
        } else {""};

        Controls {
            editing_iters: false,
            editing_location: false,
            editing_resolution: false,
            editing_color: false,
            editing_scout: false,
            editing_export: false,
            iter_step: 10,
            iter_range_min: 0, iter_range_max: settings.max_user_iter * 2,
            max_iterations: settings.max_user_iter,
            center_x, center_y, scale,
            render_res_factor: settings.render_res_factor,
            render_res_factor_during_pan: settings.render_res_factor_during_pan,
            render_sample_count: 1,
            render_jitter_strength: 0.0,
            render_sample_avg_bias: 0.9,
            palettes,
            selected_palette: Some(palette_selection),
            cycles: 1.0, cycles_max: 10.0,
            offset: 0.0, gamma: 1.0,
            color_scalar_mapping_mode: Some(ColorScalarMappingMode::Linear),
            color_scaler_mapping_strength: 1.0,
            scalar_mapping_power_range: settings.scalar_mapping_power_range,
            scalar_mapping_log_range: settings.scalar_mapping_log_range,
            scalar_mapping_atan_range: settings.scalar_mapping_atan_range,
            smooth_coloring: false,
            use_de: false, use_stripes: false, debug_coloring: false,
            enable_glow: false,
            distance_multiplier: settings.distance_multiplier,
            distance_multiplier_range: settings.distance_multiplier_range,
            glow_intensity: settings.glow_intensity,
            neighbor_scale_multiplier: settings.neighbor_scale_multiplier,
            neighbor_scale_range: settings.neighbor_scale_range,
            ambient_intensity: settings.ambient_intensity,
            enable_key_light: false,
            key_light_intensity: settings.key_light_intensity,
            key_light_azimuth: settings.key_light_azimuth,
            key_light_elevation: settings.key_light_elevation,
            enable_fill_light: false,
            fill_light_intensity: settings.fill_light_intensity,
            fill_light_azimuth: settings.fill_light_azimuth,
            fill_light_elevation: settings.fill_light_elevation,
            enable_specular: false,
            specular_intensity: settings.specular_intensity,
            specular_power: settings.specular_power,
            specular_power_range: settings.specular_power_range,
            enable_ao: false,
            ao_darkness: settings.ao_darkness,
            stripe_density: settings.stripe_density,
            stripe_density_range: settings.stripe_density_range,
            stripe_strength: settings.stripe_strength,
            stripe_strength_range: settings.stripe_strength_range,
            stripe_gamma: settings.stripe_gamma,
            stripe_gamma_range: settings.stripe_gamma_range,
            enable_rim: false,
            rim_intensity: settings.rim_intensity,
            rim_power: settings.rim_power,
            rim_power_range: settings.rim_power_range,
            auto_start: settings.auto_start,
            ref_iters_multiplier:  settings.ref_iters_multiplier,
            num_samples_to_infer_dir: settings.num_samples_to_infer_direction,
            distance_error_threshold: settings.distance_error_threshold,
            export_img_format: Some(ExportImgFormat::Png),
            export_img_dir: default_export_directory,
            export_img_file_name: settings.default_export_filename.clone(),
            export_img_png_compression: Some(PngCompression::Default),
            export_img_jpeg_quality: 90,
            use_alternate_dimensions: false,
            append_timestamp_to_filename: true,
            export_img_width: scene.borrow().width().to_string(),
            export_img_height: scene.borrow().height().to_string(),
            debug_msg: debug_msg.to_string(), info_text,
            scene: scene.clone()
        }
    }

    pub fn update(&mut self, message: Message) {
        trace!("Update {:?}", message);
        match message {
            Message::DisplayInfo => {
                if self.debug_msg.len() == self.info_text.len() {
                    self.debug_msg = String::new();
                }
                else {
                    self.debug_msg = self.info_text.clone();
                }
            }
            Message::EditingItersChanged(toggle) => {
                self.editing_iters = toggle;
            }
            Message::EditingLocationChanged(toggle) => {
                self.editing_location = toggle;
            }
            Message::EditingResolutionChanged(toggle) => {
                self.editing_resolution = toggle;
            }
            Message::EditingColorChanged(toggle) => {
                self.editing_color = toggle;
            }
            Message::EditingScoutConfigChanged(toggle) => {
                self.editing_scout = toggle;
            }
            Message::EditingExportChanged(toggle) => {
                self.editing_export = toggle;
                if !self.use_alternate_dimensions {
                    self.export_img_width = self.scene.borrow().width().to_string();
                    self.export_img_height = self.scene.borrow().height().to_string();
                }
            }
            Message::IterStepChanged(iter_inc) => {
                if let Ok(v) = iter_inc.parse::<u32>() {
                    self.iter_step = v;
                }
            }
            Message::IterRangeMinChanged(iter_min) => {
                if let Ok(v) = iter_min.parse::<u32>() {
                    self.iter_range_min = v;
                }
            }
            Message::IterRangeMaxChanged(iter_max) => {
                if let Ok(v) = iter_max.parse::<u32>() {
                    self.iter_range_max = v;
                }
            }
            Message::IterValueChanged(iter_value) => {
                self.max_iterations = iter_value;
                self.scene.borrow_mut().set_max_iterations(self.max_iterations);
            }
            Message::CenterXChanged(x_str) => {
                self.center_x = x_str;
            }
            Message::CenterYChanged(y_str) => {
                self.center_y = y_str;
            }
            Message::ScaleChanged(scale_str) => {
                self.scale = scale_str;
            }
            Message::PollFromScene => {
                let scene_b = self.scene.borrow();
                let center = scene_b.center();
                self.center_x = center.real().to_string_radix(10, Some(10));
                self.center_y = center.imag().to_string_radix(10, Some(10));
                self.scale = scene_b.scale().to_string_radix(10, Some(6));
            }
            Message::ApplyCenterScale => {
                let prec = self.scene.borrow().scale().prec();
                let center_x = if let Ok(re) = Float::parse(self.center_x.clone()) {
                    Float::with_val(prec, re)
                } else {Float::with_val(prec, 0)};

                let center_y = if let Ok(im) = Float::parse(self.center_y.clone()) {
                    Float::with_val(prec, im)
                } else {Float::with_val(prec, 0)};

                let scale = if let Ok(s) = Float::parse(self.scale.clone()) {
                    Float::with_val(prec, s)
                } else {Float::with_val(prec, 0)};

                let center = Complex::with_val(prec, (center_x, center_y));

                let mut scene_b = self.scene.borrow_mut();
                scene_b.set_center(center);
                scene_b.set_scale(scale);
                // Update scout engine with new position
                scene_b.take_camera_snapshot();
            }
            Message::RestoreFromPng => {
                if let Some(path) = FileDialog::new()
                    .add_filter("PNG Image", &["png"])
                    .pick_file() {
                    info!("Attempting to restore scene from PNG file: {:?}", path);
                    if let Some(meta) = load_metadata_from_png(path.to_str().unwrap()) {
                        debug!("Found metadata in PNG: {:?}", meta);
                        let mut scene_b = self.scene.borrow_mut();

                        scene_b.apply_metadata(meta);
                        self.debug_msg = format!("Successfully restored scene from PNG file {:?}!", path);
                        info!("{}", self.debug_msg);
                        self.center_x = scene_b.center().real().to_string_radix(10, Some(10));
                        self.center_y = scene_b.center().imag().to_string_radix(10, Some(10));
                        self.scale = scene_b.scale().to_string_radix(10, Some(6));
                    }
                    else {
                        self.debug_msg =
                            format!("Failed to restore scene from selected file {:?}\n\tPNG text header metadata may not be present.\n\tNOTE, only PNG files saved by this program can be restored in this manor.",
                                    path);
                        warn!("{}", self.debug_msg);
                    }
                }
            }
            Message::RenderResFactorChanged(res_fac) => {
                self.render_res_factor = res_fac;
                self.scene.borrow_mut().set_render_res_factor(res_fac);
            }
            Message::RenderResFactorDuringPanChanged(res_fac) => {
                self.render_res_factor_during_pan = res_fac;
                self.scene.borrow_mut().set_render_res_factor_during_pan(res_fac);
            }
            Message::RenderSampleCountChanged(count) => {
                self.render_sample_count = count;
                self.scene.borrow_mut().set_sample_count(count);
            }
            Message::RenderJitterStrengthChanged(strength) => {
                self.render_jitter_strength = strength;
                self.scene.borrow_mut().set_jitter_strength(strength);
            }
            Message::RenderSampleAvgBiasChanged(bias) => {
                self.render_sample_avg_bias = bias;
                self.scene.borrow_mut().set_sample_avg_bias(bias);
            }
            Message::SelectedPaletteChanged(selected_palette) => {
                let key = selected_palette.key.clone();
                let mut scene_b = self.scene.borrow_mut();
                self.selected_palette = Some(selected_palette);
                scene_b.set_selected_palette(&key);
            }
            Message::ImportPalette => {
                if let Some(path) = FileDialog::new()
                    .add_filter("MAP palette", &["map"])
                    .set_title("Select a MAP color palette file")
                    .pick_file() {
                    info!("Attempting to import MAP palette: {:?}", path);
                    let map_palette =
                        ExtPalette::parse_map(&path);
                    match map_palette {
                        Ok(palette) => {
                            let key = path.file_name().unwrap().to_str().unwrap().to_string();
                            let selection = PaletteSelection{
                                key: key.clone() , name: palette.name.clone()
                            };
                            self.palettes.push(selection.clone());
                            self.selected_palette = Some(selection.clone());
                            self.scene.borrow_mut().add_palette(&key, palette.to_color_palette());
                            self.debug_msg = format!("Successfully imported MAP palette with name {} and {} colors",
                                 selection.name, palette.colors.len());
                            info!("{}", self.debug_msg);
                        }
                        Err(e) => {
                            warn!("Failed to import palette: {:?}", e);
                        }
                    }
                }
            }
            Message::CyclesChanged(cycles) => {
                self.cycles = cycles;
                self.scene.borrow_mut().set_palette_cycles(cycles);
            }
            Message::CyclesMaxChanged(cyc_max) => {
                if let Ok(v) = cyc_max.parse::<f32>() {
                    self.cycles_max = v;
                }
            }
            Message::OffsetChanged(offset) => {
                self.offset = offset;
                self.scene.borrow_mut().set_palette_offset(offset);
            }
            Message::GammaChanged(gamma) => {
                self.gamma = gamma;
                self.scene.borrow_mut().set_palette_gamma(gamma);
            }
            Message::ColorScalarMappingModeChanged(mode) => {
                self.color_scalar_mapping_mode = Some(mode);
                self.scene.borrow_mut().set_color_scalar_mapping_mode(mode.into());
            }
            Message::ColorScalerMappingStrengthChanged(strength) => {
                self.color_scaler_mapping_strength = strength;
                self.scene.borrow_mut().set_color_scalar_mapping_strength(strength);
            }
            Message::SmoothColoringChanged(coloring) => {
                self.smooth_coloring = coloring;
                self.scene.borrow_mut().set_smooth_coloring(coloring);
            }
            Message::UseDEChanged(use_de) => {
                self.use_de = use_de;
                self.scene.borrow_mut().set_use_de(use_de);
            }
            Message::UseStripesChanged(use_stripes) => {
                self.use_stripes = use_stripes;
                self.scene.borrow_mut().set_use_stripes(use_stripes);
            }
            Message::DebugColoringChanged(coloring) => {
                self.debug_coloring = coloring;
                self.scene.borrow_mut().set_debug_coloring(coloring);
            }
            Message::EnableGlowChanged(enable_glow) => {
                self.enable_glow = enable_glow;
                self.scene.borrow_mut().set_enable_glow(enable_glow);
            }
            Message::DistanceMultiplierChanged(distance) => {
                self.distance_multiplier = distance;
                self.scene.borrow_mut().set_distance_multiplier(distance);
            }
            Message::GlowIntensityChanged(intensity) => {
                self.glow_intensity = intensity;
                self.scene.borrow_mut().set_glow_intensity(intensity);
            }
            Message::NeighborScaleChanged(scale) => {
                self.neighbor_scale_multiplier = scale;
                self.scene.borrow_mut().set_neighbor_scale(scale);
            }
            Message::AmbientIntensityChanged(intensity) => {
                self.ambient_intensity = intensity;
                self.scene.borrow_mut().set_ambient_intensity(intensity);
            }
            Message::EnableKeyLightChanged(enable_keylight) => {
                self.enable_key_light = enable_keylight;
                self.scene.borrow_mut().set_enable_key_light(enable_keylight);
            }
            Message::KeyLightIntensityChanged(intensity) => {
                self.key_light_intensity = intensity;
                self.scene.borrow_mut().set_key_light_intensity(intensity);
            }
            Message::KeyLightAzimuthChanged(azimuth) => {
                self.key_light_azimuth = azimuth;
                self.scene.borrow_mut().set_key_light_azimuth(azimuth);
            }
            Message::KeyLightElevationChanged(elevation) => {
                self.key_light_elevation = elevation;
                self.scene.borrow_mut().set_key_light_elevation(elevation);
            }
            Message::EnableFillLightChanged(enable_filllight) => {
                self.enable_fill_light = enable_filllight;
                self.scene.borrow_mut().set_enable_fill_light(enable_filllight);
            }
            Message::FillLightIntensityChanged(intensity) => {
                self.fill_light_intensity = intensity;
                self.scene.borrow_mut().set_fill_light_intensity(intensity);
            }
            Message::FillLightAzimuthChanged(azimuth) => {
                self.fill_light_azimuth = azimuth;
                self.scene.borrow_mut().set_fill_light_azimuth(azimuth);
            }
            Message::FillLightElevationChanged(elevation) => {
                self.fill_light_elevation = elevation;
                self.scene.borrow_mut().set_fill_light_elevation(elevation);
            }
            Message::EnableSpecularChanged(specular) => {
                self.enable_specular = specular;
                self.scene.borrow_mut().set_enable_specular(specular);
            }
            Message::SpecularIntensityChanged(intensity) => {
                self.specular_intensity = intensity;
                self.scene.borrow_mut().set_specular_intensity(intensity);
            }
            Message::SpecularPowerChanged(specular_power) => {
                self.specular_power = specular_power;
                self.scene.borrow_mut().set_specular_power(specular_power);
            }
            Message::EnableAoChanged(enable_ao) => {
                self.enable_ao = enable_ao;
                self.scene.borrow_mut().set_enable_ao(enable_ao);
            }
            Message::AoDarknessChanged(ao_darkness) => {
                self.ao_darkness = ao_darkness;
                self.scene.borrow_mut().set_ao_darkness(ao_darkness);
            }
            Message::StripeDensityChanged(density) => {
                self.stripe_density = density;
                self.scene.borrow_mut().set_stripe_density(density);
            }
            Message::StripeStrengthChanged(strength) => {
                self.stripe_strength = strength;
                self.scene.borrow_mut().set_stripe_strength(strength);
            }
            Message::StripeGammaChanged(gamma) => {
                self.stripe_gamma = gamma;
                self.scene.borrow_mut().set_stripe_gamma(gamma);
            }
            Message::EnableRimChanged(rim) => {
                self.enable_rim = rim;
                self.scene.borrow_mut().set_enable_rim(rim);
            }
            Message::RimIntensityChanged(intensity) => {
                self.rim_intensity = intensity;
                self.scene.borrow_mut().set_rim_intensity(intensity);
            }
            Message::RimPowerChanged(power) => {
                self.rim_power = power;
                self.scene.borrow_mut().set_rim_power(power);
            }
            Message::ResetScoutEngine => {
                self.scene.borrow_mut().send_scout_signal(ScoutSignal::ResetEngine);
            }
            Message::GoScout => {
                let mut scene_b = self.scene.borrow_mut();
                let config = scene_b.scout_config().lock().clone();
                scene_b.send_scout_signal(ScoutSignal::ExploreSignal(config));
            }
            Message::AutoStartChanged(auto_start) => {
                self.auto_start = auto_start;
                self.scene.borrow_mut().set_scout_auto_start(auto_start);
            }
            Message::RefItersMultiplierChanged(ref_iters_multiplier) => {
                self.ref_iters_multiplier = ref_iters_multiplier;
                self.scene.borrow_mut().set_ref_iters_multiplier(ref_iters_multiplier);
            }
            Message::SamplesToInferDirChanged(spawn_per_eval) => {
                self.num_samples_to_infer_dir = spawn_per_eval;
                self.scene.borrow_mut().set_num_samples_to_infer_direction(spawn_per_eval);
            }
            Message::DistanceErrorThresholdChanged(distance_error_threshold) => {
                self.distance_error_threshold = distance_error_threshold;
                self.scene.borrow_mut().set_distance_error_threshold(distance_error_threshold);
            }
            Message::ExportImgFormatChanged(img_format) => {
                self.export_img_format = Some(img_format);
            }
            Message::ExportImgDirChanged(img_dir) => {
                self.export_img_dir = img_dir;
            }
            Message::ExportImgFileNameChanged(img_file_name) => {
                self.export_img_file_name = img_file_name;
            }
            Message::ExportImgPngCompressionChanged(png_compression) => {
                self.export_img_png_compression = Some(png_compression);
            }
            Message::ExportImgJpegQualityChanged(jpeg_quality) => {
                self.export_img_jpeg_quality = jpeg_quality;
            }
            Message::UseAlternateDimensionsChanged(use_alt_dims) => {
                self.use_alternate_dimensions = use_alt_dims;
                if !use_alt_dims {
                    self.export_img_width = self.scene.borrow().width().to_string();
                    self.export_img_height = self.scene.borrow().height().to_string();
                }
            }
            Message::AppendTimestampToFileNameChanged(append_timestamp_to_file_name) => {
                self.append_timestamp_to_filename = append_timestamp_to_file_name;
            }
            Message::ExportImgWidthChanged(width) => {
                self.export_img_width = width;
            }
            Message::ExportImgHeightChanged(height) => {
                self.export_img_height = height;
            }
            Message::ExportImage => {
                let file_ext =  if let Some(img_fmt) = self.export_img_format && img_fmt == ExportImgFormat::Png {
                    "png".to_string()
                } else if let Some(img_fmt) = self.export_img_format && img_fmt == ExportImgFormat::Jpeg {
                    "jpg".to_string()
                } else {"".to_string()};

                let ts = if self.append_timestamp_to_filename {
                    Local::now().format("_%y%m%d_%H%M%S").to_string()
                } else {"".to_string()};

                let full_filename = format!("{}/{}{}.{}", self.export_img_dir, self.export_img_file_name, ts, file_ext);
                match File::create(full_filename.as_str()) {
                    Ok(file) => {
                        let mut img_width = match self.export_img_width.parse::<u32>() {
                            Ok(w) => w,
                            Err(e) => {
                                self.debug_msg =
                                    format!("Provided image width is invalid: {}", e.to_string());
                                return;
                            }
                        };
                        let mut img_height = match self.export_img_height.parse::<u32>() {
                            Ok(h) => h,
                            Err(e) => {
                                self.debug_msg = format!("Provided image height is invalid: {}", e.to_string());
                                return;
                            }
                        };

                        img_width = self.scene.borrow().max_width().min(img_width);
                        img_height = self.scene.borrow().max_height().min(img_height);
                        self.debug_msg = format!("Exporting image with width {} and height {} to image file {}\n\texport will run in the background...",
                             img_width, img_height, full_filename);
                        info!("{}", self.debug_msg);

                        if let Some(img_fmt) = self.export_img_format && img_fmt == ExportImgFormat::Png {
                            render_scene_to_png(self.scene.clone(), img_width, img_height, file, self.export_img_png_compression.unwrap().into())
                        }
                        else if let Some(img_fmt) = self.export_img_format && img_fmt == ExportImgFormat::Jpeg {
                            render_scene_to_jpeg(self.scene.clone(), img_width, img_height, file, self.export_img_jpeg_quality)
                        }
                    }
                    Err(e) => {
                        self.debug_msg = e.to_string();
                    }
                }
            }
            Message::UpdateDebugText(dbg_msg) => {
                self.debug_msg = dbg_msg;
            }
        }
    }

    pub fn view(&self) -> Element<'_, Message, Theme, Renderer> {
        let dbg_row = row![
            text(&self.debug_msg).color(Color::WHITE)
            .wrapping(Wrapping::Word)
            .size(10)
        ]
        .align_y(Alignment::Start);


        let toggles_row = row![
            button(
                text!("\u{1F6C8}")
                .font(Font {weight: font::Weight::Semibold, ..Font::default()})
            ).on_press(Message::DisplayInfo)
                .style(button::text),
            button("Iters")
                .on_press(Message::EditingItersChanged(!self.editing_iters))
                .style(if self.editing_iters {button::primary} else {button::text}),
            space().width(Length::Fixed(20.0)),
            button("Loc")
                .on_press(Message::EditingLocationChanged(!self.editing_location))
                .style(if self.editing_location {button::primary} else {button::text}),
            space().width(Length::Fixed(20.0)),
            button("Res")
                .on_press(Message::EditingResolutionChanged(!self.editing_resolution))
                .style(if self.editing_resolution {button::primary} else {button::text}),
            space().width(Length::Fixed(20.0)),
            button("Color")
                .on_press(Message::EditingColorChanged(!self.editing_color))
                .style(if self.editing_color {button::primary} else {button::text}),
            space().width(Length::Fixed(20.0)),
            button("Scout")
                .on_press(Message::EditingScoutConfigChanged(!self.editing_scout))
                .style(if self.editing_scout {button::primary} else {button::text}),
            space().width(Length::Fixed(20.0)),
            button("Save")
                .on_press(Message::EditingExportChanged(!self.editing_export))
                .style(if self.editing_export {button::primary} else {button::text}),
        ]
        .align_y(Alignment::Start);

        let mut primary_panel = column![
                dbg_row,
                row![]
                .height(Length::Fill)
                .align_y(Alignment::End),
                toggles_row,
            ]
            .height(Length::Fill)
            .width(Length::Fill)
            .padding(10)
            .spacing(10);

        if self.editing_iters {
            primary_panel = primary_panel.push(
                container(self.render_iterations_row())
                    .style(outer_container_style)
                    .padding(10)
            );
        }
        if self.editing_location {
            primary_panel = primary_panel.push(
                container(self.render_edit_location_row())
                    .style(outer_container_style)
                    .padding(10)
            );
        }
        if self.editing_resolution {
            primary_panel = primary_panel.push(
                container(self.render_edit_render_resolution())
                    .style(outer_container_style)
                    .padding(10)
            )
        }
        if self.editing_color {
            primary_panel = primary_panel.push(
                container(self.render_color_controls())
                    .style(outer_container_style)
                    .padding(10)
            );
        }
        if self.editing_scout {
            primary_panel = primary_panel.push(
                container(self.render_scout_config_row())
                    .style(outer_container_style)
                    .padding(10)
            );
        }
        if self.editing_export {
            primary_panel = primary_panel.push(
                container(self.render_export_config())
                    .style(outer_container_style)
                    .padding(10)
            );
        }

        row![
            primary_panel
        ]
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    }

    fn render_iterations_row(&self) -> Row<'_, Message, Theme, Renderer> {
        row![
            text("step: ")
                .align_y(Alignment::Center),
            text_input("...", &self.iter_step.to_string())
                .on_input(Message::IterStepChanged)
                .width(Length::Fixed(40.0)),
            space().width(Length::Fixed(15.0)),
            text_input("...", &self.iter_range_min.to_string())
                .on_input(Message::IterRangeMinChanged)
                .width(Length::Fixed(50.0))
                .align_x(Alignment::End),
            space().width(Length::Fixed(10.0)),
            slider(self.iter_range_min..=self.iter_range_max,
                self.max_iterations, Message::IterValueChanged)
                .step(self.iter_step)
                .width(Length::Fixed(240.0)),
            space().width(Length::Fixed(10.0)),
            text_input("...", &self.iter_range_max.to_string())
                .on_input(Message::IterRangeMaxChanged)
                .width(Length::Fixed(50.0))
                .align_x(Alignment::Start),
            space().width(Length::Fixed(15.0)),
            text("max iterations: ")
                .align_y(Alignment::Center)
                .size(16),
            text(self.max_iterations.to_string())
                .align_y(Alignment::Center)
                .size(18),
        ]
            .align_y(Alignment::Center)
    }

    fn render_edit_location_row(&self) -> Row<'_, Message, Theme, Renderer> {
        row![
            button("Poll")
                .on_press(Message::PollFromScene)
                .width(Length::Fixed(50.0)),

            text("real: ")
                .width(Length::Fixed(50.0))
                .align_y(Alignment::Center)
                .align_x(Alignment::End),
            text_input("Placeholder...", &self.center_x)
                .on_input(Message::CenterXChanged)
                .width(Length::Fixed(140.0)),

            text("imag: ")
                .width(Length::Fixed(60.0))
                .align_y(Alignment::Center)
                .align_x(Alignment::End),
            text_input("Placeholder...", &self.center_y)
                .on_input(Message::CenterYChanged)
                .width(Length::Fixed(140.0)),

            text("scale: ")
                .width(Length::Fixed(60.0))
                .align_y(Alignment::Center)
                .align_x(Alignment::End),
            text_input("Placeholder...", &self.scale)
                .on_input(Message::ScaleChanged)
                .width(Length::Fixed(110.0)),

            space().width(Length::Fixed(10.0)),

            button("Apply")
                .on_press(Message::ApplyCenterScale)
                .width(Length::Fixed(65.0)),

            space().width(Length::Fixed(10.0)),

            button(
                text("Restore From PNG")
                .wrapping(Wrapping::Word)
                .align_x(Alignment::Center)
                .size(9))
            .on_press(Message::RestoreFromPng)
            .width(Length::Fixed(90.0))
        ]
            .align_y(Alignment::Center)
    }

    fn render_edit_render_resolution(&self) -> Column<'_, Message, Theme, Renderer> {
        column![
            container(row![
                text("Resolution Factor")
                    .width(Length::Fixed(140.0))
                    .align_y(Alignment::Center)
                    .align_x(Alignment::Center),
                space().width(Length::Fixed(5.0)),
                slider(0.25..=2.0,
                    self.render_res_factor, Message::RenderResFactorChanged)
                    .step(0.05)
                    .width(Length::Fixed(100.0)),
                space().width(Length::Fixed(5.0)),
                text(format!("{:<3.2}", self.render_res_factor))
                    .width(Length::Fixed(30.0))
                    .align_y(Alignment::Center),
                space().width(Length::Fixed(15.0)),

                text("RF During Pan")
                    .width(Length::Fixed(120.0))
                    .align_y(Alignment::Center)
                    .align_x(Alignment::Center),
                space().width(Length::Fixed(5.0)),
                slider(0.1..=1.0,
                    self.render_res_factor_during_pan, Message::RenderResFactorDuringPanChanged)
                    .step(0.05)
                    .width(Length::Fixed(100.0)),
                space().width(Length::Fixed(5.0)),
                text(format!("{:<3.2}", self.render_res_factor_during_pan))
                    .width(Length::Fixed(30.0))
                    .align_y(Alignment::Center),
                ]
                .align_y(Alignment::Center)
            )
            .style(inner_container_style)
            .padding(10),
            container(row![
                text("Samples")
                    .width(Length::Fixed(80.0))
                    .align_y(Alignment::Center)
                    .align_x(Alignment::Center),
                space().width(Length::Fixed(5.0)),
                slider(1..=16,
                    self.render_sample_count, Message::RenderSampleCountChanged)
                    .step(1_u32)
                    .width(Length::Fixed(60.0)),
                space().width(Length::Fixed(5.0)),
                text(format!("{:<2}", self.render_sample_count))
                    .width(Length::Fixed(10.0))
                    .align_y(Alignment::Center),
                space().width(Length::Fixed(10.0)),

                text("Jitter")
                    .width(Length::Fixed(60.0))
                    .align_y(Alignment::Center)
                    .align_x(Alignment::Center),
                space().width(Length::Fixed(5.0)),
                slider(0.0..=1.0,
                    self.render_jitter_strength, Message::RenderJitterStrengthChanged)
                    .step(0.01)
                    .width(Length::Fixed(60.0)),
                space().width(Length::Fixed(5.0)),
                text(format!("{:<1.2}", self.render_jitter_strength))
                    .width(Length::Fixed(30.0))
                    .align_y(Alignment::Center),
                space().width(Length::Fixed(10.0)),

                text("Averaging Bias")
                    .width(Length::Fixed(120.0))
                    .align_y(Alignment::Center)
                    .align_x(Alignment::Center),
                space().width(Length::Fixed(5.0)),
                slider(0.0..=1.0,
                    self.render_sample_avg_bias, Message::RenderSampleAvgBiasChanged)
                    .step(0.01)
                    .width(Length::Fixed(60.0)),
                space().width(Length::Fixed(5.0)),
                text(format!("{:<1.2}", self.render_sample_avg_bias))
                    .width(Length::Fixed(30.0))
                    .align_y(Alignment::Center),
                ]
                .align_y(Alignment::Center)
            )
            .style(inner_container_style)
            .padding(10)
        ]
            .spacing(5)
            .padding(5)
    }

    fn render_color_controls(&self) -> Column<'_, Message, Theme, Renderer> {
        let color_scalar_mappings: Vec<ColorScalarMappingMode> = ColorScalarMappingMode::iter().collect();
        let scalar_mapping_strength_range = match self.color_scalar_mapping_mode.unwrap() {
            ColorScalarMappingMode::Linear => self.scalar_mapping_power_range,
            ColorScalarMappingMode::Power => self.scalar_mapping_power_range,
            ColorScalarMappingMode::Log => self.scalar_mapping_log_range,
            ColorScalarMappingMode::Atan => self.scalar_mapping_atan_range,
        };

        let mut color_controls = column![
            container(row![
                checkbox(self.use_de)
                    .on_toggle(Message::UseDEChanged),
                space().width(Length::Fixed(5.0)),
                text("Distance Estimation")
                    .align_y(Alignment::Center),
                space().width(Length::Fixed(15.0)),

                checkbox(self.smooth_coloring)
                    .on_toggle(Message::SmoothColoringChanged),
                space().width(Length::Fixed(5.0)),
                text("Smooth Coloring")
                    .align_y(Alignment::Center),
                space().width(Length::Fixed(15.0)),

                checkbox(self.use_stripes)
                    .on_toggle(Message::UseStripesChanged),
                space().width(Length::Fixed(5.0)),
                text("Stripe Averaging")
                    .align_y(Alignment::Center),
                space().width(Length::Fixed(15.0)),
                
                checkbox(self.debug_coloring)
                    .on_toggle(Message::DebugColoringChanged),
                    space().width(Length::Fixed(5.0)),
                text("Debug Coloring")
                .align_y(Alignment::Center),
            ])
                .style(inner_container_style)
                .padding(10),
            container(row![
                column![
                    row![
                        text("Palette")
                            .width(Length::Fixed(100.0))
                            .align_y(Alignment::Center)
                            .align_x(Alignment::Center),
                        pick_list(self.palettes.clone(),
                            self.selected_palette.clone(),
                            Message::SelectedPaletteChanged)
                            .width(Length::Fixed(220.0)),
                        space().width(Length::Fixed(10.0)),
                        button(text("Import").size(11))
                            .on_press(Message::ImportPalette)
                        ]
                    .padding(5),
                    row![
                        text("Scalar Mapping")
                            .width(Length::Fixed(100.0))
                            .align_y(Alignment::Center)
                            .align_x(Alignment::Center),
                        pick_list(color_scalar_mappings,
                            self.color_scalar_mapping_mode,
                            Message::ColorScalarMappingModeChanged)
                            .width(Length::Fixed(220.0))
                    ].padding(5),
                    row![
                        text("Mapping Strength")
                            .width(Length::Fixed(100.0))
                            .align_y(Alignment::Center)
                            .align_x(Alignment::Center),
                        slider(scalar_mapping_strength_range.0..=scalar_mapping_strength_range.1,
                                self.color_scaler_mapping_strength, Message::ColorScalerMappingStrengthChanged)
                            .step((scalar_mapping_strength_range.1 - scalar_mapping_strength_range.0) / 1000.0)
                            .width(Length::Fixed(160.0)),
                        space().width(Length::Fixed(5.0)),
                        text(format!("{:<4.2}", self.color_scaler_mapping_strength))
                            .width(Length::Fixed(40.0))
                            .align_y(Alignment::Center),
                    ].padding(5),
                ]
                .padding(5),
                column![
                    row![
                        text("cycles: ")
                            .width(Length::Fixed(90.0))
                            .align_y(Alignment::Center)
                            .align_x(Alignment::End),
                        slider(0.0..=self.cycles_max, self.cycles, Message::CyclesChanged)
                            .step(self.cycles_max / 1000.0)
                            .width(Length::Fixed(120.0)),
                        space().width(Length::Fixed(5.0)),
                        text_input("...", &self.cycles_max.to_string())
                            .on_input(Message::CyclesMaxChanged)
                            .width(Length::Fixed(40.0))
                            .align_x(Alignment::Start),
                        space().width(Length::Fixed(10.0)),
                        text(format!("{:<4.2}", self.cycles))
                            .width(Length::Fixed(35.0))
                            .align_y(Alignment::Center)
                    ].padding(5),
                    row![
                        text("offset: ")
                            .width(Length::Fixed(90.0))
                            .align_y(Alignment::Center)
                            .align_x(Alignment::End),
                        slider(0.0..=1.0, self.offset, Message::OffsetChanged)
                            .step(0.001)
                            .width(Length::Fixed(160.0)),
                        space().width(Length::Fixed(5.0)),
                        text(format!("{:<3.2}", self.offset))
                            .align_y(Alignment::Center),
                    ].padding(5),
                    row![
                        text("gamma: ")
                            .width(Length::Fixed(90.0))
                            .align_x(Alignment::End),
                        slider(0.01..=2.0, self.gamma, Message::GammaChanged)
                            .step(0.01)
                            .width(Length::Fixed(160.0)),
                        space().width(Length::Fixed(5.0)),
                        text(format!("{:<3.2}", self.gamma))
                            .width(Length::Fixed(40.0))
                            .align_y(Alignment::Center),
                    ].padding(5),
                ].padding(5),
            ].align_y(Alignment::Center))
                .style(inner_container_style)
                .padding(10),
        ].spacing(5)
            .padding(5);
        
        if self.use_de {
            color_controls = color_controls.push(
                container(self.render_de_controls())
                    .style(inner_container_style)
                    .padding(10),
            );
        }
        if self.use_stripes {
            color_controls = color_controls.push(
                container(
                    row![
                        text("Stripe density")
                            .width(Length::Fixed(70.0))
                            .align_y(Alignment::Center)
                            .align_x(Alignment::Center),
                        space().width(Length::Fixed(5.0)),
                        slider(self.stripe_density_range.0..=self.stripe_density_range.1,
                            self.stripe_density, Message::StripeDensityChanged)
                            .step((self.stripe_density_range.1 - self.stripe_density_range.0) / 1000.0)
                            .width(Length::Fixed(100.0)),
                        space().width(Length::Fixed(5.0)),
                        text(format!("{:<2.1}", self.stripe_density))
                            .width(Length::Fixed(30.0))
                            .align_y(Alignment::Center),

                        text("Stripe strength")
                            .width(Length::Fixed(70.0))
                            .align_y(Alignment::Center)
                            .align_x(Alignment::Center),
                        space().width(Length::Fixed(5.0)),
                        slider(self.stripe_strength_range.0..=self.stripe_strength_range.1,
                            self.stripe_strength, Message::StripeStrengthChanged)
                            .step((self.stripe_strength_range.1 - self.stripe_strength_range.0) / 5000.0)
                            .width(Length::Fixed(100.0)),
                        space().width(Length::Fixed(5.0)),
                        text(format!("{:<3.2}", self.stripe_strength))
                            .width(Length::Fixed(30.0))
                            .align_y(Alignment::Center),

                        text("Stripe gamma")
                            .width(Length::Fixed(70.0))
                            .align_y(Alignment::Center)
                            .align_x(Alignment::Center),
                        space().width(Length::Fixed(5.0)),
                        slider(self.stripe_gamma_range.0..=self.stripe_gamma_range.1,
                            self.stripe_gamma, Message::StripeGammaChanged)
                            .step((self.stripe_gamma_range.1 - self.stripe_gamma_range.0) / 1000.0)
                            .width(Length::Fixed(100.0)),
                        space().width(Length::Fixed(5.0)),
                        text(format!("{:<2.1}", self.stripe_gamma))
                            .width(Length::Fixed(30.0))
                            .align_y(Alignment::Center),
                    ].padding(10)
                )
                    .style(inner_container_style)
                    .padding(10)
            );
        }
        
        color_controls
    }
    fn render_scout_config_row(&self) -> row::Wrapping<'_, Message, Theme, Renderer> {
        row![
            button("Reset Scout")
            .on_press(Message::ResetScoutEngine)
            .width(Length::Fixed(60.0)),

            space().width(Length::Fixed(10.0)),
            text("Ref Iters Multiplier")
                .size(10)
                .width(Length::Fixed(50.0))
                .align_y(Alignment::Center)
                .align_x(Alignment::Center),
            space().width(Length::Fixed(5.0)),
            slider(1.0 ..= 5.0,
                self.ref_iters_multiplier, Message::RefItersMultiplierChanged)
                .step(0.1)
                .width(Length::Fixed(50.0)),
            space().width(Length::Fixed(5.0)),
            text(format!("{:<2.1}", self.ref_iters_multiplier))
                .width(Length::Fixed(30.0))
                .align_y(Alignment::Center),
            space().width(Length::Fixed(5.0)),

            text("Num Seeds per eval")
                .size(10)
                .width(Length::Fixed(50.0))
                .align_y(Alignment::Center)
                .align_x(Alignment::Center),
            space().width(Length::Fixed(5.0)),
            slider(1 ..= 32,
                self.num_samples_to_infer_dir, Message::SamplesToInferDirChanged)
                .step(1_u32)
                .width(Length::Fixed(50.0)),
            space().width(Length::Fixed(5.0)),
            text(format!("{:<2}", self.num_samples_to_infer_dir))
                .width(Length::Fixed(30.0))
                .align_y(Alignment::Center),

            text("Distance error thresh")
                .size(10)
                .width(Length::Fixed(50.0))
                .align_y(Alignment::Center)
                .align_x(Alignment::Center),
            space().width(Length::Fixed(5.0)),
            slider(2.0 ..= 8.0,
                self.distance_error_threshold, Message::DistanceErrorThresholdChanged)
                .step(0.1)
                .width(Length::Fixed(50.0)),
            space().width(Length::Fixed(5.0)),
            text(format!("{:<2.1}", self.distance_error_threshold))
                .width(Length::Fixed(30.0))
                .align_y(Alignment::Center),

            space().width(Length::Fixed(10.0)),
            checkbox(self.auto_start)
                .on_toggle(Message::AutoStartChanged),
            space().width(Length::Fixed(10.0)),
            text("Auto Start")
                .width(Length::Fixed(35.0))
                .align_y(Alignment::Center),

            space().width(Length::Fixed(20.0)),

            button("Scout!")
                .on_press(Message::GoScout)
                .width(Length::Fixed(70.0)),
        ]
            .align_y(Alignment::Center)
            .wrap().vertical_spacing(10)
    }

    fn render_export_config(&self) -> Column<'_, Message, Theme, Renderer> {
        let mut export_controls = column![
            row![
                text("Save Directory:")
                    .width(Length::Fixed(120.0))
                    .align_y(Alignment::Center)
                    .align_x(Alignment::End),
                space().width(Length::Fixed(10.0)),
                text_input("Placeholder...", &self.export_img_dir)
                    .on_input(Message::ExportImgDirChanged)
                    .align_x(Alignment::Start)
                    .width(Length::Fixed(300.0)),
                space().width(Length::Fixed(15.0)),
                text("(file extension will be added automatically)")
                .width(Length::Fixed(100.0))
                    .size(9)
                    .wrapping(Wrapping::Word)
                    .align_y(Alignment::Center),
            ],
            row![
                text("Filename:")
                    .width(Length::Fixed(120.0))
                    .align_y(Alignment::Center)
                    .align_x(Alignment::End),
                space().width(Length::Fixed(10.0)),
                text_input("Placeholder...", &self.export_img_file_name)
                    .on_input(Message::ExportImgFileNameChanged)
                    .align_x(Alignment::Start)
                    .width(Length::Fixed(300.0)),
                space().width(Length::Fixed(25.0)),
                radio("Png", ExportImgFormat::Png, self.export_img_format, Message::ExportImgFormatChanged),
                space().width(Length::Fixed(15.0)),
                radio("Jpeg", ExportImgFormat::Jpeg, self.export_img_format, Message::ExportImgFormatChanged),

            ]
        ];

        if let Some(img_fmt) = self.export_img_format && img_fmt == ExportImgFormat::Png {
            export_controls = export_controls.push(
                row![
                    text("Compression:")
                    .align_y(Alignment::Center),
                    space().width(Length::Fixed(20.0)),
                    radio("None", PngCompression::None, self.export_img_png_compression, Message::ExportImgPngCompressionChanged),
                    space().width(Length::Fixed(15.0)),
                    radio("Fast", PngCompression::Fast, self.export_img_png_compression, Message::ExportImgPngCompressionChanged),
                    space().width(Length::Fixed(15.0)),
                    radio("Default", PngCompression::Default, self.export_img_png_compression, Message::ExportImgPngCompressionChanged),
                    space().width(Length::Fixed(15.0)),
                    radio("Best", PngCompression::Best, self.export_img_png_compression, Message::ExportImgPngCompressionChanged)
                ]
            );
        }
        else if let Some(img_fmt) = self.export_img_format && img_fmt == ExportImgFormat::Jpeg {
            export_controls = export_controls.push(
                row![
                    text("Quality").align_y(Alignment::Center),
                    space().width(Length::Fixed(20.0)),
                    slider(10..=100,
                       self.export_img_jpeg_quality, Message::ExportImgJpegQualityChanged)
                    .step(1)
                    .width(Length::Fixed(75.0)),
                    space().width(Length::Fixed(10.0)),
                    text(format!("{:<3}", self.export_img_jpeg_quality))
                    .width(Length::Fixed(30.0))
                    .align_y(Alignment::Center),
                ]
            );
        }

        let mut dimensions_row = row![
            checkbox(self.use_alternate_dimensions)
                .on_toggle(Message::UseAlternateDimensionsChanged),
            space().width(Length::Fixed(10.0)),
            text("Use alternate dimensions (from current window size)")
                .size(9)
                .width(Length::Fixed(100.0))
                .wrapping(Wrapping::Word)
                .align_y(Alignment::Center),
            space().width(Length::Fixed(10.0))
        ];
        if self.use_alternate_dimensions {
            dimensions_row = dimensions_row.push(
                text("width").align_y(Alignment::Center)
            );
            dimensions_row = dimensions_row.push(
                space().width(Length::Fixed(10.0))
            );
            dimensions_row = dimensions_row.push(
                text_input("Placeholder...", &self.export_img_width)
                    .on_input(Message::ExportImgWidthChanged)
                    .width(Length::Fixed(75.0)),
            );
            dimensions_row = dimensions_row.push(
                space().width(Length::Fixed(10.0))
            );
            dimensions_row = dimensions_row.push(
                text("height").align_y(Alignment::Center)
            );
            dimensions_row = dimensions_row.push(
                space().width(Length::Fixed(10.0))
            );
            dimensions_row = dimensions_row.push(
                text_input("Placeholder...", &self.export_img_height)
                    .on_input(Message::ExportImgHeightChanged)
                    .width(Length::Fixed(75.0)),
            );
            dimensions_row = dimensions_row.push(
                space().width(Length::Fixed(5.0))
            );
        }

        dimensions_row = dimensions_row.push(
            checkbox(self.append_timestamp_to_filename)
                .on_toggle(Message::AppendTimestampToFileNameChanged)
        );
        dimensions_row = dimensions_row.push(
            space().width(Length::Fixed(10.0))
        );
        dimensions_row = dimensions_row.push(
            text("Append timestamp to filename")
                .width(Length::Fixed(70.0))
                .size(9)
                .wrapping(Wrapping::Word)
                .align_y(Alignment::Center),
        );
        dimensions_row = dimensions_row.push(
            space().width(Length::Fixed(10.0))
        );

        dimensions_row = dimensions_row.push(
            button("Save")
                .on_press(Message::ExportImage)
                .width(Length::Fixed(65.0)),
        );

        export_controls = export_controls.push(
            dimensions_row
        );
        export_controls
            .spacing(5)
    }
    
    fn render_de_controls(&self) -> Column<'_, Message, Theme, Renderer> {
        let mut glow_row = row![
            checkbox(self.enable_glow)
                    .on_toggle(Message::EnableGlowChanged),
                space().width(Length::Fixed(5.0)),
                text("Glow")
                    .align_y(Alignment::Center),
                space().width(Length::Fixed(15.0))
        ].padding(5);
        if self.enable_glow {
            glow_row = glow_row.push(
                text("intensity: ")
                    .align_y(Alignment::Center));
            glow_row = glow_row.push(
                slider(0.0..=1.0, self.glow_intensity, Message::GlowIntensityChanged)
                    .step(0.01)
                    .width(Length::Fixed(120.0)));
            glow_row = glow_row.push(
                space().width(Length::Fixed(5.0)));
            glow_row = glow_row.push(
                text(format!("{:<3.2}", self.glow_intensity))
                    .align_y(Alignment::Center));
        }
        
        let mut de_controls = column![
            container(glow_row)
                .style(inner_container_style)
                .padding(10),
            container(row![
                text("Ambient intensity")
                    .width(Length::Fixed(70.0))
                    .align_x(Alignment::Center)
                    .align_y(Alignment::Center),
                space().width(Length::Fixed(5.0)),
                slider(0.0..=2.0, self.ambient_intensity, Message::AmbientIntensityChanged)
                    .step(0.01)
                    .width(Length::Fixed(120.0)),
                space().width(Length::Fixed(5.0)),
                text(format!("{:<3.2}", self.ambient_intensity))
                    .width(Length::Fixed(30.0))
                    .align_y(Alignment::Center),
                space().width(Length::Fixed(15.0)),
                
                text("Neighbor normal scale")
                    .width(Length::Fixed(130.0))
                    .align_x(Alignment::Center)
                    .align_y(Alignment::Center),
                space().width(Length::Fixed(5.0)),
                slider(self.neighbor_scale_range.0..=self.neighbor_scale_range.1,
                    self.neighbor_scale_multiplier, Message::NeighborScaleChanged)
                    .step(1.0)
                    .width(Length::Fixed(60.0)),
                space().width(Length::Fixed(5.0)),
                text(format!("{:<3.2}", self.neighbor_scale_multiplier))
                    .width(Length::Fixed(30.0))
                    .align_y(Alignment::Center),
                space().width(Length::Fixed(15.0)),
                
                text("Distance multiplier")
                    .width(Length::Fixed(70.0))
                    .align_x(Alignment::Center)
                    .align_y(Alignment::Center),
                space().width(Length::Fixed(5.0)),
                slider(self.distance_multiplier_range.0..=self.distance_multiplier_range.1,
                    self.distance_multiplier, Message::DistanceMultiplierChanged)
                    .step((self.distance_multiplier_range.1 - self.distance_multiplier_range.0) / 1000.0)
                    .width(Length::Fixed(60.0)),
                space().width(Length::Fixed(5.0)),
                text(format!("{:<3.1}", self.distance_multiplier))
                    .width(Length::Fixed(40.0))
                    .align_y(Alignment::Center),
            ])
                .style(inner_container_style)
                .padding(10),
            container(row![
                checkbox(self.enable_key_light)
                    .on_toggle(Message::EnableKeyLightChanged),
                space().width(Length::Fixed(5.0)),
                text("Key Light")
                    .align_y(Alignment::Center),
                space().width(Length::Fixed(15.0)),

                checkbox(self.enable_fill_light)
                    .on_toggle(Message::EnableFillLightChanged),
                space().width(Length::Fixed(5.0)),
                text("Fill Light")
                    .align_y(Alignment::Center),
                space().width(Length::Fixed(15.0)),

                checkbox(self.enable_specular)
                    .on_toggle(Message::EnableSpecularChanged),
                space().width(Length::Fixed(5.0)),
                text("Specular")
                    .align_y(Alignment::Center),
                space().width(Length::Fixed(15.0)),
                
                checkbox(self.enable_ao)
                    .on_toggle(Message::EnableAoChanged),
                    space().width(Length::Fixed(5.0)),
                text("AO")
                    .align_y(Alignment::Center),
                space().width(Length::Fixed(15.0)),

                checkbox(self.enable_rim)
                    .on_toggle(Message::EnableRimChanged),
                    space().width(Length::Fixed(5.0)),
                text("Rim")
                    .align_y(Alignment::Center),
            ])
                .style(inner_container_style)
                .padding(10),
        ].spacing(5).padding(5);
        if self.enable_key_light {
            de_controls = de_controls.push(
                container(row![
                    text("Key Light intensity: ")
                        .width(Length::Fixed(150.0))
                        .align_x(Alignment::End)
                        .align_y(Alignment::Center),
                    slider(0.0..=2.0, self.key_light_intensity, Message::KeyLightIntensityChanged)
                        .step(0.01)
                        .width(Length::Fixed(60.0)),
                    space().width(Length::Fixed(5.0)),
                    text(format!("{:<3.2}", self.key_light_intensity))
                        .width(Length::Fixed(30.0))
                        .align_y(Alignment::Center),
                    space().width(Length::Fixed(20.0)),
                    
                    text("azimuth: ")
                        .align_y(Alignment::Center),
                    slider(0.0..=360.0, self.key_light_azimuth, Message::KeyLightAzimuthChanged)
                        .step(1.0)
                        .width(Length::Fixed(80.0)),
                    space().width(Length::Fixed(5.0)),
                    text(format!("{:<3}", self.key_light_azimuth))
                        .width(Length::Fixed(30.0))
                        .align_y(Alignment::Center),
                    space().width(Length::Fixed(20.0)),
                    
                    text("elevation: ")
                        .align_y(Alignment::Center),
                    slider(0.0..=90.0, self.key_light_elevation, Message::KeyLightElevationChanged)
                        .step(1.0)
                        .width(Length::Fixed(80.0)),
                    space().width(Length::Fixed(5.0)),
                    text(format!("{:<3}", self.key_light_elevation))
                        .width(Length::Fixed(30.0))
                        .align_y(Alignment::Center)
                ])
                    .style(inner_container_style)
                    .padding(10)
            );
        }
        if self.enable_fill_light {
            de_controls = de_controls.push(
                container(row![
                    text("Fill Light intensity: ")
                        .width(Length::Fixed(150.0))
                        .align_x(Alignment::End)
                        .align_y(Alignment::Center),
                    slider(0.0..=2.0, self.fill_light_intensity, Message::FillLightIntensityChanged)
                        .step(0.01)
                        .width(Length::Fixed(60.0)),
                    space().width(Length::Fixed(5.0)),
                    text(format!("{:<3.2}", self.fill_light_intensity))
                        .width(Length::Fixed(30.0))
                        .align_y(Alignment::Center),
                    space().width(Length::Fixed(20.0)),
                    
                    text("azimuth: ")
                        .align_y(Alignment::Center),
                    slider(0.0..=360.0, self.fill_light_azimuth, Message::FillLightAzimuthChanged)
                        .step(1.0)
                        .width(Length::Fixed(80.0)),
                    space().width(Length::Fixed(5.0)),
                    text(format!("{:<3}", self.fill_light_azimuth))
                        .width(Length::Fixed(30.0))
                        .align_y(Alignment::Center),
                    space().width(Length::Fixed(20.0)),
                    
                    text("elevation: ")
                        .align_y(Alignment::Center),
                    slider(0.0..=90.0, self.fill_light_elevation, Message::FillLightElevationChanged)
                        .step(1.0)
                        .width(Length::Fixed(80.0)),
                    space().width(Length::Fixed(5.0)),
                    text(format!("{:<3}", self.fill_light_elevation))
                        .width(Length::Fixed(30.0))
                        .align_y(Alignment::Center)
                ])
                    .style(inner_container_style)
                    .padding(10)
            );
        }
        if self.enable_specular {
            de_controls = de_controls.push(
                container(row![
                    text("Specular intensity: ")
                        .width(Length::Fixed(150.0))
                        .align_x(Alignment::End)
                        .align_y(Alignment::Center),
                    slider(0.0..=2.0, self.specular_intensity, Message::SpecularIntensityChanged)
                        .step(0.01)
                        .width(Length::Fixed(60.0)),
                    space().width(Length::Fixed(5.0)),
                    text(format!("{:<3.2}", self.specular_intensity))
                        .width(Length::Fixed(30.0))
                        .align_y(Alignment::Center),
                    space().width(Length::Fixed(20.0)),

                    text("power: ")
                        .align_y(Alignment::Center),
                    slider(self.specular_power_range.0..=self.specular_power_range.1,
                        self.specular_power, Message::SpecularPowerChanged)
                        .step((self.specular_power_range.1 - self.specular_power_range.0) / 1000.0)
                        .width(Length::Fixed(100.0)),
                    space().width(Length::Fixed(5.0)),
                    text(format!("{:<4.1}", self.specular_power))
                        .width(Length::Fixed(40.0))
                        .align_y(Alignment::Center),
                ])
                    .style(inner_container_style)
                    .padding(10)
            );
        }
        if self.enable_ao {
            de_controls = de_controls.push(
                container(row![
                    text("AO darkness: ")
                        .width(Length::Fixed(150.0))
                        .align_x(Alignment::End)
                        .align_y(Alignment::Center),
                    slider(0.0..=0.8, self.ao_darkness, Message::AoDarknessChanged)
                        .step(0.001)
                        .width(Length::Fixed(120.0)),
                    space().width(Length::Fixed(5.0)),
                    text(format!("{:<4.3}", self.ao_darkness))
                        .align_y(Alignment::Center),
                ])
                    .style(inner_container_style)
                    .padding(10)
            );
        }
        if self.enable_rim {
            de_controls = de_controls.push(
                container(row![
                    text("Rim intensity: ")
                    .width(Length::Fixed(120.0))
                        .align_x(Alignment::End)
                        .align_y(Alignment::Center),
                    slider(0.0..=4.0, self.rim_intensity, Message::RimIntensityChanged)
                        .step(0.01)
                        .width(Length::Fixed(60.0)),
                    space().width(Length::Fixed(5.0)),
                    text(format!("{:<3.2}", self.rim_intensity))
                        .width(Length::Fixed(30.0))
                        .align_y(Alignment::Center),
                    space().width(Length::Fixed(20.0)),

                    text("power: ")
                        .align_y(Alignment::Center),
                    slider(self.rim_power_range.0..=self.rim_power_range.1,
                        self.rim_power, Message::RimPowerChanged)
                        .step((self.rim_power_range.1 - self.rim_power_range.0) / 1000.0)
                        .width(Length::Fixed(100.0)),
                    space().width(Length::Fixed(5.0)),
                    text(format!("{:<6.3}", self.rim_power))
                        .width(Length::Fixed(40.0))
                        .align_y(Alignment::Center),
                ])
                    .style(inner_container_style)
                    .padding(10)
            );
        }
        
        de_controls
    }
}

fn outer_container_style(theme: &Theme) -> container::Style {
    let palette = theme.extended_palette();

    container::Style {
        background: Some(palette.background.neutral.color.scale_alpha(0.65).into()),
        text_color: Some(Color::WHITE),
        border: border::rounded(10).width(1).color(Color::WHITE),
        ..container::Style::default()
    }
}

fn inner_container_style(theme: &Theme) -> container::Style {
    let palette = theme.extended_palette();

    container::Style {
        background: Some(palette.background.strong.color.scale_alpha(0.80).into()),
        text_color: Some(Color::WHITE),
        border: border::rounded(5).width(1).color(color!(0x111111)),
        ..container::Style::default()
    }
}