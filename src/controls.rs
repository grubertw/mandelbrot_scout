use super::scene::Scene;

use iced_wgpu::Renderer;
use iced_widget::{text_input, column, row, text, button, space, Row, slider, pick_list, checkbox};
use iced_widget::core::{Alignment, Color, Element, Theme, Length};

use log::{trace};

use std::rc::Rc;
use std::cell::RefCell;
use iced_wgpu::core::text::Wrapping;
use rug::{Complex, Float};
use crate::scout_engine::ScoutSignal;

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

#[derive(Clone)]
pub struct Controls {
    // Bools that show/hide rows
    editing_iters: bool,
    editing_location: bool,
    editing_color: bool,
    editing_scout: bool,

    // Iterations config
    iter_step: u32,
    iter_range_min: u32,
    iter_range_max: u32,
    max_iterations: u32,

    // Location config
    center_x: String,
    center_y: String,
    scale: String,

    // Color config
    palettes: Vec<PaletteSelection>,
    selected_palette: Option<PaletteSelection>,
    frequency: f32,
    frequency_min: f32,
    frequency_max: f32,
    offset: f32,
    gamma: f32,
    debug_coloring: bool,

    // Scout config
    ref_iters_multiplier: String,
    spawn_per_eval: String,
    num_qualified_orbits: String,
    glitch_fix: bool,

    debug_msg: String,
    scene: Rc<RefCell<Scene>>
}

#[derive(Debug, Clone)]
pub enum Message {
    EditingItersChanged(bool),
    EditingLocationChanged(bool),
    EditingColorChanged(bool),
    EditingScoutConfigChanged(bool),

    IterStepChanged(String),
    IterRangeMinChanged(String),
    IterRangeMaxChanged(String),
    IterValueChanged(u32),

    CenterXChanged(String),
    CenterYChanged(String),
    ScaleChanged(String),
    PollFromScene,
    ApplyCenterScale,

    SelectedPaletteChanged(PaletteSelection),
    FrequencyChanged(f32),
    OffsetChanged(f32),
    GammaChanged(f32),
    DebugColoringChanged(bool),

    ResetScoutEngine,
    GoScout,
    RefItersMultiplierChanged(String),
    SpawnPerEvalChanged(String),
    NumQualifiedOrbits(String),
    GlitchFixChanged(bool),

    UpdateDebugText(String),
}

impl Controls {
    pub fn new(s: Rc<RefCell<Scene>>) -> Controls {
        let scene_b = s.borrow();
        let center = scene_b.center().clone();
        let center_x = center.real().to_string_radix(10, Some(10));
        let center_y = center.imag().to_string_radix(10, Some(10));
        let scale = scene_b.scale().to_string_radix(10, Some(6));
        let max_iters = scene_b.max_iterations();
        let sc = scene_b.scout_config();
        let scout_cfg_g = sc.lock();

        let mut palettes: Vec<PaletteSelection> = scene_b.get_palette_list()
            .iter()
            .map(|( key, name)|
                PaletteSelection::new(key.clone(), name.clone())
            ).collect();
        palettes.sort_by(|a, b| a.key.cmp(&b.key));

        let palette_selection =
            PaletteSelection::new("default".to_string(), "Default".to_string());
        let palette = scene_b.get_palette(&palette_selection.key);

        Controls {
            editing_iters: false, editing_location: false, editing_color: false, editing_scout: false,
            iter_step: 10,
            iter_range_min: 0, iter_range_max: max_iters * 2,
            max_iterations: max_iters,
            center_x, center_y, scale,
            palettes,
            selected_palette: Some(palette_selection),
            frequency: palette.frequency,
            frequency_min: palette.frequency_range.0,
            frequency_max: palette.frequency_range.1,
            offset: 0.0, gamma: 1.0, debug_coloring: false,
            ref_iters_multiplier:  scout_cfg_g.ref_iters_multiplier.to_string(),
            spawn_per_eval: scout_cfg_g.num_seeds_to_spawn_per_eval.to_string(),
            num_qualified_orbits: scout_cfg_g.num_qualified_orbits.to_string(),
            glitch_fix: false,
            debug_msg: "debug info loading...".to_string(),
            scene: s.clone()
        }
    }

    pub fn update(&mut self, message: Message) {
        trace!("Update {:?}", message);
        match message {
            Message::EditingItersChanged(toggle) => {
                self.editing_iters = toggle;
            }
            Message::EditingLocationChanged(toggle) => {
                self.editing_location = toggle;
            }
            Message::EditingColorChanged(toggle) => {
                self.editing_color = toggle;
            }
            Message::EditingScoutConfigChanged(toggle) => {
                self.editing_scout = toggle;
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
            Message::SelectedPaletteChanged(selected_palette) => {
                let key = selected_palette.key.clone();
                let mut scene_b = self.scene.borrow_mut();
                self.selected_palette = Some(selected_palette);
                scene_b.set_selected_palette(&key);
                // Must overwrite GUI settings with stored values from Scene
                let incoming_palette = scene_b.get_palette(&key);
                self.frequency = incoming_palette.frequency;
                self.frequency_min = incoming_palette.frequency_range.0;
                self.frequency_max = incoming_palette.frequency_range.1;
                self.offset = incoming_palette.offset;
                self.gamma = incoming_palette.gamma;
            }
            Message::FrequencyChanged(frequency) => {
                self.frequency = frequency;
                if let Some(palette) = self.selected_palette.as_ref() {
                    self.scene.borrow_mut().set_palette_frequency(&palette.key, frequency);
                }
            }
            Message::OffsetChanged(offset) => {
                self.offset = offset;
                if let Some(palette) = self.selected_palette.as_ref() {
                    self.scene.borrow_mut().set_palette_offset(&palette.key, offset);
                }
            }
            Message::GammaChanged(gamma) => {
                self.gamma = gamma;
                if let Some(palette) = self.selected_palette.as_ref() {
                    self.scene.borrow_mut().set_palette_gamma(&palette.key, gamma);
                }
            }
            Message::DebugColoringChanged(coloring) => {
                self.debug_coloring = coloring;
                self.scene.borrow_mut().set_debug_coloring(coloring);
            }
            Message::ResetScoutEngine => {
                self.scene.borrow_mut().send_scout_signal(ScoutSignal::ResetEngine);
            }
            Message::GoScout => {
                let mut scene_b = self.scene.borrow_mut();
                let mut config = scene_b.scout_config().lock().clone();
                config.ref_iters_multiplier = if let Ok(v) =
                    self.ref_iters_multiplier.parse::<f64>() { v } else { 0.0 };
                config.num_seeds_to_spawn_per_eval = if let Ok(v) =
                    self.spawn_per_eval.parse::<u32>() { v } else { 0 };
                config.num_qualified_orbits = if let Ok(v) =
                    self.num_qualified_orbits.parse::<u32>() { v } else { 0 };

                scene_b.send_scout_signal(ScoutSignal::ExploreSignal(config));
            }
            Message::RefItersMultiplierChanged(ref_iters_multiplier) => {
                self.ref_iters_multiplier = ref_iters_multiplier;
            }
            Message::SpawnPerEvalChanged(spawn_per_tile) => {
                self.spawn_per_eval = spawn_per_tile;
            }
            Message::NumQualifiedOrbits(num_qualified_orbits) => {
                self.num_qualified_orbits = num_qualified_orbits;
            }
            Message::GlitchFixChanged(glitch_fix) => {
                self.glitch_fix = glitch_fix;
                self.scene.borrow_mut().set_glitch_fix(glitch_fix);
            }
            Message::UpdateDebugText(dbg_msg) => {
                self.debug_msg = dbg_msg;
            }
        }
    }

    pub fn view(&self) -> Element<'_, Message, Theme, Renderer> {
        trace!("View");

        let dbg_row = row![
            text(&self.debug_msg).color(Color::WHITE)
            .wrapping(Wrapping::Word)
            .size(10)
        ]
        .align_y(Alignment::Start);

        let toggles_row = row![
            button("Iterations")
            .on_press(Message::EditingItersChanged(!self.editing_iters))
            .style(if self.editing_iters {button::primary} else {button::text}),
            space().width(Length::Fixed(20.0)),
            button("Location")
            .on_press(Message::EditingLocationChanged(!self.editing_location))
            .style(if self.editing_location {button::primary} else {button::text}),
            space().width(Length::Fixed(20.0)),
            button("Color")
            .on_press(Message::EditingColorChanged(!self.editing_color))
            .style(if self.editing_color {button::primary} else {button::text}),
            space().width(Length::Fixed(20.0)),
            button("Scout")
            .on_press(Message::EditingScoutConfigChanged(!self.editing_scout))
            .style(if self.editing_scout {button::primary} else {button::text}),
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
            primary_panel = primary_panel.push(self.render_iterations_row());
        }
        if self.editing_location {
            primary_panel = primary_panel.push(self.render_edit_location_row())
        }
        if self.editing_color {
            primary_panel = primary_panel.push(self.render_colors_row());
        }
        if self.editing_scout {
            primary_panel = primary_panel.push(self.render_scout_config_row());
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
                .color(Color::WHITE),
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
            space().width(Length::Fixed(30.0)),
            text("max iterations: ")
                .color(Color::WHITE)
                .size(16),
            text(self.max_iterations.to_string())
                .color(Color::WHITE)
                .size(15),
        ]
            .padding(5.0)
    }

    fn render_edit_location_row(&self) -> Row<'_, Message, Theme, Renderer> {
        row![
            button("Poll")
            .on_press(Message::PollFromScene)
            .width(Length::Fixed(50.0))
            .height(Length::Shrink),

            text("real: ")
            .color(Color::WHITE)
            .width(Length::Fixed(60.0))
            .align_x(Alignment::End),
            text_input("Placeholder...", &self.center_x)
                .on_input(Message::CenterXChanged)
                .width(Length::Fixed(140.0)),

            text("imag: ")
            .color(Color::WHITE)
            .width(Length::Fixed(60.0))
            .align_x(Alignment::End),
            text_input("Placeholder...", &self.center_y)
                .on_input(Message::CenterYChanged)
                .width(Length::Fixed(140.0)),

            text("scale: ")
            .color(Color::WHITE)
            .width(Length::Fixed(60.0))
            .align_x(Alignment::End),
            text_input("Placeholder...", &self.scale)
                .on_input(Message::ScaleChanged)
                .width(Length::Fixed(110.0)),

            space().width(Length::Fixed(20.0)),

            button("Apply")
            .on_press(Message::ApplyCenterScale)
            .width(Length::Fixed(65.0))
            .height(Length::Shrink),
        ]
            .padding(5.0)
    }

    fn render_colors_row(&self) -> row::Wrapping<'_, Message, Theme, Renderer> {
        row![
            text("palette: ")
                .color(Color::WHITE),
            pick_list(self.palettes.clone(), self.selected_palette.clone(), Message::SelectedPaletteChanged)
                .placeholder("Select palette"),
            text("frequency: ")
                .color(Color::WHITE)
                .width(Length::Fixed(90.0))
                .align_x(Alignment::End),
            slider(self.frequency_min..=self.frequency_max, self.frequency, Message::FrequencyChanged)
                .step((self.frequency_max - self.frequency_min) / 1000.0)
                .width(Length::Fixed(180.0)),
            space().width(Length::Fixed(5.0)),
            text(format!("{:<4.3}", self.frequency))
                .color(Color::WHITE)
                .width(Length::Fixed(35.0))
                .align_x(Alignment::Start),

            text("offset: ")
                .color(Color::WHITE)
                .width(Length::Fixed(70.0))
                .align_x(Alignment::End),
            slider(0.0..=1.0, self.offset, Message::OffsetChanged)
                .step(0.001)
                .width(Length::Fixed(70.0)),
            space().width(Length::Fixed(5.0)),
            text(format!("{:<3.2}", self.offset))
                .color(Color::WHITE)
                .width(Length::Fixed(30.0))
                .align_x(Alignment::Start),

            text("gamma: ")
                .color(Color::WHITE)
                .width(Length::Fixed(75.0))
                .align_x(Alignment::End),
            slider(0.3..=2.5, self.gamma, Message::GammaChanged)
                .step(0.001)
                .width(Length::Fixed(80.0)),
            space().width(Length::Fixed(5.0)),
            text(format!("{:<3.2}", self.gamma))
                .color(Color::WHITE)
                .width(Length::Fixed(40.0))
                .align_x(Alignment::Start),
            space().width(Length::Fixed(15.0)),

            checkbox(self.debug_coloring)
                .on_toggle(Message::DebugColoringChanged),
            space().width(Length::Fixed(5.0)),
            text("Debug Coloring")
                .color(Color::WHITE)
        ]
            .padding(5.0)
            .align_y(Alignment::End)
            .wrap().vertical_spacing(10)
    }
    fn render_scout_config_row(&self) -> Row<'_, Message, Theme, Renderer> {
        row![
            button("Reset Scout")
            .on_press(Message::ResetScoutEngine)
            .width(Length::Fixed(110.0))
            .height(Length::Shrink),

            text("ref iters multiplier: ")
            .color(Color::WHITE)
            .width(Length::Fixed(65.0))
            .wrapping(Wrapping::Word)
            .size(12)
            .align_x(Alignment::End),
            space().width(Length::Fixed(5.0)),
            text_input("Placeholder...", &self.ref_iters_multiplier)
                .on_input(Message::RefItersMultiplierChanged)
                .width(Length::Fixed(60.0)),

            text("spawn count per eval: ")
            .color(Color::WHITE)
            .width(Length::Fixed(75.0))
            .wrapping(Wrapping::Word)
            .size(12)
            .align_x(Alignment::End),
            space().width(Length::Fixed(5.0)),
            text_input("Placeholder...", &self.spawn_per_eval)
                .on_input(Message::SpawnPerEvalChanged)
                .width(Length::Fixed(40.0)),

            text("max ref orbs: ")
            .color(Color::WHITE)
            .width(Length::Fixed(70.0))
            .wrapping(Wrapping::Word)
            .size(12)
            .align_x(Alignment::End),
            space().width(Length::Fixed(5.0)),
            text_input("Placeholder...", &self.num_qualified_orbits)
                .on_input(Message::NumQualifiedOrbits)
                .width(Length::Fixed(30.0)),
            space().width(Length::Fixed(10.0)),

            checkbox(self.glitch_fix)
                .on_toggle(Message::GlitchFixChanged),
            space().width(Length::Fixed(5.0)),
            text("Rebase using same RefOrb")
                .color(Color::WHITE)
                .width(Length::Fixed(65.0))
                .wrapping(Wrapping::Word)
                .size(11),

            space().width(Length::Fixed(20.0)),

            button("Scout!")
            .on_press(Message::GoScout)
            .width(Length::Fixed(70.0))
            .height(Length::Shrink),
        ]
            .padding(5.0)
    }
}