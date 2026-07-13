//! The Custom Palette Editor's second window.
//!
//! Two pieces:
//! - [`PaletteEditor`]: the Iced UI state (a working [`EditablePalette`] over the
//!   scene's currently-selected palette). This is the palette-editor analogue of
//!   `controls::Controls`.
//! - [`PaletteWindow`]: the winit/wgpu/iced resource bundle for the editor's own
//!   window, mirroring the manual Iced integration in `main.rs`. It shares the
//!   scene's `wgpu::Device`/`Queue` (same thread, `Rc<RefCell<Scene>>`).
//!
//! Step 3 scope (see `docs/palette_editor.md`): open/close the window, load the
//! selected palette, and prove the live-recolor loop with a real `Reverse` edit.
//! The stop list, preview, graphs, and generators arrive in later steps.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use iced_wgpu::graphics::{Shell, Viewport};
use iced_wgpu::{Engine, Renderer};
use iced_widget::canvas::{self, Action, Canvas, LineDash, Path, Program, Stroke};
use iced_widget::{button, column, container, pick_list, row, scrollable, slider, space, text, Column};
use iced_widget::core::{
    border, Alignment, Background, Border, Color, Element, Length, Point, Rectangle, Theme,
};
use strum::IntoEnumIterator;
use iced_winit::core::renderer;
use iced_winit::core::{window, Event, Font, Pixels, Size};
use iced_winit::core::mouse;
use iced_winit::runtime::user_interface::{self, UserInterface};
use iced_winit::{conversion, Clipboard};

use iced_winit::winit;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::ModifiersState;
use winit::window::{Window, WindowAttributes, WindowId};

use iced_wgpu::wgpu;

use rfd::FileDialog;

use crate::palette_editor::{EditablePalette, Stop};
use crate::palette_generators::{self, hsv_to_rgb, rgb_to_hsv, GenParams, Generator};
use crate::scene::Scene;

/// Cleared background of the editor window (dark instrument surround).
const EDITOR_BG: Color = Color::from_rgb(0.11, 0.12, 0.14);

/// Subtle boxed panel used to group sections of the editor.
fn panel_style(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(Color::from_rgba(1.0, 1.0, 1.0, 0.03))),
        border: border::rounded(8).width(1.0).color(Color::from_rgba(1.0, 1.0, 1.0, 0.12)),
        ..container::Style::default()
    }
}

// ---------------------------------------------------------------------------
// Canvas widgets: preview bar + R/G/B channel graphs (display-only in v1).
// ---------------------------------------------------------------------------

fn u8_color(c: [u8; 4]) -> Color {
    Color::from_rgb(c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0)
}

fn lerp_color(a: [u8; 4], b: [u8; 4], t: f32) -> Color {
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) / 255.0;
    Color::from_rgb(mix(a[0], b[0]), mix(a[1], b[1]), mix(a[2], b[2]))
}

/// Color of stop `i`, or opaque black if out of range / empty.
fn stop_color(p: &EditablePalette, i: usize) -> [u8; 4] {
    p.stops.get(i).map(|s| s.color).unwrap_or([0, 0, 0, 255])
}

/// Horizontal gradient of the flattened palette (linear interpolation, matching
/// the GPU's default `mix`; one vertical strip per pixel column).
struct PalettePreview {
    colors: Vec<[u8; 4]>,
}

impl<Message> Program<Message, Theme, Renderer> for PalettePreview {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let w = bounds.width;
        let h = bounds.height;
        let n = self.colors.len();

        if n == 1 {
            frame.fill_rectangle(Point::new(0.0, 0.0), bounds.size(), u8_color(self.colors[0]));
        } else if n >= 2 {
            let cols = w.ceil().max(1.0) as usize;
            let span = (n - 1) as f32;
            for x in 0..cols {
                let t = x as f32 / (w - 1.0).max(1.0);
                let fpos = (t * span).clamp(0.0, span);
                let i0 = fpos.floor() as usize;
                let i1 = (i0 + 1).min(n - 1);
                let frac = fpos - i0 as f32;
                let color = lerp_color(self.colors[i0], self.colors[i1], frac);
                frame.fill_rectangle(Point::new(x as f32, 0.0), Size::new(1.0, h), color);
            }
        }

        vec![frame.into_geometry()]
    }
}

/// Three stacked R/G/B channel graphs (value 0..255 across the palette index).
/// `boundaries` are flattened-slot indices where a new stop begins (drawn as
/// faint dotted verticals so the stop structure is visible on the curves).
struct ChannelGraphs {
    colors: Vec<[u8; 4]>,
    boundaries: Vec<usize>,
}

impl ChannelGraphs {
    const GAP: f32 = 4.0; // background gap between the three bands
    const PAD: f32 = 4.0; // inner padding so min/max lines don't hug the edges
    const MAX_BOUNDARY_LINES: usize = 48; // skip stop markers on dense palettes
}

impl<Message> Program<Message, Theme, Renderer> for ChannelGraphs {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let w = bounds.width;
        let h = bounds.height;
        let n = self.colors.len();
        let band_h = ((h - 2.0 * Self::GAP) / 3.0).max(1.0);

        let band_bg = Color::from_rgba(1.0, 1.0, 1.0, 0.04);
        let box_col = Color::from_rgba(0.6, 0.6, 0.65, 0.5); // weak gray data box
        let bound_col = Color::from_rgba(1.0, 1.0, 1.0, 0.22); // weak dotted stop marks
        let chan_col = [
            Color::from_rgb(0.95, 0.35, 0.35), // R
            Color::from_rgb(0.35, 0.85, 0.45), // G
            Color::from_rgb(0.45, 0.60, 1.00), // B
        ];
        let dash = [2.0f32, 3.0];
        let show_bounds = n >= 2 && self.boundaries.len() <= Self::MAX_BOUNDARY_LINES;

        for ci in 0..3usize {
            let band_top = ci as f32 * (band_h + Self::GAP);
            frame.fill_rectangle(Point::new(0.0, band_top), Size::new(w, band_h), band_bg);

            // Inset data rect (gives the curves headroom at 0 and 255).
            let dx = Self::PAD;
            let dy = band_top + Self::PAD;
            let dw = (w - 2.0 * Self::PAD).max(1.0);
            let dh = (band_h - 2.0 * Self::PAD).max(1.0);

            // Faint gray box around the plotted area.
            frame.stroke(
                &Path::rectangle(Point::new(dx, dy), Size::new(dw, dh)),
                Stroke::default().with_color(box_col).with_width(1.0),
            );

            // Dotted verticals at stop boundaries.
            if show_bounds {
                for &b in &self.boundaries {
                    let x = dx + (b as f32) / (n as f32 - 1.0) * dw;
                    frame.stroke(
                        &Path::line(Point::new(x, dy), Point::new(x, dy + dh)),
                        Stroke {
                            line_dash: LineDash { segments: &dash, offset: 0 },
                            ..Stroke::default().with_color(bound_col).with_width(1.0)
                        },
                    );
                }
            }

            // Channel polyline (value 1.0 at the top of the data rect).
            let stroke = Stroke::default().with_color(chan_col[ci]).with_width(1.5);
            let y_of = |v: u8| dy + dh - (v as f32 / 255.0) * dh;

            if n == 1 {
                let y = y_of(self.colors[0][ci]);
                frame.stroke(&Path::line(Point::new(dx, y), Point::new(dx + dw, y)), stroke);
            } else if n >= 2 {
                let path = Path::new(|b| {
                    for (i, c) in self.colors.iter().enumerate() {
                        let x = dx + (i as f32) / (n as f32 - 1.0) * dw;
                        let y = y_of(c[ci]);
                        if i == 0 {
                            b.move_to(Point::new(x, y));
                        } else {
                            b.line_to(Point::new(x, y));
                        }
                    }
                });
                frame.stroke(&path, stroke);
            }
        }

        vec![frame.into_geometry()]
    }
}

/// Drag state shared by the interactive picker canvases.
#[derive(Default)]
struct DragState {
    dragging: bool,
}

/// Saturation/Value square for the current hue. Click/drag to pick; publishes
/// [`Message::PickerSV`]. Drawn as a coarse cell grid (cheap; redraws only on
/// interaction under `ControlFlow::Wait`).
struct SvSquare {
    hue: f32,
    s: f32,
    v: f32,
}

impl SvSquare {
    const CELLS: usize = 24;

    fn message_at(p: Point, bounds: Rectangle) -> Message {
        let s = ((p.x - bounds.x) / bounds.width).clamp(0.0, 1.0);
        let vy = ((p.y - bounds.y) / bounds.height).clamp(0.0, 1.0);
        Message::PickerSV(s, 1.0 - vy)
    }
}

impl Program<Message, Theme, Renderer> for SvSquare {
    type State = DragState;

    fn update(
        &self,
        state: &mut DragState,
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<Action<Message>> {
        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let p = cursor.position()?;
                if bounds.contains(p) {
                    state.dragging = true;
                    return Some(Action::publish(Self::message_at(p, bounds)));
                }
                None
            }
            Event::Mouse(mouse::Event::CursorMoved { .. }) if state.dragging => {
                let p = cursor.position()?;
                Some(Action::publish(Self::message_at(p, bounds)))
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                state.dragging = false;
                None
            }
            _ => None,
        }
    }

    fn draw(
        &self,
        _state: &DragState,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let w = bounds.width;
        let h = bounds.height;
        let cw = w / Self::CELLS as f32;
        let ch = h / Self::CELLS as f32;

        for cy in 0..Self::CELLS {
            for cx in 0..Self::CELLS {
                let s = (cx as f32 + 0.5) / Self::CELLS as f32;
                let v = 1.0 - (cy as f32 + 0.5) / Self::CELLS as f32;
                let color = u8_color(hsv_to_rgb([self.hue, s, v]));
                frame.fill_rectangle(
                    Point::new(cx as f32 * cw, cy as f32 * ch),
                    Size::new(cw + 1.0, ch + 1.0),
                    color,
                );
            }
        }

        // Selection marker (black outer + white inner square for contrast).
        let mx = self.s.clamp(0.0, 1.0) * w;
        let my = (1.0 - self.v.clamp(0.0, 1.0)) * h;
        let r = 5.0;
        let rect = Path::rectangle(Point::new(mx - r, my - r), Size::new(2.0 * r, 2.0 * r));
        frame.stroke(&rect, Stroke::default().with_color(Color::BLACK).with_width(2.0));
        frame.stroke(&rect, Stroke::default().with_color(Color::WHITE).with_width(1.0));

        vec![frame.into_geometry()]
    }
}

/// Horizontal hue strip. Click/drag to pick; publishes [`Message::PickerHue`].
struct HueStrip {
    hue: f32,
}

impl HueStrip {
    const SEGMENTS: usize = 72;

    fn message_at(p: Point, bounds: Rectangle) -> Message {
        Message::PickerHue(((p.x - bounds.x) / bounds.width).clamp(0.0, 1.0))
    }
}

impl Program<Message, Theme, Renderer> for HueStrip {
    type State = DragState;

    fn update(
        &self,
        state: &mut DragState,
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<Action<Message>> {
        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let p = cursor.position()?;
                if bounds.contains(p) {
                    state.dragging = true;
                    return Some(Action::publish(Self::message_at(p, bounds)));
                }
                None
            }
            Event::Mouse(mouse::Event::CursorMoved { .. }) if state.dragging => {
                let p = cursor.position()?;
                Some(Action::publish(Self::message_at(p, bounds)))
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                state.dragging = false;
                None
            }
            _ => None,
        }
    }

    fn draw(
        &self,
        _state: &DragState,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let w = bounds.width;
        let h = bounds.height;
        let sw = w / Self::SEGMENTS as f32;

        for i in 0..Self::SEGMENTS {
            let hue = (i as f32 + 0.5) / Self::SEGMENTS as f32;
            let color = u8_color(hsv_to_rgb([hue, 1.0, 1.0]));
            frame.fill_rectangle(Point::new(i as f32 * sw, 0.0), Size::new(sw + 1.0, h), color);
        }

        let mx = self.hue.rem_euclid(1.0) * w;
        let marker = Path::line(Point::new(mx, 0.0), Point::new(mx, h));
        frame.stroke(&marker, Stroke::default().with_color(Color::BLACK).with_width(3.0));
        frame.stroke(&marker, Stroke::default().with_color(Color::WHITE).with_width(1.0));

        vec![frame.into_geometry()]
    }
}

// ---------------------------------------------------------------------------
// PaletteEditor: the Iced UI state.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Message {
    /// Select the stop being edited by the color picker.
    SelectStop(usize),
    /// Grow / shrink a stop's slot count (count stepper).
    GrowStop(usize),
    ShrinkStop(usize),
    /// Insert a new stop after this one (copies its color) / delete this stop.
    InsertAfter(usize),
    DeleteStop(usize),
    /// Selected-stop color edits from the RGB sliders (each 0..255).
    RChanged(f32),
    GChanged(f32),
    BChanged(f32),
    /// Selected-stop color edits from the HSV sliders / picker (each 0..1).
    HueChanged(f32),
    SatChanged(f32),
    ValChanged(f32),
    /// From the SV square (saturation, value) and hue strip.
    PickerSV(f32, f32),
    PickerHue(f32),
    /// Generator panel: choose algorithm, set K, roll a new seed, tune params.
    GeneratorChanged(Generator),
    ColorsKChanged(f32),
    Roll,
    CubeStartChanged(f32),
    CubeRotationChanged(f32),
    CubeGammaChanged(f32),
    IqFrequencyChanged(f32),
    NoiseScaleChanged(f32),
    /// Re-pull the working palette from the scene's current selection.
    Reload,
    /// Reverse the stop order and apply live (FZ's "Reversed").
    ReverseStops,
    /// Save the palette to a MAP file (rfd dialog) and register it.
    Export,
    /// Ask the Runner to close this window.
    Close,
}

pub struct PaletteEditor {
    scene: Rc<RefCell<Scene>>,
    palette: EditablePalette,
    /// Index of the stop targeted by the color picker.
    selected: usize,
    /// Picker HSV state (h,s,v in 0..1). Kept independent of the stop's RGB so
    /// hue survives dragging saturation/value to zero.
    picker_hsv: [f32; 3],
    /// Selected generator + its params (for the Roll / generate panel).
    generator: Generator,
    gen_params: GenParams,
    close_requested: bool,
}

impl PaletteEditor {
    pub fn new(scene: Rc<RefCell<Scene>>) -> Self {
        let palette = Self::load_from_scene(&scene);
        let picker_hsv = rgb_to_hsv(stop_color(&palette, 0));
        Self {
            scene,
            palette,
            selected: 0,
            picker_hsv,
            generator: Generator::GoldenRatio,
            gen_params: GenParams::default(),
            close_requested: false,
        }
    }

    /// Replace the working palette with the current generator's output (keeping
    /// the palette name) and apply live. Returns true (scene changed).
    fn regenerate(&mut self) -> bool {
        let colors = palette_generators::generate(self.generator, &self.gen_params);
        let name = self.palette.name.clone();
        self.palette = EditablePalette::from_colors(name, &colors);
        self.sync_selection();
        self.apply();
        true
    }

    fn load_from_scene(scene: &Rc<RefCell<Scene>>) -> EditablePalette {
        let s = scene.borrow();
        EditablePalette::from_colors(
            s.selected_palette_display_name(),
            &s.selected_palette_colors(),
        )
    }

    /// Push the working palette to the scene (live recolor of the fractal).
    fn apply(&self) {
        self.scene
            .borrow_mut()
            .set_selected_palette_colors(self.palette.flatten());
    }

    fn selected_color(&self) -> [u8; 4] {
        stop_color(&self.palette, self.selected)
    }

    /// Clamp the selection after the stop list changes and re-sync the picker.
    fn sync_selection(&mut self) {
        let n = self.palette.stops.len();
        self.selected = self.selected.min(n.saturating_sub(1));
        self.picker_hsv = rgb_to_hsv(self.selected_color());
    }

    /// Write an RGB color to the selected stop, sync the picker, and apply live.
    fn set_selected_rgb(&mut self, rgb: [u8; 4]) {
        self.palette.set_color(self.selected, rgb);
        self.picker_hsv = rgb_to_hsv(rgb);
        self.apply();
    }

    /// Write the current picker HSV to the selected stop and apply live.
    fn apply_picker_hsv(&mut self) {
        let rgb = hsv_to_rgb(self.picker_hsv);
        self.palette.set_color(self.selected, rgb);
        self.apply();
    }

    /// Whether the editor asked to close its window (consumed by the Runner).
    pub fn take_close_request(&mut self) -> bool {
        std::mem::take(&mut self.close_requested)
    }

    /// Returns true if the scene changed (so the Runner redraws the main window).
    pub fn update(&mut self, message: Message) -> bool {
        match message {
            Message::SelectStop(i) => {
                self.selected = i.min(self.palette.stops.len().saturating_sub(1));
                self.picker_hsv = rgb_to_hsv(self.selected_color());
                false
            }
            Message::GrowStop(i) => {
                self.palette.grow(i);
                self.apply();
                true
            }
            Message::ShrinkStop(i) => {
                self.palette.shrink(i);
                self.apply();
                true
            }
            Message::InsertAfter(i) => {
                self.palette.insert_after(i);
                self.selected = (i + 1).min(self.palette.stops.len().saturating_sub(1));
                self.picker_hsv = rgb_to_hsv(self.selected_color());
                self.apply();
                true
            }
            Message::DeleteStop(i) => {
                self.palette.delete(i);
                self.sync_selection();
                self.apply();
                true
            }
            Message::RChanged(x) => {
                let mut c = self.selected_color();
                c[0] = x.round() as u8;
                self.set_selected_rgb(c);
                true
            }
            Message::GChanged(x) => {
                let mut c = self.selected_color();
                c[1] = x.round() as u8;
                self.set_selected_rgb(c);
                true
            }
            Message::BChanged(x) => {
                let mut c = self.selected_color();
                c[2] = x.round() as u8;
                self.set_selected_rgb(c);
                true
            }
            Message::HueChanged(h) => {
                self.picker_hsv[0] = h;
                self.apply_picker_hsv();
                true
            }
            Message::SatChanged(s) => {
                self.picker_hsv[1] = s;
                self.apply_picker_hsv();
                true
            }
            Message::ValChanged(v) => {
                self.picker_hsv[2] = v;
                self.apply_picker_hsv();
                true
            }
            Message::PickerSV(s, v) => {
                self.picker_hsv[1] = s;
                self.picker_hsv[2] = v;
                self.apply_picker_hsv();
                true
            }
            Message::PickerHue(h) => {
                self.picker_hsv[0] = h;
                self.apply_picker_hsv();
                true
            }
            Message::GeneratorChanged(g) => {
                self.generator = g;
                self.regenerate()
            }
            Message::ColorsKChanged(k) => {
                self.gen_params.colors = (k.round() as u32).max(1);
                self.regenerate()
            }
            Message::Roll => {
                self.gen_params.seed = rand::random::<u64>();
                self.regenerate()
            }
            Message::CubeStartChanged(x) => {
                self.gen_params.cube_start = x;
                self.regenerate()
            }
            Message::CubeRotationChanged(x) => {
                self.gen_params.cube_rotation = x;
                self.regenerate()
            }
            Message::CubeGammaChanged(x) => {
                self.gen_params.cube_gamma = x;
                self.regenerate()
            }
            Message::IqFrequencyChanged(x) => {
                self.gen_params.iq_frequency = x;
                self.regenerate()
            }
            Message::NoiseScaleChanged(x) => {
                self.gen_params.noise_scale = x;
                self.regenerate()
            }
            Message::Reload => {
                self.palette = Self::load_from_scene(&self.scene);
                self.sync_selection();
                false
            }
            Message::ReverseStops => {
                self.palette.stops.reverse();
                self.sync_selection();
                self.apply();
                true
            }
            Message::Export => self.export_map(),
            Message::Close => {
                self.close_requested = true;
                false
            }
        }
    }

    /// Open a save dialog and write the palette as a MAP file (one `R G B` line
    /// per flattened slot). The filename stem becomes the palette name and the
    /// full filename its key; the saved palette is registered and selected.
    /// Returns true so the main window refreshes its pick-list.
    fn export_map(&mut self) -> bool {
        let (dir, stem) = {
            let s = self.scene.borrow();
            let stem = if self.palette.name.is_empty() {
                "palette".to_string()
            } else {
                self.palette.name.clone()
            };
            (s.palette_export_dir().to_string(), stem)
        };

        let Some(path) = FileDialog::new()
            .add_filter("MAP palette", &["map"])
            .set_title("Save palette as MAP")
            .set_directory(&dir)
            .set_file_name(format!("{stem}.map"))
            .save_file()
        else {
            return false;
        };

        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let colors = self.palette.flatten();
        let contents: String = colors
            .iter()
            .map(|c| format!("{} {} {}\n", c[0], c[1], c[2]))
            .collect();

        if std::fs::write(&path, contents).is_err() {
            return false;
        }

        let key = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("palette.map")
            .to_string();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("palette")
            .to_string();

        self.palette.name = name.clone();
        self.scene.borrow_mut().save_palette(key, name, colors);
        true
    }

    /// One editable row in the stop list.
    fn stop_row(&self, i: usize, stop: &Stop) -> Element<'_, Message, Theme, Renderer> {
        let col = u8_color(stop.color);
        let selected = i == self.selected;

        let swatch = button(space().width(Length::Fixed(22.0)).height(Length::Fixed(22.0)))
            .on_press(Message::SelectStop(i))
            .style(move |theme, status| {
                let mut s = button::primary(theme, status);
                s.background = Some(Background::Color(col));
                s.border = Border {
                    color: Color::from_rgb(0.3, 0.3, 0.35),
                    width: 1.0,
                    radius: 3.0.into(),
                };
                s
            });

        let hex = text(format!(
            "#{:02X}{:02X}{:02X}",
            stop.color[0], stop.color[1], stop.color[2]
        ))
        .size(12)
        .width(Length::Fixed(70.0));

        let row_content = row![
            swatch,
            space().width(Length::Fixed(6.0)),
            hex,
            button(text("-").size(12)).on_press(Message::ShrinkStop(i)),
            text(format!("{}", stop.count))
                .size(12)
                .width(Length::Fixed(28.0))
                .align_x(Alignment::Center),
            button(text("+").size(12)).on_press(Message::GrowStop(i)),
            space().width(Length::Fixed(8.0)),
            button(text("Ins").size(11)).on_press(Message::InsertAfter(i)),
            button(text("Del").size(11)).on_press(Message::DeleteStop(i)),
        ]
        .spacing(4)
        .align_y(Alignment::Center);

        // A rounded white box marks the selected row (which the picker edits).
        // Non-selected rows keep the same padding + transparent border so the
        // list doesn't shift as selection moves.
        container(row_content)
            .padding(4)
            .style(move |_theme| container::Style {
                border: Border {
                    color: if selected { Color::WHITE } else { Color::TRANSPARENT },
                    width: 1.5,
                    radius: 6.0.into(),
                },
                ..container::Style::default()
            })
            .into()
    }

    /// The hand-rolled color picker for the selected stop (SV square + hue strip
    /// + RGB/HSV sliders + a large swatch and hex).
    fn color_picker(&self) -> Element<'_, Message, Theme, Renderer> {
        let sel = self.selected_color();
        let [h, s, v] = self.picker_hsv;

        let sv: Element<'_, Message, Theme, Renderer> =
            Canvas::new(SvSquare { hue: h, s, v })
                .width(Length::Fixed(150.0))
                .height(Length::Fixed(150.0))
                .into();
        let hue_strip: Element<'_, Message, Theme, Renderer> =
            Canvas::new(HueStrip { hue: h })
                .width(Length::Fixed(150.0))
                .height(Length::Fixed(18.0))
                .into();

        let big_swatch = container(space())
            .width(Length::Fixed(56.0))
            .height(Length::Fixed(56.0))
            .style(move |_theme| container::Style {
                background: Some(Background::Color(u8_color(sel))),
                border: border::rounded(4).width(1).color(Color::WHITE),
                ..container::Style::default()
            });

        let channel = |label: &'static str, value: f32, max: f32, on: fn(f32) -> Message, shown: String| {
            row![
                text(label).size(12).width(Length::Fixed(16.0)),
                slider(0.0..=max, value, on).step(max / 255.0).width(Length::Fixed(150.0)),
                space().width(Length::Fixed(6.0)),
                text(shown).size(12).width(Length::Fixed(40.0)),
            ]
            .align_y(Alignment::Center)
        };

        let rgb_sliders = column![
            channel("R", sel[0] as f32, 255.0, Message::RChanged, format!("{}", sel[0])),
            channel("G", sel[1] as f32, 255.0, Message::GChanged, format!("{}", sel[1])),
            channel("B", sel[2] as f32, 255.0, Message::BChanged, format!("{}", sel[2])),
        ]
        .spacing(4);

        let hsv_sliders = column![
            channel("H", h, 1.0, Message::HueChanged, format!("{:.0}\u{00B0}", h * 360.0)),
            channel("S", s, 1.0, Message::SatChanged, format!("{:.0}%", s * 100.0)),
            channel("V", v, 1.0, Message::ValChanged, format!("{:.0}%", v * 100.0)),
        ]
        .spacing(4);

        let picker = column![
            row![
                column![sv, hue_strip].spacing(6),
                space().width(Length::Fixed(14.0)),
                column![
                    big_swatch,
                    text(format!("#{:02X}{:02X}{:02X}", sel[0], sel[1], sel[2])).size(13),
                ]
                .spacing(8),
            ],
            space().height(Length::Fixed(8.0)),
            rgb_sliders,
            space().height(Length::Fixed(4.0)),
            hsv_sliders,
        ]
        .spacing(6);

        // Rounded white box pairs the picker with the selected (white-boxed) row.
        container(picker)
            .padding(10)
            .style(|_theme| container::Style {
                border: border::rounded(8).width(1.5).color(Color::WHITE),
                ..container::Style::default()
            })
            .into()
    }

    /// The "Generate" panel: algorithm dropdown, Colors (K), Roll, and any
    /// per-generator param sliders.
    fn generator_panel(&self) -> Element<'_, Message, Theme, Renderer> {
        let generators: Vec<Generator> = Generator::iter().collect();
        let p = &self.gen_params;

        let param = |label: &'static str,
                     range: std::ops::RangeInclusive<f32>,
                     value: f32,
                     on: fn(f32) -> Message,
                     shown: String| {
            row![
                text(label).size(12).width(Length::Fixed(76.0)),
                slider(range, value, on).step(0.01).width(Length::Fixed(160.0)),
                space().width(Length::Fixed(6.0)),
                text(shown).size(12).width(Length::Fixed(48.0)),
            ]
            .align_y(Alignment::Center)
        };

        let top = row![
            text("Generator").size(13).width(Length::Fixed(70.0)),
            pick_list(generators, Some(self.generator), Message::GeneratorChanged)
                .width(Length::Fixed(170.0)),
            space().width(Length::Fixed(12.0)),
            text("Colors").size(13),
            slider(1.0..=64.0, p.colors as f32, Message::ColorsKChanged)
                .step(1.0)
                .width(Length::Fixed(130.0)),
            text(format!("{}", p.colors)).size(12).width(Length::Fixed(26.0)),
            space().width(Length::Fixed(12.0)),
            button(text("Roll").size(13)).on_press(Message::Roll),
        ]
        .spacing(6)
        .align_y(Alignment::Center);

        let mut col = column![top].spacing(8);
        if self.generator.has_cubehelix_params() {
            col = col.push(param(
                "Start",
                0.0..=9.0,
                p.cube_start,
                Message::CubeStartChanged,
                format!("{:.2}", p.cube_start),
            ));
            col = col.push(param(
                "Rotation",
                -5.0..=5.0,
                p.cube_rotation,
                Message::CubeRotationChanged,
                format!("{:.2}", p.cube_rotation),
            ));
            col = col.push(param(
                "Gamma",
                0.1..=1.5,
                p.cube_gamma,
                Message::CubeGammaChanged,
                format!("{:.2}", p.cube_gamma),
            ));
        }
        if self.generator.has_frequency_param() {
            col = col.push(param(
                "Frequency",
                0.25..=3.0,
                p.iq_frequency,
                Message::IqFrequencyChanged,
                format!("{:.2}", p.iq_frequency),
            ));
        }
        if self.generator.has_noise_param() {
            col = col.push(param(
                "Noise Scale",
                0.3..=4.0,
                p.noise_scale,
                Message::NoiseScaleChanged,
                format!("{:.2}", p.noise_scale),
            ));
        }

        container(col).style(panel_style).padding(10).width(Length::Fill).into()
    }

    pub fn view(&self) -> Element<'_, Message, Theme, Renderer> {
        let stops = self.palette.stops.len();
        let slots = self.palette.total_len();
        let colors = self.palette.flatten();

        // Flattened-slot index where each stop (after the first) begins.
        let mut boundaries = Vec::new();
        let mut acc = 0usize;
        for stop in &self.palette.stops[..stops.saturating_sub(1)] {
            acc += stop.count as usize;
            boundaries.push(acc);
        }

        let preview: Element<'_, Message, Theme, Renderer> =
            Canvas::new(PalettePreview { colors: colors.clone() })
                .width(Length::Fill)
                .height(Length::Fixed(40.0))
                .into();

        let graphs: Element<'_, Message, Theme, Renderer> =
            Canvas::new(ChannelGraphs { colors, boundaries })
                .width(Length::Fill)
                .height(Length::Fixed(140.0))
                .into();

        // Panel 1: preview bar + channel graphs.
        let preview_panel = container(column![preview, graphs].spacing(6))
            .style(panel_style)
            .padding(10)
            .width(Length::Fill);

        // Panel 2: stop list (left, top-aligned) and color picker (right-aligned).
        let stop_rows: Vec<Element<'_, Message, Theme, Renderer>> = self
            .palette
            .stops
            .iter()
            .enumerate()
            .map(|(i, stop)| self.stop_row(i, stop))
            .collect();
        let stop_col = column![text("Stops").size(14), Column::with_children(stop_rows).spacing(4)]
            .spacing(8);
        let picker_col = column![text("Color").size(14), self.color_picker()].spacing(8);

        let editor_panel = container(
            row![stop_col, space().width(Length::Fill), picker_col].align_y(Alignment::Start),
        )
        .style(panel_style)
        .padding(10)
        .width(Length::Fill);

        let content = column![
            row![
                text(format!("Editing: {}", self.palette.name)).size(14),
                space().width(Length::Fixed(25.0)),
                text(format!("{stops} stops \u{2022} {slots} slots")).size(12),
            ].align_y(Alignment::Center),
            space().height(Length::Fixed(4.0)),
            self.generator_panel(),
            preview_panel,
            editor_panel,
            space().height(Length::Fixed(4.0)),
            row![
                button(text("Save MAP").size(13)).on_press(Message::Export),
                space().width(Length::Fixed(10.0)),
                button(text("Reverse").size(13)).on_press(Message::ReverseStops),
                space().width(Length::Fixed(10.0)),
                button(text("Reload").size(13)).on_press(Message::Reload),
                space().width(Length::Fixed(10.0)),
                button(text("Close").size(13)).on_press(Message::Close),
            ]
            .align_y(Alignment::Center),
        ]
        .spacing(10)
        .padding(20)
        .width(Length::Fill);

        scrollable(content).width(Length::Fill).height(Length::Fill).into()
    }
}

// ---------------------------------------------------------------------------
// PaletteWindow: the winit/wgpu/iced resource bundle for the editor window.
// ---------------------------------------------------------------------------

/// What a handled window event asks the Runner to do afterward.
pub struct PaletteWindowOutcome {
    pub close: bool,
    pub scene_changed: bool,
}

pub struct PaletteWindow {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    format: wgpu::TextureFormat,
    renderer: Renderer,
    viewport: Viewport,
    cache: user_interface::Cache,
    cursor: mouse::Cursor,
    clipboard: Clipboard,
    events: Vec<Event>,
    modifiers: ModifiersState,
    resized: bool,
    editor: PaletteEditor,
}

impl PaletteWindow {
    pub fn window_id(&self) -> WindowId {
        self.window.id()
    }

    /// Create the editor window on demand. Shares the scene's device/queue; the
    /// surface reuses the main window's `format` (same adapter).
    pub fn new(
        event_loop: &ActiveEventLoop,
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        scene: &Rc<RefCell<Scene>>,
        format: wgpu::TextureFormat,
    ) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("Palette Editor")
                        .with_inner_size(LogicalSize::new(650.0, 860.0)),
                )
                .expect("create palette editor window"),
        );

        let size = window.inner_size();
        let viewport = Viewport::with_physical_size(
            Size::new(size.width, size.height),
            window.scale_factor() as f32,
        );
        let clipboard = Clipboard::connect(window.clone());

        let surface = instance
            .create_surface(window.clone())
            .expect("create palette editor surface");
        surface.configure(
            scene.borrow().device(),
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width.max(1),
                height: size.height.max(1),
                present_mode: wgpu::PresentMode::AutoVsync,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        let renderer = {
            let engine = Engine::new(
                adapter,
                scene.borrow().device().clone(),
                scene.borrow().queue().clone(),
                format,
                None,
                Shell::headless(),
            );
            Renderer::new(engine, Font::default(), Pixels::from(16))
        };

        let editor = PaletteEditor::new(Rc::clone(scene));
        window.request_redraw();

        Self {
            window,
            surface,
            format,
            renderer,
            viewport,
            cache: user_interface::Cache::new(),
            cursor: mouse::Cursor::Unavailable,
            clipboard,
            events: Vec::new(),
            modifiers: ModifiersState::default(),
            resized: false,
            editor,
        }
    }

    /// Handle one winit event for the editor window.
    pub fn handle_event(
        &mut self,
        event: WindowEvent,
        scene: &Rc<RefCell<Scene>>,
    ) -> PaletteWindowOutcome {
        let mut outcome = PaletteWindowOutcome { close: false, scene_changed: false };

        match event {
            WindowEvent::RedrawRequested => {
                self.redraw(scene);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = mouse::Cursor::Available(conversion::cursor_position(
                    position,
                    self.viewport.scale_factor(),
                ));
            }
            WindowEvent::ModifiersChanged(new_modifiers) => {
                self.modifiers = new_modifiers.state();
            }
            WindowEvent::Resized(_) => {
                self.resized = true;
                self.window.request_redraw();
            }
            WindowEvent::CloseRequested => {
                outcome.close = true;
                return outcome;
            }
            _ => {}
        }

        if let Some(iced_event) = conversion::window_event(
            event,
            self.window.scale_factor() as f32,
            self.modifiers,
        ) {
            self.events.push(iced_event);
        }

        if !self.events.is_empty() {
            let mut interface = UserInterface::build(
                self.editor.view(),
                self.viewport.logical_size(),
                std::mem::take(&mut self.cache),
                &mut self.renderer,
            );

            let mut messages = Vec::new();
            let _ = interface.update(
                &self.events,
                self.cursor,
                &mut self.renderer,
                &mut self.clipboard,
                &mut messages,
            );
            self.events.clear();
            self.cache = interface.into_cache();

            for message in messages {
                if self.editor.update(message) {
                    outcome.scene_changed = true;
                }
            }
            if self.editor.take_close_request() {
                outcome.close = true;
            }
            self.window.request_redraw();
        }

        outcome
    }

    fn redraw(&mut self, scene: &Rc<RefCell<Scene>>) {
        if self.resized {
            let size = self.window.inner_size();
            self.viewport = Viewport::with_physical_size(
                Size::new(size.width, size.height),
                self.window.scale_factor() as f32,
            );
            self.surface.configure(
                scene.borrow().device(),
                &wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: self.format,
                    width: size.width.max(1),
                    height: size.height.max(1),
                    present_mode: wgpu::PresentMode::AutoVsync,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                    desired_maximum_frame_latency: 2,
                },
            );
            self.resized = false;
        }

        match self.surface.get_current_texture() {
            Ok(frame) => {
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut interface = UserInterface::build(
                    self.editor.view(),
                    self.viewport.logical_size(),
                    std::mem::take(&mut self.cache),
                    &mut self.renderer,
                );

                let (state, _) = interface.update(
                    &[Event::Window(window::Event::RedrawRequested(Instant::now()))],
                    self.cursor,
                    &mut self.renderer,
                    &mut self.clipboard,
                    &mut Vec::new(),
                );

                if let user_interface::State::Updated { mouse_interaction, .. } = state {
                    if let Some(icon) = conversion::mouse_interaction(mouse_interaction) {
                        self.window.set_cursor(icon);
                        self.window.set_cursor_visible(true);
                    } else {
                        self.window.set_cursor_visible(false);
                    }
                }

                interface.draw(
                    &mut self.renderer,
                    &Theme::Dark,
                    // Default text color is black; the main window overrides per
                    // widget, but the editor's default should be light-on-dark.
                    &renderer::Style { text_color: Color::WHITE },
                    self.cursor,
                );
                self.cache = interface.into_cache();

                self.renderer.present(
                    Some(EDITOR_BG),
                    frame.texture.format(),
                    &view,
                    &self.viewport,
                );
                frame.present();
            }
            Err(_) => {
                self.window.request_redraw();
            }
        }
    }
}
