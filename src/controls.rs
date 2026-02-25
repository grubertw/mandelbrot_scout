use super::scene::Scene;

use iced_wgpu::Renderer;
use iced_widget::{text_input, column, row, text, button, horizontal_space};
use iced_widget::core::{Alignment, Color, Element, Theme, Length};
use iced_winit::runtime::{Program, Task};

use log::{trace};

use std::rc::Rc;
use std::cell::RefCell;
use iced_wgpu::core::text::Wrapping;
use rug::{Complex, Float};
use crate::scene::policy::{COVERAGE_TO_ANCHOR, EXPLORATION_BUDGET, SMALLEST_TILE_PIXEL_SPAN, STARTING_TILE_PIXEL_SPAN};
use crate::scout_engine::ScoutSignal;

#[derive(Clone)]
pub struct Controls {
    iter_increment: String,
    max_iterations: u32,
    center_x: String,
    center_y: String,
    scale: String,
    starting_tile_span: String,
    smallest_tile_span: String,
    coverage_to_anchor: String,
    explore_budget: String,
    debug_msg: String,
    scene: Rc<RefCell<Scene>>
}

#[derive(Debug, Clone)]
pub enum Message {
    IterIncrementChanged(String),
    AddIters,
    SubtractIters,
    CenterXChanged(String),
    CenterYChanged(String),
    ScaleChanged(String),
    PollFromScene,
    ApplyCenterScale,
    ResetScoutEngine,
    GoScout,
    StartingTileSpanChanged(String),
    SmallestTileSpanChanged(String),
    CoverageToAnchorChanged(String),
    ExploreBudgetChanged(String),
    UpdateDebugText(String),
}

impl Controls {
    pub fn new(s: Rc<RefCell<Scene>>) -> Controls {
        let center = s.borrow().center().clone();
        let center_x = center.real().to_string_radix(10, Some(10));
        let center_y = center.imag().to_string_radix(10, Some(10));
        let scale = s.borrow().scale().to_string_radix(10, Some(6));
        let max_iters = s.borrow().max_iterations();

        Controls {
            iter_increment: "10".to_string(),
            max_iterations: max_iters,
            center_x, center_y, scale,
            starting_tile_span: STARTING_TILE_PIXEL_SPAN.to_string(),
            smallest_tile_span: SMALLEST_TILE_PIXEL_SPAN.to_string(),
            coverage_to_anchor: COVERAGE_TO_ANCHOR.to_string(),
            explore_budget: EXPLORATION_BUDGET.to_string(),
            debug_msg: "debug info loading...".to_string(),
            scene: s
        }
    }
}

impl Program for Controls {
    type Theme = Theme;
    type Message = Message;
    type Renderer = Renderer;

    fn update(&mut self, message: Message) -> Task<Message> {
        trace!("Update {:?}", message);
        match message {
            Message::IterIncrementChanged(iter_inc) => {
                self.iter_increment = iter_inc;
            }
            Message::AddIters => {
                if let Ok(inc) = self.iter_increment.parse::<u32>() {
                    self.max_iterations += inc;
                    self.scene.borrow_mut().set_max_iterations(self.max_iterations);
                }
            }
            Message::SubtractIters => {
                if let Ok(inc) = self.iter_increment.parse::<u32>() {
                    self.max_iterations -= inc;
                    self.scene.borrow_mut().set_max_iterations(self.max_iterations);
                }
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
            Message::ResetScoutEngine => {
                self.scene.borrow_mut().send_scout_signal(ScoutSignal::ResetEngine);
            }
            Message::GoScout => {
                let mut scene_b = self.scene.borrow_mut();
                let mut config = scene_b.scout_config().lock().clone();
                config.starting_tile_pixel_span = if let Ok(v) =
                    self.starting_tile_span.parse::<f64>() { v } else { 0.0 };
                config.smallest_tile_pixel_span = if let Ok(v) =
                    self.smallest_tile_span.parse::<f64>() { v } else { 0.0 };
                config.coverage_to_anchor = if let Ok(v) =
                    self.coverage_to_anchor.parse::<f64>() { v } else { 0.0 };
                config.exploration_budget = if let Ok(v) =
                    self.explore_budget.parse::<i32>() { v } else { 0 };

                scene_b.send_scout_signal(ScoutSignal::ExploreSignal(config));
            }
            Message::StartingTileSpanChanged(starting_span) => {
                self.starting_tile_span = starting_span;
            }
            Message::SmallestTileSpanChanged(smallest_span) => {
                self.smallest_tile_span = smallest_span;
            }
            Message::CoverageToAnchorChanged(coverage_to_anchor) => {
                self.coverage_to_anchor = coverage_to_anchor;
            }
            Message::ExploreBudgetChanged(budget) => {
                self.explore_budget = budget;
            }
            Message::UpdateDebugText(dbg_msg) => {
                self.debug_msg = dbg_msg;
            }
        }

        Task::none()
    }

    fn view(&self) -> Element<'_, Message, Theme, Renderer> {
        trace!("View");

        let dbg_row = row![
            text(&self.debug_msg).color(Color::WHITE)
            .wrapping(Wrapping::Word)
            .size(10)
        ]
        .align_y(Alignment::Start);

        let itrs_row = row![
            button("<")
            .on_press(Message::SubtractIters)
            .height(Length::Shrink),
            horizontal_space().width(Length::Fixed(5.0)),
            text_input("Placeholder...", &self.iter_increment)
                .on_input(Message::IterIncrementChanged)
                .width(Length::Fixed(40.0)),
             horizontal_space().width(Length::Fixed(5.0)),
            button(">")
            .on_press(Message::AddIters)
            .height(Length::Shrink),
            horizontal_space().width(Length::Fixed(20.0)),
            text("max iterations: ")
            .color(Color::WHITE)
            .size(14),
            text(self.max_iterations.to_string())
            .size(14),
        ]
        .height(Length::Fixed(30.0));

        let edit_center_scale_row = row![
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
                .width(Length::Fixed(135.0)),

            text("imag: ")
            .color(Color::WHITE)
            .width(Length::Fixed(60.0))
            .align_x(Alignment::End),
            text_input("Placeholder...", &self.center_y)
                .on_input(Message::CenterYChanged)
                .width(Length::Fixed(135.0)),

            text("scale: ")
            .color(Color::WHITE)
            .width(Length::Fixed(60.0))
            .align_x(Alignment::End),
            text_input("Placeholder...", &self.scale)
                .on_input(Message::ScaleChanged)
                .width(Length::Fixed(100.0)),

            horizontal_space().width(Length::Fixed(20.0)),

            button("Apply")
            .on_press(Message::ApplyCenterScale)
            .width(Length::Fixed(65.0))
            .height(Length::Shrink),
        ]
        .height(Length::Fixed(30.0));

        let scout_config_row = row![
            button("Reset Scout")
            .on_press(Message::ResetScoutEngine)
            .width(Length::Fixed(110.0))
            .height(Length::Shrink),

            text("tile size px start: ")
            .color(Color::WHITE)
            .width(Length::Fixed(70.0))
            .wrapping(Wrapping::Word)
            .size(12)
            .align_x(Alignment::End),
            horizontal_space().width(Length::Fixed(5.0)),
            text_input("Placeholder...", &self.starting_tile_span)
                .on_input(Message::StartingTileSpanChanged)
                .width(Length::Fixed(60.0)),

            text("smallest tile px: ")
            .color(Color::WHITE)
            .width(Length::Fixed(65.0))
            .wrapping(Wrapping::Word)
            .size(12)
            .align_x(Alignment::End),
            horizontal_space().width(Length::Fixed(5.0)),
            text_input("Placeholder...", &self.smallest_tile_span)
                .on_input(Message::SmallestTileSpanChanged)
                .width(Length::Fixed(60.0)),

             text("coverage to anchor: ")
            .color(Color::WHITE)
            .width(Length::Fixed(75.0))
            .wrapping(Wrapping::Word)
            .size(12)
            .align_x(Alignment::End),
            horizontal_space().width(Length::Fixed(5.0)),
            text_input("Placeholder...", &self.coverage_to_anchor)
                .on_input(Message::CoverageToAnchorChanged)
                .width(Length::Fixed(40.0)),

            text("explore budget: ")
            .color(Color::WHITE)
            .width(Length::Fixed(65.0))
            .wrapping(Wrapping::Word)
            .size(12)
            .align_x(Alignment::End),
            horizontal_space().width(Length::Fixed(5.0)),
            text_input("Placeholder...", &self.explore_budget)
                .on_input(Message::ExploreBudgetChanged)
                .width(Length::Fixed(25.0)),

            horizontal_space().width(Length::Fixed(20.0)),

            button("Scout!")
            .on_press(Message::GoScout)
            .width(Length::Fixed(70.0))
            .height(Length::Shrink),
        ]
            .height(Length::Fixed(30.0));

        row![
            column![
                dbg_row,
                row![]
                .height(Length::Fill)
                .align_y(Alignment::End),
                itrs_row,
                scout_config_row,
                edit_center_scale_row,
            ]
            .height(Length::Fill)
            .width(Length::Fill)
            .padding(10)
            .spacing(10),
        ]
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    }
}