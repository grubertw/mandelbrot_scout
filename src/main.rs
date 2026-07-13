mod settings;
mod controls;
mod scene;
mod gpu_pipeline;

mod export;

#[allow(dead_code)]
mod numerics;

#[allow(dead_code)]
mod signals;

#[allow(dead_code)]
mod scout_engine;

#[allow(dead_code)]
mod palette_editor;

mod palette_generators;
mod palette_window;

use settings::Settings;
use palette_window::PaletteWindow;
use controls::Controls;
use controls::Message;
use scene::Scene;

use iced_wgpu::graphics::{Viewport, Shell};
use iced_wgpu::{Engine, Renderer, wgpu};
use iced_winit::Clipboard;
use iced_winit::conversion;
use iced_winit::core::{mouse};
use iced_winit::core::renderer;
use iced_winit::core::window;
use iced_winit::core::{Event, Font, Pixels, Size, Theme};
use iced_winit::runtime::user_interface::{self, UserInterface};
use futures::executor;
use iced_winit::winit;

use winit::{
    application::ApplicationHandler,
    keyboard::{ModifiersState, PhysicalKey, KeyCode},
    event::{WindowEvent, KeyEvent, MouseScrollDelta, ElementState, MouseButton},
    event_loop::{ControlFlow, EventLoop, ActiveEventLoop},
    window::{Window, WindowId, WindowAttributes},
};

use log::{debug, error, info, trace};
use chrono::Local;
use anstyle::Style;

use std::io::Write;
use std::rc::Rc;
use std::cell::RefCell;
use std::sync::Arc;
use std::process;
use std::time::Instant;
use iced_winit::winit::dpi::PhysicalPosition;
use num_traits::Signed;

pub const TITLE: &str = "Mandelbrot Scout";
pub const CLICK_THRESHOLD: f64 = 5.0;

#[derive(Debug, Clone)]
pub enum NavigationEvent {
    Zoom {
        factor: f64,
        cursor_pos: PhysicalPosition<f64>,
    },

    ZoomStep {
        zoom_in: bool,
        cursor_pos: PhysicalPosition<f64>,
    },

    Pan {
        delta_pixels: (f64, f64),
    },

    PanFinished,
}

struct NavigationState {
    cursor_pos: PhysicalPosition<f64>,
    fractal_can_receive_input: bool,

    left_pressed: bool,
    right_pressed: bool,

    press_position: Option<PhysicalPosition<f64>>,
    drag_start: Option<PhysicalPosition<f64>>,
    scene: Rc<RefCell<Scene>>,
}

impl NavigationState {
    fn new(scene: Rc<RefCell<Scene>>) -> Self {
        NavigationState {
            cursor_pos: PhysicalPosition::new(0.0, 0.0),
            fractal_can_receive_input: false,
            left_pressed: false,
            right_pressed: false,
            press_position: None,
            drag_start: None,
            scene,
        }
    }

    pub fn process_event(
        &mut self,
        event: &NavigationEvent,
    ) {
        debug!("process NavigationEvent: {:?}", event);
        match event {
            NavigationEvent::Zoom {
                factor,
                cursor_pos,
            } => {
                self.scene.borrow_mut()
                    .zoom_step_toward((cursor_pos.x, cursor_pos.y), factor.is_negative());
            }

            NavigationEvent::ZoomStep {
                zoom_in,
                cursor_pos,
            } => {
                self.scene.borrow_mut()
                    .zoom_step_toward((cursor_pos.x, cursor_pos.y), *zoom_in);
            }

            NavigationEvent::Pan {
                delta_pixels,
            } => {
                self.scene.borrow_mut()
                    .pan_pixels(*delta_pixels);
            }

            NavigationEvent::PanFinished => {
                self.scene.borrow_mut().recalculate();
                self.scene.borrow_mut().take_camera_snapshot();
            }
        }
    }

    pub fn is_pressed(&self) -> bool {
        self.press_position.is_some()
    }

    pub fn is_dragging(&self) -> bool {
        self.drag_start.is_some()
    }
}

#[allow(clippy::large_enum_variant)]
enum Runner {
    Loading {
        settings: Settings,
    },
    Ready {
        window: Arc<Window>,
        surface: wgpu::Surface<'static>,
        format: wgpu::TextureFormat,
        renderer: Renderer,
        scene: Rc<RefCell<Scene>>,
        controls: Controls,
        events: Vec<Event>,
        cursor: mouse::Cursor,
        cache: user_interface::Cache,
        clipboard: Clipboard,
        viewport: Viewport,
        modifiers: ModifiersState,
        resized: bool,
        // For tracking mouse/touch/trackpad interaction to abstract platform-specific behavior
        navigation_state: NavigationState,
        // Kept so the Custom Palette Editor can spin up a second surface + iced
        // renderer on demand (same adapter/device as the main window).
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        // The editor's second window, when open. Shares the scene (same thread).
        palette_window: Option<PaletteWindow>,
    },
}

impl ApplicationHandler for Runner {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        debug!("Runner.resumed");
        if let Self::Loading{settings} = self {
            let window = Arc::new(event_loop.create_window(
                WindowAttributes::default()).unwrap_or_else(|e| {
                    error!("Failed to Create window .. {}", e);
                    process::exit(1);
                }));

            window.set_title(TITLE);
            let physical_size = window.inner_size();
            let viewport = Viewport::with_physical_size(
                Size::new(physical_size.width, physical_size.height),
                window.scale_factor() as f32);

            let clipboard = Clipboard::connect(window.clone());

            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
            let surface = instance
                .create_surface(window.clone())
                .expect("Create window surface");

            debug!("Runner.resumed - physical_size={:?} scale_fac={:?}",
                physical_size, window.scale_factor());

            let (format, adapter, device, queue) =
                executor::block_on(async {
                    let adapter = wgpu::util::initialize_adapter_from_env_or_default(
                                    &instance, Some(&surface))
                        .await
                        .expect("Create adapter");

                    let adapter_features = 
                        adapter.features() & wgpu::Features::default();
                    let capabilities = surface.get_capabilities(&adapter);

                    let (device, queue) = adapter
                        .request_device(&wgpu::DeviceDescriptor {label: None,
                            required_features: adapter_features,
                            required_limits: wgpu::Limits::default(),
                            memory_hints: wgpu::MemoryHints::MemoryUsage,
                            trace: wgpu::Trace::Off,
                            experimental_features:
                            wgpu::ExperimentalFeatures::disabled(),
                        })
                        .await
                        .expect("Request device");

                    (capabilities.formats.iter().copied().find(wgpu::TextureFormat::is_srgb)
                            .or_else(|| {capabilities.formats.first().copied()})
                            .expect("Get preferred format"),
                        adapter, device, queue)
                });
            debug!("Runner.resumed - format={:?} adapter={:?} device={:?} queue={:?}",
                format, adapter, device, queue);

            surface.configure(&device, &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format,
                width: physical_size.width,
                height: physical_size.height,
                present_mode: wgpu::PresentMode::AutoVsync,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
                desired_maximum_frame_latency: 2});

            // Initialize scene and GUI controls
            let scene = Rc::new(RefCell::new(
                Scene::new(window.clone(), device, queue, format,
                    physical_size.width.into(),
                    physical_size.height.into(),
                   &settings
                )));
            // Take a snapshot of where the camera is in the beginning to send to ScoutEngine
            // So that an initial reference orbit can be computed.
            scene.borrow_mut().take_camera_snapshot();
            
            let controls = Controls::new(&settings, Rc::clone(&scene));

            // Initialize iced
            let renderer = {
                let engine = Engine::new(
                    &adapter,
                    scene.borrow().device().clone(),
                    scene.borrow().queue().clone(),
                    format,
                    None,
                    Shell::headless(),
                );

                Renderer::new(engine, Font::default(), Pixels::from(16))
            };

            // Change this to render continuously
            event_loop.set_control_flow(ControlFlow::Wait);

            info!("ApplicationHandler Runner Initialized");
            *self = Self::Ready {window, surface, format, renderer,
                scene: Rc::clone(&scene), controls, events: Vec::new(), cursor: mouse::Cursor::Unavailable,
                cache: user_interface::Cache::new(), modifiers: ModifiersState::default(),
                clipboard, viewport, resized: false,
                navigation_state: NavigationState::new(Rc::clone(&scene)),
                instance, adapter, palette_window: None};
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop,
            window_id: WindowId, event: WindowEvent) {
        let Self::Ready {window, surface, format, renderer, scene,
            controls, events, viewport, cursor, cache, modifiers, clipboard, resized,
            navigation_state, instance, adapter, palette_window} = self
        else {
            return;
        };

        // Route events for the editor's second window to it. It shares the scene,
        // so a live edit there redraws the main (fractal) window.
        if palette_window.as_ref().is_some_and(|pw| pw.window_id() == window_id) {
            let outcome = palette_window.as_mut().unwrap().handle_event(event, scene);
            if outcome.scene_changed {
                window.request_redraw();
            }
            if outcome.close {
                *palette_window = None;
            }
            return;
        }

        //trace!("Runner.window_event - {:?} {:?}", window_id, event);
        let mut nav_events: Vec<NavigationEvent> = Vec::new();

        match event {
            WindowEvent::RedrawRequested => {
                // The editor (separate window) may have saved a new palette.
                if scene.borrow_mut().take_palettes_dirty() {
                    controls.refresh_palettes();
                }
                if *resized {
                    let size = window.inner_size();
                    scene.borrow_mut().set_window_size(
                        size.width.into(),
                        size.height.into());
                    *viewport = Viewport::with_physical_size(Size::new(size.width, size.height),
                        window.scale_factor() as f32);

                    surface.configure(scene.borrow().device(), &wgpu::SurfaceConfiguration {
                            format: *format,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                            width: size.width,
                            height: size.height,
                            present_mode: wgpu::PresentMode::AutoVsync,
                            alpha_mode: wgpu::CompositeAlphaMode::Auto,
                            view_formats: vec![],
                            desired_maximum_frame_latency: 2});

                    *resized = false;
                }

                match surface.get_current_texture() {
                    Ok(frame) => {
                        let view = frame.texture.create_view(
                            &wgpu::TextureViewDescriptor::default());

                        {
                            let mut s = scene.borrow_mut();

                            // Ask ScoutEngine for it's current tile orbits and push to the GPU
                            s.query_qualified_orbits();

                            {
                                trace!("Reading scout diagnostics...");
                                let scout_diags = s.read_scout_diagnostics();
                                let mut scout_diags_g = scout_diags.lock();
                                if !scout_diags_g.consumed {
                                    controls.update(Message::UpdateDebugText(scout_diags_g.message.clone()));
                                    scout_diags_g.consumed = true;
                                }
                                trace!("scout diagnostics read successfully!");
                            }

                            let size = window.inner_size();
                            let (w, h) = if navigation_state.is_dragging() {
                                (size.width as f64 * s.render_res_factor_during_pan(),
                                 size.height as f64 * s.render_res_factor_during_pan())
                            }
                            else {
                                (size.width as f64 * s.render_res_factor(),
                                 size.height as f64 * s.render_res_factor())
                            };

                            // Draw the scene (contains both fragment render and compute passes)
                            s.render(w as u32, h as u32, Some(&view));
                        }

                        // Draw Iced on top
                        let mut interface = UserInterface::build(
                            controls.view(),
                            viewport.logical_size(),
                            std::mem::take(cache),
                            renderer,
                        );

                        let (state, _) = interface.update(
                            &[Event::Window(
                                window::Event::RedrawRequested(
                                    Instant::now(),
                                ),
                            )],
                            *cursor,
                            renderer,
                            clipboard,
                            &mut Vec::new(),
                        );

                        // Update the mouse cursor
                        if let user_interface::State::Updated {
                            mouse_interaction,
                            ..
                        } = state
                        {
                            navigation_state.fractal_can_receive_input
                                = mouse_interaction == mouse::Interaction::None;

                            // Update the mouse cursor
                            if let Some(icon) =
                                conversion::mouse_interaction(
                                    mouse_interaction,
                                )
                            {
                                window.set_cursor(icon);
                                window.set_cursor_visible(true);
                            } else {
                                window.set_cursor_visible(false);
                            }
                        }

                        // Draw the interface
                        interface.draw(
                            renderer,
                            &Theme::Dark,
                            &renderer::Style::default(),
                            *cursor,
                        );
                        *cache = interface.into_cache();

                        renderer.present(
                            None,
                            frame.texture.format(),
                            &view,
                            viewport,
                        );

                        // Present the frame
                        frame.present();
                    }
                    Err(error) => match error {
                        wgpu::SurfaceError::OutOfMemory => {
                            panic!("Swapchain error: {error}. Rendering cannot continue.")
                        }
                        _ => {
                            // Try rendering again next frame.
                            window.request_redraw();
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                *cursor = mouse::Cursor::Available(conversion::cursor_position(
                    position,
                    viewport.scale_factor(),
                ));

                if navigation_state.left_pressed
                    && position != navigation_state.press_position.unwrap()
                    && navigation_state.drag_start.is_none() {
                    navigation_state.drag_start = Some(position);
                }

                navigation_state.cursor_pos = position;

                if let Some(previous) = navigation_state.drag_start {
                    let dx = position.x - previous.x;
                    let dy = position.y - previous.y;

                    navigation_state.drag_start = Some(position);

                    nav_events.push(
                        NavigationEvent::Pan {
                            delta_pixels: (dx, dy),
                        }
                    );
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let MouseScrollDelta::LineDelta(_, h) = delta {
                    if navigation_state.fractal_can_receive_input {
                        nav_events.push(NavigationEvent::Zoom {
                            factor: h as f64,
                            cursor_pos: navigation_state.cursor_pos,
                        });
                    }
                }
            }
            WindowEvent::MouseInput { state, button, ..} => {
                if navigation_state.fractal_can_receive_input {
                    match state {
                        ElementState::Pressed => {
                            match button {
                                MouseButton::Left => {
                                    navigation_state.left_pressed = true;
                                    navigation_state.press_position = Some(navigation_state.cursor_pos);
                                }
                                MouseButton::Right => {
                                    navigation_state.right_pressed = true;
                                    navigation_state.press_position = Some(navigation_state.cursor_pos);
                                }
                                _ => {}
                            }
                        }
                        ElementState::Released => {
                            if navigation_state.is_pressed() {
                                if !navigation_state.is_dragging() && navigation_state.left_pressed {
                                    nav_events.push(NavigationEvent::ZoomStep {
                                        zoom_in: true,
                                        cursor_pos: navigation_state.cursor_pos,
                                    });
                                } else if !navigation_state.is_dragging() && navigation_state.right_pressed {
                                    nav_events.push(NavigationEvent::ZoomStep {
                                        zoom_in: false,
                                        cursor_pos: navigation_state.cursor_pos,
                                    });
                                } else if navigation_state.left_pressed {
                                    nav_events.push(NavigationEvent::PanFinished)
                                }
                                navigation_state.press_position = None;
                                navigation_state.left_pressed = false;
                                navigation_state.right_pressed = false;
                                navigation_state.drag_start = None;
                            }
                        }
                    }
                }
            }
            WindowEvent::KeyboardInput { device_id: _, ref event, .. } => {
                let KeyEvent{physical_key, ..} = event;
                let PhysicalKey::Code (c) = physical_key else {
                    return;
                };
                
                match c {
                    KeyCode::ArrowUp => {
                        nav_events.push(NavigationEvent::Zoom {
                            factor: 1.0,
                            cursor_pos: navigation_state.cursor_pos,
                        });
                    }
                    KeyCode::ArrowDown => {
                        nav_events.push(NavigationEvent::Zoom {
                            factor: -1.0,
                            cursor_pos: navigation_state.cursor_pos,
                        });
                    }
                    _ => {}
                }
            }
            WindowEvent::PinchGesture { device_id: _, delta, .. } => {
                nav_events.push(
                    NavigationEvent::Zoom {
                        factor: -delta,
                        cursor_pos: navigation_state.cursor_pos,
                    }
                );
            }
            WindowEvent::ModifiersChanged(new_modifiers) => {
                *modifiers = new_modifiers.state();
            }
            WindowEvent::Resized(_) => {
                *resized = true;
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            _ => {}
        }

        // Handle (Application Generated) navigation events
        for nav_event in nav_events {
            navigation_state.process_event(&nav_event);
        }

        // Map window event to iced event
        if let Some(event) = conversion::window_event(
            event,
            window.scale_factor() as f32,
            *modifiers,
        ) {
            events.push(event);
        }

        // If there are events pending
        if !events.is_empty() {
            // We process them
            let mut interface = UserInterface::build(
                controls.view(),
                viewport.logical_size(),
                std::mem::take(cache),
                renderer,
            );

            let mut messages = Vec::new();

            let _ = interface.update(
                events,
                *cursor,
                renderer,
                clipboard,
                &mut messages,
            );

            events.clear();
            *cache = interface.into_cache();

            // update our UI with any messages
            for message in messages {
                controls.update(message);
            }

            // Open the Custom Palette Editor window if the user asked (only the
            // event loop can create a window, so Controls just raises a flag).
            if controls.take_open_palette_editor() && palette_window.is_none() {
                *palette_window = Some(PaletteWindow::new(
                    event_loop, instance, adapter, scene, *format,
                ));
            }

            // and request a redraw
            window.request_redraw();
        }
    }
}

pub fn main() -> Result<(), winit::error::EventLoopError> {
    // Define a better log format!
    env_logger::builder()
        .format(|buf, record| {
            // Color the level
            let level_style = buf.default_level_style(record.level()).bold();
            let bold = Style::new().bold();

            // Time: local, with microseconds
            let ts = Local::now().format("%y-%m-%d %H:%M:%S:%6f");

            // Thread ID/truncated name
            let thread = std::thread::current();
            let thread_id = format!("{:?}", thread.id());

            // Level, module (target)
            writeln!(
                buf,
                "{bold}[{} {{{}}}{bold:#} {level_style}{}{level_style:#} {bold}{}]{bold:#} {}",
                ts,
                thread_id,
                record.level(),
                record.target(),
                record.args()
            )
        })
        .init();

    // Load app settings from settings file
    let settings = Settings::new().unwrap();
    info!("Loaded application settings");

    // Initialize winit
    let event_loop = EventLoop::new().expect("Opening winit Event Loop");
    let mut runner = Runner::Loading{settings};

    info!("Starting Winit event loop");

    // Run the event loop forever
    event_loop.run_app(&mut runner)
}