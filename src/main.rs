mod controls;
mod scene;
mod gpu_pipeline;

#[allow(dead_code)]
mod numerics;

#[allow(dead_code)]
mod signals;

#[allow(dead_code)]
mod scout_engine;

use controls::Controls;
use controls::Message;
use scene::Scene;

use iced_wgpu::graphics::{Viewport, Shell};
use iced_wgpu::{Engine, Renderer, wgpu};
use iced_winit::Clipboard;
use iced_winit::conversion;
use iced_winit::core::mouse;
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

use log::{debug, error, info};
use chrono::Local;
use anstyle::Style;

use std::io::Write;
use std::rc::Rc;
use std::cell::RefCell;
use std::sync::Arc;
use std::process;
use std::time::Instant;

#[allow(clippy::large_enum_variant)]
enum Runner {
    Loading,
    Ready {
        window: Arc<Window>,
        queue: wgpu::Queue,
        device: wgpu::Device,
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
        // For tracking mouse movment and shifting (offsetting) the fractal
        prev_pos: (f64, f64),
        mouse_lb_state: ElementState,
        mouse_rb_state: ElementState,
    },
}

impl ApplicationHandler for Runner {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        debug!("Runner.resumed");
        if let Self::Loading = self {
            let window = Arc::new(event_loop.create_window(
                WindowAttributes::default()).unwrap_or_else(|e| {
                    error!("Failed to Create window .. {}", e);
                    process::exit(1);
                }));

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
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
                desired_maximum_frame_latency: 2});

            // Initialize scene and GUI controls
            let scene = Rc::new(RefCell::new(
                Scene::new(window.clone(), &device, format,
                    physical_size.width.into(),
                    physical_size.height.into()
                )));
            // Take a snapshot of where the camera is in the beginning to send to ScoutEngine
            // So that an initial reference orbit can be computed.
            scene.borrow_mut().take_camera_snapshot();
            
            let controls = Controls::new(Rc::clone(&scene));

            // Initialize iced
            let renderer = {
                let engine = Engine::new(
                    &adapter,
                    device.clone(),
                    queue.clone(),
                    format,
                    None,
                    Shell::headless(),
                );

                Renderer::new(engine, Font::default(), Pixels::from(16))
            };

            // Change this to render continuously
            event_loop.set_control_flow(ControlFlow::Wait);
            
            let prev_pos = (-1.0, -1.0);
            let mouse_lb_state = ElementState::Released;
            let mouse_rb_state = ElementState::Released;

            info!("ApplicationHandler Runner Initialized");
            *self = Self::Ready {window, device, queue, surface, format, renderer,
                scene: Rc::clone(&scene), controls, events: Vec::new(), cursor: mouse::Cursor::Unavailable,
                cache: user_interface::Cache::new(), modifiers: ModifiersState::default(),
                clipboard, viewport, resized: false, prev_pos, mouse_lb_state, mouse_rb_state};
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop,
            _window_id: WindowId, event: WindowEvent) {
        let Self::Ready {window, device, queue, surface, format, renderer, scene,
            controls, events, viewport, cursor, cache, modifiers, clipboard, resized,
            mouse_lb_state, mouse_rb_state, prev_pos} = self
        else {
            return;
        };
        //trace!("Runner.window_event - {:?} {:?}", _window_id, event);
        
        match event {
            WindowEvent::RedrawRequested => {
                if *resized {
                    let size = window.inner_size();
                    scene.borrow_mut().set_window_size(
                        size.width.into(), 
                        size.height.into());

                    *viewport = Viewport::with_physical_size(Size::new(size.width, size.height),
                        window.scale_factor() as f32);

                    surface.configure(device, &wgpu::SurfaceConfiguration {
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
                            s.query_tile_orbits(queue);

                            let scout_diags = s.read_scout_diagnostics();
                            let mut scout_diags_g = scout_diags.lock();
                            if !scout_diags_g.consumed {
                                controls.update(Message::UpdateDebugText(scout_diags_g.message.clone()));
                                scout_diags_g.consumed = true;
                            }

                            // Draw the scene (contains both fragment render and compute passes)
                            s.draw(&device, &queue, &view);

                            s.stamp_frame();
                            s.read_debug(&device, &queue);
                            s.read_grid_feedback(&device, &queue);
                            s.read_tile_feedback(&device, &queue);
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

                match mouse_rb_state {
                    ElementState::Pressed => {
                        if prev_pos.0 > 0.0 {
                            let diff = ((position.x - prev_pos.0) as f64,
                                        (position.y - prev_pos.1) as f64);

                            debug!("CursorMoved & ElementState::Pressed prev_pos={:?} new_pos={:?} diff={:?}", 
                                prev_pos, position, diff);

                            scene.borrow_mut().change_center(diff);
                        } else {
                            debug!("CursorMoved & ElementState::Pressed starting pos={:?}", position);
                        }

                        prev_pos.0 = position.x;
                        prev_pos.1 = position.y;
                        
                        window.request_redraw();
                    }
                    ElementState::Released => {
                        if prev_pos.0 > 0.0 {
                            scene.borrow_mut().take_camera_snapshot();
                        }

                        *prev_pos = (-1.0, -1.0);
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let MouseScrollDelta::LineDelta(_, h) = delta {
                    let new_scale = scene.borrow_mut().change_scale(if h > 0.0 {true} else {false});
                    scene.borrow_mut().take_camera_snapshot();

                    debug!("MouseWheel & MouseScrollDelta::LineDelta ---> h={} scale={}", 
                        h, new_scale);
                }
            }
            WindowEvent::MouseInput { state, button, ..} => {
                match button {
                    MouseButton::Left => {
                        *mouse_lb_state = state;
                    }
                    MouseButton::Right => {
                        *mouse_rb_state = state;
                    }
                    _ => {}
                }
            }
            WindowEvent::KeyboardInput { device_id: _, ref event, .. } => {
                let KeyEvent{physical_key, ..} = event;
                let PhysicalKey::Code (c) = physical_key else {
                    return;
                };
                let new_scale: String;
                
                match c {
                    KeyCode::ArrowUp => {
                        new_scale = scene.borrow_mut().change_scale(true);
                        scene.borrow_mut().take_camera_snapshot();
                    }
                    KeyCode::ArrowDown => {
                        new_scale = scene.borrow_mut().change_scale(false);
                        scene.borrow_mut().take_camera_snapshot();
                    }
                    _ => {
                        new_scale = "".to_string();
                    }
                }

                debug!("Arrow Key Pressed! new scale={}", new_scale);
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

    // Initialize winit
    let event_loop = EventLoop::new().expect("Opening winit Event Loop");
    let mut runner = Runner::Loading;

    info!("Starting Winit event loop");

    // Run the event loop forever
    event_loop.run_app(&mut runner)
}