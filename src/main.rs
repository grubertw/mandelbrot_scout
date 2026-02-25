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

use iced_wgpu::graphics::Viewport;
use iced_wgpu::{Engine, Renderer, wgpu};
use iced_winit::Clipboard;
use iced_winit::conversion;
use iced_winit::core::event;
use iced_winit::core::mouse;
use iced_winit::core::renderer;
use iced_winit::core::{Font, Pixels, Size, Theme, Color};
use futures::executor;
use iced_winit::runtime::program;
use iced_winit::runtime::Debug;
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

#[allow(clippy::large_enum_variant)]
enum Runner {
    Loading,
    Ready {
        window: Arc<Window>,
        queue: wgpu::Queue,
        device: wgpu::Device,
        surface: wgpu::Surface<'static>,
        format: wgpu::TextureFormat,
        engine: Engine,
        renderer: Renderer,
        debug: Debug,
        scene: Rc<RefCell<Scene>>,
        state: program::State<Controls>,
        cursor_position: Option<winit::dpi::PhysicalPosition<f64>>,
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
                window.scale_factor() as f64);            

            let clipboard = Clipboard::connect(window.clone());
            let backend = wgpu::util::backend_bits_from_env().unwrap_or_default();

            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: backend,
                ..Default::default()});
            let surface = instance
                .create_surface(window.clone())
                .expect("Create window surface");

            debug!("Runner.resumed - physical_size={:?} scale_fac={:?} backend={:?}", 
                physical_size, window.scale_factor(), backend);

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
                            required_limits: wgpu::Limits::default()}, None)
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
            let scene = Rc::new(RefCell::new(Scene::new(window.clone(), &device, format, 
                physical_size.width.into(), 
                physical_size.height.into())));
            // Take a snapshot of where the camera is in the beginning to send to ScoutEngine
            // So that an initial reference orbit can be computed.
            scene.borrow_mut().take_camera_snapshot();
            
            let controls = Controls::new(Rc::clone(&scene));

            // Initialize iced
            let mut debug = Debug::new();
            let engine = Engine::new(&adapter, &device, &queue, format, None);
            let mut renderer = Renderer::new(&device, &engine, Font::default(), Pixels::from(16));
            let state = program::State::new(controls, viewport.logical_size(), &mut renderer, &mut debug);

            // Change this to render continuously
            event_loop.set_control_flow(ControlFlow::Wait);
            
            let prev_pos = (-1.0, -1.0);
            let mouse_lb_state = ElementState::Released;
            let mouse_rb_state = ElementState::Released;

            info!("ApplicationHandler Runner Initialized");
            *self = Self::Ready {window, device, queue, surface, format, engine, renderer, debug, 
                scene: Rc::clone(&scene), state, cursor_position: None, modifiers: ModifiersState::default(),
                clipboard, viewport, resized: false, prev_pos, mouse_lb_state, mouse_rb_state};
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop,
            _window_id: WindowId, event: WindowEvent) {
        let Self::Ready {window, device, queue, surface, format, engine, renderer, debug, scene, 
            state, viewport, cursor_position, modifiers, clipboard, resized, 
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
                        window.scale_factor() as f64);

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
                                state.queue_message(Message::UpdateDebugText(scout_diags_g.message.clone()));
                                scout_diags_g.consumed = true;
                            }

                            // Draw the scene (contains both fragment render and compute passes)
                            s.draw(&device, &queue, &view);

                            s.stamp_frame();
                            s.read_debug(&device, &queue);
                            //s.read_gpu_feedback(&device, &queue);
                            s.read_orbit_feedback(&device, &queue);
                            //s.read_gpu_texture_feedback(&device, &queue);
                        }

                        let mut encoder = device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor { label: None });

                        // Draw Iced on top
                        renderer.present(engine, &device, &queue, &mut encoder, 
                            None, frame.texture.format(), &view, viewport, &debug.overlay());
                        engine.submit(queue, encoder);

                        //debug!("draw frame={:?}", frame);
                        frame.present();

                        // Update the mouse cursor
                        window.set_cursor(iced_winit::conversion::mouse_interaction(
                            state.mouse_interaction()));
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
                *cursor_position = Some(position);

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

        // Map window event to iced event, and filter mouse movements when no mouse
        // buttons are preseed 
        if let Some(event) = conversion::window_event(event, window.scale_factor(), *modifiers) {
            let mut queue_event = true;

            if let event::Event::Mouse(me) = event {
                if let mouse::Event::CursorMoved{..} = me {
                    if   *mouse_lb_state == ElementState::Released 
                      || *mouse_lb_state == ElementState::Released {
                        queue_event = false;
                    }
                }
            }

            if queue_event {
                debug!("Queing window event={:?}", event);
                state.queue_event(event);
            }
        }

        // If there are events pending
        if !state.is_queue_empty() {
            debug!{"State Q is not empty, so requesting redraw"};

            // We update iced
            let _ = state.update(viewport.logical_size(), cursor_position
                    .map(|p| {conversion::cursor_position(p,viewport.scale_factor())})
                    .map(mouse::Cursor::Available)
                    .unwrap_or(mouse::Cursor::Unavailable),
                renderer, &Theme::Dark, &renderer::Style {text_color: Color::WHITE},
                clipboard, debug);

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