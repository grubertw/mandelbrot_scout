pub mod import;

use std::collections::HashMap;
use crate::settings::Settings;
use crate::gpu_pipeline::builder::*;
use crate::gpu_pipeline::structs::*;
use crate::signals::*;
use crate::scout_engine::{ScoutEngineConfig, ScoutEngine, ScoutConfig, ScoutSignal};
use crate::scout_engine::orbit::OrbitId;
use crate::scout_engine::utils::complex_delta;

use futures::channel;
use futures::executor;
use iced_wgpu::wgpu;
use iced_wgpu::wgpu::BufferAsyncError;
use iced_winit::winit::window::Window;

use rug::{Float, Complex};
use log::{trace, debug, info, warn};
use std::sync::Arc;
use std::time;
use parking_lot::Mutex;
use crate::scene::import::{FractalMetadata, RefOrbitMetadata, META_VERSION};
use crate::TITLE;

#[derive(Debug, Clone)]
struct LoadedOrbit {
    rank: u32,
    orbit_id: OrbitId,
    orbit_c_ref: Complex,
    center_offset: Complex,
    gpu_orb_loc_info: GpuRefOrbitLocation
}

#[derive(Debug, Clone)]
pub struct Rgba8Palette {
    pub name: String,
    palette: Vec<[u8; 4]>,
}

#[derive(Debug)]
pub struct Scene {
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    frame_id: u64,
    frame_timestamp: time::Instant,
    center: Complex, // scaled and shifted with mouse drag
    scale: Float,
    scale_factor: Float,
    width: f64,
    height: f64,
    render_res_factor: f64,
    render_res_factor_during_pan: f64,
    max_orbits_per_frame: u32,
    max_palette_colors: u32,
    scout_engine: ScoutEngine,
    loaded_orbits: Vec<LoadedOrbit>,
    uniform: SceneUniform,
    pipeline: PipelineBundle,
    selected_palette: String,
    palette_changed: bool,
    color_palettes: HashMap<String, Rgba8Palette>,
    recalc_fractal: bool,
    recalc_color: bool
}

impl Scene {
    pub fn new(
        window: Arc<Window>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        texture_format: wgpu::TextureFormat,
        width: f64, height: f64,
        settings: &Settings
    ) -> Scene {
        let center = Complex::with_val(settings.init_rug_precision, settings.center);
        let scale = Float::with_val(settings.init_rug_precision, settings.complex_span / width);

        // Configure ScoutEngine (our single source of truth for reference orbits)
        let scout_config =  ScoutEngineConfig {
            max_live_orbits: settings.max_live_orbits,
            max_user_iters: settings.max_user_iter,
            max_ref_orbit_iters: settings.max_ref_orbit,
            auto_start: settings.auto_start,
            starting_scale: settings.starting_scale,
            ref_iters_multiplier: settings.ref_iters_multiplier,
            num_seeds_to_spawn_per_eval: settings.num_seeds_to_spawn_per_eval,
            num_qualified_orbits: settings.num_qualified_orbits,
            rug_precision: settings.init_rug_precision,
            exploration_budget: settings.exploration_budget,
        };

        let scout_engine = ScoutEngine::new(
            scout_config,
            window.clone(),
            CameraSnapshot::new(
                FrameStamp::new(), center.clone(), scale.clone(),
                width.max(height) / 2.0,
            ));

        let color_palettes =
            build_color_palettes_from_settings(settings);
        let default_palette = color_palettes.get("default").unwrap();

        let uniform = SceneUniform {
            center_x: center.real().to_f32(),
            center_y: center.imag().to_f32(),
            scale: scale.to_f32(),
            view_width: width as f32,
            view_height: height as f32,
            render_width: width as u32,
            render_height: height as u32,
            render_tex_width: settings.render_tex_width as f32,
            render_tex_height: settings.render_tex_height as f32,
            max_iter: settings.max_user_iter,
            ref_orb_count: 0,
            grid_size: settings.screen_grid_size,
            grid_width: width as u32 / settings.screen_grid_size,
            render_flags: 0,
            stripe_density: settings.stripe_density,
            stripe_strength: settings.stripe_strength,
            stripe_gamma: settings.stripe_gamma,
            color_scalar_mapping_mode: 0,
            color_scaler_mapping_strength: 1.0,
            palette_tex_width: settings.max_palette_colors,
            palette_len: default_palette.palette.len() as u32,
            palette_cycles: 1.0,
            palette_offset: 0.0,
            palette_gamma: 1.0,
            distance_multiplier: settings.distance_multiplier,
            glow_intensity: settings.glow_intensity,
            neighbor_scale_multiplier: settings.neighbor_scale_multiplier,
            ambient_intensity: settings.ambient_intensity,
            key_light_intensity: settings.key_light_intensity,
            key_light_azimuth: settings.key_light_azimuth,
            key_light_elevation: settings.key_light_elevation,
            fill_light_intensity: settings.fill_light_intensity,
            fill_light_azimuth: settings.fill_light_azimuth,
            fill_light_elevation: settings.fill_light_elevation,
            specular_intensity: settings.specular_intensity,
            specular_power: settings.specular_power,
            ao_darkness: settings.ao_darkness,
            rim_intensity: settings.rim_intensity,
            rim_power: settings.rim_power,
        };

        // Configure and initialize all WGPU resources for render passes.
        let pipeline = PipelineBundle::build_pipelines(
            &device, &uniform, texture_format, settings);

        Scene {
            window, device, queue,
            frame_id: 0, frame_timestamp: time::Instant::now(),
            center, scale, scale_factor: Float::with_val(settings.init_rug_precision, 1.05),
            width, height,
            render_res_factor: settings.render_res_factor,
            render_res_factor_during_pan: settings.render_res_factor_during_pan,
            max_orbits_per_frame: settings.max_orbits_per_frame,
            max_palette_colors: settings.max_palette_colors,
            scout_engine,
            loaded_orbits: Vec::new(),
            uniform, pipeline,
            selected_palette: "default".to_string(),
            palette_changed: true, color_palettes,
            recalc_fractal: true, recalc_color: true,
        }
    }

    pub fn render(
        &mut self,
        width: u32,
        height: u32,
        target: Option<&wgpu::TextureView>
    ) {
        // set the render size and compute shader workgroup size depending on desired size for this
        // render pass. If target texture is None, it is assumed that a read for image data in the
        // render texture will be performed after (i.e. an invocation from the export module).
        self.uniform.render_width = width.min(self.uniform.render_tex_width as u32);
        self.uniform.render_height = height.min(self.uniform.render_tex_height as u32);
        self.uniform.grid_width = width / self.uniform.grid_size;
        let gx = (self.uniform.render_width  + 15) / 16;
        let gy = (self.uniform.render_height + 15) / 16;

        // If invoked from export (i.e. target=None), ensure render & view dimensions match
        if target.is_none() {
            self.uniform.view_width = self.uniform.render_width as f32;
            self.uniform.view_height = self.uniform.render_height as f32;
        }

        // Upload color palette texture, if changed.
        self.upload_color_palette();

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render encoder"),
        });

        trace!("Render width={} height={} with uniform={:?}", 
            self.uniform.render_width, self.uniform.render_height, self.uniform);
        // Uniforms must be updated on every draw operation.
        self.queue.write_buffer(&self.pipeline.uniform_buff, 0, bytemuck::cast_slice(&[self.uniform]));

        if self.recalc_fractal {
            {
                trace!("Compute pass clear storage textures. gx={} gy={}", gx, gy);
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("clear feedback pass"),
                    timestamp_writes: None
                });

                cpass.set_pipeline(&self.pipeline.clear_pipeline);
                cpass.set_bind_group(0, &self.pipeline.clear_bg, &[]);
                cpass.dispatch_workgroups(gx, gy, 1);
            }
            {
                trace!("Compute pass mandelbrot calculate. gx={} gy={}.", gx, gy);
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Calculate Fractal pass"),
                    timestamp_writes: None
                });

                // Update/refresh orbit locations before use.
                self.upload_orbit_locations();
                cpass.set_pipeline(&self.pipeline.calc_mandel_pipeline);
                cpass.set_bind_group(0, &self.pipeline.calc_bg, &[]);
                cpass.set_bind_group(1, &self.pipeline.debug_bg, &[]);
                cpass.dispatch_workgroups(gx, gy, 1);
            }
            {
                trace!("Compute pass reduce/aggregate. gx={} gy={}", gx, gy);
                let mut rpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("reduce tiles pass"),
                    timestamp_writes: None
                });

                rpass.set_pipeline(&self.pipeline.reduce_pipeline);
                rpass.set_bind_group(0, &self.pipeline.reduce_bg, &[]);
                rpass.dispatch_workgroups(gx, gy, 1);
            }
        }
        if self.recalc_color || self.recalc_fractal {
            self.recalc_color = false;
            trace!("Begin color pass");
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("color pass"),
                timestamp_writes: None
            });

            cpass.set_pipeline(&self.pipeline.color_pipeline);
            cpass.set_bind_group(0, &self.pipeline.color_bg, &[]);
            cpass.dispatch_workgroups(gx, gy, 1);
        }
        if let Some(t) = target {
            trace!("Begin display pass");
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Display pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: t,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear({
                            wgpu::Color::BLACK
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline.display_pipeline);
            render_pass.set_bind_group(0, &self.pipeline.display_bg, &[]);
            render_pass.draw(0..3, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.stamp_frame();

        if self.recalc_fractal {
            self.recalc_fractal = false;

            self.read_debug();
            self.read_grid_feedback();
            self.read_orbit_feedback();
        }

        // Reset the view dimensions when finished
        if target.is_none() {
            self.uniform.view_width = self.width as f32;
            self.uniform.view_height = self.height as f32;
            // Ensure export does not affect the viewport
            self.recalc_fractal = true;
        }
    }

    pub fn read_grid_feedback(&mut self) {
        let byte_size = 
            (self.uniform.grid_width
            * (self.uniform.render_height / self.uniform.grid_size)) as u64
            * std::mem::size_of::<GridFeedbackOut>() as u64;
        if byte_size == 0 {
            warn!("This frame had no grid feedback!");
            return;
        }

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("grid feedback copy encoder"),
            },
        );

        encoder.copy_buffer_to_buffer(
            &self.pipeline.grid_feedback_buffer,
            0,
            &self.pipeline.grid_feedback_readback,
            0,
            byte_size,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.pipeline.grid_feedback_readback.slice(0..byte_size);

        let (sender, receiver) = futures::channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            sender.send(res).ok();
        });

        self.device.poll(wgpu::PollType::Wait{submission_index: None, timeout: None})
            .expect("Failed to poll grid feedback");

        let feedback = executor::block_on(async {
            match receiver.await {
                Ok(Ok(())) => {
                    let data = buffer_slice.get_mapped_range();
                    // SAFETY: TileFeedbackOut is Pod + repr(C)
                    let slice: &[GridFeedbackOut] =
                        bytemuck::cast_slice(&data);

                    let result = slice.to_vec();
                    drop(data);
                    self.pipeline.grid_feedback_readback.unmap();
                    result
                }
                _ => {
                    self.pipeline.grid_feedback_readback.unmap();
                    Vec::new()
                }
            }
        });

        let mut trace_str = String::from(format!("Scene read {} GpuSamples.\n", feedback.len()).as_str());

        let grid_samples: Vec<GpuGridSample> = feedback
            .iter()
            .filter_map(|sample| {
                // If the sample was perturbed, then we must derive it's 'c' value from center-offset
                // info found within the GpuOrbitLocation struct
                let best_sample= if sample.perturbed() {
                    let delta_c_pair =
                        self.build_delta_c_from_orbit_location(
                            sample.best_pixel_x, sample.best_pixel_y, sample.orbit_idx()
                        );
                    if delta_c_pair.is_some() {
                        delta_c_pair.unwrap()
                    }
                    else {
                        return None
                    }
                }
                else {
                    self.build_c_from_scene(sample.best_pixel_x, sample.best_pixel_y)
                };

                let sample_iters = sample.iter();
                let escaped = sample.escaped();

                if !escaped {
                    trace_str.push_str(
                        format!("[{:<3}, {:<3}]\tc={:<56} depth={:<3} escaped={} perturbed={} \
                        max_iters_reached={} orbit_idx={}\traw_flags={}\n",
                            sample.best_pixel_x,
                            sample.best_pixel_y,
                            best_sample.to_string_radix(10, Some(18)),
                            sample_iters, escaped,
                            sample.perturbed(),
                            sample.max_iters_reached(),
                            sample.orbit_idx(),
                            sample.best_pixel_flags
                    ).as_str());
                }

               Some( GpuGridSample {
                    frame_stamp: FrameStamp {
                        frame_id: self.frame_id,
                        timestamp: self.frame_timestamp
                    },
                    location: best_sample,
                    iters_reached: sample_iters,
                    escaped,
                    max_user_iters: self.uniform.max_iter
                })
            })
            .collect();

        //trace!("{}", trace_str);
        self.scout_engine.set_grid_samples(grid_samples)
    }

    pub fn read_orbit_feedback(&mut self) {
        let byte_size = self.uniform.ref_orb_count as u64 * size_of::<OrbitFeedbackOut>() as u64;
        if byte_size == 0 {
            trace!("This frame had no orbit feedback. Tile count was zero!");
            return;
        }

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("orbit feedback copy encoder"),
            },
        );

        encoder.copy_buffer_to_buffer(
            &self.pipeline.orbit_feedback_buffer,
            0,
            &self.pipeline.orbit_feedback_readback,
            0,
            byte_size,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.pipeline.orbit_feedback_readback.slice(0..byte_size);

        let (sender, receiver) = futures::channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            sender.send(res).ok();
        });

        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .expect("Failed to poll orbit feedback");

        let feedback = executor::block_on(async {
            match receiver.await {
                Ok(Ok(())) => {
                    let data = buffer_slice.get_mapped_range();
                    // SAFETY: OrbitFeedbackOut is Pod + repr(C)
                    let slice: &[OrbitFeedbackOut] =
                        bytemuck::cast_slice(&data);

                    let result = slice.to_vec();
                    drop(data);
                    self.pipeline.orbit_feedback_readback.unmap();
                    result
                }
                _ => {
                    self.pipeline.orbit_feedback_readback.unmap();
                    Vec::new()
                }
            }
        });

        if self.loaded_orbits.len() == self.uniform.ref_orb_count as usize {
            let observations: Vec<OrbitObservation> = feedback
                .iter()
                .enumerate()
                .map(|(i, v)| OrbitObservation {
                    frame_stamp: FrameStamp {
                        frame_id: self.frame_id,
                        timestamp: self.frame_timestamp
                    },
                    orbit_id: self.loaded_orbits[i].orbit_id,
                    feedback: *v
                })
                .collect();

            self.scout_engine.submit_orbit_observations(observations);
        }
    }

    pub fn read_render_feedback(&mut self, width: u32, height: u32) -> Vec<u8> {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render buffer copy encoder"),
        });

        let padded_bytes_per_row = ((4 * width + 255) / 256) * 256;

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.pipeline.render_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.pipeline.render_readback_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // submit the copy
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.pipeline.render_readback_buf.slice(..);

        let (sender, receiver) = channel::oneshot::channel::<Result<(), BufferAsyncError>>();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        self.device.poll(wgpu::PollType::Wait{submission_index: None, timeout: None})
            .expect("Failed to poll render copy buffer");

        executor::block_on(async {
            match receiver.await {
                Ok(Ok(())) => {
                    let data = buffer_slice.get_mapped_range();
                    let slice : &[u8] = bytemuck::cast_slice(&data);
                    let result = slice.to_vec();
                    drop(data);
                    self.pipeline.render_readback_buf.unmap();
                    result
                }
                _ => {
                    self.pipeline.render_readback_buf.unmap();
                    Vec::new()
                }
            }
        })
    }

    pub fn read_debug(&self) {
        // 1) create encoder, copy storage -> readback
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("debug copy encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &self.pipeline.debug_buffer, // src
            0,
            &self.pipeline.debug_readback, // dst
            0,
            std::mem::size_of::<DebugOut>() as u64,
        );

        // submit the copy
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.pipeline.debug_readback.slice(..);

        let (sender, receiver) = channel::oneshot::channel::<Result<(), BufferAsyncError>>();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        self.device.poll(wgpu::PollType::Wait{submission_index: None, timeout: None})
            .expect("Failed to poll debug copy buffer");

        executor::block_on(async {
            if let Ok(Ok(_)) = receiver.await {
                let data = buffer_slice.get_mapped_range();
                let dbg = bytemuck::from_bytes::<DebugOut>(&data[..]).clone();

                debug!("FROM CPU (Scene struct)");
                debug!("  center = {:?}", self.center);
                debug!("  scale  = {:?}", self.scale);
                debug!("  width  = {}\theight = {}", self.width, self.height);
                debug!("FROM GPU:");
                debug!("  center = (re:{}, im:{})", dbg.center_x, dbg.center_y);
                debug!("  max_iters    = {}", dbg.max_iters);
                debug!("  fi           = {}", dbg.fi);
                debug!("  distance     = {}", dbg.distance);
                debug!("  stripe_avg   = {}", dbg.stripe_avg);
                debug!("  flags        = {}", dbg.flags);
    
                drop(data);
                self.pipeline.debug_readback.unmap();
            }
        });
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn scout(&self) -> &ScoutEngine {
        &self.scout_engine
    }

    pub fn scout_config(&self) -> ScoutConfig {
        self.scout_engine.config()
    }

    pub fn center(&self) -> &Complex {
        &self.center
    }

    pub fn scale(&self) -> &Float {
        &self.scale
    }

    pub fn width(&self) -> u32 {
        self.uniform.render_width
    }

    pub fn height(&self) -> u32 {
        self.uniform.render_height
    }
    
    pub fn max_width(&self) -> u32 {
        self.uniform.render_tex_width as u32
    }
    
    pub fn max_height(&self) -> u32 {
        self.uniform.render_tex_height as u32
    }

    pub fn render_res_factor(&self) -> f64 {
        self.render_res_factor
    }

    pub fn render_res_factor_during_pan(&self) -> f64 {
        self.render_res_factor_during_pan
    }

    pub fn send_scout_signal(&mut self, signal: ScoutSignal) {
        self.scout_engine.submit_scout_signal(signal);
    }

    pub fn read_scout_diagnostics(&self) -> Arc<Mutex<ScoutDiagnostics>> {
        self.scout_engine.read_diagnostics()
    }
    
    pub fn set_render_res_factor(&mut self, res_factor: f64) {
        self.render_res_factor = res_factor;
        self.recalc_fractal = true;
    }
    
    pub fn set_render_res_factor_during_pan(&mut self, res_factor: f64) {
        self.render_res_factor_during_pan = res_factor;
        self.recalc_fractal = true;
    }

    pub fn recalculate(&mut self) {
        self.recalc_fractal = true;
    }

    pub fn set_max_iterations(&mut self, max_iterations: u32) {
        self.uniform.max_iter = max_iterations;
        self.recalc_fractal = true;
        self.scout_engine.set_max_user_iterations(max_iterations)
    }

    pub fn set_window_size(&mut self, width: f64, height: f64) {
        self.width = width;
        self.height = height;
        self.uniform.view_width = width as f32;
        self.uniform.view_height = height as f32;
        self.recalc_fractal = true;
        self.update_window_title();
        debug!("Window size changed w={} h={}", width, height);
    }

    pub fn set_scale(&mut self, scale: Float) {
        self.scale = scale;
        self.recalc_fractal = true;
        self.uniform.scale = self.scale.to_f32();
        self.update_window_title();
    }

    pub fn change_scale(&mut self, increase: bool) -> String {
        if increase {
            self.scale *= &self.scale_factor;
        } else {
            self.scale /= &self.scale_factor;
        }

        self.uniform.scale = self.scale.to_f32();
        self.recalc_fractal = true;
        let s = self.scale.to_string_radix(10, None);
        debug!("Scale changed {}", s);
        self.update_window_title();
        s
    }

    fn update_center_offsets(&mut self) {
        let center = self.center().clone();
        self.uniform.center_x = center.real().to_f32();
        self.uniform.center_y = center.imag().to_f32();

        for loadout in self.loaded_orbits.iter_mut() {
            loadout.center_offset = complex_delta(&center, &loadout.orbit_c_ref);

            loadout.gpu_orb_loc_info.center_offset_re = loadout.center_offset.real().to_f32();
            loadout.gpu_orb_loc_info.center_offset_im = loadout.center_offset.imag().to_f32();
        }
    }

    pub fn set_center(&mut self, new_center: Complex) {
        self.center = new_center;
        self.recalc_fractal = true;
        self.update_center_offsets();
        self.update_window_title();
    }

    pub fn change_center(&mut self, center_diff: (f64, f64)) -> String {
        let dx = self.scale.clone() * center_diff.0;
        let dy = self.scale.clone() * center_diff.1;

        let (real, imag) = self.center.as_mut_real_imag();
        *real -= &dx;
        *imag += &dy;

        self.update_center_offsets();
        self.recalc_fractal = true;
        let c = self.center.to_string_radix(10, None);
        debug!("Center changed {:?} ----- diff ({:?} {:?})", 
            c, dx, dy);
        self.update_window_title();
        c
    }
    pub fn set_debug_coloring(&mut self, debug_coloring: bool) {
        self.uniform.set_debug_coloring(debug_coloring);
        self.recalc_color = true;
    }
    
    pub fn set_glitch_fix(&mut self, glitch_fix: bool) {
        self.uniform.set_glitch_fix(glitch_fix);
        self.recalc_fractal = true;
    }
    
    pub fn set_smooth_coloring(&mut self, smooth_coloring: bool) {
        self.uniform.set_smooth_coloring(smooth_coloring);
        self.recalc_fractal = true;
    }

    pub fn set_use_de(&mut self, use_de: bool) {
        self.uniform.set_use_de(use_de);
        self.recalc_fractal = true;
    }

    pub fn set_use_stripes(&mut self, use_stripes: bool) {
        self.uniform.set_use_stripes(use_stripes);
        self.recalc_fractal = true;
    }

    pub fn set_enable_glow(&mut self, enable_glow: bool) {
        self.uniform.set_enable_glow(enable_glow);
        self.recalc_color = true;
    }

    pub fn set_enable_key_light(&mut self, enable_key_light: bool) {
        self.uniform.set_enable_key_light(enable_key_light);
        self.recalc_color = true;
    }

    pub fn set_enable_fill_light(&mut self, enable_fill_light: bool) {
        self.uniform.set_enable_fill_light(enable_fill_light);
        self.recalc_color = true;
    }

    pub fn set_enable_specular(&mut self, enable_specular: bool) {
        self.uniform.set_enable_specular(enable_specular);
        self.recalc_color = true;
    }

    pub fn set_enable_ao(&mut self, enable_ao: bool) {
        self.uniform.set_enable_ao(enable_ao);
        self.recalc_color = true;
    }

    pub fn set_enable_rim(&mut self, enable_rim: bool) {
        self.uniform.set_enable_rim(enable_rim);
        self.recalc_color = true;
    }
    
    // Obtain an enumerated list/mapping of color palettes
    pub fn get_palette_list(&self) -> Vec<(String, String)> {
        let mut palette_list = Vec::new();
        self.color_palettes.
            iter()
            .for_each(|(key, palette)| {
            palette_list.push((key.to_string(), palette.name.clone()));
        });
        palette_list
    }

    pub fn add_palette(&mut self, key: &String, palette: Rgba8Palette) {
        self.color_palettes.insert(key.clone(), palette);
        self.set_selected_palette(key);
    }

    pub fn set_selected_palette(&mut self, key: &String) {
        self.selected_palette = key.clone();
        self.palette_changed = true;
        self.recalc_color = true;
    }

    pub fn set_palette_cycles(&mut self, cycles: f32) {
        self.uniform.palette_cycles = cycles;
        self.recalc_color = true;
    }
    
    pub fn set_palette_offset(&mut self, offset: f32) {
        self.uniform.palette_offset = offset;
        self.recalc_color = true;
    }
    
    pub fn set_palette_gamma(&mut self, gamma: f32) {
        self.uniform.palette_gamma = gamma;
        self.recalc_color = true;
    }

    pub fn set_color_scalar_mapping_mode(&mut self, mode: u32) {
        self.uniform.color_scalar_mapping_mode = mode;
        self.recalc_color = true;
    }

    pub fn set_color_scalar_mapping_strength(&mut self, strength: f32) {
        self.uniform.color_scaler_mapping_strength = strength;
        self.recalc_color = true;
    }

    pub fn set_distance_multiplier(&mut self, distance_multiplier: f32) {
        self.uniform.distance_multiplier = distance_multiplier;
        self.recalc_color = true;
    }

    pub fn set_glow_intensity(&mut self, glow_intensity: f32) {
        self.uniform.glow_intensity = glow_intensity;
        self.recalc_color = true;
    }

    pub fn set_neighbor_scale(&mut self, neighbor_scale: f32) {
        self.uniform.neighbor_scale_multiplier = neighbor_scale;
        self.recalc_color = true;
    }

    pub fn set_ambient_intensity(&mut self, ambient_intensity: f32) {
        self.uniform.ambient_intensity = ambient_intensity;
        self.recalc_color = true;
    }

    pub fn set_key_light_intensity(&mut self, key_light_intensity: f32) {
        self.uniform.key_light_intensity = key_light_intensity;
        self.recalc_color = true;
    }

    pub fn set_key_light_azimuth(&mut self, key_light_azimuth: f32) {
        self.uniform.key_light_azimuth = key_light_azimuth;
        self.recalc_color = true;
    }

    pub fn set_key_light_elevation(&mut self, key_light_elevation: f32) {
        self.uniform.key_light_elevation = key_light_elevation;
        self.recalc_color = true;
    }

    pub fn set_fill_light_intensity(&mut self, fill_light_intensity: f32) {
        self.uniform.fill_light_intensity = fill_light_intensity;
        self.recalc_color = true;
    }

    pub fn set_fill_light_azimuth(&mut self, fill_light_azimuth: f32) {
        self.uniform.fill_light_azimuth = fill_light_azimuth;
        self.recalc_color = true;
    }

    pub fn set_fill_light_elevation(&mut self, fill_light_elevation: f32) {
        self.uniform.fill_light_elevation = fill_light_elevation;
        self.recalc_color = true;
    }

    pub fn set_specular_intensity(&mut self, specular_intensity: f32) {
        self.uniform.specular_intensity = specular_intensity;
        self.recalc_color = true;
    }

    pub fn set_specular_power(&mut self, specular_power: f32) {
        self.uniform.specular_power = specular_power;
        self.recalc_color = true;
    }

    pub fn set_ao_darkness(&mut self, ao_darkness: f32) {
        self.uniform.ao_darkness = ao_darkness;
        self.recalc_color = true;
    }

    pub fn set_stripe_density(&mut self, stripe_density: f32) {
        self.uniform.stripe_density = stripe_density;
        self.recalc_fractal = true;
    }

    pub fn set_stripe_strength(&mut self, stripe_strength: f32) {
        self.uniform.stripe_strength = stripe_strength;
        self.recalc_fractal = true;
    }

    pub fn set_stripe_gamma(&mut self, stripe_gamma: f32) {
        self.uniform.stripe_gamma = stripe_gamma;
        self.recalc_fractal = true;
    }
    
    pub fn set_rim_intensity(&mut self, rim_intensity: f32) {
        self.uniform.rim_intensity = rim_intensity;
        self.recalc_color = true;
    }
    
    pub fn set_rim_power(&mut self, rim_power: f32) {
        self.uniform.rim_power = rim_power;
        self.recalc_color = true;
    }

    pub fn stamp_frame(&mut self) {
        self.frame_id += 1;
        self.frame_timestamp = time::Instant::now();
    }

    fn build_camera_snapshot(&mut self) -> CameraSnapshot {
        CameraSnapshot::new(
            FrameStamp {
                frame_id: self.frame_id,
                timestamp: self.frame_timestamp
            },
            self.center.clone(),
            self.scale.clone(),
            self.width.max(self.height) / 2.0
        )
    }

    pub fn take_camera_snapshot(&mut self) {
        let snapshot = self.build_camera_snapshot();
        self.scout_engine.submit_camera_snapshot(snapshot);
    }

    pub fn query_qualified_orbits(&mut self) {
        // First check if ScoutEngine's context has changed. 
        if self.scout_engine.context_changed() {
            self.recalc_fractal = true;
            let orbits = self.scout_engine.query_qualified_orbits();

            // Upload qualified orbits to a Texture2D
            self.upload_orbits(&orbits);

            // Build GPU-specific orbit locations from scout-engine's tile-orbit pairs
            self.loaded_orbits =
                build_gpu_orbit_loadout_from_qualified_orbits(&orbits, self.center());
            info!("Rebuilt orbit loadout! len={}", self.loaded_orbits.len());

            self.upload_orbit_locations();
        }
    }

    fn upload_orbits(&mut self, orbits: &Vec<QualifiedOrbit>) {
        if orbits.len() == 0 {
            trace!("Orbit Count reduced to zero! Scout likely shutdown!");
            self.uniform.ref_orb_count = 0;
            return;
        }

        let mut orbit_count: u32 = 0;
        if orbits.len() > 0 {
            let ranked_orb = &orbits[0];
            self.queue.write_buffer(&self.pipeline.rank_one_orbit_buf, 0, bytemuck::cast_slice(&ranked_orb.orbit));
            orbit_count += 1;
            trace!("Wrote Rank One RefOrb to GPU! Wrote {} bytes to storage buffer for {} orbits! ",
                ranked_orb.orbit.len() * 8, ranked_orb.orbit.len());
        }

        if orbits.len() > 1 {
            let ranked_orb = &orbits[1];
            self.queue.write_buffer(&self.pipeline.rank_two_orbit_buf, 0, bytemuck::cast_slice(&ranked_orb.orbit));
            orbit_count += 1;
            trace!("Wrote Rank Two RefOrb to GPU! Wrote {} bytes to storage buffer for {} orbits! ",
                ranked_orb.orbit.len() * 8, ranked_orb.orbit.len());
        }

        self.uniform.ref_orb_count = orbit_count;
    }

    fn upload_orbit_locations(&self) {
        if self.loaded_orbits.len() > 0 && self.loaded_orbits.len() <= self.max_orbits_per_frame as usize {
            let mut trace_str = String::from(
                format!("Orbit Loadout has {} orbits. Mapping follows...\n",
                        self.loaded_orbits.len()).as_str());

            for load in self.loaded_orbits.iter() {
                trace_str.push_str(
                    format!("  Rank #{:<2}\tOrbitId={:<3?} center_offset={} orbit_c_ref={}\n",
                        load.rank, load.orbit_id,
                        load.center_offset.to_string_radix(10, Some(18)),
                        load.orbit_c_ref.to_string_radix(10, Some(18))
                    ).as_str());
            }
            trace!("{}", trace_str);

            let orb_locations: Vec<GpuRefOrbitLocation> = self.loaded_orbits
                .iter()
                .map(|l| l.gpu_orb_loc_info)
                .collect();
            self.queue.write_buffer(&self.pipeline.ref_orbit_location_buf, 0, bytemuck::cast_slice(&orb_locations));
        }
    }

    pub fn upload_color_palette(&mut self) {
        if let Some(palette) = self.color_palettes.get(&self.selected_palette)
                && self.palette_changed {
            self.palette_changed = false;

            let tex_width = self.max_palette_colors as usize;
            let src = &palette.palette;
            self.uniform.palette_len = src.len() as u32;

            // Create a repeating cycle of the palette within the texture. This works best
            // for texture color sampling, and for frequency/offset changes.
            let mut full_palette = Vec::<[u8; 4]>::with_capacity(tex_width);
            for i in 0..tex_width {
                full_palette.push(src[i % src.len()]);
            }

            let palette_bytes: &[u8] = bytemuck::cast_slice(&full_palette);

            trace!("Uploading rgba8 palette {}. len={} cycles={} offset={} gamma={} render_flags={} tex_width={} total_bytes={}",
                palette.name, self.uniform.palette_len,
                self.uniform.palette_cycles,
                self.uniform.palette_offset,
                self.uniform.palette_gamma,
                self.uniform.render_flags,
                self.max_palette_colors,
                palette_bytes.len());

            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.pipeline.palette_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                palette_bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * self.max_palette_colors),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: self.max_palette_colors,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    // Pixels from the GPU MUST first undergo f32 reconstruction first before
    // being taken to high precision. Otherwise, the collected sample is NOT being
    // faithful to 'c' that underwent escape-time calculation using its f32 arithmetic.
    fn build_c_from_scene(&self, pix_x: i32, pix_y: i32) -> Complex {
        let rw = self.uniform.render_width as f32;
        let rh = self.uniform.render_height as f32;

        let vw = self.uniform.view_width;
        let vh = self.uniform.view_height;

        let u = (pix_x as f32 + 0.5) / rw;
        let v = (pix_y as f32 + 0.5) / rh;

        let cu = u - 0.5;
        //let cv = 0.5 - v;
        let cv = v - 0.5;

        let off_x = cu * vw * self.uniform.scale;
        let off_y = cv * vh * self.uniform.scale;

        let c_re = off_x + self.uniform.center_x;
        let c_im = off_y + self.uniform.center_y;

        let prec = self.scale.prec();
        Complex::with_val(prec, (c_re, c_im))
    }

    // Under perturbation, f32 math must be specifically avoided.
    // GPU is using CPU-derived deltas anyway, so stay in high precision
    fn build_delta_c_from_orbit_location(
        &self,
        pix_x: i32,
        pix_y: i32,
        orbit_idx: u32
    ) -> Option<Complex> {
        let rw = self.uniform.render_width as f32;
        let rh = self.uniform.render_height as f32;

        let vw = self.uniform.view_width;
        let vh = self.uniform.view_height;

        let u = (pix_x as f32 + 0.5) / rw;
        let v = (pix_y as f32 + 0.5) / rh;

        let cu = u - 0.5;
        //let cv = 0.5 - v;
        let cv = v - 0.5;

        let off_x_f32 = (cu * vw) * self.uniform.scale;
        let off_y_f32 = (cv * vh) * self.uniform.scale;

        let prec = self.scale.prec();
        let off_x = Float::with_val(prec, off_x_f32);
        let off_y = Float::with_val(prec, off_y_f32);

        if let Some(loadout_loc) = self.loaded_orbits.get(orbit_idx as usize) {
            let mut c_re = loadout_loc.center_offset.real().clone();
            c_re += &off_x;

            let mut c_im = loadout_loc.center_offset.imag().clone();
            c_im += &off_y;

            c_re += loadout_loc.orbit_c_ref.real();
            c_im += loadout_loc.orbit_c_ref.imag();

            Some(Complex::with_val(prec, (c_re, c_im)))
        } else {
            None
        }
    }

    fn update_window_title(&self) {
        let title_str =
            format!("re: {} im: {}  scale: {}  w: {} h: {}",
                self.center.real().to_string_radix(10, Some(8)),
                self.center.imag().to_string_radix(10, Some(8)),
                self.scale.to_string_radix(10, Some(4)),
                self.width.to_string(), self.height.to_string()
            );
        self.window.set_title(&title_str);
    }

    pub fn build_metadata(&self) -> FractalMetadata {
        let center_re = self.center.real().to_string_radix(10, None);
        let center_im = self.center.imag().to_string_radix(10, None);
        let scale = self.scale.to_string_radix(10, None);

        let ref_orbit = self.loaded_orbits.first().map(|orb| {
            let c_ref_re = orb.orbit_c_ref.real().to_string_radix(10, None);
            let c_ref_im = orb.orbit_c_ref.imag().to_string_radix(10, None);

            let center_offset_re = orb.center_offset.real().to_string_radix(10, None);
            let center_offset_im = orb.center_offset.imag().to_string_radix(10, None);

            RefOrbitMetadata {
                c_ref_re, c_ref_im,
                center_offset_re, center_offset_im,
                max_ref_iters: orb.gpu_orb_loc_info.max_ref_iters,
            }
        });

        FractalMetadata {
            program_name: TITLE.to_string(),
            version: META_VERSION.to_string(),
            center_re, center_im, scale,
            max_iter: self.uniform.max_iter,
            ref_orbit,
        }
    }

    pub fn apply_metadata(&mut self, meta: FractalMetadata) {
        let re = Float::parse(meta.center_re).unwrap();
        let im = Float::parse(meta.center_im).unwrap();

        self.center = Complex::with_val(self.scale.prec(), (re, im));
        self.update_center_offsets();
        
        self.scale = Float::with_val(self.scale.prec(), Float::parse(meta.scale).unwrap());
        self.uniform.scale = self.scale.to_f32();
        
        self.uniform.max_iter = meta.max_iter;
        self.update_window_title();
        self.recalc_fractal = true;
    }
}

fn build_gpu_orbit_loadout_from_qualified_orbits(
    orbits: &Vec<QualifiedOrbit>,
    center: &Complex
) -> Vec<LoadedOrbit> {
    orbits.iter()
        .map(|orb| {
            let orbit_offset_from_center = complex_delta(center, &orb.c_ref);
            LoadedOrbit {
                rank: orb.rank,
                orbit_id: orb.orbit_id,
                orbit_c_ref: orb.c_ref.clone(),
                center_offset: orbit_offset_from_center.clone(),
                gpu_orb_loc_info: GpuRefOrbitLocation {
                    c_ref_re: orb.c_ref_32.re,
                    c_ref_im: orb.c_ref_32.im,
                    r_valid: orb.r_valid,
                    max_ref_iters: orb.escape_index.unwrap_or(orb.orbit.len() as u32),
                    center_offset_re: orbit_offset_from_center.real().to_f32(),
                    center_offset_im: orbit_offset_from_center.imag().to_f32(),
            }
        }
        })
        .collect()
}

fn build_color_palettes_from_settings(settings: &Settings) -> HashMap<String, Rgba8Palette> {
    let mut color_palettes: HashMap<String, Rgba8Palette> = HashMap::new();
    settings.palettes.iter().for_each(|(key, palette)| {
        // First validate the palette is well-formed
        if palette.array.len() % 3 != 0 {
            warn!("Palette {} array is not a multiple of 3! Skipping!", key);
        }
        else {
            color_palettes.insert(
                key.clone(),
                Rgba8Palette {
                    name: palette.name.clone(),
                    palette: palette.array
                        .iter()
                        .as_slice()
                        .chunks(3)
                        .map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
                        .collect(),
                }
            );
        }
    });
    color_palettes
}
