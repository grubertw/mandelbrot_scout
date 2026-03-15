use std::collections::HashMap;
use crate::settings::Settings;
use crate::gpu_pipeline::builder::*;
use crate::gpu_pipeline::structs::*;
use crate::numerics::*;
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

#[derive(Debug, Clone)]
struct LoadedOrbit {
    rank: u32,
    orbit_id: OrbitId,
    orbit_c_ref: Complex,
    center_offset: Complex,
    gpu_orb_loc_info: GpuRefOrbitLocation
}

#[derive(Debug, Clone)]
pub struct ColorPalette {
    pub name: String,
    palette: Vec<[u8; 4]>,
    pub offset: f32,
    pub frequency: f32,
    pub frequency_range: (f32, f32),
    pub gamma: f32,
}

#[derive(Debug)]
pub struct Scene {
    frame_id: u64,
    frame_timestamp: time::Instant,
    center: Complex, // scaled and shifted with mouse drag
    scale: Float,
    scale_factor: Float,
    width: f64,
    height: f64,
    max_screen_width: u32,
    max_screen_height: u32,
    max_orbits_per_frame: u32,
    max_palette_colors: u32,
    scout_engine: ScoutEngine,
    loaded_orbits: Vec<LoadedOrbit>,
    uniform: SceneUniform,
    pipeline: PipelineBundle,
    selected_palette: String,
    palette_changed: bool,
    color_palettes: HashMap<String, ColorPalette>,
}

impl Scene {
    pub fn new(window: Arc<Window>, device: &wgpu::Device, 
        texture_format: wgpu::TextureFormat,
        width: f64, height: f64,
        settings: &Settings
    ) -> Scene {
        let center = Complex::with_val(settings.init_rug_precision, settings.center);
        let c_df = ComplexDf::from_complex(&center);

        let scale = Float::with_val(settings.init_rug_precision, settings.complex_span / width);
        let scale_df = Df::from_float(&scale);

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
            window,
            CameraSnapshot::new(
                FrameStamp::new(), center.clone(), scale.clone(),
                width.max(height) / 2.0,
            ));

        let color_palettes =
            build_color_palettes_from_settings(settings);
        let default_palette = color_palettes.get("default").unwrap();

        let uniform = SceneUniform {
            center_x_hi: c_df.re.hi, center_x_lo: c_df.re.lo,
            center_y_hi: c_df.im.hi, center_y_lo: c_df.im.lo,
            scale_hi:    scale_df.hi, scale_lo:    scale_df.lo,
            screen_width: width as u32,
            screen_height: height as u32,
            max_iter: settings.max_user_iter,
            ref_orb_count: 0,
            grid_size: settings.screen_grid_size,
            grid_width: width as u32 / settings.screen_grid_size,
            palette_frequency: default_palette.frequency,
            palette_offset: 0.0,
            palette_gamma: 1.0,
            render_flags: 0,
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
            stripe_density: settings.stripe_density,
            stripe_strength: settings.stripe_strength,
            stripe_gamma: settings.stripe_gamma,
            rim_intensity: settings.rim_intensity,
            rim_power: settings.rim_power,
        };

        // Configure and initialize all WGPU resources for render passes.
        let pipeline = PipelineBundle::build_pipelines(
            device, &uniform, texture_format, settings);

        Scene { 
            frame_id: 0, frame_timestamp: time::Instant::now(), 
            center, scale, scale_factor: Float::with_val(settings.init_rug_precision, 1.05),
            width, height,
            max_screen_width: settings.max_screen_width,
            max_screen_height: settings.max_screen_height,
            max_orbits_per_frame: settings.max_orbits_per_frame,
            max_palette_colors: settings.max_palette_colors,
            scout_engine,
            loaded_orbits: Vec::new(),
            uniform, pipeline,
            selected_palette: "default".to_string(),
            palette_changed: true, color_palettes
        }
    }

    pub fn draw<'a>(&'a self, device: &wgpu::Device, queue: &wgpu::Queue, target: &'a wgpu::TextureView) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame encoder"),
            });
        let gx = (self.max_screen_width + 15) / 16;
        let gy = (self.max_screen_height + 15) / 16;

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
            trace!("Begin render pass");
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            trace!("Draw with uniform={:?}", self.uniform);
            // Uniforms must be updated on every draw operation.
            queue.write_buffer(&self.pipeline.uniform_buff, 0, bytemuck::cast_slice(&[self.uniform]));
            // So must orbit locations, if any exist
            self.upload_orbit_locations(queue);

            render_pass.set_pipeline(&self.pipeline.render_pipeline);
            render_pass.set_bind_group(0, &self.pipeline.scene_bg, &[]);
            render_pass.set_bind_group(1, &self.pipeline.ref_orbit_bg, &[]);
            render_pass.set_bind_group(2, &self.pipeline.palette_bg, &[]);
            render_pass.set_bind_group(3, &self.pipeline.debug_bg, &[]);
            render_pass.draw(0..3, 0..1);
        }

        {
            trace!("Compute pass reduce/aggregate. gx={} gy={}", gx, gy);

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce tiles pass"),
                timestamp_writes: None
            });

            cpass.set_pipeline(&self.pipeline.reduce_pipeline);
            cpass.set_bind_group(0, &self.pipeline.reduce_bg, &[]);

            cpass.dispatch_workgroups(gx, gy, 1);
        }

        queue.submit(Some(encoder.finish()));
    }

    pub fn read_grid_feedback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let byte_size = 
            (self.uniform.grid_width
            * (self.uniform.screen_height / self.uniform.grid_size)) as u64
            * std::mem::size_of::<GridFeedbackOut>() as u64;
        if byte_size == 0 {
            warn!("This frame had no grid feedback!");
            return;
        }

        let mut encoder = device.create_command_encoder(
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

        queue.submit(Some(encoder.finish()));

        let buffer_slice = self.pipeline.grid_feedback_readback.slice(0..byte_size);

        let (sender, receiver) = futures::channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            sender.send(res).ok();
        });

        device.poll(wgpu::PollType::Wait{submission_index: None, timeout: None})
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
                        perturbed_err={} max_iters_reached={} orbit_idx={}\traw_flags={}\n",
                            sample.best_pixel_x,
                            sample.best_pixel_y,
                            best_sample.to_string_radix(10, Some(18)),
                            sample_iters, escaped,
                            sample.perturbed(),
                            sample.perturb_err(),
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

    pub fn read_orbit_feedback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let byte_size = self.uniform.ref_orb_count as u64 * size_of::<OrbitFeedbackOut>() as u64;
        if byte_size == 0 {
            warn!("This frame had no orbit feedback. Tile count was zero!");
            return;
        }

        let mut encoder = device.create_command_encoder(
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

        queue.submit(Some(encoder.finish()));

        let buffer_slice = self.pipeline.orbit_feedback_readback.slice(0..byte_size);

        let (sender, receiver) = futures::channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            sender.send(res).ok();
        });

        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .expect("Failed to poll orbit feedback");

        let feedback = futures::executor::block_on(async {
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

    pub fn read_debug(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // 1) create encoder, copy storage -> readback
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
        queue.submit(Some(encoder.finish()));

        let buffer_slice = self.pipeline.debug_readback.slice(..);

        let (sender, receiver) = channel::oneshot::channel::<Result<(), BufferAsyncError>>();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        device.poll(wgpu::PollType::Wait{submission_index: None, timeout: None})
            .expect("Failed to poll debug copy buffer");

        executor::block_on(async {
            if let Ok(Ok(_)) = receiver.await {
                let data = buffer_slice.get_mapped_range();
                let dbg = bytemuck::from_bytes::<DebugOut>(&data[..]).clone();

                debug!("FROM CPU (Secne struct)");
                debug!("  center = {:?}", self.center);
                debug!("  scale  = {:?}", self.scale);
                debug!("  width  = {}\theight = {}", self.width, self.height);
                debug!("FROM GPU:");
                debug!("  center = (({},{}) ({},{}))", dbg.center_x_hi, dbg.center_x_lo, dbg.center_y_hi, dbg.center_y_lo);
                debug!("  max_iters    = {}", dbg.max_iters);
                debug!("  iter         = {}", dbg.iter);
                debug!("  nu_iter      = {}", dbg.nu_iter);
                debug!("  distance     = {}", dbg.distance);
                debug!("  t            = {}", dbg.t);
    
                drop(data);
                self.pipeline.debug_readback.unmap();
            }
        });
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

    pub fn send_scout_signal(&mut self, signal: ScoutSignal) {
        self.scout_engine.submit_scout_signal(signal);
    }

    pub fn read_scout_diagnostics(&self) -> Arc<Mutex<ScoutDiagnostics>> {
        self.scout_engine.read_diagnostics()
    }

    pub fn set_max_iterations(&mut self, max_iterations: u32) {
        self.uniform.max_iter = max_iterations;
        self.scout_engine.set_max_user_iterations(max_iterations)
    }

    pub fn set_window_size(&mut self, width: f64, height: f64) {
        self.width = width;
        self.height = height;
        self.uniform.screen_width = width as u32;
        self.uniform.screen_height = height as u32;
        self.uniform.grid_width = width as u32 / self.uniform.grid_size;

        debug!("Window size changed w={} h={}", width, height);
    }

    pub fn set_scale(&mut self, scale: Float) {
        self.scale = scale;

        let scale_df = Df::from_float(&self.scale);
        self.uniform.scale_hi = scale_df.hi;
        self.uniform.scale_lo = scale_df.lo;
    }

    pub fn change_scale(&mut self, increase: bool) -> String {
        if increase {
            self.scale *= &self.scale_factor;
        } else {
            self.scale /= &self.scale_factor;
        }

        let scale_df = Df::from_float(&self.scale);
        self.uniform.scale_hi = scale_df.hi;
        self.uniform.scale_lo = scale_df.lo;
        
        let s = self.scale.to_string_radix(10, None);
        debug!("Scale changed {}", s);
        s
    }

    fn update_center_offsets(&mut self) {
        let center = self.center().clone();
        let center_df = ComplexDf::from_complex(&center);

        self.uniform.center_x_hi = center_df.re.hi;
        self.uniform.center_x_lo = center_df.re.lo;
        self.uniform.center_y_hi = center_df.im.hi;
        self.uniform.center_y_lo = center_df.im.lo;

        for loadout in self.loaded_orbits.iter_mut() {
            loadout.center_offset = complex_delta(&center, &loadout.orbit_c_ref);

            let center_offset_df = ComplexDf::from_complex(&loadout.center_offset);
            loadout.gpu_orb_loc_info.center_offset_re_hi = center_offset_df.re.hi;
            loadout.gpu_orb_loc_info.center_offset_re_lo = center_offset_df.re.lo;
            loadout.gpu_orb_loc_info.center_offset_im_hi = center_offset_df.im.hi;
            loadout.gpu_orb_loc_info.center_offset_im_lo = center_offset_df.im.lo;
        }
    }

    pub fn set_center(&mut self, new_center: Complex) {
        self.center = new_center;

        self.update_center_offsets();
    }

    pub fn change_center(&mut self, center_diff: (f64, f64)) -> String {
        let dx = self.scale.clone() * center_diff.0;
        let dy = self.scale.clone() * center_diff.1;

        let (real, imag) = self.center.as_mut_real_imag();
        *real -= &dx;
        *imag += &dy;

        self.update_center_offsets();

        let c = self.center.to_string_radix(10, None);
        debug!("Center changed {:?} ----- diff ({:?} {:?})", 
            c, dx, dy);

        c
    }
    
    pub fn set_debug_coloring(&mut self, debug_coloring: bool) {
        self.uniform.set_debug_coloring(debug_coloring);
    }
    
    pub fn set_glitch_fix(&mut self, glitch_fix: bool) {
        self.uniform.set_glitch_fix(glitch_fix);
    }
    
    pub fn set_smooth_coloring(&mut self, smooth_coloring: bool) {
        self.uniform.set_smooth_coloring(smooth_coloring);
    }

    pub fn set_use_de(&mut self, use_de: bool) {
        self.uniform.set_use_de(use_de);
    }

    pub fn set_use_stripes(&mut self, use_stripes: bool) {
        self.uniform.set_use_stripes(use_stripes);
    }

    pub fn set_enable_glow(&mut self, enable_glow: bool) {
        self.uniform.set_enable_glow(enable_glow);
    }

    pub fn set_enable_key_light(&mut self, enable_key_light: bool) {
        self.uniform.set_enable_key_light(enable_key_light);
    }

    pub fn set_enable_fill_light(&mut self, enable_fill_light: bool) {
        self.uniform.set_enable_fill_light(enable_fill_light);
    }

    pub fn set_enable_specular(&mut self, enable_specular: bool) {
        self.uniform.set_enable_specular(enable_specular);
    }

    pub fn set_enable_ao(&mut self, enable_ao: bool) {
        self.uniform.set_enable_ao(enable_ao);
    }

    pub fn set_enable_rim(&mut self, enable_rim: bool) {
        self.uniform.set_enable_rim(enable_rim);
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
    
    pub fn get_palette(&self, key: &String) -> &ColorPalette {
        &self.color_palettes[key]
    }

    pub fn set_selected_palette(&mut self, key: &String) {
        self.selected_palette = key.clone();
        self.palette_changed = true;
    }

    pub fn set_palette_frequency(&mut self, key: &String, frequency: f32) {
        self.color_palettes.get_mut(key).unwrap().frequency = frequency;
        self.uniform.palette_frequency = frequency;
    }
    
    pub fn set_palette_offset(&mut self, key: &String, offset: f32) {
        self.color_palettes.get_mut(key).unwrap().offset = offset;
        self.uniform.palette_offset = offset;
    }
    
    pub fn set_palette_gamma(&mut self, key: &String, gamma: f32) {
        self.color_palettes.get_mut(key).unwrap().gamma = gamma;
        self.uniform.palette_gamma = gamma;
    }

    pub fn set_distance_multiplier(&mut self, distance_multiplier: f32) {
        self.uniform.distance_multiplier = distance_multiplier;
    }

    pub fn set_glow_intensity(&mut self, glow_intensity: f32) {
        self.uniform.glow_intensity = glow_intensity;
    }

    pub fn set_neighbor_scale(&mut self, neighbor_scale: f32) {
        self.uniform.neighbor_scale_multiplier = neighbor_scale;
    }

    pub fn set_ambient_intensity(&mut self, ambient_intensity: f32) {
        self.uniform.ambient_intensity = ambient_intensity;
    }

    pub fn set_key_light_intensity(&mut self, key_light_intensity: f32) {
        self.uniform.key_light_intensity = key_light_intensity;
    }

    pub fn set_key_light_azimuth(&mut self, key_light_azimuth: f32) {
        self.uniform.key_light_azimuth = key_light_azimuth;
    }

    pub fn set_key_light_elevation(&mut self, key_light_elevation: f32) {
        self.uniform.key_light_elevation = key_light_elevation;
    }

    pub fn set_fill_light_intensity(&mut self, fill_light_intensity: f32) {
        self.uniform.fill_light_intensity = fill_light_intensity;
    }

    pub fn set_fill_light_azimuth(&mut self, fill_light_azimuth: f32) {
        self.uniform.fill_light_azimuth = fill_light_azimuth;
    }

    pub fn set_fill_light_elevation(&mut self, fill_light_elevation: f32) {
        self.uniform.fill_light_elevation = fill_light_elevation;
    }

    pub fn set_specular_intensity(&mut self, specular_intensity: f32) {
        self.uniform.specular_intensity = specular_intensity;
    }

    pub fn set_specular_power(&mut self, specular_power: f32) {
        self.uniform.specular_power = specular_power;
    }

    pub fn set_ao_darkness(&mut self, ao_darkness: f32) {
        self.uniform.ao_darkness = ao_darkness;
    }

    pub fn set_stripe_density(&mut self, stripe_density: f32) {
        self.uniform.stripe_density = stripe_density;
    }

    pub fn set_stripe_strength(&mut self, stripe_strength: f32) {
        self.uniform.stripe_strength = stripe_strength;
    }

    pub fn set_stripe_gamma(&mut self, stripe_gamma: f32) {
        self.uniform.stripe_gamma = stripe_gamma;
    }
    
    pub fn set_rim_intensity(&mut self, rim_intensity: f32) {
        self.uniform.rim_intensity = rim_intensity;
    }
    
    pub fn set_rim_power(&mut self, rim_power: f32) {
        self.uniform.rim_power = rim_power;
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

    pub fn query_qualified_orbits(&mut self, queue: &wgpu::Queue) {
        // First check if ScoutEngine's context has changed. 
        if self.scout_engine.context_changed() {
            let orbits = self.scout_engine.query_qualified_orbits();

            // Upload qualified orbits to a Texture2D
            self.upload_orbits(&orbits, queue);

            // Build GPU-specific orbit locations from scout-engine's tile-orbit pairs
            self.loaded_orbits =
                build_gpu_orbit_loadout_from_qualified_orbits(&orbits, self.center());
            info!("Rebuilt orbit loadout! len={}", self.loaded_orbits.len());

            self.upload_orbit_locations(queue);
        }
    }

    fn upload_orbits(&mut self, orbits: &Vec<QualifiedOrbit>, queue: &wgpu::Queue) {
        if orbits.len() == 0 {
            trace!("Orbit Count reduced to zero! Scout likely shutdown!");
            self.uniform.ref_orb_count = 0;
            return;
        }

        let mut orbit_count: u32 = 0;
        if orbits.len() > 0 {
            let ranked_orb = &orbits[0];
            queue.write_buffer(&self.pipeline.rank_one_orbit_buf, 0, bytemuck::cast_slice(&ranked_orb.orbit));
            orbit_count += 1;
            trace!("Wrote Rank One RefOrb to GPU! Wrote {} bytes to storage buffer for {} orbits! ",
                ranked_orb.orbit.len() * 16, ranked_orb.orbit.len());
        }

        if orbits.len() > 1 {
            let ranked_orb = &orbits[1];
            queue.write_buffer(&self.pipeline.rank_two_orbit_buf, 0, bytemuck::cast_slice(&ranked_orb.orbit));
            orbit_count += 1;
            trace!("Wrote Rank Two RefOrb to GPU! Wrote {} bytes to storage buffer for {} orbits! ",
                ranked_orb.orbit.len() * 16, ranked_orb.orbit.len());
        }

        self.uniform.ref_orb_count = orbit_count;
    }

    fn upload_orbit_locations(&self, queue: &wgpu::Queue) {
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
            queue.write_buffer(&self.pipeline.ref_orbit_location_buf, 0, bytemuck::cast_slice(&orb_locations));
        }
    }

    pub fn upload_color_palette(&mut self, queue: &wgpu::Queue) {
        if let Some(palette) = self.color_palettes.get(&self.selected_palette)
                && self.palette_changed {
            self.palette_changed = false;
            self.uniform.palette_frequency = palette.frequency;
            self.uniform.palette_offset = palette.offset;
            self.uniform.palette_gamma = palette.gamma;

            let max = self.max_palette_colors as usize;
            let src = &palette.palette;

            // Create a repeating cycle of the palette within the texture. This works best
            // for texture color sampling, and for frequency/offset changes.
            let mut full_palette = Vec::<[u8; 4]>::with_capacity(max);
            for i in 0..max {
                full_palette.push(src[i % src.len()]);
            }

            let palette_bytes: &[u8] = bytemuck::cast_slice(&full_palette);

            trace!("Uploading color palette {}. freq={} offset={} gamma={} render_flags={} len={} bytes={}",
                palette.name, self.uniform.palette_frequency,
                self.uniform.palette_offset, self.uniform.palette_gamma,
                self.uniform.render_flags,
                self.max_palette_colors, palette_bytes.len());

            queue.write_texture(
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

    // Pixels from the GPU MUST first undergo Df reconstruction first before
    // being taken to high precision. Otherwise, the collected sample is NOT being
    // faithful to 'c' that underwent escape-time calculation using its Df arithmetic.
    fn build_c_from_scene(&self, pix_x: i32, pix_y: i32) -> Complex {
        let half_width = self.width / 2.0;
        let half_height = self.height / 2.0;

        let dx = pix_x - half_width as i32;
        let dy = half_height as i32 - pix_y;

        let scale_df = Df::new(
            self.uniform.scale_hi, self.uniform.scale_lo
        );

        let off_x = scale_df * Df::new(dx as f32, 0.0);
        let off_y = scale_df * Df::new(dy as f32, 0.0);

        let center_re = Df::new(
            self.uniform.center_x_hi, self.uniform.center_x_lo
        );
        let center_im = Df::new(
            self.uniform.center_y_hi, self.uniform.center_y_lo
        );
        let c_re_df = off_x + center_re;
        let c_im_df = off_y + center_im;

        let prec = self.scale.prec();
        // No high precision math here, just conversion at the end!
        Complex::with_val(prec, (c_re_df.to_float(prec), c_im_df.to_float(prec)))
    }

    // Under perturbation, Df math must be specifically avoided.
    // GPU is using CPU-derived deltas anyway, so stay in high precision
    fn build_delta_c_from_orbit_location(&self, pix_x: i32, pix_y: i32, orbit_idx: u32) -> Option<Complex> {
        let half_width = self.width / 2.0;
        let half_height = self.height / 2.0;

        let dx = pix_x - half_width as i32;
        let dy = half_height as i32 - pix_y;

        let mut off_x = self.scale.clone();
        off_x *= dx;

        let mut off_y = self.scale.clone();
        off_y *= dy;


        if let Some (loadout_loc) = self.loaded_orbits.get(orbit_idx as usize) {
            let mut c_re = loadout_loc.center_offset.real().clone();
            c_re += &off_x;

            let mut c_im = loadout_loc.center_offset.imag().clone();
            c_im += &off_y;

            c_re += loadout_loc.orbit_c_ref.real();
            c_im += loadout_loc.orbit_c_ref.imag();

            let prec = self.scale.prec();
            Some (
                Complex::with_val(prec, (c_re, c_im))
            )
        }
        else {
            None
        }
    }
}

fn build_gpu_orbit_loadout_from_qualified_orbits(
    orbits: &Vec<QualifiedOrbit>,
    center: &Complex
) -> Vec<LoadedOrbit> {
    orbits.iter()
        .map(|orb| {
            let orbit_offset_from_center = complex_delta(center, &orb.c_ref);
            let center_offset_df = ComplexDf::from_complex(&orbit_offset_from_center);
            LoadedOrbit {
                rank: orb.rank,
                orbit_id: orb.orbit_id,
                orbit_c_ref: orb.c_ref.clone(),
                center_offset: orbit_offset_from_center,
                gpu_orb_loc_info: GpuRefOrbitLocation {
                    c_ref_re_hi: orb.c_ref_df.re.hi,
                    c_ref_re_lo: orb.c_ref_df.re.lo,
                    c_ref_im_hi: orb.c_ref_df.im.hi,
                    c_ref_im_lo: orb.c_ref_df.im.lo,
                    max_ref_iters: orb.escape_index.unwrap_or(orb.orbit.len() as u32),
                    center_offset_re_hi: center_offset_df.re.hi,
                    center_offset_re_lo: center_offset_df.re.lo,
                    center_offset_im_hi: center_offset_df.im.hi,
                    center_offset_im_lo: center_offset_df.im.lo,
            }
        }
        })
        .collect()
}

fn build_color_palettes_from_settings(settings: &Settings) -> HashMap<String, ColorPalette> {
    let mut color_palettes: HashMap<String, ColorPalette> = HashMap::new();
    settings.palettes.iter().for_each(|(key, palette)| {
        // First validate the palette is well-formed
        if palette.array.len() % 3 != 0 {
            warn!("Palette {} array is not a multiple of 3! Skipping!", key);
        }
        else {
            color_palettes.insert(
                key.clone(),
                ColorPalette {
                    name: palette.name.clone(),
                    palette: palette.array
                        .iter()
                        .as_slice()
                        .chunks(3)
                        .map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
                        .collect(),
                    offset: 0.0,
                    frequency: palette.frequency,
                    frequency_range: palette.frequency_range,
                    gamma: 1.0,
                }
            );
        }
    });
    color_palettes
}
