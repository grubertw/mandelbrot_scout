pub mod policy;

use crate::scene::policy::*;
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

#[derive(Debug)]
pub struct Scene {
    frame_id: u64,
    frame_timestamp: time::Instant,
    center: Complex, // scaled and shifted with mouse drag
    scale: Float,
    scale_factor: Float,
    width: f64,
    height: f64,
    scout_engine: ScoutEngine,
    loaded_orbits: Vec<LoadedOrbit>,
    uniform: SceneUniform,
    pipeline: PipelineBundle,
}

impl Scene {
    pub fn new(window: Arc<Window>, device: &wgpu::Device, 
        texture_format: wgpu::TextureFormat,
        width: f64, height: f64,
    ) -> Scene {
        let center = Complex::with_val(INIT_RUG_PRECISION, CENTER);
        let c_df = ComplexDf::from_complex(&center);

        let scale = Float::with_val(INIT_RUG_PRECISION, COMPLEX_SPAN / width);
        let scale_df = Df::from_float(&scale);

        // Configure ScoutEngine (our single source of truth for reference orbits)
        let scout_config =  ScoutEngineConfig {
            max_live_orbits: MAX_LIVE_ORBITS,
            max_user_iters: MAX_USER_ITER,
            max_ref_orbit_iters: MAX_REF_ORBIT,
            auto_start: AUTO_START,
            starting_scale: STARTING_SCALE,
            ref_iters_multiplier: REF_ITERS_MULTIPLIER,
            num_seeds_to_spawn_per_eval: NUM_SEEDS_TO_SPAWN_PER_EVAL,
            num_qualified_orbits: NUM_QUALIFIED_ORBITS,
            rug_precision: INIT_RUG_PRECISION,
            exploration_budget: EXPLORATION_BUDGET,
        };

        let scout_engine = ScoutEngine::new(
            scout_config,
            window,
            CameraSnapshot::new(
                FrameStamp::new(), center.clone(), scale.clone(),
                width.max(height) / 2.0,
            ));

        let uniform = SceneUniform {
            center_x_hi: c_df.re.hi, center_x_lo: c_df.re.lo,
            center_y_hi: c_df.im.hi, center_y_lo: c_df.im.lo,
            scale_hi:    scale_df.hi, scale_lo:    scale_df.lo,
            screen_width: width as u32,
            screen_height: height as u32,
            max_iter: MAX_USER_ITER,
            ref_orb_count: 0,
            grid_size: SCREEN_GRID_SIZE,
            grid_width: width as u32 / SCREEN_GRID_SIZE,
        };

        // Configure and initialize all WGPU resources for render passes.
        let pipeline = PipelineBundle::build_pipelines(
            device, &uniform, texture_format);

        Scene { 
            frame_id: 0, frame_timestamp: time::Instant::now(), 
            center, scale, scale_factor: Float::with_val(INIT_RUG_PRECISION, 1.05),
            width, height, scout_engine, 
            loaded_orbits: Vec::new(),
            uniform, pipeline
        }
    }

    pub fn draw<'a>(&'a self, device: &wgpu::Device, queue: &wgpu::Queue, target: &'a wgpu::TextureView) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame encoder"),
            });
        let gx = (MAX_SCREEN_WIDTH + 15) / 16;
        let gy = (MAX_SCREEN_HEIGHT + 15) / 16;

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
            render_pass.set_bind_group(2, &self.pipeline.debug_bg, &[]);
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
            * (self.uniform.screen_height / SCREEN_GRID_SIZE)) as u64
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

        trace!("{}", trace_str);
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
                debug!("  scale  = ({}, {})", dbg.scale_hi, dbg.scale_lo);
                debug!("  width  = {}\theight = {}", dbg.screen_width, dbg.screen_height);
                debug!("  ref_orb_count = {}", dbg.ref_orb_count);
                debug!("  orbit_idx   = {}", dbg.orbit_idx);
    
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
    pub fn max_iterations(&self) -> u32 {
        self.uniform.max_iter
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
        self.uniform.grid_width = width as u32 / SCREEN_GRID_SIZE;

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
            self.load_orbits_to_texture(&orbits, queue);

            // Build GPU-specific orbit locations from scout-engine's tile-orbit pairs
            self.loaded_orbits =
                build_gpu_orbit_loadout_from_qualified_orbits(&orbits, self.center());
            info!("Rebuilt orbit loadout! len={}", self.loaded_orbits.len());

            self.upload_orbit_locations(queue);
        }
    }

    fn load_orbits_to_texture(&mut self, orbits: &Vec<QualifiedOrbit>, queue: &wgpu::Queue) {
        if orbits.len() == 0 {
            trace!("Orbit Count reduced to zero! Scout likely shutdown!");
            self.uniform.ref_orb_count = 0;
            return;
        }

        let largest_orb_len = orbits
            .iter()
            .fold(0, |acc, orb| {
                acc.max(orb.orbit_re_hi.len())
            });

        let orbit_count = orbits.len().min(MAX_ORBITS_PER_FRAME as usize);

        let row_count = orbit_count * 4;
        let orb_len = largest_orb_len.min(MAX_REF_ORBIT as usize);

        let mut texture_data = Vec::<f32>::with_capacity(orb_len * row_count);

        for orb in orbits {
            // re_hi row
            for it in 0..orb_len {
                texture_data.push(orb.orbit_re_hi.get(it).copied().unwrap_or(0.0));
            }

            // re_lo row
            for it in 0..orb_len {
                texture_data.push(orb.orbit_re_lo.get(it).copied().unwrap_or(0.0));
            }

            // im_hi row
            for it in 0..orb_len {
                texture_data.push(orb.orbit_im_hi.get(it).copied().unwrap_or(0.0));
            }

            // im_lo row
            for it in 0..orb_len {
                texture_data.push(orb.orbit_im_lo.get(it).copied().unwrap_or(0.0));
            }
        }
        
        let tex_height = row_count as u32;
        let tex_width = orb_len as u32;

        trace!("Text height = {}, Tex width = {}, row_count = {}, largest_orb_len={} texture_data.len={}",
            tex_height, tex_width, row_count, largest_orb_len, texture_data.len());

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.pipeline.ref_orbit_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&texture_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(
                    tex_width * std::mem::size_of::<f32>() as u32
                ).unwrap().into()),
                rows_per_image: Some(std::num::NonZeroU32::new(
                    tex_height
                ).unwrap().into()),
            },
            wgpu::Extent3d {
                width: tex_width,
                height: tex_height,
                depth_or_array_layers: 1,
            },
        );

        info!("Uploading qualified orbits into texture. width={} height={}. {} orbits uploaded of byte size {}",
            tex_width, tex_height, orbit_count, &texture_data.len() * 4);
        self.uniform.ref_orb_count = orbit_count as u32;
    }

    fn upload_orbit_locations(&self, queue: &wgpu::Queue) {
        if self.loaded_orbits.len() > 0 && self.loaded_orbits.len() <= MAX_ORBITS_PER_FRAME as usize {
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
                    center_offset_re_hi: center_offset_df.re.hi,
                    center_offset_re_lo: center_offset_df.re.lo,
                    center_offset_im_hi: center_offset_df.im.hi,
                    center_offset_im_lo: center_offset_df.im.lo,
            }
        }
        })
        .collect()
}
