pub mod policy;

use crate::scene::policy::*;
use crate::gpu_pipeline::builder::*;
use crate::gpu_pipeline::structs::*;
use crate::numerics::*;
use crate::signals::*;
use crate::scout_engine::{ScoutEngineConfig, ScoutEngine, ScoutConfig, ScoutSignal};
use crate::scout_engine::orbit::OrbitId;
use crate::scout_engine::tile::TileId;

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
    loaded_tiles: Vec<(TileId, OrbitId)>,
    uniform: SceneUniform,
    reduce_uniform: ReduceUniform,
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
            max_tile_levels: MAX_TILE_LEVELS,
            ideal_tile_size: IDEAL_TILE_SIZE,
            ref_iters_multiplier: REF_ITERS_MULTIPLIER,
            num_seeds_to_spawn_per_tile_eval: NUM_SEEDS_TO_SPAWN_PER_TILE_EVAL,
            contraction_threshold: CONTRACTION_THRESHOLD,
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
            screen_width: width as f32, 
            screen_height: height as f32,
            max_iter: MAX_USER_ITER, tile_count: 0,
        };
        
        let reduce_uniform = ReduceUniform {
            screen_width: uniform.screen_width as u32,
            screen_height: uniform.screen_height as u32,
            grid_size: SCREEN_GRID_SIZE,
            grid_width: uniform.screen_width as u32 / SCREEN_GRID_SIZE,
        };

        // Configure and initialize all WGPU resources for render passes.
        let pipeline = PipelineBundle::build_pipelines(
            device, &uniform, &reduce_uniform, texture_format);

        Scene { 
            frame_id: 0, frame_timestamp: time::Instant::now(), 
            center, scale, scale_factor: Float::with_val(INIT_RUG_PRECISION, 1.05),
            width, height, scout_engine, 
            loaded_tiles: Vec::new(),
            uniform, reduce_uniform, pipeline
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
            queue.write_buffer(&self.pipeline.reduce_uniform_buff, 0, bytemuck::cast_slice(&[self.reduce_uniform]));

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
            (self.reduce_uniform.grid_width 
            * (self.reduce_uniform.screen_height / SCREEN_GRID_SIZE)) as u64 
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
            .map(|sample| {
                let best_sample =self.build_c_from_scene(sample.best_pixel_x, sample.best_pixel_y);
                let sample_iters = sample.iter();
                let escaped = sample.escaped();

                if !escaped {
                    trace_str.push_str(
                        format!("\t({:<3}, {:<3}) c={:<40} depth={:<3} escaped={} perturbed={} \
                        perturbed_err={} max_iters_reached={} tile_id={}\traw_flags={}\n",
                           sample.best_pixel_x, sample.best_pixel_y,
                           best_sample.to_string_radix(10, Some(10)),
                           sample_iters, escaped,
                            sample.perturbed(),
                            sample.perturb_err(),
                            sample.max_iters_reached(),
                            sample.tile_id(),
                            sample.best_pixel_flags
                    ).as_str());
                }

                GpuGridSample {
                    frame_stamp: FrameStamp {
                        frame_id: self.frame_id,
                        timestamp: self.frame_timestamp
                    },
                    best_sample,
                    best_sample_iters: sample_iters,
                    best_sample_escaped: escaped,
                    max_user_iters: self.uniform.max_iter
            }})
            .collect();

        trace!("{}", trace_str);
        self.scout_engine.set_grid_samples(grid_samples)
    }

    pub fn read_tile_feedback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let byte_size = self.uniform.tile_count as u64 * std::mem::size_of::<TileFeedbackOut>() as u64;
        if byte_size == 0 {
            warn!("This frame had no orbit feedback. Tile count was zero!");
            return;
        }

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("grid feedback copy encoder"),
            },
        );

        encoder.copy_buffer_to_buffer(
            &self.pipeline.tile_feedback_buffer,
            0,
            &self.pipeline.tile_feedback_readback,
            0,
            byte_size,
        );

        queue.submit(Some(encoder.finish()));

        let buffer_slice = self.pipeline.tile_feedback_readback.slice(0..byte_size);

        let (sender, receiver) = futures::channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            sender.send(res).ok();
        });

        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .expect("Failed to poll tile feedback");

        let feedback = futures::executor::block_on(async {
            match receiver.await {
                Ok(Ok(())) => {
                    let data = buffer_slice.get_mapped_range();
                    // SAFETY: TileFeedbackOut is Pod + repr(C)
                    let slice: &[TileFeedbackOut] =
                        bytemuck::cast_slice(&data);

                    let result = slice.to_vec();
                    drop(data);
                    self.pipeline.tile_feedback_readback.unmap();
                    result
                }
                _ => {
                    self.pipeline.tile_feedback_readback.unmap();
                    Vec::new()
                }
            }
        });

        if self.loaded_tiles.len() == self.uniform.tile_count as usize {
            let observations: Vec<TileObservation> = feedback
                .iter()
                .enumerate()
                .map(|(i, v)| TileObservation {
                    frame_stamp: FrameStamp {
                        frame_id: self.frame_id,
                        timestamp: self.frame_timestamp
                    },
                    tile_id: self.loaded_tiles[i].0,
                    orbit_id: self.loaded_tiles[i].1,
                    feedback: *v
                })
                .collect();

            self.scout_engine.submit_tile_observations(observations);
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
                debug!("  tile_count = {}", dbg.tile_count);
                debug!("  tile_idx   = {}", dbg.tile_idx);
    
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
        self.uniform.screen_width = width as f32;
        self.uniform.screen_height = height as f32;
        self.reduce_uniform.screen_width = width as u32;
        self.reduce_uniform.screen_height = height as u32;
        self.reduce_uniform.grid_width = width as u32 / SCREEN_GRID_SIZE;

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

    pub fn set_center(&mut self, new_center: Complex) {
        self.center = new_center;

        let center_df = ComplexDf::from_complex(&self.center);
        self.uniform.center_x_hi = center_df.re.hi;
        self.uniform.center_x_lo = center_df.re.lo;
        self.uniform.center_y_hi = center_df.im.hi;
        self.uniform.center_y_lo = center_df.im.lo;
    }

    pub fn change_center(&mut self, center_diff: (f64, f64)) -> String {
        let dx = self.scale.clone() * center_diff.0;
        let dy = self.scale.clone() * center_diff.1;

        let (real, imag) = self.center.as_mut_real_imag();
        *real -= &dx;
        *imag += &dy;

        let center_df = ComplexDf::from_complex(&self.center);
        self.uniform.center_x_hi = center_df.re.hi;
        self.uniform.center_x_lo = center_df.re.lo;
        self.uniform.center_y_hi = center_df.im.hi;
        self.uniform.center_y_lo = center_df.im.lo;

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

    pub fn query_tile_orbits(&mut self, queue: &wgpu::Queue) {
        // First check if ScoutEngine's context has changed. 
        if self.scout_engine.context_changed() {
            let snapshot = self.build_camera_snapshot();
            let tiles = self.scout_engine.query_tiles_for_orbits(&snapshot);

            // Upload orbit atlas and orbit meta to GPU
            self.upload_tile_orbits(&tiles, queue);

            // Build GPU-specific tile geometry from scout-engine's tile-orbit pairs
            let gpu_tile_geometry_mappings = 
                build_gpu_tile_geometry_from_tile_views(
                    &tiles, &self.scale, self.width, self.height
                );

            self.upload_tile_geometry(&gpu_tile_geometry_mappings, queue);

            let zipped_for_log: Vec<(TileOrbitViewDf, GpuTileGeometry)> = tiles
                .iter()
                .cloned()
                .zip(gpu_tile_geometry_mappings)
                .collect();

            let mut trace_str = String::from(
            format!("Number of Tile Geometry Mappings is {}. Mapping follows...\n", 
                zipped_for_log.len()).as_str());
            
            for (i, (tile, geo)) in zipped_for_log.iter().enumerate() {
                trace_str.push_str(
                    format!("#{:>2} {:>3?} {:>6}\tmin_xy=({:>3.1},{:>3.1}) max_xy=({:>3.1},{:>3.1}) anchor_c_ref=(({},{}),({},{}))\n",
                    i, tile.id, tile.orbit.orbit_id, 
                    geo.tile_screen_min_x, geo.tile_screen_min_y, geo.tile_screen_max_x, geo.tile_screen_max_y,
                    geo.anchor_c_ref_re_hi, geo.anchor_c_ref_re_lo, geo.anchor_c_ref_im_hi, geo.anchor_c_ref_im_lo,
                ).as_str());
            }
            trace!("{}", trace_str);
        }
    }

    fn upload_tile_orbits(&mut self, tiles: &Vec<TileOrbitViewDf>, queue: &wgpu::Queue) {
        self.loaded_tiles.clear();
        if tiles.len() == 0 {
            self.uniform.tile_count = 0;
            return;
        }

        let largest_orb_len = tiles
            .iter()
            .fold(0, |acc, tile| {
                acc.max(tile.orbit.orbit_re_hi.len())
            });

        let tile_count = tiles.len().min(MAX_ORBITS_PER_FRAME as usize);

        let row_count = tile_count * 4;
        let orb_len = largest_orb_len.min(MAX_REF_ORBIT as usize);

        let mut texture_data = Vec::<f32>::with_capacity(orb_len * row_count);

        for tile in tiles {
            // re_hi row
            for it in 0..orb_len {
                texture_data.push(tile.orbit.orbit_re_hi.get(it).copied().unwrap_or(0.0));
            }

            // re_lo row
            for it in 0..orb_len {
                texture_data.push(tile.orbit.orbit_re_lo.get(it).copied().unwrap_or(0.0));
            }

            // im_hi row
            for it in 0..orb_len {
                texture_data.push(tile.orbit.orbit_im_hi.get(it).copied().unwrap_or(0.0));
            }

            // im_lo row
            for it in 0..orb_len {
                texture_data.push(tile.orbit.orbit_im_lo.get(it).copied().unwrap_or(0.0));
            }
            self.loaded_tiles.push((tile.id.clone(), tile.orbit.orbit_id));
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

        info!("Uploading tile anchor orbits into texture. width={} height={}. {} orbits uploaded of byte size {}", 
            tex_width, tex_height, tile_count, &texture_data.len() * 4);
        self.uniform.tile_count = tile_count as u32;
    }

    fn upload_tile_geometry(&mut self, tile_geometry: &Vec<GpuTileGeometry>, queue: &wgpu::Queue) {  
        if tile_geometry.len() > 0 && tile_geometry.len() <= MAX_ORBITS_PER_FRAME as usize {
            debug!("Upload GpuTileGeometry(s).len={}", tile_geometry.len());
            queue.write_buffer(&self.pipeline.tile_geometry_buf, 0, bytemuck::cast_slice(&tile_geometry));
        }
    }

    fn build_c_from_scene(&self, pix_x: i32, pix_y: i32) -> Complex {
        let half_width = self.width / 2.0;
        let half_height = self.height / 2.0;

        let dx = pix_x - half_width as i32;
        let dy = half_height as i32 - pix_y;

        let mut off_x = self.scale.clone();
        off_x *= dx;

        let mut off_y = self.scale.clone();
        off_y *= dy;

        let mut pix_c = self.center.clone();
        let (re, im) = pix_c.as_mut_real_imag();
        *re += &off_x;
        *im += &off_y;

        pix_c
    }
}

pub fn build_gpu_tile_geometry_from_tile_views(
    tiles: &Vec<TileOrbitViewDf>,
    scale: &Float, 
    width: f64, 
    height: f64,
) -> Vec<GpuTileGeometry> {
    let half_width = width / 2.0;
    let half_height = height / 2.0;

    tiles.iter()
        .map(|tile| {
            let tile_x_center = (tile.delta_from_center.real().clone() / scale).to_f64() + half_width;
            let tile_y_center = -(tile.delta_from_center.imag().clone() / scale).to_f64() + half_height;
            let r_pix = (tile.geometry.radius().clone() / scale).to_f64();

            GpuTileGeometry {
                anchor_c_ref_re_hi: tile.orbit.c_ref.re.hi,
                anchor_c_ref_re_lo: tile.orbit.c_ref.re.lo,
                anchor_c_ref_im_hi: tile.orbit.c_ref.im.hi,
                anchor_c_ref_im_lo: tile.orbit.c_ref.im.lo,
                center_offset_re_hi: tile.delta_from_center_to_anchor.re.hi,
                center_offset_re_lo: tile.delta_from_center_to_anchor.re.lo,
                center_offset_im_hi: tile.delta_from_center_to_anchor.im.hi,
                center_offset_im_lo: tile.delta_from_center_to_anchor.im.lo,
                tile_screen_min_x: (tile_x_center - r_pix) as f32,
                tile_screen_max_x: (tile_x_center + r_pix) as f32,
                tile_screen_min_y: (tile_y_center - r_pix) as f32,
                tile_screen_max_y: (tile_y_center + r_pix) as f32,
            }
        })
        .collect()
}
