#![allow(unused)]

pub mod policy;

use crate::scene::policy::*;
use crate::gpu_pipeline::builder::*;
use crate::gpu_pipeline::structs::*;
use crate::numerics::*;
use crate::signals::*;
use crate::scout_engine::{ScoutConfig, HeuristicConfig, ScoutEngine};
use crate::scout_engine::tile::{TileId, TileLevel};

use futures::channel;
use futures::executor;
use iced_wgpu::wgpu;
use iced_wgpu::wgpu::BufferAsyncError;
use iced_winit::winit::window::Window;

use rug::{Float, Complex};
use log::{trace, debug, info, warn};
use std::collections::HashSet;
use std::hash::{DefaultHasher, Hasher};
use std::sync::Arc;
use std::mem::size_of;
use std::time;

#[derive(Debug)]
pub struct Scene {
    frame_id: u64,
    frame_timestamp: time::Instant,
    scale: Float,
    scale_factor: Float,
    center: Complex, // scaled and shifted with mouse drag
    width: f64,
    height: f64,
    pix_dx: Float,
    pix_dy: Float,
    scout_engine: ScoutEngine,
    loaded_orbits: Vec<(TileId, u64)>,
    feedback_hash: u64, // Hash of loaded orbits, used to send feedback to scout engine only when it's changed.
    uniform: SceneUniform,
    pipeline: PipelineBundle,
}

impl Scene {
    pub fn new(window: Arc<Window>, device: &wgpu::Device, 
        texture_format: wgpu::TextureFormat,
        width: f64, height: f64,
    ) -> Scene {
        let center = Complex::with_val(INIT_RUG_PRECISION, (-0.75, 0.0));
        let c_df = ComplexDf::from_complex(&center);

        let scale = Float::with_val(INIT_RUG_PRECISION, 3.5);
        let scale_df = Df::from_float(&scale);

        let pix_dx = Float::with_val(INIT_RUG_PRECISION, &scale / width);
        let pix_dy = Float::with_val(INIT_RUG_PRECISION, &scale / height);
        let pix_dx_df = Df::from_float(&pix_dx);
        let pix_dy_df = Df::from_float(&pix_dy);

        // Configure ScoutEngine (our single source of truth for reference orbits)
        let scout_config = ScoutConfig {
            max_live_orbits: MAX_LIVE_ORBITS,
            max_orbit_iters: MAX_REF_ORBIT,
            heuristic_config: HeuristicConfig {
                frame_decay_increment: FRAME_DECAY_INCREMENT,
                tile_deficiency_threshold: TILE_DEFICIENCY_THRESHOLD,
            },
            orbit_rng_seed: ORBIT_RNG_SEED,
            init_rug_precision: INIT_RUG_PRECISION,
            base_tile_size: BASE_COMPLEX_TILE_SIZE,
            tile_level_addition_increment: STARTING_NUM_TILE_LEVELS,
            ideal_tile_pix_width: IDEAL_TILE_PIX_WIDTH,
            num_orbits_to_spawn_per_tile: NUM_ORBITS_PER_TILE_SPAWN,
            initial_max_orbits_per_tile: MAX_ORBITS_PER_TILE,
            exploration_budget: EXPLORATION_BUDGET,
        };

        let scout_engine = ScoutEngine::new(window, scout_config);
        let loaded_orbits = Vec::<(TileId, u64)>::new();

        let uniform = SceneUniform {
            center_x_hi: c_df.re.hi, center_x_lo: c_df.re.lo,
            center_y_hi: c_df.im.hi, center_y_lo: c_df.im.lo,
            scale_hi:    scale_df.hi, scale_lo:    scale_df.lo,
            pix_dx_hi:   pix_dx_df.hi, pix_dx_lo:   pix_dx_df.lo,
            pix_dy_hi:   pix_dy_df.hi, pix_dy_lo:   pix_dy_df.lo,
            screen_width: width as f32, 
            screen_height: height as f32,
            screen_tile_size: SCREEN_TILE_SIZE as f32,
            max_iter: 500, ref_orb_len: 0, ref_orb_count: 0,
        };

        // Configure and initialize all WGPU resources for render passes.
        let pipeline = PipelineBundle::build_pipelines(device, &uniform, texture_format);

        Scene { 
            frame_id: 0, frame_timestamp: time::Instant::now(), 
            scale, scale_factor: Float::with_val(80, 1.04),
            center, width, height, pix_dx, pix_dy, scout_engine, 
            loaded_orbits, feedback_hash: 0,
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

            trace!("Draw with uniform={:?} size={} bytes: {:?}", 
                self.uniform, size_of::<SceneUniform>(), &bytemuck::bytes_of(&self.uniform));
            // Uniforms must be updated on every draw operation.
            queue.write_buffer(&self.pipeline.uniform_buff, 0, bytemuck::cast_slice(&[self.uniform]));

            render_pass.set_pipeline(&self.pipeline.render_pipeline);
            render_pass.set_bind_group(0, &self.pipeline.scene_bg, &[]);
            render_pass.set_bind_group(1, &self.pipeline.frame_feedback_bg, &[]);
            render_pass.set_bind_group(2, &self.pipeline.ref_orbit_bg, &[]);
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

    pub fn read_gpu_feedback<'a>(&'a self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // 1) create encoder, copy storage -> readback
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu feedback copy encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &self.pipeline.frame_feedback_buffer, // src
            0,
            &self.pipeline.frame_feedback_readback, // dst
            0,
            std::mem::size_of::<FrameFeedbackOut>() as u64,
        );

        // submit the copy
        queue.submit(Some(encoder.finish()));

        let buffer_slice = self.pipeline.frame_feedback_readback.slice(..);

        let (sender, receiver) = channel::oneshot::channel::<Result<(), BufferAsyncError>>();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);

        executor::block_on(async {
            if let Ok(Ok(_)) = receiver.await {
                let data = buffer_slice.get_mapped_range();
                let dbg = bytemuck::from_bytes::<FrameFeedbackOut>(&data[..]).clone();

                debug!("FROM GPU (via GPU feedback buffer):");
                debug!("  max_lambda = ({}, {}) ({}, {})", dbg.max_lambda_re_hi, dbg.max_lambda_re_lo, dbg.max_lambda_im_hi, dbg.max_lambda_im_lo);
                debug!("  max_delta_z = ({}, {}) ({}, {})", dbg.max_delta_z_re_hi, dbg.max_delta_z_re_lo, dbg.max_delta_z_im_hi, dbg.max_delta_z_im_lo);
                debug!("  escape_ratio = {}", dbg.escape_ratio);
                
                drop(data);
                self.pipeline.frame_feedback_readback.unmap();
            }
        });
    }

    pub fn read_orbit_feedback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let byte_size = self.uniform.ref_orb_count as u64 * std::mem::size_of::<OrbitFeedbackOut>() as u64;
        if byte_size == 0 {
            warn!("This frame had no orbit feedback. RefOrb count was zero!");
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

        device.poll(wgpu::Maintain::Wait);

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

        // Determine if the GPU orbit-slot configuration has changed.
        let mut hasher = DefaultHasher::new();
        for o in &self.loaded_orbits {
            hasher.write_i64(o.0.tx);
            hasher.write_i64(o.0.ty);
            hasher.write_u64(o.1);
        }
        let orbit_slot_config_hash = hasher.finish();

        // Only send orbit observactiosn of the GPU orbit-slot config has changed.
        if orbit_slot_config_hash != self.feedback_hash && 
           self.loaded_orbits.len() == self.uniform.ref_orb_count as usize {
            debug!("FROM GPU: (Valid OrbitFeedback, by slot #/orbit_idx)");
            for (i, fb) in feedback.iter().enumerate() {
                if fb.perturb_attempted_count > 0 {
                    debug!("  Orbit Slot #{} feedback={:?}", i, fb);
                }
            }

            let observations: Vec<OrbitObservation> = feedback
                .iter()
                .enumerate()
                .map(|(i, v)| OrbitObservation{
                    frame_stamp: FrameStamp {
                        frame_id: self.frame_id,
                        timestamp: self.frame_timestamp
                    },
                    tile_id: self.loaded_orbits[i].0,
                    orbit_id: self.loaded_orbits[i].1,
                    feedback: *v
                })
                .collect(); 

            self.scout_engine.submit_orbit_observations(observations);
            self.feedback_hash = orbit_slot_config_hash;
        } else {
            trace!("Orbit Slot config unchanged. Will not send observations this frame. num loaded orbits is {}",
                self.loaded_orbits.len());
        }
    }

    pub fn read_debug<'a>(&'a self, device: &wgpu::Device, queue: &wgpu::Queue) {
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

        device.poll(wgpu::Maintain::Wait);

        executor::block_on(async {
            if let Ok(Ok(_)) = receiver.await {
                let data = buffer_slice.get_mapped_range();
                let dbg = bytemuck::from_bytes::<DebugOut>(&data[..]).clone();

                debug!("FROM CPU (Secne struct)");
                debug!("  center = {:?}", self.center);
                debug!("  width = {}\theight = {}", self.width, self.height);
                debug!("FROM CPU (scene uniform):");
                debug!("  pix_dx = ({}, {})", self.uniform.pix_dx_hi, self.uniform.pix_dx_lo);
                debug!("  pix_dy = ({}, {})", self.uniform.pix_dy_hi, self.uniform.pix_dy_lo);
                debug!("  scale  = ({}, {})", self.uniform.scale_hi, self.uniform.scale_lo);
                debug!("  screen_tile_size = {}", self.uniform.screen_tile_size);
                debug!("  ref_orb_len = {}", self.uniform.ref_orb_len);
                debug!("  ref_orb_count = {}", self.uniform.ref_orb_count);
                debug!("FROM GPU (via debug buffer at center pixel):");
                debug!("  c_ref   = (({}, {}) ({}, {}))", dbg.c_ref_re_hi, dbg.c_ref_re_lo, dbg.c_ref_im_hi, dbg.c_ref_im_lo);
                debug!("  delta_c = (({}, {}) ({}, {}))", dbg.delta_c_re_hi, dbg.delta_c_re_lo, dbg.delta_c_im_hi, dbg.delta_c_im_lo);
                debug!("  orbit_idx = {}", dbg.orbit_idx);
                debug!("  orbit_meta_ref_len = {}", dbg.orbit_meta_ref_len);
                debug!("  peturb_flags = {}", dbg.perturb_escape_seq);
                debug!("  last_valid_i = {}", dbg.last_valid_i);
                debug!("  abs_i = {}", dbg.abs_i);
                debug!("  last_valid_z = ({},{}) ({},{})", dbg.last_valid_z_re_hi, dbg.last_valid_z_re_lo, dbg.last_valid_z_im_hi,dbg.last_valid_z_im_lo);
    
                drop(data);
                self.pipeline.debug_readback.unmap();
            }
        });
    }

    pub fn set_max_iterations(&mut self, max_iterations: u32) {
        self.uniform.max_iter = max_iterations;
    }

    pub fn set_window_size(&mut self, width: f64, height: f64) {
        self.width = width;
        self.height = height;
        self.uniform.screen_width = width as f32;
        self.uniform.screen_height = height as f32;

        debug!("Window size changed w={} h={}", width, height);
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
        
        self.pix_dx = self.scale.clone() / self.width;
        self.pix_dy = self.scale.clone() / self.height;

        let pix_dx_df = Df::from_float(&self.pix_dx);
        let pix_dy_df = Df::from_float(&self.pix_dy);

        self.uniform.pix_dx_hi = pix_dx_df.hi;
        self.uniform.pix_dx_lo = pix_dx_df.lo;
        self.uniform.pix_dy_hi = pix_dy_df.hi;
        self.uniform.pix_dy_lo = pix_dy_df.lo;

        let s = self.scale.to_string_radix(10, None);
        let s_pix_dx = self.pix_dx.to_string_radix(10, None);
        let s_pix_dy = self.pix_dy.to_string_radix(10, None);

        debug!("Scale changed {} --- pix_dx={} pix_dy={}", s, s_pix_dx, s_pix_dy);
        s
    }

    pub fn set_center(&mut self, center_diff: (f64, f64)) -> String {
        let dx = self.pix_dx.clone() * center_diff.0;
        let dy = self.pix_dy.clone() * center_diff.1;

        let (real, imag) = self.center.as_mut_real_imag();
        *real -= &dx;
        *imag -= &dy;

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

    pub fn take_camera_snapshot(&mut self) {
        let cam_snap = CameraSnapshot {
            frame_stamp: FrameStamp {
                frame_id: self.frame_id,
                timestamp: self.frame_timestamp
            },
            center: self.center.clone(),
            // ScoutEngine needs pixel scale, not viewport scale.
            // Pick the bigger one, so the other dimension has some 
            // overlap, which is desireable! 
            scale: self.pix_dx.clone().max(&self.pix_dy),
            screen_extent_multiplier: self.width.max(self.height) / 2.0
        };

        self.scout_engine.submit_camera_snapshot(cam_snap);
    }

    pub fn query_tile_orbits(&mut self, queue: &wgpu::Queue) {
        // First check if ScoutEngine's context has changed. 
        if self.scout_engine.context_changed() {
            // Query for all complex tiles that fall within the viewport
            let mapper = PixelToComplexMapper::new(self.width, self.height, self.center.clone(), self.scale.clone());
            let vp_c_min = mapper.pixel_to_complex(0.0, 0.0);
            let vp_c_max = mapper.pixel_to_complex(self.width, self.height);
            let tiles = self.scout_engine.query_tiles_in_bounding_box(&vp_c_min, &vp_c_max);

            // Flatten complex tiles into orbit slots for the orbit (atlas) texture
            let slots = build_gpu_orbit_slots_from_tiles(tiles, self.uniform.scale_hi as f64);

            // Upload orbit atlas and orbit meta to GPU
            self.upload_reference_orbits(&slots, queue);
            self.upload_reference_orbit_meta(&slots, queue);

            // Shader needs a way to index into the orbit atlas, which is done based 
            // on screen-space tile locations.
            self.upload_tile_orbit_index(&slots, queue);
        }
    }

    fn upload_reference_orbits(&mut self, orbit_slots: &Vec<GpuOrbitSlot>, queue: &wgpu::Queue) {
        self.loaded_orbits.clear();
        if orbit_slots.len() == 0 {
            warn!("Upload reference orbits found no orbit slots for this frame update!");
            return;
        }

        let orbit_count = orbit_slots.len().min(MAX_ORBITS_PER_FRAME as usize);

        let orb_row_count = orbit_count * 4;
        let orb_len = MAX_REF_ORBIT as usize;

        let mut texture_data = Vec::<f32>::with_capacity(orb_len * orb_row_count);

        for slot in orbit_slots {
            self.loaded_orbits.push((slot.tile.clone(), slot.orbit.orbit_id));
                       
            texture_data.extend(slot.orbit.orbit_re_hi.iter());
            texture_data.extend(slot.orbit.orbit_re_lo.iter());
            texture_data.extend(slot.orbit.orbit_im_hi.iter());
            texture_data.extend(slot.orbit.orbit_im_lo.iter());
        }

        let tex_height = orb_row_count as u32;
        let tex_width = MAX_REF_ORBIT;

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.pipeline.ref_orbit_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&texture_data),
            wgpu::ImageDataLayout {
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

        info!("Upload reference orbits into texture. width={} height={}. {} orbits uploaded of byte size {}", 
            tex_width, tex_height, orbit_count, &texture_data.len() * 4);
        self.uniform.ref_orb_len = tex_width;
        self.uniform.ref_orb_count = orbit_count as u32;
    }

    fn upload_reference_orbit_meta(&mut self, orbit_slots: &Vec<GpuOrbitSlot>, queue: &wgpu::Queue) {
        let orbit_count = orbit_slots.len().min(MAX_ORBITS_PER_FRAME as usize);
        let mut ref_orbit_meta = Vec::<GpuOrbitMeta>::with_capacity(orbit_count);

        for slot in orbit_slots {
            ref_orbit_meta.push(slot.meta.clone());
        }

        queue.write_buffer(&self.pipeline.ref_orbit_meta_buf, 0, bytemuck::cast_slice(&ref_orbit_meta));
    }

    fn upload_tile_orbit_index(&mut self, orbit_slots: &Vec<GpuOrbitSlot>, queue: &wgpu::Queue) {
        let tiles_x = (self.width / SCREEN_TILE_SIZE).ceil() as usize + 1;
        let tiles_y = (self.height / SCREEN_TILE_SIZE).ceil() as usize + 1;
        
        let mut tile_indices = Vec::<u32>::with_capacity(tiles_x * tiles_y);
        let mapper = PixelToComplexMapper::new(self.width, self.height, self.center.clone(), self.scale.clone());
        let mut orbits_found = HashSet::<u32>::new();
        let mut tiles_with_orbits = Vec::<(usize, usize, u32)>::new();

        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                // Screen-space bounds of this tile
                let px0 = tx as f64 * SCREEN_TILE_SIZE;
                let py0 = ty as f64 * SCREEN_TILE_SIZE;
                let px1 = (px0 + SCREEN_TILE_SIZE).min(self.width);
                let py1 = (py0 + SCREEN_TILE_SIZE).min(self.height);

                // Convert to complex-space bounds
                let c_min = mapper.pixel_to_complex(px0, py0);
                let c_max = mapper.pixel_to_complex(px1, py1);

                // Query ScoutEngine based on bounding box of screen-space tile
                let complex_tiles = self.scout_engine.query_tiles_in_bounding_box(&c_min, &c_max);
                
                // Select the best orbit for this SSTile with a very simple scoring
                // (mainly based on distance)
                let orbit_slot = select_best_orbit_slot(&mapper, orbit_slots, complex_tiles, px0, py0, px1, py1);
                tile_indices.push(orbit_slot);

                if orbit_slot != NO_ORBIT {
                    orbits_found.insert(orbit_slot);
                    tiles_with_orbits.push((ty, tx, orbit_slot));
                }
            }
        }

        let mut slots_sorted: Vec<u32> = orbits_found.iter().cloned().collect();
        slots_sorted.sort();

        info!("Upload tile orbit indexes for SSTiles. tiles_x={} tiles_y={}. Total orbits found = {}. Slot #'s: {:?}", 
            tiles_x, tiles_y, orbits_found.len(), slots_sorted);

        let mut trace_str = String::from("Tiles with orbits graph:");
        let mut curr_row = 65535;
        for tworb in tiles_with_orbits {
            if curr_row != tworb.0 {
                trace_str.push_str("\n\t");
                curr_row = tworb.0;
            }
            trace_str.push_str(format!("{:>3?} ", tworb ).as_str());
        }
        trace!("{}", trace_str);

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.pipeline.tile_orbit_index_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&tile_indices),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(
                    std::num::NonZeroU32::new(tiles_x as u32 * 4).unwrap().into()
                ),
                rows_per_image: Some(
                    std::num::NonZeroU32::new(tiles_y as u32 * 4).unwrap().into()
                ),
            },
            wgpu::Extent3d {
                width: tiles_x as u32,
                height: tiles_y as u32,
                depth_or_array_layers: 1,
            },
        );
    }

}

