#![allow(unexpected_cfgs)]
#![allow(unused)]

use bytemuck;
use bitmask::bitmask;

use super::numerics::{Df, ComplexDf};
use super::signals::{FrameStamp, CameraSnapshot, GpuFeedback, TileOrbitViewDf, ReferenceOrbitDf};
use super::scout_engine::{ScoutConfig, HeuristicConfig, TileId, TileLevel, ScoutEngine};

use futures::channel;
use futures::executor;
use iced_wgpu::wgpu;
use iced_wgpu::wgpu::BufferAsyncError;
use wgpu::util::DeviceExt;
use iced_winit::winit::window::Window;

use rug::{Float, Complex};
use log::{trace, debug, info};
use std::collections::HashSet;
use std::sync::Arc;
use std::mem::size_of;
use std::time;

const MAX_ORBITS_PER_FRAME: u32 = 64;
const MAX_REF_ORBIT: u32 = 8192;
const ROWS_PER_ORBIT: u32 = 4;
const INIT_RUG_PRECISION: u32 = 128;
const K: f32 = 0.25; // Multiplied by scale, and often used as a radius of validity test
const PERTURB_THRESHOLD: f32 = 1e-5;
const DEEP_ZOOM_THRESHOLD: f64 = 1e-8;
const SHORT_THRESHOLD: u32 = 32;
const MIN_PERTURB_ITERS: u32 = 32; // or 64 later
const SCREEN_TILE_SIZE: f64 = 128.0;
const MAX_SCREEN_WIDTH:  u32 = 3840; // Support for a 4k display
const MAX_SCREEN_HEIGHT: u32 = 2160; // Support for a 4k display
const MAX_SCREEN_TILES_X: u32 = 256; // Support for a 4k display
const MAX_SCREEN_TILES_Y: u32 = 160; // Support for a 4k display
const NO_ORBIT: u32 = 0xFFFF_FFFF;   // Sentinel: means "no perturbation"

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniform {
    center_x_hi:        f32,
    center_x_lo:        f32,
    center_y_hi:        f32,
    center_y_lo:        f32,
    scale_hi:           f32,
    scale_lo:           f32,
    pix_dx_hi:          f32, 
    pix_dx_lo:          f32,
    pix_dy_hi:          f32, 
    pix_dy_lo:          f32,
    screen_width:       f32,
    screen_height:      f32,
    screen_tile_size:   f32,
    max_iter:           u32,
    ref_orb_len:        u32,
    ref_orb_count:      u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct FrameFeedbackOut {
    max_lambda_re_hi:   f32,
    max_lambda_re_lo:   f32,
    max_lambda_im_hi:   f32,
    max_lambda_im_lo:   f32,
    max_delta_z_re_hi:  f32,
    max_delta_z_re_lo:  f32,
    max_delta_z_im_hi:  f32,
    max_delta_z_im_lo:  f32,
    escape_ratio:       f32,
}
unsafe impl bytemuck::Pod for FrameFeedbackOut {}
unsafe impl bytemuck::Zeroable for FrameFeedbackOut {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OrbitFeedbackOut {
    // --- Perturbation stability ---
    pub min_last_valid_i: u32,         // Worst-case perturbation validity
    pub max_last_valid_i: u32,         // Best-case     
    
    // --- Perturbation Flag counts ---
    pub perturb_attempted_count: u32,
    pub perturb_valid_count: u32,
    pub perturb_collapsed_count: u32,
    pub perturb_escaped_count: u32,
    pub max_iter_reached_count: u32,
    pub absolute_fallback_count: u32,
    pub absolute_escaped_count: u32,
}
unsafe impl bytemuck::Pod for OrbitFeedbackOut {}
unsafe impl bytemuck::Zeroable for OrbitFeedbackOut {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DebugOut {
    c_ref_re_hi: f32,
    c_ref_re_lo: f32,
    c_ref_im_hi: f32,
    c_ref_im_lo: f32,
    delta_c_re_hi: f32,
    delta_c_re_lo: f32,
    delta_c_im_hi: f32,
    delta_c_im_lo: f32,
    orbit_idx:     u32,
    orbit_meta_ref_len: u32,
    perturb_escape_seq: u32,
    last_valid_i: u32,
    abs_i: u32,
    last_valid_z_re_hi: f32,
    last_valid_z_re_lo: f32,
    last_valid_z_im_hi: f32,
    last_valid_z_im_lo: f32,
}
unsafe impl bytemuck::Pod for DebugOut {}
unsafe impl bytemuck::Zeroable for DebugOut {}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuOrbitMeta {
    pub ref_len: u32,
    pub escape_index: u32,
    pub flags: u32,
    pub pad: u32,
}

bitmask! {
    pub mask OrbitMetaMask: u32 where flags OrbitMetaFlags {
        OrbitEscapes    = 0b00000001,
        OrbitInterior   = 0b00000010,
        OrbitShort      = 0b00000100,
        OrbitUnstable   = 0b00001000,
        OrbitUsable     = 0b00010000
    }
}

impl GpuOrbitMeta {
    pub fn new(ref_len: u32, escape_index: Option<u32>, orbit_meta_mask: OrbitMetaMask) -> Self {
        Self {
            ref_len,
            escape_index: escape_index.unwrap_or(u32::MAX),
            flags: *orbit_meta_mask,
            pad: 0
        }
    }

    pub fn is_usable(&self) -> bool {
        self.flags & *OrbitMetaFlags::OrbitUsable > 0
    }
}

#[derive(Clone, Debug)]
struct GpuOrbitSlot {
    tile: TileId,
    orbit: ReferenceOrbitDf,
    meta: GpuOrbitMeta,
}

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
    uniform: SceneUniform,
    uniform_buff: wgpu::Buffer,
    frame_feedback_buffer: wgpu::Buffer,
    frame_feedback_readback: wgpu::Buffer,
    orbit_feedback_buffer: wgpu::Buffer,
    orbit_feedback_readback: wgpu::Buffer,
    debug_buffer: wgpu::Buffer,
    debug_readback: wgpu::Buffer,
    ref_orbit_texture: wgpu::Texture,
    ref_orbit_meta_buf: wgpu::Buffer,
    tile_orbit_index_texture: wgpu::Texture,
    clear_bg: wgpu::BindGroup,
    scene_bg: wgpu::BindGroup,
    frame_feedback_bg: wgpu::BindGroup,
    ref_orbit_bg: wgpu::BindGroup,
    debug_bg: wgpu::BindGroup,
    reduce_bg: wgpu::BindGroup,
    clear_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    reduce_pipeline: wgpu::ComputePipeline
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

        let (uniform_buff, frame_feedback_buffer, frame_feedback_readback, 
                orbit_feedback_buffer, orbit_feedback_readback,
                debug_buffer, debug_readback,
                ref_orbit_texture, ref_orbit_meta_buf, tile_orbit_index_texture, 
                clear_bg, scene_bg, frame_feedback_bg, ref_orbit_bg, debug_bg, reduce_bg,
                clear_pipeline, render_pipeline, reduce_pipeline) = 
            build_pipelines(device, uniform, texture_format);

        // Configure ScoutEngine (our single source of truth for reference orbits)
        let scout_config = ScoutConfig {
            max_orbits: MAX_REF_ORBIT,
            max_iterations_ref: uniform.max_iter,
            rug_precision: INIT_RUG_PRECISION,
            heuristic_config: HeuristicConfig {
                weight_1: 0.0
            },
            tile_levels: vec![
                //TileLevel {
                //    level: 0,
                //    tile_size: Float::with_val(INIT_RUG_PRECISION, K),
                //    max_orbits_per_tile: 3
                //},
                TileLevel {
                    level: 0,
                    tile_size: Float::with_val(INIT_RUG_PRECISION, 5.0 * PERTURB_THRESHOLD),
                    influence_radius_factor: 1.25,
                    max_orbits_per_tile: 3,
                },
            ],
            exploration_budget: 5.0,
        };

        let scout_engine = ScoutEngine::new(window, scout_config);
        let loaded_orbits = Vec::<(TileId, u64)>::new();

        Scene { 
            frame_id: 0, frame_timestamp: time::Instant::now(), scale, scale_factor: Float::with_val(80, 1.04),
            center, width, height, pix_dx, pix_dy, scout_engine, loaded_orbits,
            uniform, uniform_buff, frame_feedback_buffer, frame_feedback_readback, 
            orbit_feedback_buffer, orbit_feedback_readback,
            debug_buffer, debug_readback,
            ref_orbit_texture, ref_orbit_meta_buf, tile_orbit_index_texture, 
            clear_bg, scene_bg, frame_feedback_bg, ref_orbit_bg, debug_bg, reduce_bg,
            clear_pipeline, render_pipeline, reduce_pipeline
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

            cpass.set_pipeline(&self.clear_pipeline);
            cpass.set_bind_group(0, &self.clear_bg, &[]);
            
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
            queue.write_buffer(&self.uniform_buff, 0, bytemuck::cast_slice(&[self.uniform]));

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.scene_bg, &[]);
            render_pass.set_bind_group(1, &self.frame_feedback_bg, &[]);
            render_pass.set_bind_group(2, &self.ref_orbit_bg, &[]);
            render_pass.set_bind_group(3, &self.debug_bg, &[]);
            render_pass.draw(0..3, 0..1);
        }

        {
            trace!("Compute pass reduce/aggregate. gx={} gy={}", gx, gy);

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce tiles pass"),
                timestamp_writes: None
            });

            cpass.set_pipeline(&self.reduce_pipeline);
            cpass.set_bind_group(0, &self.reduce_bg, &[]);

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
            &self.frame_feedback_buffer, // src
            0,
            &self.frame_feedback_readback, // dst
            0,
            std::mem::size_of::<FrameFeedbackOut>() as u64,
        );

        // submit the copy
        queue.submit(Some(encoder.finish()));

        let buffer_slice = self.frame_feedback_readback.slice(..);

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
                self.frame_feedback_readback.unmap();
            }
        });
    }

    pub fn read_orbit_feedback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Vec<OrbitFeedbackOut> {
        let byte_size = self.uniform.ref_orb_count as u64 * std::mem::size_of::<OrbitFeedbackOut>() as u64;

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("orbit feedback copy encoder"),
            },
        );

        encoder.copy_buffer_to_buffer(
            &self.orbit_feedback_buffer,
            0,
            &self.orbit_feedback_readback,
            0,
            byte_size,
        );

        queue.submit(Some(encoder.finish()));

        let buffer_slice = self.orbit_feedback_readback.slice(0..byte_size);

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
                    self.orbit_feedback_readback.unmap();

                    result
                }
                _ => {
                    self.orbit_feedback_readback.unmap();
                    Vec::new()
                }
            }
        });

        debug!("FROM GPU: (Valid OrbitFeedback, by slot #/orbit_idx)");

        for (i, fb) in feedback.iter().enumerate() {
            if fb.perturb_attempted_count > 0 {
                debug!("  Orbit Slot #{} feedback={:?}", i, fb);
            }
        }

        feedback
    }


    pub fn read_debug<'a>(&'a self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // 1) create encoder, copy storage -> readback
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("debug copy encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &self.debug_buffer, // src
            0,
            &self.debug_readback, // dst
            0,
            std::mem::size_of::<DebugOut>() as u64,
        );

        // submit the copy
        queue.submit(Some(encoder.finish()));

        let buffer_slice = self.debug_readback.slice(..);

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
                self.debug_readback.unmap();
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
            scale: self.scale.clone(),
        };

        self.scout_engine.submit_camera_snapshot(cam_snap);
    }

    pub fn query_tile_orbits(&mut self, queue: &wgpu::Queue) {
        // First query for all complex tiles that fall within the viewport
        let vp_c_min = self.pixel_to_complex(0.0, 0.0);
        let vp_c_max = self.pixel_to_complex(self.width, self.height);
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

    fn upload_reference_orbits(&mut self, orbit_slots: &Vec<GpuOrbitSlot>, queue: &wgpu::Queue) {
        self.loaded_orbits.clear();
        if orbit_slots.len() == 0 {
            debug!("Upload reference orbits found no orbit slots for this frame update!");
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
                texture: &self.ref_orbit_texture,
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

        queue.write_buffer(&self.ref_orbit_meta_buf, 0, bytemuck::cast_slice(&ref_orbit_meta));
    }

    fn upload_tile_orbit_index(&mut self, orbit_slots: &Vec<GpuOrbitSlot>, queue: &wgpu::Queue) {
        let tiles_x = (self.width / SCREEN_TILE_SIZE).ceil() as usize + 1;
        let tiles_y = (self.height / SCREEN_TILE_SIZE).ceil() as usize + 1;
        
        let mut tile_indices = Vec::<u32>::with_capacity(tiles_x * tiles_y);
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
                let c_min = self.pixel_to_complex(px0, py0);
                let c_max = self.pixel_to_complex(px1, py1);

                // Query ScoutEngine based on bounding box of screen-space tile
                let complex_tiles = self.scout_engine.query_tiles_in_bounding_box(&c_min, &c_max);
                
                // Select the best orbit for this SSTile with a very simple scoring
                // (mainly based on distance)
                let orbit_slot = self.select_best_orbit_slot(orbit_slots, complex_tiles, px0, py0, px1, py1);
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
            trace_str.push_str(format!("{:?} ", tworb ).as_str());
        }
        trace!("{}", trace_str);

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.tile_orbit_index_texture,
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

    fn pixel_to_complex(&self, px: f64, py: f64) -> Complex {
        let off_x = self.pix_dx.clone() * (px - (self.width / 2.0));
        let off_y = self.pix_dy.clone() * (py - (self.height / 2.0));

        let mut c = self.center.clone();
        let (c_re, c_im) = c.as_mut_real_imag();
        *c_re += off_x;
        *c_im += off_y;

        c
    }

    fn select_best_orbit_slot(&self, 
        orbit_slots: &Vec<GpuOrbitSlot>, complex_tiles: Vec<TileOrbitViewDf>,
        px0: f64, py0: f64, px1: f64, py1: f64,
    ) -> u32 {
        if orbit_slots.is_empty() || complex_tiles.is_empty() {
            return NO_ORBIT;
        }

        // Screen-tile center in complex space
        let cx = (px0 + px1) * 0.5;
        let cy = (py0 + py1) * 0.5;
        let c_center = self.pixel_to_complex(cx, cy);

        let mut best_score = f64::INFINITY;
        let mut best_slot = NO_ORBIT;

        for tile in complex_tiles {
            for orb in tile.orbits {
                // Find GPU slot
                let slot_idx = if let Some(i) = find_slot_index(orbit_slots, orb.orbit_id) {
                    i
                } else { 
                    continue;
                };

                // Policy: must be usable
                let meta = &orbit_slots[slot_idx as usize].meta;
                if !meta.is_usable() {
                    continue;
                }

                // Distance from screen-tile center
                let dr = (orb.c_ref.re.hi as f64) - c_center.real().to_f64();
                let di = (orb.c_ref.im.hi as f64) - c_center.imag().to_f64();
                let dist2 = dr * dr + di * di;

                // Prefer deeper perturbation validity
                let depth_bonus = -(orb.max_valid_perturb_index as f64);
               
                // Simple weighted score
                let score = dist2 * 1.0 + depth_bonus * 0.01;

                if score < best_score {
                    best_score = score;
                    best_slot = slot_idx;
                }
            }
        }
      
        best_slot
    }
}

fn build_gpu_orbit_slots_from_tiles(tiles: Vec<TileOrbitViewDf>, scale: f64) -> Vec<GpuOrbitSlot> {
    let max_orb_count = MAX_ORBITS_PER_FRAME as usize;
    let mut gpu_orbits = Vec::<GpuOrbitSlot>::new();
    let mut slot_num: u32 = 0;

    for tile in tiles {
        for orb in &tile.orbits {
            let mut flags = OrbitMetaMask::none();

            match orb.escape_index {
                Some(esc_idx) => {
                    flags.set(OrbitMetaFlags::OrbitEscapes);

                    if esc_idx < SHORT_THRESHOLD {
                        flags.set(OrbitMetaFlags::OrbitShort);
                    } 
                }
                None => {
                    flags.set(OrbitMetaFlags::OrbitInterior);
                }
            }
            // Orbit stability is primarily driven by how well perturbation iteration goes,
            // and for ScoutEngine, it starts from a maximum (i.e. MAX_REF_ORBIT), but may
            // be reduced as the reference orbit gets re-used and min_last_valid_i is taken
            // into account.
            if orb.max_valid_perturb_index >= MIN_PERTURB_ITERS {
                flags.set(OrbitMetaFlags::OrbitUsable);
            }

            let meta = GpuOrbitMeta::new(
                orb.max_valid_perturb_index,
                orb.escape_index,
                flags
            );
            debug!("Built GpuOrbitSlot {} for {:?}\twith orbit_id={}\tand meta={:?}", 
                slot_num, &tile.tile, &orb.orbit_id, &meta);

            gpu_orbits.push(GpuOrbitSlot {
                tile: tile.tile.clone(), 
                orbit: orb.clone(),
                meta
            });
            
            if gpu_orbits.len() == max_orb_count {
                break;
            }
            slot_num += 1;
        }
        if gpu_orbits.len() == max_orb_count {
            break;
        }
    }
    gpu_orbits
}

fn find_slot_index(
    orbit_slots: &Vec<GpuOrbitSlot>,
    orbit_id: u64,
) -> Option<u32> {
    orbit_slots
        .iter()
        .position(|s| s.orbit.orbit_id == orbit_id)
        .map(|i| i as u32)
}

fn build_pipelines(
    device: &wgpu::Device,
    uniform: SceneUniform,
    texture_format: wgpu::TextureFormat,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, 
      wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer,
      wgpu::Texture, wgpu::Buffer, wgpu::Texture,
      wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup, 
    wgpu::ComputePipeline, wgpu::RenderPipeline, wgpu::ComputePipeline) {
    // Shader for the Clear (compute) pipeline
    let clear_shader = device.create_shader_module(wgpu::include_wgsl!("clear_feedback.wgsl"));

    // Shader for the Render (vertex + fragment) pipeline
    let frag_shader = device.create_shader_module(wgpu::include_wgsl!("mandelbrot.wgsl"));

    // shader for Reduce (compute) pipeline
    let reduce_shader = device.create_shader_module(wgpu::include_wgsl!("reduce_feedback.wgsl"));

    /////////////////////////////////////////////////////////
    // Buffers & Textures to be shared accorss three piplines 
    /////////////////////////////////////////////////////////
    let uniform_buff = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // --- Frame feedback (sampled) ---
    let frame_feedback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame_feedback_buffer"),
        size: std::mem::size_of::<FrameFeedbackOut>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // --- Per-pixel feedback textures ---
    let last_valid_i_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("last_valid_i_tex"),
        size: wgpu::Extent3d {
            width: MAX_SCREEN_WIDTH,
            height: MAX_SCREEN_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let orbit_idx_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("orbit_idx_tex"),
        size: wgpu::Extent3d {
            width: MAX_SCREEN_WIDTH,
            height: MAX_SCREEN_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let flags_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("flags_tex"),
        size: wgpu::Extent3d {
            width: MAX_SCREEN_WIDTH,
            height: MAX_SCREEN_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    // --- Reduce output buffer ---
    let orbit_feedback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("orbit_feedback_buffer"),
        size: MAX_ORBITS_PER_FRAME as u64
            * std::mem::size_of::<OrbitFeedbackOut>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let debug_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("debug_buffer"),
        size: std::mem::size_of::<DebugOut>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });


    /////////////////////////////////////////////////////////
    // Pipeline 1 Bind Group Layout & Pipeline (clear/compute)
    /////////////////////////////////////////////////////////
    let clear_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("clear_bgl"),
        entries: &[
            // last_valid_i_tex
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            // orbit_idx_tex
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            // flags_tex
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            // orbit_feedback
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let clear_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("clear_feedback_bg"),
        layout: &clear_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &last_valid_i_tex.create_view(&Default::default())
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &orbit_idx_tex.create_view(&Default::default())
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(
                    &flags_tex.create_view(&Default::default())
                ),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: orbit_feedback_buffer.as_entire_binding(),
            },
        ],
    });


    let clear_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clear_pipeline_layout"),
            push_constant_ranges: &[],
            bind_group_layouts: &[&clear_bgl],
        });

    let clear_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clear_pipeline"),
            layout: Some(&clear_pipeline_layout),
            module: &clear_shader,
            entry_point: "cs_main",
        });

    
    /////////////////////////////////////////////////////////
    // Pipeline 2 Bind Group Layout & Pipeline (render/frag,vert)
    /////////////////////////////////////////////////////////
    /// 
    // Start with definitions for input (read) textures (and buffers),
    //  all for use in render pipeline's bind group 2.
    let ref_orbit_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ref_orbit_texture"),
        size: wgpu::Extent3d {
            width: MAX_REF_ORBIT,
            height: MAX_ORBITS_PER_FRAME * ROWS_PER_ORBIT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let ref_orbit_texture_view =
        ref_orbit_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let ref_orbit_meta_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ref_orbit_meta"),
        size: (MAX_ORBITS_PER_FRAME as usize * std::mem::size_of::<GpuOrbitMeta>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let tile_orbit_index_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tile_orbit_index_texture"),
        size: wgpu::Extent3d {
            width: MAX_SCREEN_TILES_X,
            height: MAX_SCREEN_TILES_Y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let tile_orbit_index_view =
        tile_orbit_index_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None
                },
                count: None,
            }
        ],
    });

     let frame_fb_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("frame feedback bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let ref_orbit_bgl = device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("ref_orbit_texture_bgl"),
            entries: &[
                // Reference Orbit Atlas
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Orbit Metadata buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Screen-space tile → orbit index
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Per-pixel Last-valid-i output texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Per-pixel (chosen) orbit_idx output texture
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Per-pixel perturbation flags (flags_tex) output texture
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        }
    );

    let debug_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("debug bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        ],
    });

    let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &scene_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buff.as_entire_binding()
            }
        ],
        label: None
    });

    let frame_feedback_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu feedback bind group"),
        layout: &frame_fb_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_feedback_buffer.as_entire_binding(),
            },
        ]
    });

    let ref_orbit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pipeline2_ref_orbit_bg"),
        layout: &ref_orbit_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&ref_orbit_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: ref_orbit_meta_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&tile_orbit_index_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(
                    &last_valid_i_tex.create_view(&Default::default())
                ),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(
                    &orbit_idx_tex.create_view(&Default::default())
                ),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(
                    &flags_tex.create_view(&Default::default())
                ),
            },
        ],
    });

    let debug_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("debug bind group"),
        layout: &debug_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: debug_buffer.as_entire_binding(),
            }
        ]
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render_pipeline_layout"),
        push_constant_ranges: &[],
        bind_group_layouts: &[
            &scene_bgl,
            &frame_fb_bgl,
            &ref_orbit_bgl,
            &debug_bgl,
        ],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &frag_shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &frag_shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: texture_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None
    });

    ///////////////////////////////////////////////////////
    // Pipeline 3 layouts and pipeline (reduce/compute)
    ///////////////////////////////////////////////////////
    let reduce_bgl = device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("reduce_feedback_bgl"),
            entries: &[
                // last_valid_i_tex
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // orbit_idx_tex
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // flags_tex
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // orbit_feedback buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        },
    );

    let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reduce_feedback_bg"),
        layout: &reduce_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &last_valid_i_tex.create_view(&Default::default())
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &orbit_idx_tex.create_view(&Default::default())
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(
                    &flags_tex.create_view(&Default::default())
                ),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: orbit_feedback_buffer.as_entire_binding(),
            },
        ],
    });

    let reduce_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reduce_feedback_pipeline_layout"),
            bind_group_layouts: &[&reduce_bgl],
            push_constant_ranges: &[],
        });

    let reduce_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reduce_feedback_pipeline"),
            layout: Some(&reduce_pipeline_layout),
            module: &reduce_shader,
            entry_point: "main",
        });

    ///////////////////////////////////////////////////////
    // Readback buffers
    ///////////////////////////////////////////////////////
    let frame_feedback_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame_feedback_readback"),
        size: std::mem::size_of::<FrameFeedbackOut>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let orbit_feedback_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("orbit_feedback_readback"),
        size: (MAX_ORBITS_PER_FRAME as usize
            * std::mem::size_of::<OrbitFeedbackOut>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let debug_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("debug_readback"),
        size: std::mem::size_of::<DebugOut>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    (uniform_buff, frame_feedback_buffer, frame_feedback_readback, 
        orbit_feedback_buffer, orbit_feedback_readback,
        debug_buffer, debug_readback,
        ref_orbit_texture, ref_orbit_meta_buf, tile_orbit_index_texture, 
        clear_bg, scene_bg, frame_feedback_bg, ref_orbit_bg, debug_bg, reduce_bg,
        clear_pipeline, render_pipeline, reduce_pipeline)
}

fn make_r32uint_storage_tex(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC, // readback happens later
        view_formats: &[],
    });

    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}
