use crate::gpu_pipeline::structs::*;

use iced_wgpu::wgpu;
use log::trace;
use wgpu::util::DeviceExt;
use crate::numerics::ComplexDf;
use crate::settings::Settings;

#[derive(Debug)]
pub struct PipelineBundle {
    pub uniform_buff: wgpu::Buffer,
    pub grid_feedback_buffer: wgpu::Buffer,
    pub grid_feedback_readback: wgpu::Buffer,
    pub orbit_feedback_buffer: wgpu::Buffer,
    pub orbit_feedback_readback: wgpu::Buffer,
    pub debug_buffer: wgpu::Buffer,
    pub debug_readback: wgpu::Buffer,
    pub ref_orbit_location_buf: wgpu::Buffer,
    pub rank_one_orbit_buf: wgpu::Buffer,
    pub rank_two_orbit_buf: wgpu::Buffer,
    pub palette_texture: wgpu::Texture,
    pub clear_bg: wgpu::BindGroup,
    pub scene_bg: wgpu::BindGroup,
    pub ref_orbit_bg: wgpu::BindGroup,
    pub palette_bg: wgpu::BindGroup,
    pub debug_bg: wgpu::BindGroup,
    pub reduce_bg: wgpu::BindGroup,
    pub clear_pipeline: wgpu::ComputePipeline,
    pub render_pipeline: wgpu::RenderPipeline,
    pub reduce_pipeline: wgpu::ComputePipeline
}

impl PipelineBundle {
    pub fn build_pipelines(
        device: &wgpu::Device,
        uniform: &SceneUniform,
        texture_format: wgpu::TextureFormat,
        settings: &Settings
    ) -> Self {
        let (mandel_out_tex, grid_feedback_buffer, orbit_feedback_buffer) =
            create_shared_buffers(device, settings);

        //
        // Build WGPU Bind Group and Pipeline descriptors
        // --- 
        // The functions below can best be thought of as a memory contract for each
        // shader pass, where the wgsl source executes on the GPU. Note the buffers 
        // above are shared with each shader stage, acting as imputs and outputs.
        //
        let (
            clear_bg, clear_pipeline
        ) = build_clear_pipeline(device, &mandel_out_tex, &grid_feedback_buffer, &orbit_feedback_buffer);

        let (
            uniform_buff,
            ref_orbit_location_buf, rank_one_orbit_buf, rank_two_orbit_buf,
            palette_texture,
            debug_buffer, debug_readback,
            scene_bg, ref_orbit_bg, palette_bg, debug_bg,
            render_pipeline
        ) = build_shader_calc_pipeline(device, uniform, texture_format, &mandel_out_tex, &settings);

        let (
            grid_feedback_readback, orbit_feedback_readback,
            reduce_bg, reduce_pipeline
        ) = build_reduce_pipeline(device, &uniform_buff, &mandel_out_tex,
                  &grid_feedback_buffer, &orbit_feedback_buffer, &settings);

        Self {
            uniform_buff,
            grid_feedback_buffer,
            grid_feedback_readback,
            orbit_feedback_buffer,
            orbit_feedback_readback,
            debug_buffer, debug_readback,
            ref_orbit_location_buf, rank_one_orbit_buf, rank_two_orbit_buf,
            palette_texture,
            clear_bg, scene_bg, ref_orbit_bg, palette_bg, debug_bg, reduce_bg,
            clear_pipeline, render_pipeline, reduce_pipeline
        }
    }
}

//
// Create WGPU Buffers & Textures to be shared accorss three piplines 
//
fn create_shared_buffers(device: &wgpu::Device, settings: &Settings) 
-> (wgpu::Texture, wgpu::Buffer, wgpu::Buffer) {
    // --- Per-pixel feedback textures ---
    let mandel_out_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mandel_out_tex"),
        size: wgpu::Extent3d {
            width: settings.max_screen_width,
            height: settings.max_screen_height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    // --- Reduce output buffer ---
    let grid_feedback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grid_feedback_buffer"),
        size: ((settings.max_screen_width / settings.screen_grid_size) * (settings.max_screen_height / settings.screen_grid_size)) as u64
            * std::mem::size_of::<GridFeedbackOut>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let orbit_feedback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tile_feedback_buffer"),
        size: settings.max_orbits_per_frame as u64
            * std::mem::size_of::<OrbitFeedbackOut>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    (mandel_out_tex, grid_feedback_buffer, orbit_feedback_buffer)
}

//
// Pipeline 1 Bind Group Layout & Pipeline (clear/compute)
//
fn build_clear_pipeline(
    device: &wgpu::Device,
    mandel_out_tex: &wgpu::Texture,
    grid_feedback_buffer: &wgpu::Buffer,
    orbit_feedback_buffer: &wgpu::Buffer,
) -> (wgpu::BindGroup, wgpu::ComputePipeline) {
    // Shader for the Clear (compute) pipeline
    let clear_shader = device.create_shader_module(wgpu::include_wgsl!("clear_feedback.wgsl"));  

    let clear_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("clear_bgl"),
        entries: &[
            // mandel_out_tex
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            // grid feedback
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // orbit feedback
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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
                    &mandel_out_tex.create_view(&Default::default())
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: grid_feedback_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
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
            entry_point: Option::from("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
    
    (clear_bg, clear_pipeline)
}

//
// Pipeline 2 Bind Group Layout & Pipeline (render/frag,vert)
//
fn build_shader_calc_pipeline(
    device: &wgpu::Device,
    uniform: &SceneUniform,
    texture_format: wgpu::TextureFormat,
    mandel_out_tex: &wgpu::Texture,
    settings: &Settings
) -> (
    wgpu::Buffer,
    wgpu::Buffer, wgpu::Buffer, wgpu::Buffer,
    wgpu::Texture,
    wgpu::Buffer, wgpu::Buffer,
    wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup,
    wgpu::RenderPipeline
) {
    // Shader for the Render (vertex + fragment) pipeline
    let frag_shader = device.create_shader_module(wgpu::include_wgsl!("mandelbrot_df.wgsl"));

    let uniform_buff = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(uniform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let orbit_location_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("orbit_location_buf"),
        size: (settings.max_orbits_per_frame as usize * size_of::<GpuRefOrbitLocation>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let rank_one_orbit_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rank_one_orbit_buf"),
        size: (settings.max_ref_orbit as usize * size_of::<ComplexDf>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let rank_two_orbit_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rank_two_orbit_buf"),
        size: (settings.max_ref_orbit as usize * size_of::<ComplexDf>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    trace!("Settings.max_palette_colors={}", settings.max_palette_colors);

    let palette_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("palette_texture"),
        size: wgpu::Extent3d {
            width: settings.max_palette_colors,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let palette_view = palette_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let palette_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        ..Default::default()
    });

    let debug_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("debug_buffer"),
        size: size_of::<DebugOut>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let debug_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("debug_readback"),
        size: size_of::<DebugOut>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

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

    let ref_orbit_bgl = device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("ref_orbit_texture_bgl"),
            entries: &[
                // Per-pixel perturbation flags (flags_tex) output texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Orbit Location buffer
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
                // Rank 1 orbit buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Rank 2 orbit buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        }
    );

    let palette_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("color palette bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            }
        ],
    });

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

    let ref_orbit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ref_orbit_bg"),
        layout: &ref_orbit_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &mandel_out_tex.create_view(&Default::default())
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: orbit_location_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: rank_one_orbit_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: rank_two_orbit_buf.as_entire_binding(),
            },
        ],
    });

    let palette_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("palette_bg"),
        layout: &palette_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &palette_view
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&palette_sampler),
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
            &ref_orbit_bgl,
            &palette_bgl,
            &debug_bgl,
        ],
    });

    let shader_calc_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Shader Calc Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &frag_shader,
            entry_point: Option::from("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &frag_shader,
            entry_point: Option::from("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: texture_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    (
        uniform_buff,
        orbit_location_buf, rank_one_orbit_buf, rank_two_orbit_buf,
        palette_texture,
        debug_buffer, debug_readback,
        scene_bg, ref_orbit_bg, palette_bg, debug_bg,
        shader_calc_pipeline
    )
}
//
// Pipeline 3 layouts and pipeline (reduce/compute)
//
fn build_reduce_pipeline(
    device: &wgpu::Device,
    uniform_buff: &wgpu::Buffer,
    mandel_out_tex: &wgpu::Texture,
    grid_feedback_buffer: &wgpu::Buffer,
    orbit_feedback_buffer: &wgpu::Buffer,
    settings: &Settings
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup, wgpu::ComputePipeline) {
    // shader for Reduce (compute) pipeline
    let reduce_shader = device.create_shader_module(wgpu::include_wgsl!("reduce_feedback.wgsl"));

    let grid_feedback_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grid_feedback_readback"),
        size: (((settings.max_screen_width / settings.screen_grid_size) * (settings.max_screen_height / settings.screen_grid_size)) as usize
            * std::mem::size_of::<GridFeedbackOut>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let orbit_feedback_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tile_feedback_readback"),
        size: (settings.max_orbits_per_frame as usize
            * size_of::<OrbitFeedbackOut>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let reduce_bgl = device.create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("reduce_feedback_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None,
                },
                // flags_tex
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
                // grid feedback buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // orbit feedback buffer
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
                resource: uniform_buff.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &mandel_out_tex.create_view(&Default::default())
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: grid_feedback_buffer.as_entire_binding(),
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
            entry_point: Option::from("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    
    (grid_feedback_readback, orbit_feedback_readback, reduce_bg, reduce_pipeline)
}
