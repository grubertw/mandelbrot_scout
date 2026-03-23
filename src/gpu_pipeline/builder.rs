use crate::gpu_pipeline::structs::*;

use iced_wgpu::wgpu;
use log::trace;
use wgpu::util::DeviceExt;
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
    pub calc_bg: wgpu::BindGroup,
    pub debug_bg: wgpu::BindGroup,
    pub color_bg: wgpu::BindGroup,
    pub reduce_bg: wgpu::BindGroup,
    pub clear_pipeline: wgpu::ComputePipeline,
    pub calc_mandel_pipeline: wgpu::ComputePipeline,
    pub color_pipeline: wgpu::RenderPipeline,
    pub reduce_pipeline: wgpu::ComputePipeline
}

impl PipelineBundle {
    pub fn build_pipelines(
        device: &wgpu::Device,
        uniform: &SceneUniform,
        texture_format: wgpu::TextureFormat,
        settings: &Settings
    ) -> Self {
        let (uniform_buff, mandel_out_tex, grid_feedback_buffer, orbit_feedback_buffer) =
            create_shared_buffers(device, uniform, settings);

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
            ref_orbit_location_buf, rank_one_orbit_buf, rank_two_orbit_buf,
            debug_buffer, debug_readback,
            calc_bg, debug_bg,
            calc_mandel_pipeline
        ) = build_shader_calc_pipeline(device, &uniform_buff, &mandel_out_tex, &settings);

        let (
            palette_texture,
            color_bg,
            color_pipeline
        ) = build_color_pipeline(device, &uniform_buff, &mandel_out_tex, texture_format, &settings);

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
            clear_bg, calc_bg, debug_bg, color_bg, reduce_bg,
            clear_pipeline,
            calc_mandel_pipeline, color_pipeline, reduce_pipeline
        }
    }
}

//
// Create WGPU Buffers & Textures to be shared accorss three piplines 
//
fn create_shared_buffers(device: &wgpu::Device, uniform: &SceneUniform, settings: &Settings)
-> (wgpu::Buffer, wgpu::Texture, wgpu::Buffer, wgpu::Buffer) {
    // Allocate Scene Uniforms buffer, where most shader settings reside
    let uniform_buff = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(uniform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

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

    (uniform_buff, mandel_out_tex, grid_feedback_buffer, orbit_feedback_buffer)
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
                    format: wgpu::TextureFormat::Rgba32Float,
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
// Pipeline 2 mandelbrot computation (compute) shader
//
fn build_shader_calc_pipeline(
    device: &wgpu::Device,
    uniform_buff: &wgpu::Buffer,
    mandel_out_tex: &wgpu::Texture,
    settings: &Settings
) -> (
    wgpu::Buffer, wgpu::Buffer, wgpu::Buffer,
    wgpu::Buffer, wgpu::Buffer,
    wgpu::BindGroup, wgpu::BindGroup,
    wgpu::ComputePipeline,
) {
    let calc_mandel_f32_shader = device.create_shader_module(wgpu::include_wgsl!("mandelbrot_f32.wgsl"));

    let orbit_location_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("orbit_location_buf"),
        size: (settings.max_orbits_per_frame as usize * size_of::<GpuRefOrbitLocation>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let rank_one_orbit_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rank_one_orbit_buf"),
        size: (settings.max_ref_orbit as usize * size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let rank_two_orbit_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rank_two_orbit_buf"),
        size: (settings.max_ref_orbit as usize * size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
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

    let calc_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene bind group layout"),
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
            // Per-pixel mandelbrot() results output texture
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            // Orbit Location buffer
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Rank 1 orbit buffer
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Rank 2 orbit buffer
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]}
    );

    let debug_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("debug bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        ],
    });

    let calc_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Calc bind group"),
        layout: &calc_bgl,
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
                resource: orbit_location_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: rank_one_orbit_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: rank_two_orbit_buf.as_entire_binding(),
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

    let calc_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("calc_pipeline_layout"),
        push_constant_ranges: &[],
        bind_group_layouts: &[
            &calc_bgl,
            &debug_bgl,
        ],
    });

    let calc_mandel_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("calc_mandel_f32_pipeline"),
            layout: Some(&calc_pipeline_layout),
            module: &calc_mandel_f32_shader,
            entry_point: Option::from("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    (
        orbit_location_buf, rank_one_orbit_buf, rank_two_orbit_buf,
        debug_buffer, debug_readback,
        calc_bg, debug_bg,
        calc_mandel_pipeline,
    )
}

//
// Pipeline 3 color (vert+fragment) shader pipeline
//
fn build_color_pipeline(
    device: &wgpu::Device,
    uniform_buff: &wgpu::Buffer,
    mandel_calc_tex: &wgpu::Texture,
    texture_format: wgpu::TextureFormat,
    settings: &Settings
) -> (
    wgpu::Texture,
    wgpu::BindGroup,
    wgpu::RenderPipeline
) {
    // Shader for the Render (vertex + fragment) pipeline
    let color_shader = device.create_shader_module(wgpu::include_wgsl!("color.wgsl"));

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

    let color_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Color bind group layout"),
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
            },
            // mandel calc output 2d (per-pixel) texture
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // Color palette 1d texture
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            }
        ],
    });

    let color_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("color_bg"),
        layout: &color_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buff.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &mandel_calc_tex.create_view(&Default::default())
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(
                    &palette_view
                ),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&palette_sampler),
            },
        ],
    });

    let color_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("color_pipeline_layout"),
        push_constant_ranges: &[],
        bind_group_layouts: &[
            &color_bgl,
        ],
    });

    let color_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Color Pipeline"),
        layout: Some(&color_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &color_shader,
            entry_point: Option::from("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &color_shader,
            entry_point: Option::from("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: texture_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent::REPLACE,
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });

    (palette_texture, color_bg, color_pipeline)
}

//
// Pipeline 4 reduce (compute) shader
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
                // mandel calc output texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
