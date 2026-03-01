use crate::gpu_pipeline::structs::*;
use crate::scene::policy::*;

use iced_wgpu::wgpu;
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub struct PipelineBundle {
    pub uniform_buff: wgpu::Buffer,
    pub reduce_uniform_buff: wgpu::Buffer,
    pub grid_feedback_buffer: wgpu::Buffer,
    pub grid_feedback_readback: wgpu::Buffer,
    pub tile_feedback_buffer: wgpu::Buffer,
    pub tile_feedback_readback: wgpu::Buffer,
    pub debug_buffer: wgpu::Buffer,
    pub debug_readback: wgpu::Buffer,
    pub ref_orbit_texture: wgpu::Texture,
    pub tile_geometry_buf: wgpu::Buffer,
    pub clear_bg: wgpu::BindGroup,
    pub scene_bg: wgpu::BindGroup,
    pub ref_orbit_bg: wgpu::BindGroup,
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
        reduce_uniform: &ReduceUniform,
        texture_format: wgpu::TextureFormat,
    ) -> Self {
        let (flags_tex, grid_feedback_buffer, tile_feedback_buffer) = 
            create_shared_buffers(device);

        //
        // Build WGPU Bind Group and Pipeline descriptors
        // --- 
        // The functions below can best be thought of as a memory contract for each
        // shader pass, where the wgsl source executes on the GPU. Note the buffers 
        // above are shared with each shader stage, acting as imputs and outputs.
        //

        let (
            clear_bg, clear_pipeline
        ) = build_clear_pipeline(device, &flags_tex, &grid_feedback_buffer, &tile_feedback_buffer);

        let (
            uniform_buff,
            ref_orbit_texture, tile_geometry_buf,
            debug_buffer, debug_readback,
            scene_bg, ref_orbit_bg, debug_bg, 
            render_pipeline
        ) = build_render_pipeline(device, uniform, texture_format, &flags_tex);

        let (
            reduce_uniform_buff, grid_feedback_readback, tile_feedback_readback, 
            reduce_bg, reduce_pipeline
        ) = build_reduce_pipeline(device, reduce_uniform, &flags_tex, 
                                  &grid_feedback_buffer, &tile_feedback_buffer);

        Self {
            uniform_buff, reduce_uniform_buff,
            grid_feedback_buffer,
            grid_feedback_readback,
            tile_feedback_buffer,
            tile_feedback_readback,
            debug_buffer, debug_readback,
            ref_orbit_texture, tile_geometry_buf,
            clear_bg, scene_bg, ref_orbit_bg, debug_bg, reduce_bg,
            clear_pipeline, render_pipeline, reduce_pipeline
        }
    }
}

//
// Create WGPU Buffers & Textures to be shared accorss three piplines 
//
fn create_shared_buffers(device: &wgpu::Device) 
-> (wgpu::Texture, wgpu::Buffer, wgpu::Buffer) {
    // --- Per-pixel feedback textures ---
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
    let grid_feedback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grid_feedback_buffer"),
        size: ((MAX_SCREEN_WIDTH / SCREEN_GRID_SIZE) * (MAX_SCREEN_HEIGHT / SCREEN_GRID_SIZE)) as u64
            * std::mem::size_of::<GridFeedbackOut>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let tile_feedback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tile_feedback_buffer"),
        size: MAX_ORBITS_PER_FRAME as u64
            * std::mem::size_of::<TileFeedbackOut>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    (flags_tex, grid_feedback_buffer, tile_feedback_buffer)
}

//
// Pipeline 1 Bind Group Layout & Pipeline (clear/compute)
//
fn build_clear_pipeline(
    device: &wgpu::Device,
    flags_tex: &wgpu::Texture,
    grid_feedback_buffer: &wgpu::Buffer,
    tile_feedback_buffer: &wgpu::Buffer,
) -> (wgpu::BindGroup, wgpu::ComputePipeline) {
    // Shader for the Clear (compute) pipeline
    let clear_shader = device.create_shader_module(wgpu::include_wgsl!("clear_feedback.wgsl"));  

    let clear_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("clear_bgl"),
        entries: &[
            // flags_tex
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
            // tile feedback
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
                    &flags_tex.create_view(&Default::default())
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: grid_feedback_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: tile_feedback_buffer.as_entire_binding(),
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
fn build_render_pipeline(
    device: &wgpu::Device, 
    uniform: &SceneUniform, 
    texture_format: wgpu::TextureFormat,
    flags_tex: &wgpu::Texture
) -> (
    wgpu::Buffer,
    wgpu::Texture, wgpu::Buffer,
    wgpu::Buffer, wgpu::Buffer,
    wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup,
    wgpu::RenderPipeline
) {
    // Shader for the Render (vertex + fragment) pipeline
    let frag_shader = device.create_shader_module(wgpu::include_wgsl!("mandelbrot.wgsl"));

    let uniform_buff = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(uniform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

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

    let tile_geometry_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tile_geometry_buf"),
        size: (MAX_ORBITS_PER_FRAME as usize * std::mem::size_of::<GpuTileGeometry>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let debug_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("debug_buffer"),
        size: std::mem::size_of::<DebugOut>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let debug_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("debug_readback"),
        size: std::mem::size_of::<DebugOut>() as u64,
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
                // Ref Orbit Tile Anchors
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
                // Tile Geometry buffer
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
                // Per-pixel perturbation flags (flags_tex) output texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
                resource: tile_geometry_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
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
            &ref_orbit_bgl,
            &debug_bgl,
        ],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
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
        ref_orbit_texture, tile_geometry_buf,
        debug_buffer, debug_readback,
        scene_bg, ref_orbit_bg, debug_bg, 
        render_pipeline
    )
}
//
// Pipeline 3 layouts and pipeline (reduce/compute)
//
fn build_reduce_pipeline(
    device: &wgpu::Device,
    reduce_uniform: &ReduceUniform,
    flags_tex: &wgpu::Texture,
    grid_feedback_buffer: &wgpu::Buffer,
    tile_feedback_buffer: &wgpu::Buffer,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup, wgpu::ComputePipeline) {
    // shader for Reduce (compute) pipeline
    let reduce_shader = device.create_shader_module(wgpu::include_wgsl!("reduce_feedback.wgsl"));

    let reduce_uniform_buff = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(reduce_uniform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let grid_feedback_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grid_feedback_readback"),
        size: (((MAX_SCREEN_WIDTH / SCREEN_GRID_SIZE) * (MAX_SCREEN_HEIGHT / SCREEN_GRID_SIZE)) as usize
            * std::mem::size_of::<GridFeedbackOut>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let tile_feedback_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tile_feedback_readback"),
        size: (MAX_ORBITS_PER_FRAME as usize
            * std::mem::size_of::<TileFeedbackOut>()) as u64,
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
                // tile feedback buffer
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
                resource: reduce_uniform_buff.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &flags_tex.create_view(&Default::default())
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: grid_feedback_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: tile_feedback_buffer.as_entire_binding(),
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
    
    (reduce_uniform_buff, grid_feedback_readback, tile_feedback_readback, reduce_bg, reduce_pipeline)
}
