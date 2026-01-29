use crate::gpu_pipeline::structs::*;
use crate::scene::policy::*;

use iced_wgpu::wgpu;
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub struct PipelineBundle {
    pub uniform_buff: wgpu::Buffer,
    pub frame_feedback_buffer: wgpu::Buffer,
    pub frame_feedback_readback: wgpu::Buffer,
    pub orbit_feedback_buffer: wgpu::Buffer,
    pub orbit_feedback_readback: wgpu::Buffer,
    pub debug_buffer: wgpu::Buffer,
    pub debug_readback: wgpu::Buffer,
    pub ref_orbit_texture: wgpu::Texture,
    pub ref_orbit_meta_buf: wgpu::Buffer,
    pub tile_orbit_index_texture: wgpu::Texture,
    pub clear_bg: wgpu::BindGroup,
    pub scene_bg: wgpu::BindGroup,
    pub frame_feedback_bg: wgpu::BindGroup,
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
        texture_format: wgpu::TextureFormat,
    ) -> Self {
        let (
            last_valid_i_tex, orbit_idx_tex, flags_tex, orbit_feedback_buffer, 
        ) = create_shared_buffers(device);

        //
        // Build WGPU Bind Group and Pipeline descriptors
        // --- 
        // The functions below can best be thought of as a memory contract for each
        // shader pass, where the wgsl source executes on the GPU. Note the buffers 
        // above are shared with each shader stage, acting as imputs and outputs.
        //

        let (
            clear_bg, clear_pipeline
        ) = build_clear_pipeline(device, 
            &last_valid_i_tex, &orbit_idx_tex, &flags_tex, &orbit_feedback_buffer);

        let (
            uniform_buff, frame_feedback_buffer, frame_feedback_readback,
            ref_orbit_texture, ref_orbit_meta_buf, tile_orbit_index_texture,
            debug_buffer, debug_readback,
            scene_bg, frame_feedback_bg, ref_orbit_bg, debug_bg, 
            render_pipeline
        ) = build_render_pipeline(device, &uniform, texture_format,
            &last_valid_i_tex, &orbit_idx_tex, &flags_tex
        );

        let (
            orbit_feedback_readback, reduce_bg, reduce_pipeline
        ) = build_reduce_pipeline(device, 
            &last_valid_i_tex, &orbit_idx_tex, &flags_tex, &orbit_feedback_buffer);

        Self {
            uniform_buff,
            frame_feedback_buffer, frame_feedback_readback,
            orbit_feedback_buffer, orbit_feedback_readback,
            debug_buffer, debug_readback,
            ref_orbit_texture, ref_orbit_meta_buf, tile_orbit_index_texture,
            clear_bg, scene_bg, frame_feedback_bg, ref_orbit_bg, debug_bg, reduce_bg,
            clear_pipeline, render_pipeline, reduce_pipeline
        }
    }
}

//
// Create WGPU Buffers & Textures to be shared accorss three piplines 
//
fn create_shared_buffers(device: &wgpu::Device) 
-> (wgpu::Texture, wgpu::Texture, wgpu::Texture, wgpu::Buffer) {
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

    (last_valid_i_tex, orbit_idx_tex, flags_tex, orbit_feedback_buffer)
}

//
// Pipeline 1 Bind Group Layout & Pipeline (clear/compute)
//
fn build_clear_pipeline(device: &wgpu::Device, 
    last_valid_i_tex: &wgpu::Texture, orbit_idx_tex: &wgpu::Texture, flags_tex: &wgpu::Texture, 
    orbit_feedback_buffer: &wgpu::Buffer, 
) -> (wgpu::BindGroup, wgpu::ComputePipeline) {
    // Shader for the Clear (compute) pipeline
    let clear_shader = device.create_shader_module(wgpu::include_wgsl!("clear_feedback.wgsl"));  

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
    
    (clear_bg, clear_pipeline)
}

//
// Pipeline 2 Bind Group Layout & Pipeline (render/frag,vert)
//
fn build_render_pipeline(device: &wgpu::Device, uniform: &SceneUniform, texture_format: wgpu::TextureFormat,
    last_valid_i_tex: &wgpu::Texture, orbit_idx_tex: &wgpu::Texture, flags_tex: &wgpu::Texture
) -> (
    wgpu::Buffer, wgpu::Buffer, wgpu::Buffer,
    wgpu::Texture, wgpu::Buffer, wgpu::Texture,
    wgpu::Buffer, wgpu::Buffer,
    wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup,
    wgpu::RenderPipeline
) {
    // Shader for the Render (vertex + fragment) pipeline
    let frag_shader = device.create_shader_module(wgpu::include_wgsl!("mandelbrot.wgsl"));

    let uniform_buff = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(uniform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // --- Frame feedback (sampled) ---
    let frame_feedback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame_feedback_buffer"),
        size: std::mem::size_of::<FrameFeedbackOut>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let frame_feedback_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame_feedback_readback"),
        size: std::mem::size_of::<FrameFeedbackOut>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
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

    (
        uniform_buff, frame_feedback_buffer, frame_feedback_readback,
        ref_orbit_texture, ref_orbit_meta_buf, tile_orbit_index_texture,
        debug_buffer, debug_readback,
        scene_bg, frame_feedback_bg, ref_orbit_bg, debug_bg, 
        render_pipeline
    )
}
//
// Pipeline 3 layouts and pipeline (reduce/compute)
//
fn build_reduce_pipeline(device: &wgpu::Device, 
    last_valid_i_tex: &wgpu::Texture, orbit_idx_tex: &wgpu::Texture, flags_tex: &wgpu::Texture, 
    orbit_feedback_buffer: &wgpu::Buffer,
) -> (wgpu::Buffer, wgpu::BindGroup, wgpu::ComputePipeline) {
    // shader for Reduce (compute) pipeline
    let reduce_shader = device.create_shader_module(wgpu::include_wgsl!("reduce_feedback.wgsl"));

    let orbit_feedback_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("orbit_feedback_readback"),
        size: (MAX_ORBITS_PER_FRAME as usize
            * std::mem::size_of::<OrbitFeedbackOut>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

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
    
    (orbit_feedback_readback, reduce_bg, reduce_pipeline)
}
