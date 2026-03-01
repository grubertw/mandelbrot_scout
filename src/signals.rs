use crate::gpu_pipeline::structs::{TileFeedbackOut};
use crate::numerics::{ComplexDf, Df};
use crate::scout_engine::orbit::OrbitId;
use crate::scout_engine::tile::{TileId, TileGeometry};

use std::hash::{Hash, Hasher};
use std::time;

use rug::{Float, Complex};

///////////////////////////////////////////////////////////
// Consumed by Scout Engine
///////////////////////////////////////////////////////////
#[derive(Clone, Copy, Debug)]
pub struct FrameStamp {
    pub frame_id: u64,
    pub timestamp: time::Instant, // in nanoseconds
} 

#[derive(Clone, Debug)]
pub struct CameraSnapshot {
    frame_stamp: FrameStamp,
    center: Complex,
    scale: Float, // pixel scale, pix_dx.max(pix_dy)
    half_extent: Float, // scale * extent

    pub screen_extent_multiplier: f64, // width.max(height)
}

impl CameraSnapshot {
    pub fn new(
        frame_stamp: FrameStamp,
        center: Complex,
        scale: Float,
        screen_extent_multiplier: f64
    ) -> Self {
        let half_extent = scale.clone() * screen_extent_multiplier;

        Self {
            frame_stamp, center, scale, 
            screen_extent_multiplier, half_extent,
        }
    }

    pub fn frame_stamp(&self) -> &FrameStamp {
        &self.frame_stamp
    }

    pub fn center(&self) -> &Complex {
        &self.center
    }

    pub fn scale(&self) -> &Float {
        &self.scale
    }

    pub fn half_extent(&self) -> &Float {
        &self.half_extent
    }
}

#[derive(Clone, Debug)]
pub struct GpuGridSample {
    pub frame_stamp: FrameStamp,
    
    // Gpu Reduced info about the pest sampled pixel it's sample grid
    pub best_sample: Complex,
    pub best_sample_iters: u32,
    pub best_sample_escaped: bool,
    pub max_user_iters: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct TileObservation {
    pub frame_stamp: FrameStamp,
    pub tile_id: TileId,
    pub orbit_id: OrbitId,
    pub feedback: TileFeedbackOut,
}

#[derive(Clone, Debug)]
pub struct FrameDiagnostics {
    pub frame_stamp: FrameStamp,
    pub message: String,
}

///////////////////////////////////////////////////////////
// Produced by Scout Engine
///////////////////////////////////////////////////////////
#[derive(Clone, Debug)]
pub struct TileOrbitViewDf {
    pub id: TileId,
    pub geometry: TileGeometry,
    pub delta_from_center: Complex,
    pub delta_from_center_to_anchor: ComplexDf,
    pub orbit: ReferenceOrbitDf,
}

#[derive(Clone, Debug)]
pub struct ReferenceOrbitDf {
    pub orbit_id: OrbitId,
    pub c_ref: ComplexDf,
    pub orbit_re_hi: Vec<f32>,
    pub orbit_re_lo: Vec<f32>,
    pub orbit_im_hi: Vec<f32>,
    pub orbit_im_lo: Vec<f32>,
    pub escape_index: Option<u32>,
    pub r_valid: Df,
    pub contraction: Df,
    pub created_at: FrameStamp,
}

impl Hash for ReferenceOrbitDf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.orbit_id.hash(state);
    }
}

impl PartialEq for ReferenceOrbitDf {
    fn eq(&self, other: &Self) -> bool {
        self.orbit_id == other.orbit_id
    }
}

impl Eq for ReferenceOrbitDf {}

#[derive(Clone, Debug)]
pub struct ScoutDiagnostics {
    pub message: String,
    pub consumed: bool,
}

impl FrameStamp {
    pub fn new() -> Self {
        Self {
            frame_id: u64::MAX,
            timestamp: time::Instant::now()
        }
    }
}

impl ScoutDiagnostics {
    pub fn new(message: String) -> Self {
        Self {
            message, consumed: false
        }
    }
}