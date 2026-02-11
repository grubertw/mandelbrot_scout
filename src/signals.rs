use crate::gpu_pipeline::structs::{OrbitFeedbackOut};
use crate::numerics::{Df, ComplexDf};
use crate::scout_engine::orbit::{OrbitId};
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
    pub frame_stamp: FrameStamp,
    pub center: Complex,
    pub scale: Float, // pixel scale, pix_dx.max(pix_dy)
    pub screen_extent_multiplier: f64, // width.max(height)
}

#[derive(Clone, Copy, Debug)]
pub struct GpuFeedback {
    pub frame_stamp: FrameStamp,
    pub max_lambda: Df,
    pub max_delta_z: Df,
    pub escape_ratio: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct OrbitObservation {
    pub frame_stamp: FrameStamp,
    pub tile_id: TileId,
    pub orbit_id: OrbitId,      
    pub feedback: OrbitFeedbackOut,
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
    pub tile: TileId,
    pub geometry: TileGeometry,
    pub orbits: Vec<ReferenceOrbitDf>,
}

#[derive(Clone, Debug)]
pub struct ReferenceOrbitDf {
    pub orbit_id: u64,
    pub c_ref: ComplexDf,
    pub orbit_re_hi: Vec<f32>,
    pub orbit_re_lo: Vec<f32>,
    pub orbit_im_hi: Vec<f32>,
    pub orbit_im_lo: Vec<f32>,
    pub escape_index: Option<u32>,
    pub min_valid_perturb_index: u32,
    pub max_valid_perturb_index: u32,
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
    pub timestamp: time::SystemTime,
    pub message: String,
}

impl CameraSnapshot {
    // Snapshots with MAX frame_id signify they are invalid and should NOT be used.
    pub fn new() -> Self {
        Self {
            frame_stamp: FrameStamp::new(),
            center: Complex::new(80),
            scale: Float::new(80),
            screen_extent_multiplier: 0.0,
        }
    }
}

impl FrameStamp {
    pub fn new() -> Self {
        Self {
            frame_id: u64::MAX,
            timestamp: time::Instant::now()
        }
    }
}