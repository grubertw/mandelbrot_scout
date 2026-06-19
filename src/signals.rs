use crate::gpu_pipeline::structs::{OrbitFeedbackOut};
use crate::scout_engine::orbit::OrbitId;

use std::hash::{Hash, Hasher};
use std::time;

use num_complex::Complex32;
use crate::numerics::{FixedComplex, FixedReal};

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
    center: FixedComplex,
    scale: FixedReal, // pixel scale, pix_dx.max(pix_dy)
    half_extent: FixedReal, // scale * extent

    pub screen_extent_multiplier: f64, // width.max(height)
}

impl CameraSnapshot {
    pub fn new(
        frame_stamp: FrameStamp,
        center: FixedComplex,
        scale: FixedReal,
        screen_extent_multiplier: f64
    ) -> Self {
        let half_extent = scale.scale_by_f64(screen_extent_multiplier, true);

        Self {
            frame_stamp, center, scale, 
            screen_extent_multiplier, half_extent,
        }
    }

    pub fn frame_stamp(&self) -> &FrameStamp {
        &self.frame_stamp
    }

    pub fn center(&self) -> &FixedComplex {
        &self.center
    }

    pub fn scale(&self) -> &FixedReal {
        &self.scale
    }

    pub fn half_extent(&self) -> &FixedReal {
        &self.half_extent
    }
}

#[derive(Clone, Debug)]
pub struct GpuGridSample {
    pub frame_stamp: FrameStamp,
    
    // Gpu Reduced info about the pest sampled pixel it's sample grid
    pub location: FixedComplex,
    pub iters_reached: u32,
    pub escaped: bool,
    pub max_user_iters: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct OrbitObservation {
    pub frame_stamp: FrameStamp,
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
pub struct QualifiedOrbit {
    pub rank: u32,
    pub orbit_id: OrbitId,
    pub c_ref: FixedComplex,
    pub c_ref_32: Complex32,
    pub orbit: Vec<Complex32>,
    pub escape_index: Option<u32>,
    pub created_at: FrameStamp,
}

impl Hash for QualifiedOrbit {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.orbit_id.hash(state);
    }
}

impl PartialEq for QualifiedOrbit {
    fn eq(&self, other: &Self) -> bool {
        self.orbit_id == other.orbit_id
    }
}

impl Eq for QualifiedOrbit {}

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