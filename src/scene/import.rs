use std::fs::File;
use std::io::BufReader;
use log::warn;
use serde::{Deserialize, Serialize};
use png::Decoder;

pub const META_VERSION: &str = "1";

// Used to generate JSON data for image export
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FractalMetadata {
    pub program_name: String,
    pub version: String,
    pub center_re: String,
    pub center_im: String,
    pub scale: String,
    pub max_iter: u32,
    pub ref_orbit: Option<RefOrbitMetadata>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RefOrbitMetadata {
    pub c_ref_re: String,
    pub c_ref_im: String,
    pub center_offset_re: String,
    pub center_offset_im: String,
    pub max_ref_iters: u32,
}

pub fn load_metadata_from_png(path: &str) -> Option<FractalMetadata> {
    let file = File::open(path).ok()?;
    let decoder = Decoder::new(BufReader::new(file));
    let reader = match decoder.read_info() {
        Ok(reader) => reader,
        Err(e) => {
            warn!("Failed to decode png image: {:?}", e);
            return None;
        }
    };

    for chunk in &reader.info().uncompressed_latin1_text {
        if chunk.keyword == "FractalMetadata" {
            return serde_json::from_str(&chunk.text).ok();
        }
    }

    warn!("Failed to find FractalMetadata in the png header");
    None
}