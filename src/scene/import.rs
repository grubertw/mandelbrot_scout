use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use log::warn;
use serde::{Deserialize, Serialize};
use png::Decoder;
use crate::scene::Rgba8Palette;

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

pub struct ExtPalette {
    pub name: String,
    pub colors: Vec<[u8; 3]>,
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

impl ExtPalette {
    pub fn parse_map(path: &Path) -> Result<ExtPalette, String> {
        let name = path.file_prefix().unwrap().to_str().unwrap();
        let input = match fs::read_to_string(path.to_str().unwrap()) {
            Ok(input) => input,
            Err(e) => return Err(e.to_string()),
        };

        let mut colors = Vec::new();

        for (i, line) in input.lines().enumerate() {
            let parts: Vec<_> = line.split_whitespace().collect();

            if parts.len() != 3 {
                continue; // skip junk lines
            }

            let r = parts[0].parse::<u8>().map_err(|_| format!("line {}", i))?;
            let g = parts[1].parse::<u8>().map_err(|_| format!("line {}", i))?;
            let b = parts[2].parse::<u8>().map_err(|_| format!("line {}", i))?;

            colors.push([r, g, b]);
        }

        if colors.is_empty() {
            return Err("No valid colors found in MAP file".into());
        }

        Ok(ExtPalette { name: name.to_string(), colors })
    }

    pub fn to_color_palette(&self) -> Rgba8Palette {
        let rgba = self.colors
            .iter()
            .map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
            .collect();

        Rgba8Palette {
            name: self.name.clone(),
            palette: rgba,
        }
    }
}