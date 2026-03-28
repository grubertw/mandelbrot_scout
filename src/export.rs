use std::cell::RefCell;
use std::fs::File;
use std::rc::Rc;
use image::{ExtendedColorType, ImageEncoder, ImageResult};
use image::codecs::jpeg::JpegEncoder;
use log::debug;
use png::{BitDepth, ColorType, Compression, Encoder, EncodingError};
use crate::scene::import::FractalMetadata;
use crate::scene::Scene;

fn read_image_from_texture(scene: Rc<RefCell<Scene>>, width: u32, height: u32) -> Vec<u8> {
    let mut s = scene.borrow_mut();

    if s.width() != width || s.height() != height {
        s.recalculate();
    }

    s.render(width, height, None);
    s.read_render_feedback(width, height)
}

pub fn render_scene_to_png(
    scene: Rc<RefCell<Scene>>,
    width: u32,
    height: u32,
    file: File,
    compression: Compression
) {
    let padded_img_data = read_image_from_texture(scene.clone(), width, height);
    debug!("read padded image data of {} bytes", padded_img_data.len());

    let metadata = scene.borrow().build_metadata();
    debug!("metadata for png extracted from scene: {:?}", metadata);

    scene.borrow().scout().spawn_external_task(move || {
        let img_res = process_and_encode_png(
            padded_img_data,
            width, height,
            file, compression, metadata
        );

        match img_res {
            Ok(_) => format!("Image of width {} and height {} exported successfully!",
                        width, height),
            Err(e) => format!("Export failed: {}", e),
        }
    });
}

fn process_and_encode_png(
    padded_img_data: Vec<u8>,
    width: u32,
    height: u32,
    file: File,
    compression: Compression,
    metadata: FractalMetadata,
) -> Result<(), EncodingError> {

    let mut img_data = strip_padded_img_data(width, height, &padded_img_data, false);
    linear_to_srgb_inplace_with_lut(&mut img_data, true);
    debug!("stripped image data has {} bytes", img_data.len());

    let mut encoder = Encoder::new(file, width, height);
    encoder.set_color(ColorType::Rgba);
    encoder.set_depth(BitDepth::Eight);
    encoder.set_compression(compression);

    // 🔥 Embed metadata as JSON
    let json = serde_json::to_string(&metadata).unwrap();

    encoder.add_text_chunk("FractalMetadata".to_string(), json)?;

    let mut writer = encoder.write_header()?;
    writer.write_image_data(&img_data)
}

pub fn render_scene_to_jpeg(
    scene: Rc<RefCell<Scene>>,
    width: u32,
    height: u32,
    file: File,
    quality: u8,
) {
    let padded_img_data = read_image_from_texture(scene.clone(), width, height);

    scene.borrow().scout().spawn_external_task(move || {
        let img_res = process_and_encode_jpeg(
            padded_img_data,
            width, height,
            file, quality,
        );

        match img_res {
            Ok(_) => format!("Image of width {} and height {} exported successfully!",
                        width, height),
            Err(e) => format!("Export failed: {}", e),
        }
    });
}

fn process_and_encode_jpeg(
    padded_img_data: Vec<u8>,
    width: u32,
    height: u32,
    file: File,
    quality: u8,
) -> ImageResult<()> {
    let mut img_data = strip_padded_img_data(width, height, &padded_img_data, true);
    linear_to_srgb_inplace_with_lut(&mut img_data, false);

    let encoder = JpegEncoder::new_with_quality(file, quality);
    encoder.write_image(&img_data, width, height, ExtendedColorType::Rgb8)
}

fn strip_padded_img_data(
    width: u32,
    height: u32,
    padded_img_data: &[u8],
    strip_alpha: bool,
) -> Vec<u8> {
    let src_pix_size = 4;
    let dst_pix_size = if strip_alpha { 3 } else { 4 };
    let padded_bytes_per_row = ((src_pix_size * width + 255) / 256) * 256;

    let mut img_data = vec![0u8; (width * height * dst_pix_size) as usize];

    for y in 0..height as usize {
        let src_y = height as usize - 1 - y; // flip Y

        let src_offset = src_y * padded_bytes_per_row as usize;
        let dst_offset = y * (width * dst_pix_size) as usize;

        let src_row = &padded_img_data[src_offset..src_offset + (width * src_pix_size) as usize];
        let dst_row = &mut img_data[dst_offset..dst_offset + (width * dst_pix_size) as usize];

        if strip_alpha {
            // RGBA → RGB
            for x in 0..width as usize {
                let si = x * 4;
                let di = x * 3;

                dst_row[di + 0] = src_row[si + 0];
                dst_row[di + 1] = src_row[si + 1];
                dst_row[di + 2] = src_row[si + 2];
            }
        } else {
            // RGBA → RGBA (just copy)
            dst_row.copy_from_slice(src_row);
        }
    }

    img_data
}

fn linear_to_srgb_inplace_with_lut(data: &mut [u8], has_alpha: bool) {
    let pix_size = if has_alpha {4} else {3};
    let lut = build_srgb_lut();

    for px in data.chunks_mut(pix_size) {
        px[0] = lut[px[0] as usize];
        px[1] = lut[px[1] as usize];
        px[2] = lut[px[2] as usize];
    }
}

fn build_srgb_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    for i in 0..256 {
        let v = i as f32 / 255.0;
        let srgb = if v <= 0.0031308 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        lut[i] = (srgb * 255.0) as u8;
    }
    lut
}