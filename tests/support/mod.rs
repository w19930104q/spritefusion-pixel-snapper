//! Deterministic synthetic test images, mirroring the Python prototype in
//! tools/proto_score.py. All randomness uses a fixed-seed ChaCha8 RNG so the
//! generated PNG bytes are reproducible across machines and runs.

use image::{ImageBuffer, Rgba, RgbaImage};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone, Copy)]
pub enum GridMode {
    Sharp,
    Blurred,
    LowContrast,
    TransparentEdges,
    Diagonal45,
    Noisy,
}

#[derive(Debug, Clone, Copy)]
pub enum NoGridKind {
    Smooth,
    Noise,
}

fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Build a synthetic sprite laid out on a square grid of `pitch` input pixels.
pub fn synthetic_grid(
    cells_x: usize,
    cells_y: usize,
    pitch: usize,
    mode: GridMode,
    seed: u64,
) -> RgbaImage {
    let mut rng = rng(seed);
    let w = cells_x * pitch;
    let h = cells_y * pitch;
    let mut img: RgbaImage = ImageBuffer::new(w as u32, h as u32);

    // Per-cell base colors, chosen according to the degradation mode.
    let mut colors = vec![[0u8; 3]; cells_x * cells_y];
    for j in 0..cells_y {
        for i in 0..cells_x {
            colors[j * cells_x + i] = match mode {
                GridMode::LowContrast => [
                    (90 + rng.gen_range(0..30)).clamp(0, 255),
                    (100u8 as i32 + rng.gen_range(-12..13)).clamp(0, 255) as u8,
                    (110u8 as i32 + rng.gen_range(-12..13)).clamp(0, 255) as u8,
                ],
                GridMode::Diagonal45 => {
                    if (i + j) % 2 == 0 {
                        [40, 40, 90]
                    } else {
                        [220, 220, 200]
                    }
                }
                _ => [rng.gen(), rng.gen(), rng.gen()],
            };
        }
    }

    for j in 0..cells_y {
        for i in 0..cells_x {
            // Transparent-edge mode: drop some cells entirely.
            if matches!(mode, GridMode::TransparentEdges) && rng.gen_bool(0.4) {
                continue;
            }
            let color = colors[j * cells_x + i];
            for y in j * pitch..(j + 1) * pitch {
                for x in i * pitch..(i + 1) * pitch {
                    img.put_pixel(
                        x as u32,
                        y as u32,
                        Rgba([color[0], color[1], color[2], 255]),
                    );
                }
            }
            // Semi-transparent fringe on some opaque cells.
            if matches!(mode, GridMode::TransparentEdges) && rng.gen_bool(0.3) {
                let x0 = i * pitch;
                for x in x0..x0 + pitch {
                    let p = img.get_pixel_mut(x as u32, (j * pitch) as u32);
                    if p[3] != 0 {
                        p[3] = 128;
                    }
                }
            }
        }
    }

    if matches!(mode, GridMode::Noisy) {
        // Salt noise inside cells; grid edges stay intact.
        for y in 0..h {
            for x in 0..w {
                if rng.gen_bool(0.08) {
                    let p = img.get_pixel_mut(x as u32, y as u32);
                    p[0] = rng.gen();
                    p[1] = rng.gen();
                    p[2] = rng.gen();
                    p[3] = 255;
                }
            }
        }
    }

    if matches!(mode, GridMode::Blurred) {
        // Three passes of a 3x3 box blur, pure-RGB including alpha.
        let mut work = img.clone();
        for _ in 0..3 {
            let src = work.clone();
            for y in 0..h as i32 {
                for x in 0..w as i32 {
                    let mut sums = [0u32; 4];
                    let mut count = 0u32;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let sx = (x + dx).clamp(0, w as i32 - 1) as u32;
                            let sy = (y + dy).clamp(0, h as i32 - 1) as u32;
                            let p = src.get_pixel(sx, sy);
                            for c in 0..4 {
                                sums[c] += p[c] as u32;
                            }
                            count += 1;
                        }
                    }
                    let out = work.get_pixel_mut(x as u32, y as u32);
                    for c in 0..4 {
                        out[c] = ((sums[c] + count / 2) / count) as u8;
                    }
                }
            }
        }
        img = work;
    }

    img
}

/// Build an image with no periodic grid structure.
pub fn synthetic_no_grid(size: usize, kind: NoGridKind) -> RgbaImage {
    let mut rng = rng(7);
    let mut img: RgbaImage = ImageBuffer::new(size as u32, size as u32);
    match kind {
        NoGridKind::Smooth => {
            // Horizontal gradient plus one soft circle; no periodic edges.
            let center = size as i32 / 2;
            let radius = size as i32 * 3 / 8;
            for y in 0..size as i32 {
                for x in 0..size as i32 {
                    let mut gray = (x * 255 / size as i32) as u8;
                    let dist2 = (x - center).pow(2) + (y - center).pow(2);
                    if dist2 < radius.pow(2) {
                        gray = gray.saturating_sub(80);
                    }
                    img.put_pixel(x as u32, y as u32, Rgba([gray, gray, gray, 255]));
                }
            }
        }
        NoGridKind::Noise => {
            // Independent per-pixel noise: no periodicity.
            for y in 0..size as u32 {
                for x in 0..size as u32 {
                    img.put_pixel(x, y, Rgba([rng.gen(), rng.gen(), rng.gen(), 255]));
                }
            }
        }
    }
    img
}

/// Encode to PNG bytes via the same encoder the pipeline uses.
pub fn png_bytes(img: &RgbaImage) -> Vec<u8> {
    use std::io::Cursor;
    let mut bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("png encode");
    bytes
}
