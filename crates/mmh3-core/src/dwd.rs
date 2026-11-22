//! Model geometry and checkpoint layout of the pi-H3 DWD pixel refiner.

use crate::safetensors::{DType, SafeTensors, check_int8_convrot};
use std::{collections::BTreeMap, f32::consts::TAU, ops::Range};

pub const PREFIX: &str = "dwd_pixel_refiner.";
pub const DIM: usize = 2048;
pub const HEAD: usize = 64;
pub const HEADS: usize = DIM / HEAD;
pub const FFN: usize = DIM * 4;
pub const LAYERS: usize = 8;
pub const INPUT: usize = 5376;
pub const REGISTERS: usize = 4;
pub const GLOBALS: usize = REGISTERS + 1;
pub const TIME: usize = 5;
const WINDOW: usize = 16;
pub const PATCH: usize = 16;
pub const PATCH_FRAMES: usize = 4;
pub const PIXELS: usize = PATCH * PATCH * 3;

#[derive(Default)]
pub struct Indices {
    pub queries: Vec<usize>,
    pub keys: Vec<usize>,
}

pub type Windows = BTreeMap<(usize, usize), Indices>;

pub struct Grid {
    pub time: usize,
    pub height: usize,
    pub width: usize,
}

impl Grid {
    pub fn plane(&self) -> usize {
        self.height * self.width
    }

    pub fn video_rows(&self) -> usize {
        self.time * self.plane()
    }

    pub fn rows(&self) -> usize {
        self.video_rows() + self.time.div_ceil(TIME) * GLOBALS
    }

    fn global(&self, block: usize) -> Range<usize> {
        let start = self.video_rows() + block * GLOBALS;
        start..start + GLOBALS
    }

    /// Window-local positions in `[-1, 1]` along each of the three axes.
    pub fn angles(&self, shifted: bool) -> Vec<f32> {
        let (st, s) = shifts(shifted);
        let frequencies: [f32; 8] = std::array::from_fn(|i| TAU / 100.0f32.powf(i as f32 / 8.0));
        let coordinate = |index: usize, shift, size| {
            (((index + shift) % size) as f32 + 0.5) * 2.0 / size as f32 - 1.0
        };
        let mut angles = Vec::with_capacity(self.video_rows() * 24);
        for t in 0..self.time {
            for y in 0..self.height {
                for x in 0..self.width {
                    for position in [
                        coordinate(t, st, TIME),
                        coordinate(y, s, WINDOW),
                        coordinate(x, s, WINDOW),
                    ] {
                        angles.extend(frequencies.iter().map(|f| f * position));
                    }
                }
            }
        }
        angles
    }

    /// Groups windows with identical lengths for batched attention without padding or masks.
    pub fn windows(&self, shifted: bool) -> Windows {
        let (st, s) = shifts(shifted);
        let extents = |size, shift, limit| {
            (0..usize::div_ceil(limit + shift, size)).map(move |index| {
                (index * size).saturating_sub(shift)..((index + 1) * size - shift).min(limit)
            })
        };
        let mut groups = Windows::new();
        for time in extents(TIME, st, self.time) {
            let blocks = time.start / TIME..=(time.end - 1) / TIME;
            for height in extents(WINDOW, s, self.height) {
                for width in extents(WINDOW, s, self.width) {
                    let queries = time.len() * height.len() * width.len();
                    let keys = queries + blocks.clone().count() * GLOBALS;
                    let indices = groups.entry((queries, keys)).or_default();
                    for t in time.clone() {
                        for y in height.clone() {
                            let start = t * self.plane() + y * self.width;
                            let rows = start + width.start..start + width.end;
                            indices.queries.extend(rows.clone());
                            indices.keys.extend(rows);
                        }
                    }
                    for block in blocks.clone() {
                        indices.keys.extend(self.global(block));
                    }
                }
            }
        }
        for block in 0..self.time.div_ceil(TIME) {
            let start = block * TIME * self.plane();
            let end = ((block + 1) * TIME).min(self.time) * self.plane();
            let indices = groups.entry((GLOBALS, end - start + GLOBALS)).or_default();
            let globals = self.global(block);
            indices.queries.extend(globals.clone());
            indices.keys.extend(start..end);
            indices.keys.extend(globals);
        }
        groups
    }

    pub fn output_frames(&self) -> usize {
        PATCH_FRAMES * self.time - (PATCH_FRAMES - 1) * self.time.div_ceil(TIME)
    }
}

fn shifts(shifted: bool) -> (usize, usize) {
    if shifted { (2, WINDOW / 2) } else { (0, 0) }
}

/// Checks the architecture and quantization from a converted adapter before sampling.
pub fn validate(file: &SafeTensors) -> Result<(), String> {
    let check = |name: &str, shape: &[usize], dtypes: &[DType]| {
        let name = format!("{PREFIX}{name}");
        file.get(&name)
            .filter(|info| info.shape == shape && dtypes.contains(&info.dtype))
            .map(|info| info.dtype)
            .ok_or_else(|| format!("missing or incompatible tensor {name}"))
    };
    let half = &[DType::F16];
    let linear = |name: &str, outputs, inputs, allow_int8| -> Result<(), String> {
        let weight = format!("{name}.weight");
        let dtypes = if allow_int8 {
            &[DType::F16, DType::I8][..]
        } else {
            half
        };
        let dtype = check(&weight, &[outputs, inputs], dtypes)?;
        check(&format!("{name}.bias"), &[outputs], half)?;
        if dtype == DType::I8 {
            check(
                &format!("{name}.weight_scale"),
                &[outputs, 1],
                &[DType::F32],
            )?;
            let layout = format!("{PREFIX}{name}.comfy_quant");
            let info = file
                .get(&layout)
                .ok_or_else(|| format!("missing tensor {layout}"))?;
            check_int8_convrot(&layout, file.data(info))?;
        }
        Ok(())
    };
    check("register_tokens", &[1, REGISTERS, DIM], half)?;
    check("norm_out.weight", &[DIM], half)?;
    check("norm_out.bias", &[DIM], half)?;
    linear("feature_expand", DIM * 4, INPUT, false)?;
    linear("first_frame_proj_out", PIXELS, DIM, false)?;
    linear("proj_out", PIXELS * PATCH_FRAMES, DIM, false)?;
    for index in 0..LAYERS {
        let block = format!("blocks.{index}");
        for name in ["norm1.weight", "norm2.weight", "scale1", "scale2"] {
            check(&format!("{block}.{name}"), &[DIM], half)?;
        }
        for (name, outputs, inputs) in [
            ("attn.to_qkv", 3 * DIM, DIM),
            ("attn.to_out", DIM, DIM),
            ("ff.w1", 2 * FFN, DIM),
            ("ff.w2", DIM, FFN),
        ] {
            linear(&format!("{block}.{name}"), outputs, inputs, true)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifted_windows_include_both_native_blocks() {
        let grid = Grid {
            time: 7,
            height: 2,
            width: 2,
        };
        let windows = grid.windows(true);
        assert_eq!(
            windows[&(12, 17)].keys,
            (0..12).chain(28..33).collect::<Vec<_>>()
        );
        assert_eq!(
            windows[&(16, 26)].keys,
            (12..28).chain(28..38).collect::<Vec<_>>()
        );
        assert_eq!(
            windows[&(5, 25)].keys,
            (0..20).chain(28..33).collect::<Vec<_>>()
        );
        assert_eq!(
            windows[&(5, 13)].keys,
            (20..28).chain(33..38).collect::<Vec<_>>()
        );
    }

    #[test]
    fn windows_cover_edge_tiles_and_pixel_projection_keeps_partial_chunks() {
        let mut grid = Grid {
            time: 37,
            height: 18,
            width: 20,
        };
        for shifted in [false, true] {
            let mut queries: Vec<_> = grid
                .windows(shifted)
                .into_values()
                .flat_map(|indices| indices.queries)
                .collect();
            queries.sort_unstable();
            assert_eq!(queries, (0..grid.rows()).collect::<Vec<_>>());
        }
        for (time, frames) in [(1, 1), (2, 5), (5, 17), (6, 18), (7, 22), (37, 124)] {
            grid.time = time;
            assert_eq!(grid.output_frames(), frames);
        }
    }
}
