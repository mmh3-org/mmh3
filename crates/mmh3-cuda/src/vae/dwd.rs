//! The pi-H3 pixel refiner over the final DiT's modulated video rows.

use super::{
    CudaVideoFrames, NORM_EPSILON, mmh3_vae_norm, mmh3_vae_residual_add_scaled, mmh3_vae_swiglu,
    sol_linear,
};
use crate::attention::AttentionLayout;
use crate::loader::Uploader;
use crate::model::{DeviceTensors, Error, check_quantization, host_tensor, i32_buffer};
use crate::{DeviceBuffer, check};
use mmh3_core::safetensors::{DType, SafeTensors};
use std::collections::BTreeMap;
use std::f32::consts::TAU;
use std::ffi::{c_int, c_void};
use std::ops::Range;
use std::ptr;

const PREFIX: &str = "dwd_pixel_refiner.";
const DIM: usize = 2048;
const HEAD: usize = 64;
const HEADS: usize = DIM / HEAD;
const FFN: usize = DIM * 4;
const LAYERS: usize = 8;
const INPUT: usize = 5376;
const REGISTERS: usize = 4;
const GLOBALS: usize = REGISTERS + 1;
const TIME: usize = 5;
const WINDOW: usize = 16;
const PATCH: usize = 16;
const PATCH_FRAMES: usize = 4;
const PIXELS: usize = PATCH * PATCH * 3;

unsafe extern "C" {
    pub(crate) fn mmh3_dwd_adaln(
        hidden: *const c_void,
        modulation: *const c_void,
        features: *mut c_void,
        rows: c_int,
        width: c_int,
    ) -> c_int;
    fn mmh3_dwd_expand_rows(
        expanded: *const c_void,
        registers: *const c_void,
        residual: *mut c_void,
        height: c_int,
        width: c_int,
        video_rows: c_int,
        rows: c_int,
        dim: c_int,
    ) -> c_int;
    fn mmh3_dwd_norm_rope(
        qkv: *mut c_void,
        angles: *const c_void,
        rows: c_int,
        video_rows: c_int,
        heads: c_int,
    ) -> c_int;
    fn mmh3_dwd_gather(
        qkv: *const c_void,
        indices: *const c_void,
        output: *mut c_void,
        rows: c_int,
        dim: c_int,
        first_operand: c_int,
        operands: c_int,
    ) -> c_int;
    fn mmh3_dwd_scatter(
        packed: *const c_void,
        indices: *const c_void,
        output: *mut c_void,
        rows: c_int,
        dim: c_int,
    ) -> c_int;
    fn mmh3_attention_f16_64_rectangular(
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        output: *mut c_void,
        query_tokens: c_int,
        key_tokens: c_int,
        heads: c_int,
        batch: c_int,
        layout: *const AttentionLayout,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_dwd_pixels(
        projected: *const c_void,
        rgb: *mut c_void,
        frames: c_int,
        output_frames: c_int,
        position: c_int,
        patch_frames: c_int,
        height: c_int,
        width: c_int,
    ) -> c_int;
}

/// Target video rows of the final DiT call, in (t, y, x) order after the DWD AdaLN.
pub struct DwdFeatures {
    pub(crate) values: DeviceBuffer,
    pub(crate) grid: [usize; 3],
}

#[derive(Default)]
struct Indices {
    queries: Vec<usize>,
    keys: Vec<usize>,
}

type Windows = BTreeMap<(usize, usize), Indices>;

struct Grid {
    time: usize,
    height: usize,
    width: usize,
}

impl Grid {
    fn plane(&self) -> usize {
        self.height * self.width
    }

    fn video_rows(&self) -> usize {
        self.time * self.plane()
    }

    fn rows(&self) -> usize {
        self.video_rows() + self.time.div_ceil(TIME) * GLOBALS
    }

    fn global(&self, block: usize) -> Range<usize> {
        let start = self.video_rows() + block * GLOBALS;
        start..start + GLOBALS
    }

    /// Window-local positions in `[-1, 1]` along each of the three axes.
    fn angles(&self, shifted: bool) -> Vec<f32> {
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
    fn windows(&self, shifted: bool) -> Windows {
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

    fn output_frames(&self) -> usize {
        PATCH_FRAMES * self.time - (PATCH_FRAMES - 1) * self.time.div_ceil(TIME)
    }
}

fn shifts(shifted: bool) -> (usize, usize) {
    if shifted { (2, WINDOW / 2) } else { (0, 0) }
}

struct Batch {
    query: DeviceBuffer,
    key: DeviceBuffer,
    queries: usize,
    keys: usize,
    windows: usize,
}

struct Attention {
    angles: DeviceBuffer,
    batches: Vec<Batch>,
}

impl Attention {
    fn new(grid: &Grid, shifted: bool) -> Result<Self, Error> {
        let mut batches = Vec::new();
        for ((queries, keys), indices) in grid.windows(shifted) {
            // CUDA's z dimension holds up to 65,535 windows per launch.
            for (q, k) in indices
                .queries
                .chunks(queries * 65_535)
                .zip(indices.keys.chunks(keys * 65_535))
            {
                batches.push(Batch {
                    query: i32_buffer(q)?,
                    key: i32_buffer(k)?,
                    queries,
                    keys,
                    windows: q.len() / queries,
                });
            }
        }
        Ok(Self {
            angles: DeviceBuffer::from_f32(&grid.angles(shifted))?,
            batches,
        })
    }

    fn forward(&self, grid: &Grid, work: &Workspace) -> Result<(), Error> {
        // SAFETY: the qkv projection has every row of the grid and angles cover its video rows.
        check(unsafe {
            mmh3_dwd_norm_rope(
                work.qkv.pointer(),
                self.angles.pointer(),
                grid.rows() as c_int,
                grid.video_rows() as c_int,
                HEADS as c_int,
            )
        })?;
        for batch in &self.batches {
            let q_rows = batch.queries * batch.windows;
            let k_rows = batch.keys * batch.windows;
            let layout = AttentionLayout {
                token_stride: [DIM as i64; 4],
                head_stride: [HEAD as i64; 4],
                batch_stride: [
                    (batch.queries * DIM) as i64,
                    (batch.keys * DIM) as i64,
                    (batch.keys * DIM) as i64,
                    (batch.queries * DIM) as i64,
                ],
                ..Default::default()
            };
            // SAFETY: each row index is in the grid, and the scratch holds the largest batch.
            unsafe {
                check(mmh3_dwd_gather(
                    work.qkv.pointer(),
                    batch.query.pointer(),
                    work.query.pointer(),
                    q_rows as c_int,
                    DIM as c_int,
                    0,
                    1,
                ))?;
                check(mmh3_dwd_gather(
                    work.qkv.pointer(),
                    batch.key.pointer(),
                    work.key_value.pointer(),
                    k_rows as c_int,
                    DIM as c_int,
                    1,
                    2,
                ))?;
                check(mmh3_attention_f16_64_rectangular(
                    work.query.pointer(),
                    work.key_value.pointer(),
                    work.key_value.pointer_at(k_rows * DIM * 2),
                    work.attended.pointer(),
                    batch.queries as c_int,
                    batch.keys as c_int,
                    HEADS as c_int,
                    batch.windows as c_int,
                    &layout,
                    ptr::null_mut(),
                ))?;
                check(mmh3_dwd_scatter(
                    work.attended.pointer(),
                    batch.query.pointer(),
                    work.attention.pointer(),
                    q_rows as c_int,
                    DIM as c_int,
                ))?;
            }
        }
        Ok(())
    }
}

struct Workspace {
    residual: DeviceBuffer,
    normalized: DeviceBuffer,
    qkv: DeviceBuffer,
    attention: DeviceBuffer,
    query: DeviceBuffer,
    key_value: DeviceBuffer,
    attended: DeviceBuffer,
    delta: DeviceBuffer,
    expanded: DeviceBuffer,
    activated: DeviceBuffer,
    quantized: DeviceBuffer,
    scales: DeviceBuffer,
}

impl Workspace {
    fn new(
        grid: &Grid,
        attention: &[Attention],
        quantized: bool,
        dense_ffn: bool,
    ) -> Result<Self, Error> {
        let rows = grid.rows();
        let batches = || attention.iter().flat_map(|plan| &plan.batches);
        let max_queries = batches()
            .map(|batch| batch.queries * batch.windows)
            .max()
            .unwrap();
        let max_keys = batches()
            .map(|batch| batch.keys * batch.windows)
            .max()
            .unwrap();
        Ok(Self {
            residual: DeviceBuffer::new(rows * DIM * 4)?,
            normalized: DeviceBuffer::new(rows * DIM * 2)?,
            qkv: DeviceBuffer::new(rows * 3 * DIM * 2)?,
            attention: DeviceBuffer::new(rows * DIM * 2)?,
            query: DeviceBuffer::new(max_queries * DIM * 2)?,
            key_value: DeviceBuffer::new(max_keys * 2 * DIM * 2)?,
            attended: DeviceBuffer::new(max_queries * DIM * 2)?,
            delta: DeviceBuffer::new(rows * DIM * 2)?,
            expanded: DeviceBuffer::new(if dense_ffn { rows * 2 * FFN * 2 } else { 1 })?,
            activated: DeviceBuffer::new(rows * FFN * 2)?,
            quantized: DeviceBuffer::new(if quantized { rows * FFN } else { 1 })?,
            scales: DeviceBuffer::new(if quantized { rows * 4 } else { 1 })?,
        })
    }
}

/// The eight Sol decoder blocks and their two pixel projections from pi-H3 DWD.
pub struct CudaDwdDecoder {
    tensors: DeviceTensors,
    quantized: bool,
    dense_ffn: bool,
}

impl CudaDwdDecoder {
    /// Checks the architecture and quantization from a converted adapter before sampling.
    pub fn validate(file: &SafeTensors) -> Result<(), Error> {
        let check = |name: &str, shape: &[usize], dtypes: &[DType]| {
            let name = format!("{PREFIX}{name}");
            file.get(&name)
                .filter(|info| info.shape == shape && dtypes.contains(&info.dtype))
                .map(|info| info.dtype)
                .ok_or_else(|| Error::Model(format!("missing or incompatible tensor {name}")))
        };
        let half = &[DType::F16];
        let linear = |name: &str, outputs, inputs, allow_int8| -> Result<(), Error> {
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
                    .ok_or_else(|| Error::Model(format!("missing tensor {layout}")))?;
                check_quantization(&layout, file.data(info))?;
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

    pub fn load(file: &SafeTensors) -> Result<Self, Error> {
        Self::validate(file)?;
        let mut tensors = DeviceTensors::default();
        let mut uploader = Uploader::new(file);
        let mut quantized = false;
        for info in file.tensors() {
            let Some(name) = info.name.strip_prefix(PREFIX) else {
                continue;
            };
            if name.ends_with(".comfy_quant") {
                continue;
            }
            let int8_bias = name
                .strip_suffix(".bias")
                .and_then(|layer| file.get(&format!("{PREFIX}{layer}.weight")))
                .is_some_and(|weight| weight.dtype == DType::I8);
            if int8_bias {
                let bias = host_tensor(file, &info.name)?.data;
                tensors.insert_buffer(
                    name,
                    DeviceBuffer::from_f32(&bias)?,
                    DType::F32,
                    info.shape.clone(),
                );
            } else {
                tensors.insert(name, file, info, &mut uploader)?;
            }
            quantized |= info.dtype == DType::I8;
        }
        uploader.run()?;
        let mut dense_ffn = false;
        for layer in 0..LAYERS {
            let name = format!("blocks.{layer}.ff.w1");
            if tensors.is_int8(&name) {
                for suffix in ["weight", "weight_scale", "bias"] {
                    tensors.interleave_swiglu(&format!("{name}.{suffix}"))?;
                }
            } else {
                dense_ffn = true;
            }
        }
        Ok(Self {
            tensors,
            quantized,
            dense_ffn,
        })
    }

    fn normalize(
        &self,
        name: &str,
        bias: Option<&str>,
        work: &Workspace,
        rows: usize,
    ) -> Result<(), Error> {
        let bias = bias
            .map(|name| self.tensors.pointer(name))
            .transpose()?
            .unwrap_or(ptr::null());
        // SAFETY: the workspace has at least these rows, and the norm has DIM weights and biases.
        check(unsafe {
            mmh3_vae_norm(
                work.residual.pointer(),
                self.tensors.pointer(name)?,
                bias,
                work.normalized.pointer(),
                rows as c_int,
                DIM as c_int,
                NORM_EPSILON,
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }

    fn add_scaled(&self, name: &str, work: &Workspace, rows: usize) -> Result<(), Error> {
        // SAFETY: the workspace has these rows and the scale holds DIM values.
        check(unsafe {
            mmh3_vae_residual_add_scaled(
                work.residual.pointer(),
                work.delta.pointer(),
                self.tensors.pointer(name)?,
                rows as c_int,
                DIM as c_int,
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }

    /// Expands the final features, refines the shifted windows and returns planar RGB on CUDA.
    pub fn decode(&self, features: &DwdFeatures) -> Result<CudaVideoFrames, Error> {
        let [time, height, width] = features.grid;
        let grid = Grid {
            time,
            height: height * 2,
            width: width * 2,
        };
        let rows = grid.rows();
        if grid.video_rows() == 0
            || rows > i32::MAX as usize / (2 * HEADS)
            || features.values.bytes() != grid.video_rows() / 4 * INPUT * 2
        {
            return Err(Error::Model("invalid DWD feature grid".into()));
        }
        let attention = [Attention::new(&grid, false)?, Attention::new(&grid, true)?];
        let work = Workspace::new(&grid, &attention, self.quantized, self.dense_ffn)?;
        let linear = |name: &str, input: &DeviceBuffer, output: &DeviceBuffer, count, swiglu| {
            sol_linear(
                &self.tensors,
                name,
                input,
                output,
                count,
                (&work.quantized, &work.scales),
                swiglu,
                false,
            )
        };
        linear(
            "feature_expand",
            &features.values,
            &work.normalized,
            grid.video_rows() / 4,
            false,
        )?;
        // SAFETY: feature expansion wrote the video rows and the grid includes the global rows.
        check(unsafe {
            mmh3_dwd_expand_rows(
                work.normalized.pointer(),
                self.tensors.pointer("register_tokens")?,
                work.residual.pointer(),
                grid.height as c_int,
                grid.width as c_int,
                grid.video_rows() as c_int,
                rows as c_int,
                DIM as c_int,
            )
        })?;
        for index in 0..LAYERS {
            let block = format!("blocks.{index}");
            self.normalize(&format!("{block}.norm1.weight"), None, &work, rows)?;
            linear(
                &format!("{block}.attn.to_qkv"),
                &work.normalized,
                &work.qkv,
                rows,
                false,
            )?;
            attention[index % 2].forward(&grid, &work)?;
            linear(
                &format!("{block}.attn.to_out"),
                &work.attention,
                &work.delta,
                rows,
                false,
            )?;
            self.add_scaled(&format!("{block}.scale1"), &work, rows)?;

            self.normalize(&format!("{block}.norm2.weight"), None, &work, rows)?;
            let w1 = format!("{block}.ff.w1");
            if self.tensors.is_int8(&w1) {
                linear(&w1, &work.normalized, &work.activated, rows, true)?;
            } else {
                linear(&w1, &work.normalized, &work.expanded, rows, false)?;
                // SAFETY: the dense expansion holds rows × 2 × FFN and activations rows × FFN.
                check(unsafe {
                    mmh3_vae_swiglu(
                        work.expanded.pointer(),
                        work.activated.pointer(),
                        rows as c_int,
                        FFN as c_int,
                        ptr::null_mut(),
                    )
                })?;
            }
            linear(
                &format!("{block}.ff.w2"),
                &work.activated,
                &work.delta,
                rows,
                false,
            )?;
            self.add_scaled(&format!("{block}.scale2"), &work, rows)?;
        }
        self.normalize(
            "norm_out.weight",
            Some("norm_out.bias"),
            &work,
            grid.video_rows(),
        )?;
        let output_frames = grid.output_frames();
        let (height, width) = (grid.height * PATCH, grid.width * PATCH);
        let rgb = DeviceBuffer::new(3 * output_frames * height * width * 4)?;
        let projection = DeviceBuffer::new(grid.plane() * PIXELS * PATCH_FRAMES * (TIME - 1) * 2)?;
        let mut position = 0;
        for block in (0..grid.time).step_by(TIME) {
            let tail = (grid.time - block).min(TIME) - 1;
            for (offset, count, patch_frames, name) in [
                (0, 1, 1, "first_frame_proj_out"),
                (1, tail, PATCH_FRAMES, "proj_out"),
            ] {
                if count == 0 {
                    continue;
                }
                let first = (block + offset) * grid.plane();
                let rows = count * grid.plane();
                // SAFETY: the layer takes only these normalized video rows and the view stays local.
                let input = unsafe {
                    DeviceBuffer::view(&work.normalized, first * DIM * 2, rows * DIM * 2)
                };
                linear(name, &input, &projection, rows, false)?;
                let frames = count * patch_frames;
                // SAFETY: the projection covers these frames and their interval lies in RGB.
                check(unsafe {
                    mmh3_dwd_pixels(
                        projection.pointer(),
                        rgb.pointer(),
                        frames as c_int,
                        output_frames as c_int,
                        position as c_int,
                        patch_frames as c_int,
                        height as c_int,
                        width as c_int,
                    )
                })?;
                position += frames;
            }
        }
        debug_assert_eq!(position, output_frames);
        CudaVideoFrames::from_rgb(rgb, output_frames, height, width)
    }
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

    #[test]
    #[ignore = "requires published DWD weights and reference features/RGB"]
    fn matches_reference_pixels() {
        let weights = std::env::var("MMH3_DWD_WEIGHTS").unwrap();
        let reference = std::env::var("MMH3_DWD_REFERENCE").unwrap();
        let weights = SafeTensors::open(std::path::Path::new(&weights)).unwrap();
        let reference = SafeTensors::open(std::path::Path::new(&reference)).unwrap();
        let features = host_tensor(&reference, "features").unwrap();
        let expected = host_tensor(&reference, "rgb").unwrap();
        let grid = features.shape[..3].try_into().unwrap();
        let features = DwdFeatures {
            values: super::super::f16_buffer(&features.data).unwrap(),
            grid,
        };
        let decoder = CudaDwdDecoder::load(&weights).unwrap();
        let decoded = decoder.decode(&features).unwrap();
        assert_eq!(
            expected.shape,
            [3, decoded.frames(), decoded.height(), decoded.width()]
        );
        let decoded = decoded.pixels.to_f32().unwrap();
        let mse = decoded
            .iter()
            .zip(&expected.data)
            .map(|(actual, expected)| f64::from(actual - expected).powi(2))
            .sum::<f64>()
            / decoded.len() as f64;
        let psnr = -10.0 * mse.log10();
        println!("DWD RGB PSNR: {psnr:.2} dB");
        assert!(psnr > 45.0, "DWD RGB differs from the reference");
    }
}
