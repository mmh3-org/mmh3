//! The pi-H3 pixel refiner over the final DiT's modulated video rows.

use super::MetalVideoFrames;
use crate::{
    AttentionPrecision, Device, Error, LinearPrecision, Result,
    model::Weights,
    ops::{Array, Packing, RowMap, attends_in_half},
};
use mmh3_core::{
    dwd::{
        self, DIM, GLOBALS, Grid, HEAD, HEADS, INPUT, LAYERS, PATCH, PATCH_FRAMES, PREFIX,
        REGISTERS, TIME,
    },
    safetensors::SafeTensors,
};

const NORM_EPSILON: f32 = 1e-5;

/// Target video rows of the final DiT call, in (t, y, x) order after the DWD AdaLN.
pub struct DwdFeatures {
    pub(crate) values: Array,
    pub(crate) grid: [usize; 3],
}

struct WindowBatch {
    queries: usize,
    keys: usize,
    query: RowMap,
    key: RowMap,
}

struct Attention {
    angles: Array,
    batches: Vec<WindowBatch>,
}

impl Attention {
    fn new(device: &Device, grid: &Grid, shifted: bool) -> Result<Self> {
        let mut angles = grid.angles(shifted);
        let pairs = angles.len() / grid.video_rows();
        // The registers and class token have no rotary embedding.
        angles.resize(grid.rows() * pairs, 0.0);
        let batches = grid
            .windows(shifted)
            .into_iter()
            .map(|((queries, keys), indices)| {
                Ok(WindowBatch {
                    queries,
                    keys,
                    query: RowMap::new(device, &indices.queries)?,
                    key: RowMap::new(device, &indices.keys)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            angles: Array::from_f32(device, grid.rows(), pairs, &angles)?,
            batches,
        })
    }

    fn forward(&self, qkv: &Array, norm: &Array, half: bool) -> Result<Array> {
        let inputs = qkv.attention_inputs(
            HEADS,
            (norm, norm),
            NORM_EPSILON,
            Some(&self.angles),
            half,
            true,
            false,
        )?;
        let output = Array::empty(qkv.device(), qkv.rows, DIM)?;
        for batch in &self.batches {
            inputs.attend_windows(
                (batch.queries, &batch.query),
                (batch.keys, &batch.key),
                &output,
            )?;
        }
        Ok(output)
    }
}

/// The eight Sol decoder blocks and their pixel projections from the converted DWD patch.
pub struct MetalDwdDecoder {
    weights: Weights,
}

impl MetalDwdDecoder {
    pub fn validate(file: &SafeTensors) -> Result<()> {
        dwd::validate(file).map_err(Error::new)
    }

    pub fn load(
        file: &SafeTensors,
        linear: LinearPrecision,
        attention: AttentionPrecision,
    ) -> Result<Self> {
        Self::validate(file)?;
        let mut weights = Weights::load(file, PREFIX)?;
        weights.set_precision((linear, attention))?;
        if linear == LinearPrecision::Int8 {
            for layer in 0..LAYERS {
                let name = format!("blocks.{layer}.ff.w1");
                if weights.is_int8(&name) {
                    weights.interleave_swiglu(&name)?;
                }
            }
        }
        Ok(Self { weights })
    }

    fn linear(&self, input: &Array, name: &str) -> Result<Array> {
        let w = &self.weights;
        if matches!(
            w.linear_precision,
            LinearPrecision::Int8 | LinearPrecision::Fp16
        ) && w.is_half(name)
        {
            return w.linear_fp16(input, name);
        }
        w.linear(input, name)
    }

    /// Expands and refines the final DiT features and returns planar RGB on the GPU.
    pub fn decode(&self, features: &DwdFeatures) -> Result<MetalVideoFrames> {
        let [time, height, width] = features.grid;
        let grid = Grid {
            time,
            height: height * 2,
            width: width * 2,
        };
        if grid.video_rows() == 0 || features.values.shape() != [grid.video_rows() / 4, INPUT] {
            return Err(Error::new("invalid DWD feature grid".into()));
        }
        let w = &self.weights;
        let device = &w.device;
        let attention = [
            Attention::new(device, &grid, false)?,
            Attention::new(device, &grid, true)?,
        ];
        let half = attends_in_half(device, w.attention_precision, HEAD);
        let norm = Array::from_f32(device, 1, HEAD, &[1.0; HEAD])?;
        let mut hidden = {
            let expanded = self.linear(&features.values, "feature_expand")?;
            let registers = w.vector("register_tokens")?;
            let hidden = Array::empty(device, grid.rows(), DIM)?;
            device.run(
                "dwd_expand_rows",
                &[&expanded.buffer, &registers.buffer, &hidden.buffer],
                &[
                    grid.height as u32,
                    grid.width as u32,
                    grid.video_rows() as u32,
                    grid.rows() as u32,
                    DIM as u32,
                    GLOBALS as u32,
                    REGISTERS as u32,
                ],
                hidden.len(),
                false,
            )?;
            hidden
        };
        for index in 0..LAYERS {
            let block = format!("blocks.{index}");
            {
                let normalized = w.norm(&hidden, &format!("{block}.norm1"), NORM_EPSILON)?;
                let qkv = self.linear(&normalized, &format!("{block}.attn.to_qkv"))?;
                let attended = attention[index % 2].forward(&qkv, &norm, half)?;
                let delta = self.linear(&attended, &format!("{block}.attn.to_out"))?;
                hidden = hidden.add_scaled(&delta, &w.vector(&format!("{block}.scale1"))?)?;
            }

            let normalized = w.norm(&hidden, &format!("{block}.norm2"), NORM_EPSILON)?;
            let w1 = format!("{block}.ff.w1");
            let activated = if w.linear_precision == LinearPrecision::Int8 && w.is_int8(&w1) {
                let packed = normalized.pack(Packing::Rotate, w.linear_precision)?;
                w.linear_swiglu_prepacked(&packed, None, &w1)?
            } else {
                self.linear(&normalized, &w1)?.swiglu()?
            };
            let delta = self.linear(&activated, &format!("{block}.ff.w2"))?;
            hidden = hidden.add_scaled(&delta, &w.vector(&format!("{block}.scale2"))?)?;
        }

        let video = hidden.slice(0, grid.video_rows(), 0, DIM)?;
        let normalized = w.norm(&video, "norm_out", NORM_EPSILON)?;
        let frames = grid.output_frames();
        let (height, width) = (grid.height * PATCH, grid.width * PATCH);
        let pixels = Array::empty(device, 3, frames * height * width)?;
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
                let rows = normalized.slice(
                    (block + offset) * grid.plane(),
                    count * grid.plane(),
                    0,
                    DIM,
                )?;
                let projected = self.linear(&rows, name)?;
                let values = count * patch_frames * height * width * 3;
                device.run(
                    "dwd_pixels",
                    &[&projected.buffer, &pixels.buffer],
                    &[
                        values as u32,
                        frames as u32,
                        position as u32,
                        patch_frames as u32,
                        height as u32,
                        width as u32,
                        PATCH as u32,
                    ],
                    values,
                    false,
                )?;
                position += count * patch_frames;
            }
        }
        debug_assert_eq!(position, frames);
        Ok(MetalVideoFrames {
            start: 0,
            pixels,
            frames,
            height,
            width,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmh3_core::tensor::Tensor;
    use std::path::Path;

    #[test]
    #[ignore = "requires published DWD weights and reference features/RGB"]
    fn matches_reference_pixels() {
        let weights = std::env::var("MMH3_DWD_WEIGHTS").unwrap();
        let reference = std::env::var("MMH3_DWD_REFERENCE").unwrap();
        let weights = SafeTensors::open(Path::new(&weights)).unwrap();
        let reference = SafeTensors::open(Path::new(&reference)).unwrap();
        let load = |name| Tensor::load(&reference, reference.get(name).unwrap()).unwrap();
        let features = load("features");
        let expected = load("rgb");
        let grid = features.shape[..3].try_into().unwrap();
        let device = Device::shared().unwrap();
        let values =
            Array::from_f32(&device, features.data.len() / INPUT, INPUT, &features.data).unwrap();
        let (linear, attention) = if device.supports_tensor_ops() {
            (LinearPrecision::Int8, AttentionPrecision::Fp16)
        } else {
            (LinearPrecision::MpsFp16, AttentionPrecision::Fp32)
        };
        let decoder = MetalDwdDecoder::load(&weights, linear, attention).unwrap();
        let decoded = decoder
            .decode(&DwdFeatures { values, grid })
            .unwrap()
            .to_pixels()
            .unwrap();
        assert_eq!(expected.shape, decoded.shape);
        let mse = decoded
            .data
            .iter()
            .zip(&expected.data)
            .map(|(actual, expected)| f64::from(actual - expected).powi(2))
            .sum::<f64>()
            / decoded.data.len() as f64;
        let psnr = -10.0 * mse.log10();
        println!("DWD RGB PSNR: {psnr:.2} dB");
        assert!(psnr > 45.0, "DWD RGB differs from the reference");
    }
}
