//! Sampling schedules and the step rules of the rectified flow.

use crate::dit::timestep::time_shift_sigma;

/// Video and audio sigma schedules that share one unshifted grid, ending at zero.
#[derive(Clone, Debug, PartialEq)]
pub struct Schedule {
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
}

impl Schedule {
    /// `steps` model calls on the unshifted grid linspace(1, 0, steps + 1), shifted separately per
    /// stream. This is the official scheduler's grid and the q grid of the Turbo LoRAs.
    pub fn uniform(steps: usize, shift_video: f32, shift_audio: f32) -> Self {
        let indices: Vec<usize> = (0..=steps).collect();
        Self::retained(steps + 1, &indices, shift_video, shift_audio)
    }

    /// The states at `indices` of the shifted `points`-point grid linspace(1, 0, points). The first
    /// index must be 0 and the last `points − 1`, so that sampling starts from noise and ends at
    /// zero.
    pub fn retained(points: usize, indices: &[usize], shift_video: f32, shift_audio: f32) -> Self {
        assert!(
            indices.len() >= 2
                && indices[0] == 0
                && indices[indices.len() - 1] == points - 1
                && indices.windows(2).all(|pair| pair[0] < pair[1]),
            "the retained states must rise from 0 to {}",
            points - 1
        );
        let base: Vec<f32> = indices
            .iter()
            .map(|&index| 1.0 - index as f32 / (points - 1) as f32)
            .collect();
        Schedule {
            video: base
                .iter()
                .map(|&value| time_shift_sigma(value, 1.0, shift_video))
                .collect(),
            audio: base
                .iter()
                .map(|&value| time_shift_sigma(value, 1.0, shift_audio))
                .collect(),
        }
    }

    /// The three steps TaoMate-H3's LoRA was distilled for: states 0, 16, 33 and 49 of the
    /// 50-point grid.
    pub fn taomate(shift_video: f32, shift_audio: f32) -> Self {
        Self::retained(50, &[0, 16, 33, 49], shift_video, shift_audio)
    }

    /// The pi-H3 grid starts below sigma one and uses a shorter final interval before shifting.
    pub fn pi_h3(steps: usize, shift_video: f32, shift_audio: f32) -> Option<Self> {
        let max_raw_sigma = match steps {
            2 => 0.93,
            4 => 0.96,
            _ => return None,
        };
        let stride = max_raw_sigma / (steps as f32 - 1.0 + 0.6);
        let base = (0..steps)
            .map(|index| max_raw_sigma - index as f32 * stride)
            .chain(std::iter::once(0.0));
        let shifted = |shift| {
            base.clone()
                .map(|sigma| time_shift_sigma(sigma, 1.0, shift))
                .collect()
        };
        Some(Self {
            video: shifted(shift_video),
            audio: shifted(shift_audio),
        })
    }

    pub fn steps(&self) -> usize {
        self.video.len() - 1
    }
}

/// How a step moves the latent from one sigma to the next.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sampler {
    /// The Euler step of the flow, the rule of the official pipeline.
    Euler,
    /// The multistep rule of DMD-style students: the clean latent the step predicts, noised again
    /// with fresh noise to the next sigma.
    Renoise,
}

/// One Euler step: x ← x + (sigma_next − sigma) · v, where v is the flow velocity
/// (x0 = x − sigma · v).
pub fn euler_step(latent: &mut [f32], velocity: &[f32], sigma: f32, sigma_next: f32) {
    let delta = sigma_next - sigma;
    for (value, &slope) in latent.iter_mut().zip(velocity) {
        *value += delta * slope;
    }
}

/// One re-noise step: x ← (1 − sigma_next) · x0 + sigma_next · noise, where x0 = x − sigma · v.
pub fn renoise_step(
    latent: &mut [f32],
    velocity: &[f32],
    noise: &[f32],
    sigma: f32,
    sigma_next: f32,
) {
    for ((value, &slope), &fresh) in latent.iter_mut().zip(velocity).zip(noise) {
        let clean = *value - sigma * slope;
        *value = (1.0 - sigma_next) * clean + sigma_next * fresh;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_turbo_grid() {
        let schedule = Schedule::uniform(4, 12.0, 3.0);
        let expected_video = [1.0, 0.9730, 0.9231, 0.8000, 0.0];
        let expected_audio = [1.0, 0.9000, 0.7500, 0.5000, 0.0];
        for (actual, expected) in schedule.video.iter().zip(expected_video) {
            assert!((actual - expected).abs() < 1e-4, "{:?}", schedule.video);
        }
        for (actual, expected) in schedule.audio.iter().zip(expected_audio) {
            assert!((actual - expected).abs() < 1e-4, "{:?}", schedule.audio);
        }
    }

    #[test]
    fn matches_the_taomate_states() {
        let schedule = Schedule::taomate(12.0, 3.0);
        let expected_video = [1.0, 0.961165, 0.853333, 0.0];
        let expected_audio = [1.0, 0.860870, 0.592593, 0.0];
        for (actual, expected) in schedule.video.iter().zip(expected_video) {
            assert!((actual - expected).abs() < 1e-5, "{:?}", schedule.video);
        }
        for (actual, expected) in schedule.audio.iter().zip(expected_audio) {
            assert!((actual - expected).abs() < 1e-5, "{:?}", schedule.audio);
        }
    }

    #[test]
    fn matches_the_pi_h3_schedules() {
        for (video, audio) in [
            (
                &[0.993767, 0.865340, 0.0][..],
                &[0.975524, 0.616348, 0.0][..],
            ),
            (
                &[0.996540, 0.964451, 0.899297, 0.695652, 0.0][..],
                &[0.986301, 0.871508, 0.690647, 0.363636, 0.0][..],
            ),
        ] {
            let schedule = Schedule::pi_h3(video.len() - 1, 12.0, 3.0).unwrap();
            for (actual, expected) in [(&schedule.video, video), (&schedule.audio, audio)] {
                assert_eq!(actual.len(), expected.len());
                for (&actual, &expected) in actual.iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
                }
            }
        }
        assert!(Schedule::pi_h3(3, 12.0, 3.0).is_none());
    }

    #[test]
    fn lands_on_the_clean_latent() {
        let (clean, noise) = (2.0f32, -1.0f32);
        let mut latent = [noise];
        let schedule = Schedule::uniform(3, 12.0, 3.0);
        for step in 0..schedule.steps() {
            let (sigma, next) = (schedule.video[step], schedule.video[step + 1]);
            euler_step(&mut latent, &[noise - clean], sigma, next);
        }
        assert!((latent[0] - clean).abs() < 1e-6);
    }

    #[test]
    fn renoises_the_predicted_clean_latent() {
        let (clean, noise, fresh) = (2.0f32, -1.0f32, 0.5f32);
        let mut latent = [0.25 * noise + 0.75 * clean];
        renoise_step(&mut latent, &[noise - clean], &[fresh], 0.25, 0.1);
        assert!((latent[0] - (0.9 * clean + 0.1 * fresh)).abs() < 1e-6);
        renoise_step(&mut latent, &[fresh - clean], &[fresh], 0.1, 0.0);
        assert!((latent[0] - clean).abs() < 1e-6);
    }
}
