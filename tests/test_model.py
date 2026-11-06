import numpy as np
import torch

from jitai_twin.calibration import build_linear_guidance
from jitai_twin.data import TrajectoryScaler
from jitai_twin.diffusion import DiffusionConfig, GaussianDiffusion
from jitai_twin.model import CausalConditionalDenoiser
from jitai_twin.synthetic import SyntheticConfig, make_synthetic_cohorts


def test_future_actions_do_not_change_the_generated_past():
    torch.manual_seed(2)
    model = CausalConditionalDenoiser(baseline_dim=3, hidden_dim=8, depth=2).eval()
    noisy = torch.randn(2, 20)
    baseline = torch.randn(2, 3)
    timesteps = torch.tensor([2, 4])
    first_actions = torch.zeros(2, 20)
    second_actions = first_actions.clone()
    second_actions[:, 11:] = 1.0
    with torch.no_grad():
        first = model(noisy, first_actions, baseline, timesteps)
        second = model(noisy, second_actions, baseline, timesteps)
    torch.testing.assert_close(first[:, :11], second[:, :11], rtol=0, atol=1e-6)


def test_diffusion_loss_sampling_and_guidance_are_finite():
    cohorts = make_synthetic_cohorts(
        SyntheticConfig(n_source=6, n_prior=4, n_target=4, horizon=12, baseline_dim=3)
    )
    cohort = cohorts["target"]
    scaler = TrajectoryScaler.fit(cohorts["source"])
    model = CausalConditionalDenoiser(baseline_dim=3, hidden_dim=8, depth=1)
    diffusion = GaussianDiffusion(DiffusionConfig(steps=4))
    clean = torch.from_numpy(scaler.transform(cohort.steps))
    actions = torch.from_numpy(cohort.actions)
    baseline = torch.from_numpy(cohort.baseline)
    loss = diffusion.loss(model, clean, actions, baseline)
    assert torch.isfinite(loss)
    sample = diffusion.sample(model, actions, baseline, seed=3)
    guided = diffusion.sample(
        model,
        actions,
        baseline,
        seed=3,
        guidance=torch.full((cohort.horizon,), 0.05),
    )
    assert sample.shape == clean.shape
    assert torch.isfinite(sample).all() and torch.isfinite(guided).all()
    assert not torch.equal(sample, guided)


def test_calibration_uses_profile_difference_in_model_space():
    cohort = make_synthetic_cohorts(
        SyntheticConfig(n_source=6, n_prior=4, n_target=4, horizon=12)
    )["source"]
    scaler = TrajectoryScaler.fit(cohort)
    base = np.full((5, 12), 100.0, dtype=np.float32)
    target = np.full(12, 150.0, dtype=np.float32)
    calibration = build_linear_guidance(base, target, scaler)
    assert calibration.guidance.shape == (12,)
    assert (calibration.guidance > 0).all()
