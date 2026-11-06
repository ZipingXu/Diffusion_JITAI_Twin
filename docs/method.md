# Method-to-code map

## Stage 1: observational pretraining

`training.fit_stage(..., stage="pretrain")` trains the causal outcome
backbone and baseline-conditioning path. The intervention projection and
low-rank adapter are frozen because the observational cohort has no recorded
target intervention.

## Stage 2: prior-deployment fine-tuning

`training.fit_stage(..., stage="finetune")` freezes the causal backbone and
updates the intervention path, a low-rank baseline adapter, and the input and
output projections. This mirrors the parameter-efficient adaptation used when
the prior randomized deployment is much smaller than the observational cohort.

## Temporal consistency

`CausalConditionalDenoiser` uses left-padded convolutions and normalizes only
across channels at each time point. Its output at time `t` therefore has no
computational path from action values after `t`. The test
`test_future_actions_do_not_change_the_generated_past` checks this property
directly.

Temporal consistency is an architectural non-anticipation property. It does
not by itself identify a causal intervention effect.

## Stage 3: inference-time calibration

Let `mu_target` be a pre-deployment target profile, `mu_base` the corresponding
mean from uncalibrated draws, and `v = mu_target - mu_base` in model space.
`GaussianDiffusion.sample` applies the linear reward tilt inside the reverse
process, approximating

```text
p_tilt(x) proportional to p_theta(x) exp(gamma <v, x>).
```

`gamma=0` recovers the base sampler. The public example constructs its target
from synthetic protocol facts; it never estimates the target from held-out
outcomes.

## Compact versus full-scale model

The reference denoiser is deliberately small enough for CPU CI. The study used
a forward-only state-space TimeWeaver model, T5-based survey conditioning, and
larger training and evaluation workflows. The reference implementation tests
the interfaces and invariants but is not a substitute for the study's
checkpoint when reproducing reported numbers.
