# JITAI-Twin

Privacy-safe reference code for **“A Diffusion-Model Subpopulation Digital
Twin for Mobile Health Deployment: A Case Study on the HeartSteps
Intervention.”**

The package demonstrates the paper's three-stage workflow:

1. pretrain a temporally consistent conditional diffusion model on a large
   observational cohort;
2. fine-tune intervention injection and a small low-rank adapter on a prior
   randomized deployment; and
3. calibrate the reverse diffusion process toward a pre-deployment target by
   linear reward tilting, without changing model weights.

## Privacy boundary

No All of Us, HeartSteps, or other participant-level data are included. The
repository also excludes learned checkpoints, generated study trajectories,
survey responses or embeddings, credentials, private paths, and aggregate
results computed from restricted data. Tests and examples create visibly
synthetic trajectories in memory.

The public data contract is documented in [docs/data.md](docs/data.md). To use
authorized real data, write a private adapter that constructs a `Cohort`; do
not add the underlying files to this repository.

## What this release implements

- a causal one-dimensional denoiser whose output at time `t` cannot depend on
  actions after `t`;
- baseline and intervention conditioning;
- observational pretraining and parameter-efficient fine-tuning;
- variance-preserving diffusion training and sampling;
- in-process linear reward tilting toward an expert-specified target profile;
- marginal, temporal, and between-participant fidelity metrics;
- a synthetic end-to-end example and automated privacy audit.

The paper's full-scale experiments use a state-space TimeWeaver backbone. The
model here is a compact causal convolutional reference implementation designed
to test the scientific interfaces on a CPU. See [NOTICE.md](NOTICE.md) and
[docs/method.md](docs/method.md) before claiming numerical reproduction of the
paper.

## Install

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For CPU-only environments, install the appropriate PyTorch build for your
platform before installing this package.

## Run the synthetic demonstration

```bash
jitai-twin demo --quick --output outputs/demo
```

This trains both stages, draws uncalibrated and calibrated fake trajectories,
and writes only aggregate synthetic diagnostics to `outputs/demo/summary.json`.
The output directory is ignored by Git.

## Test and audit

```bash
pytest
jitai-twin audit --root .
```

The audit fails on dataset formats, model weights, credentials, private
cluster paths, unexpected binary files, symlinks, and files larger than 2 MB.
It is also run in GitHub Actions.

## Repository layout

```text
src/jitai_twin/     package source
tests/              synthetic unit and end-to-end tests
docs/               method, data contract, and release notes
.github/workflows/  clean-install CI and release audit
```

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). Until the
article has a final journal citation, cite the title and authors recorded
there.

## License

The release is available under the MIT License. The full-scale backbone has
separate authorship and terms; see [NOTICE.md](NOTICE.md).
