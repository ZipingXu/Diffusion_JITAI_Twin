# Data contract and privacy policy

## Public contract

The package accepts an in-memory `Cohort` with four arrays:

| field | shape | meaning |
|---|---:|---|
| `steps` | participants × time | non-negative outcome trajectory |
| `actions` | participants × time | binary intervention indicator |
| `baseline` | participants × features | numeric baseline embedding |
| `observed` | participants × time | outcome observation mask |

The name `steps` follows the case study, but the algorithms only require a
non-negative longitudinal outcome.

## Private adapter pattern

Real applications should keep data access, cohort definitions, survey
harmonization, and participant identifiers in an access-controlled repository.
That private code may return a `Cohort` to this package at runtime. It should
never copy raw data, cached embeddings, generated participant trajectories, or
credentials into this repository.

## Files blocked from release

The checked-in audit rejects common tabular and array formats, serialized
models, archives, key material, environment files, unexpected binaries,
private compute paths, and large files. The `.gitignore` repeats these rules to
prevent accidental staging.

Synthetic arrays are generated at runtime by `make_synthetic_cohorts`. No
synthetic dataset is stored in Git either; this keeps the public repository's
data boundary simple and auditable.
