import numpy as np
import pytest

from jitai_twin.data import Cohort, TrajectoryScaler
from jitai_twin.synthetic import SyntheticConfig, make_synthetic_cohorts


def test_synthetic_cohorts_are_deterministic_and_valid():
    config = SyntheticConfig(
        n_source=8, n_prior=6, n_target=5, horizon=24, baseline_dim=3, seed=41
    )
    first = make_synthetic_cohorts(config)
    second = make_synthetic_cohorts(config)
    assert set(first) == {"source", "prior", "target"}
    for name in first:
        np.testing.assert_array_equal(first[name].steps, second[name].steps)
        assert first[name].steps.shape[1] == 24
        assert first[name].baseline.shape[1] == 3
    assert not first["source"].actions.any()
    assert first["prior"].actions.any()


def test_scaler_round_trip():
    cohort = make_synthetic_cohorts(
        SyntheticConfig(n_source=6, n_prior=4, n_target=4, horizon=12)
    )["source"]
    scaler = TrajectoryScaler.fit(cohort)
    recovered = scaler.inverse(scaler.transform(cohort.steps))
    np.testing.assert_allclose(recovered, cohort.steps, rtol=1e-5, atol=1e-3)


def test_data_contract_rejects_nonbinary_actions():
    with pytest.raises(ValueError, match="binary"):
        Cohort(
            steps=np.ones((2, 8)),
            actions=np.full((2, 8), 0.5),
            baseline=np.ones((2, 2)),
            observed=np.ones((2, 8), dtype=bool),
        )
