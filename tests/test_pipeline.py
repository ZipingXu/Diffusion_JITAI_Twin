import json

from jitai_twin.pipeline import DemoConfig, run_synthetic_demo


def test_quick_pipeline_runs_only_on_synthetic_data(tmp_path):
    summary = run_synthetic_demo(tmp_path, DemoConfig.quick(seed=5))
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["release"] == "synthetic-only"
    assert saved["privacy"]["external_files_read"] == 0
    assert saved["privacy"]["participant_data_written"] == 0
    assert not saved["privacy"]["target_outcomes_used_for_training_or_calibration"]
    assert set(summary["uncalibrated"]) == set(summary["calibrated"])
