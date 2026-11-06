from pathlib import Path

from jitai_twin.release_audit import audit_release


def test_repository_passes_release_audit():
    root = Path(__file__).resolve().parents[1]
    assert audit_release(root) == []


def test_audit_rejects_dataset_and_secret(tmp_path):
    (tmp_path / "participants.csv").write_text("participant_id,value\n1,2\n")
    secret_marker = "BEGIN " + "PRIVATE KEY"
    (tmp_path / "notes.txt").write_text(secret_marker + "\nnot-a-real-key\n")
    issues = audit_release(tmp_path)
    reasons = {issue.reason for issue in issues}
    assert "forbidden release suffix .csv" in reasons
    assert "private key" in reasons
