"""Fail-closed checks that keep private data and credentials out of releases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


FORBIDDEN_SUFFIXES = {
    ".csv",
    ".tsv",
    ".parquet",
    ".feather",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".pem",
    ".key",
    ".zip",
}
FORBIDDEN_NAMES = {
    ".env",
    "credentials.json",
    "gmail_token.json",
    "gmail_client_secret.json",
}
SKIP_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "checkpoints",
    "dist",
    "outputs",
}
TEXT_PATTERNS = {
    "private cluster path": re.compile(r"/(?:n/netscratch|hickory|mnt/jitai)/"),
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
}


@dataclass(frozen=True)
class AuditIssue:
    path: Path
    reason: str


def _is_skipped(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return any(part in SKIP_DIRECTORIES or part.endswith(".egg-info") for part in relative.parts)


def audit_release(root: str | Path, *, max_bytes: int = 2_000_000) -> list[AuditIssue]:
    """Return release-policy violations found below ``root``."""

    root = Path(root).resolve()
    issues: list[AuditIssue] = []
    for path in root.rglob("*"):
        if _is_skipped(path, root):
            continue
        if path.is_symlink():
            issues.append(AuditIssue(path, "symbolic links are not allowed"))
            continue
        if not path.is_file():
            continue
        name_lower = path.name.lower()
        if name_lower in FORBIDDEN_NAMES or name_lower.startswith(".env."):
            issues.append(AuditIssue(path, "credential-like filename"))
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(AuditIssue(path, f"forbidden release suffix {path.suffix}"))
        size = path.stat().st_size
        if size > max_bytes:
            issues.append(AuditIssue(path, f"file exceeds {max_bytes} bytes"))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            issues.append(AuditIssue(path, "unreviewed binary file"))
            continue
        for label, pattern in TEXT_PATTERNS.items():
            if pattern.search(text):
                issues.append(AuditIssue(path, label))
    return issues


def assert_release_clean(root: str | Path) -> None:
    issues = audit_release(root)
    if issues:
        details = "\n".join(f"- {issue.path}: {issue.reason}" for issue in issues)
        raise RuntimeError(f"release audit failed:\n{details}")
