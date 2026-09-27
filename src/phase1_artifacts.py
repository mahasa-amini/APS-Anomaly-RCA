"""Exclusive run outputs, hashes and runtime provenance for Phase 1."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

from .config import ROOT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def runtime_environment():
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl", "xgboost"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=platform.python_version(), executable=sys.executable,
                platform=platform.platform(), packages=packages)


def provenance():
    def git(*args):
        result = subprocess.run(
            ["git", "--no-optional-locks", *args], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    sources = [ROOT / "main_phase1.py", *sorted((ROOT / "src").glob("*.py"))]
    return dict(
        utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        git_commit=git("rev-parse", "HEAD"), git_status=git("status", "--short"),
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sources},
        environment=runtime_environment(),
    )


def public_provenance(local):
    """Explicit allowlist for commit-friendly metadata; never copy local context."""
    return dict(
        utc=local["utc"], git_commit=local["git_commit"],
        source_sha256=dict(local["source_sha256"]),
        environment=dict(python=local["environment"]["python"],
                         packages=dict(local["environment"]["packages"])),
    )


def run_paths(run_id, artifact_root, result_root):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("Run ID must contain only letters, digits, hyphens and underscores.")
    artifact_dir, result_dir = Path(artifact_root) / run_id, Path(result_root) / run_id
    if artifact_dir.resolve() == result_dir.resolve():
        raise ValueError("Artifacts and public summaries must have separate directories.")
    return artifact_dir, result_dir


def file_record(path):
    path = Path(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha256(path))


def data_record(X, y):
    return dict(shape=list(X.shape), label_counts={str(i): int((y == i).sum()) for i in (0, 1)})
