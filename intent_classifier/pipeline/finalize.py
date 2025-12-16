"""Run finalization: normalize outputs into the new run schema and write metadata.

This module is responsible for the *last* step of a pipeline run. It assumes
that training, prediction, and evaluation have already been executed and that
all artefacts have been written under the paths derived from the active config.

Responsibilities:

- Use ``load_config_with_metadata`` to discover the active run and its
  ``run_id`` / ``paths``.
- Ensure the canonical layout under ``output/runs/{run_id}/`` exists.
- Standardise model directories to stable ``model_id`` slugs.
- Write a reproducibility package under ``meta/``.
- Generate a ``manifest.json`` with file sizes and SHA256 hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from intent_classifier.evaluation.run_docs import generate_run_readme_and_model_cards
from intent_classifier.utils.config_loader import ConfigMetadata, load_config_with_metadata
from intent_classifier.utils.file_ops import ensure_dir
from intent_classifier.utils.paths import get_repo_root
from intent_classifier.utils.slugify import slugify_model_id


def _sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _discover_model_dirs(models_root: Path) -> dict[str, Path]:
    """Discover per-model directories under the models root and map slugs to paths."""
    model_dirs: dict[str, Path] = {}
    if not models_root.exists():
        return model_dirs

    for child in models_root.iterdir():
        if not child.is_dir():
            continue
        display_name = child.name
        slug = slugify_model_id(display_name)
        # Prefer the first occurrence if there are collisions
        model_dirs.setdefault(slug, child)
    return model_dirs


def _standardise_model_dirs(models_root: Path) -> dict[str, dict[str, Any]]:
    """Rename model directories to slug ids and attach basic metadata.

    Returns a mapping: slug -> {"path": Path, "display_name": str}.
    """
    models_root = ensure_dir(models_root)
    mapping: dict[str, dict[str, Any]] = {}

    for child in sorted(models_root.iterdir()):
        if not child.is_dir():
            continue

        display_name = child.name

        # Skip and clean up any legacy/duplicate directories created by previous
        # runs of finalisation. Older versions used both single \"_obsolete\"
        # and double \"__obsolete\" suffixes, and repeated runs could nest them.
        if "obsolete" in display_name:
            try:
                shutil.rmtree(child)
            except OSError:
                # Best-effort cleanup; if removal fails we simply ignore this dir.
                pass
            continue

        slug = slugify_model_id(display_name)
        dest = models_root / slug
        # If the directory is already in slug form, keep it as-is.
        if child == dest:
            mapping[slug] = {"path": dest, "display_name": display_name}
        else:
            if dest.exists():
                # If a destination already exists, keep the existing one and
                # drop the duplicate display-name directory.
                # This situation should be rare and typically only happens
                # when re-running finalisation.
                # We simply remove the duplicate folder instead of keeping
                # another \"obsolete\" copy to avoid polluting the models dir.
                try:
                    shutil.rmtree(child)
                except OSError:
                    # Ignore best-effort cleanup failures.
                    pass
            else:
                child.rename(dest)
            mapping[slug] = {"path": dest, "display_name": display_name}

        # Write a small metadata file per model folder
        meta_path = mapping[slug]["path"] / "model_meta.json"
        if not meta_path.exists():
            payload = {
                "model_id": slug,
                "display_name": display_name,
            }
            meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return mapping


@dataclass
class MetaConfigSources:
    """Description of configuration sources used to build the resolved config."""

    files: list[str]
    sha256: dict[str, str]


def _build_config_sources(metadata: ConfigMetadata) -> MetaConfigSources:
    """Infer the configuration files that contributed to this run.

    We rely on the known layering scheme:
        config/base/defaults.yaml
        config/base/providers.yaml
        config/datasets/{dataset}.yaml
        config/experiments/{dataset}/{variant}.yaml
    """
    repo_root = get_repo_root()
    dataset_name = metadata["dataset_name"]
    config_name = metadata["config_name"]

    candidates: list[Path] = []

    defaults = repo_root / "config" / "base" / "defaults.yaml"
    providers = repo_root / "config" / "base" / "providers.yaml"
    dataset_cfg = repo_root / "config" / "datasets" / f"{dataset_name}.yaml"
    experiment_cfg = repo_root / "config" / "experiments" / dataset_name / f"{config_name}.yaml"

    for path in (defaults, providers, dataset_cfg, experiment_cfg):
        if path.exists():
            candidates.append(path)

    # If no experiment config exists, fall back to the concrete path we loaded.
    if not experiment_cfg.exists():
        cfg_path = metadata["config_path"]
        if cfg_path.exists() and cfg_path not in candidates:
            candidates.append(cfg_path)

    file_paths = [str(p.relative_to(repo_root)) for p in candidates]
    hashes = {str(p.relative_to(repo_root)): _sha256_file(p) for p in candidates}
    return MetaConfigSources(files=file_paths, sha256=hashes)


def _write_meta_package(run_dir: Path, metadata: ConfigMetadata) -> None:
    """Create the reproducibility package under meta/."""
    repo_root = get_repo_root()
    config = metadata["config"]
    paths_cfg = config.get("paths", {})
    run_id = config.get("resolved", {}).get("run_id") or metadata["config_name"]

    meta_dir = ensure_dir(run_dir / "meta")

    # 1) Resolved config (YAML)
    import yaml

    resolved_cfg_path = meta_dir / "config_resolved.yaml"
    with resolved_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # 2) Config sources + hashes
    sources = _build_config_sources(metadata)
    (meta_dir / "config_sources.json").write_text(
        json.dumps(asdict(sources), indent=2), encoding="utf-8"
    )

    # 3) Command/environment selection
    command_info = {
        "argv": sys.argv,
        "env": {
            k: os.environ.get(k)
            for k in ["CONFIG_FILE", "DATASET", "VARIANT", "N_CLASSES", "MODELS_DIR"]
            if k in os.environ
        },
        "run_id": run_id,
        "paths": paths_cfg,
    }
    (meta_dir / "command.txt").write_text(json.dumps(command_info, indent=2), encoding="utf-8")

    # 4) Git state
    git_info: dict[str, Any] = {}
    try:

        def _run_git(args: list[str]) -> str | None:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            return result.stdout.strip() or None

        git_info = {
            "commit": _run_git(["rev-parse", "HEAD"]),
            "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "is_dirty": bool(_run_git(["status", "--porcelain"])),
        }
    except Exception:
        git_info = {"error": "git_not_available"}
    (meta_dir / "git.json").write_text(json.dumps(git_info, indent=2), encoding="utf-8")

    # 5) Environment info
    env_info: dict[str, Any] = {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
        },
    }

    # Try to capture a `uv pip freeze` style snapshot if possible.
    try:
        result = subprocess.run(
            ["uv", "pip", "freeze"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            env_info["packages"] = {
                "tool": "uv pip freeze",
                "output": result.stdout.splitlines(),
            }
    except Exception:
        # Skip silently if uv is not available – core env info is still present.
        pass

    (meta_dir / "env.json").write_text(json.dumps(env_info, indent=2), encoding="utf-8")

    # 6) Timestamps (best-effort)
    now = datetime.now(UTC).isoformat()
    timestamps = {
        "finalize_started_at": now,
    }
    (meta_dir / "timestamps.json").write_text(json.dumps(timestamps, indent=2), encoding="utf-8")


def _write_manifest(run_dir: Path) -> None:
    """Write a manifest.json with all files under the run directory."""
    manifest: list[dict[str, Any]] = []
    repo_root = get_repo_root()

    for path in run_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(repo_root)
            try:
                size = path.stat().st_size
                sha = _sha256_file(path)
            except OSError:
                size = None
                sha = None
            manifest.append(
                {
                    "path": str(rel),
                    "size": size,
                    "sha256": sha,
                }
            )

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def finalize_run() -> None:
    """Entry point: normalize the current run's outputs and write metadata."""
    metadata = load_config_with_metadata()
    config = metadata["config"]
    paths_cfg = config.get("paths", {})

    repo_root = get_repo_root()
    run_dir = repo_root / paths_cfg.get("run_dir", "output/runs/unknown")
    run_dir = ensure_dir(run_dir)

    # 1) Standardise model directories
    models_root = repo_root / paths_cfg.get("models_dir", "output/runs/unknown/models")
    _standardise_model_dirs(models_root)

    # 2) Write meta package (resolved config, sources, env, git, etc.)
    _write_meta_package(run_dir, metadata)

    # 3) Write manifest across the entire run directory
    _write_manifest(run_dir)

    # 4) Generate run-level README and per-model model cards **after**
    #    directory standardisation and metadata/compare artefacts exist.
    generate_run_readme_and_model_cards()


def main() -> None:
    """CLI entrypoint used by the pipeline script."""
    finalize_run()
