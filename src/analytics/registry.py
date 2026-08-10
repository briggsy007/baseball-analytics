"""
Model registry for persisting, versioning, and loading analytics models.

Provides a centralised store for trained model artefacts alongside JSON
metadata so that any model can be reproduced, compared, or rolled back.

Each model can be assigned a *stage* — one of ``"dev"``, ``"test"``,
``"staging"``, or ``"production"`` — to track its lifecycle from initial
experimentation through to live serving.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import joblib

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_DIR = _PROJECT_ROOT / "models"

VALID_STAGES: tuple[str, ...] = ("dev", "test", "staging", "production")
"""Allowed values for the model stage field."""

Stage = Literal["dev", "test", "staging", "production"]

# ---------------------------------------------------------------------------
# WS2.1 (2026-08-10): versioned manifest layout + alias index
# ---------------------------------------------------------------------------
# Layout:
#   models/<name>/v<YYYY.MM.DD>[-tag]/manifest.json   (write-once, immutable)
#   models/registry.json                              (the ONLY mutable file)
#
# ``registry.json`` maps ``{name: {production: vX, frozen_validated: vY,
# history: [...]}}`` (MLflow immutable-versions + mutable-aliases pattern).
# A version dir holds the manifest and either the artifact itself or a
# registry-root-relative pointer to an artifact that lives at its canonical
# path (e.g. ``defensive_pressing/xout_v1.pkl``).  Pointer form is the
# default so the WS0.1 frozen-path RuntimeError guards in the analytics
# modules keep protecting the exact same file the registry serves.
#
# ``hash_policy``:
#   "pinned"   — verify_artifacts recomputes sha256 and FAILS on mismatch
#                (frozen validation evidence).
#   "advisory" — sha256 records registration-time state only; the artifact
#                churns by design (nightly in-season retrains, gitignored),
#                so verification checks existence, not the hash.

REGISTRY_INDEX_NAME = "registry.json"
MANIFEST_NAME = "manifest.json"
REGISTRY_ALIASES: tuple[str, ...] = ("production", "frozen_validated")
HASH_POLICIES: tuple[str, ...] = ("pinned", "advisory")
REGISTRY_DIR_ENV = "BASEBALL_MODEL_REGISTRY_DIR"

_VERSION_RE = re.compile(r"^v\d{4}\.\d{2}\.\d{2}([.-][A-Za-z0-9][\w.\-]*)?$")


def _timestamp_version() -> str:
    """Generate a timestamp-based version string like ``20260409T153012``."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def default_registry_dir() -> Path:
    """The registry root: ``$BASEBALL_MODEL_REGISTRY_DIR`` or ``<project>/models``.

    The env override exists so the test suite (and any sandboxed run) can
    point registry resolution at a tmp dir and never touch the repo-level
    ``models/`` tree (WS0.2 guard).
    """
    env = os.environ.get(REGISTRY_DIR_ENV)
    return Path(env) if env else DEFAULT_REGISTRY_DIR


def sha256_of_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """Stream a file through sha256 and return the hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def current_git_sha() -> Optional[str]:
    """Best-effort HEAD sha of the project repo (None if git unavailable)."""
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
        sha = (out.stdout or "").strip()
        return sha if out.returncode == 0 and sha else None
    except Exception:
        return None


class ModelRegistry:
    """Manages model artefact storage, versioning, and metadata.

    Models are saved as::

        {registry_dir}/{name}_v{version}.pkl
        {registry_dir}/{name}_v{version}_meta.json

    Args:
        registry_dir: Root directory for stored artefacts.
                      Defaults to ``<project>/models/``.
    """

    def __init__(self, registry_dir: Path | str | None = None) -> None:
        self.registry_dir = (
            Path(registry_dir) if registry_dir else default_registry_dir()
        )
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    # ── WS2.1 versioned manifests + alias index ───────────────────────────

    @property
    def index_path(self) -> Path:
        """Path of ``registry.json`` — the only mutable registry file."""
        return self.registry_dir / REGISTRY_INDEX_NAME

    def _read_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {}
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write_index_atomic(self, index: dict[str, Any]) -> None:
        """Write ``registry.json`` atomically: temp file same dir + os.replace.

        Same-directory temp file guarantees same volume, which is the
        Windows caveat for ``os.replace`` atomicity.
        """
        payload = json.dumps(index, indent=2, default=str)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".registry_", suffix=".json.tmp", dir=str(self.registry_dir)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_name, str(self.index_path))
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def _version_dir(self, name: str, version: str) -> Path:
        return self.registry_dir / name / version

    def _manifest_path(self, name: str, version: str) -> Path:
        return self._version_dir(name, version) / MANIFEST_NAME

    def register_version(
        self,
        name: str,
        version: str,
        artifact: Path | str,
        *,
        sha256: Optional[str] = None,
        hash_policy: str = "pinned",
        train_window: Optional[str] = None,
        data_snapshot: Optional[dict[str, Any]] = None,
        training_script: Optional[str] = None,
        spec_version: Optional[str] = None,
        validation_results_ref: Optional[str] = None,
        git_sha: Optional[str] = None,
        notes: Optional[str] = None,
        copy_artifact: bool = False,
    ) -> dict[str, Any]:
        """Create the write-once version dir + manifest for an artifact.

        Args:
            name: Model identifier (e.g. ``"defensive_pressing"``).
            version: ``v<YYYY.MM.DD>`` (optional ``-tag`` suffix when two
                distinct artifacts share a freeze date).
            artifact: Path to the EXISTING artifact file — absolute, or
                relative to the registry dir.
            sha256: Expected hex digest.  When given it is verified against
                the file's actual hash; a mismatch raises.  When omitted the
                actual hash is recorded.
            hash_policy: ``"pinned"`` (immutable evidence; verification
                fails on drift) or ``"advisory"`` (churning in-season
                artifact; hash records registration-time state only).
            train_window / data_snapshot / training_script / spec_version /
                validation_results_ref / git_sha / notes: provenance fields
                per the WS2.1 manifest schema.  ``git_sha`` defaults to the
                current HEAD when resolvable.
            copy_artifact: Copy the artifact into the version dir instead of
                recording a registry-root-relative pointer.  Pointer form is
                the default so the frozen-path guards in the analytics
                modules keep protecting the served file.

        Returns:
            The manifest dict as written.

        Raises:
            FileExistsError: If the version dir already exists (write-once).
            FileNotFoundError: If the artifact does not exist.
            ValueError: On invalid version/hash_policy, or sha256 mismatch.
        """
        if not _VERSION_RE.match(version):
            raise ValueError(
                f"Invalid version {version!r}: expected v<YYYY.MM.DD> with "
                f"optional -tag suffix (e.g. v2026.08.10, v2026.08.10-inseason)"
            )
        if hash_policy not in HASH_POLICIES:
            raise ValueError(
                f"Invalid hash_policy {hash_policy!r}; must be one of {HASH_POLICIES}"
            )

        artifact_path = Path(artifact)
        if not artifact_path.is_absolute():
            artifact_path = self.registry_dir / artifact_path
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        version_dir = self._version_dir(name, version)
        if version_dir.exists():
            raise FileExistsError(
                f"Version dir already exists (versions are write-once): "
                f"{version_dir}. Register a new version instead."
            )

        actual_sha = sha256_of_file(artifact_path)
        if sha256 is not None and sha256.lower() != actual_sha:
            raise ValueError(
                f"sha256 mismatch for {artifact_path}: expected {sha256}, "
                f"actual {actual_sha}. Refusing to register."
            )

        version_dir.mkdir(parents=True, exist_ok=False)
        if copy_artifact:
            dest = version_dir / artifact_path.name
            shutil.copy2(artifact_path, dest)
            artifact_ref = artifact_path.name
        else:
            try:
                artifact_ref = artifact_path.resolve().relative_to(
                    self.registry_dir.resolve()
                ).as_posix()
            except ValueError:
                shutil.rmtree(version_dir, ignore_errors=True)
                raise ValueError(
                    f"Artifact {artifact_path} lies outside the registry dir "
                    f"{self.registry_dir}; pointer registration requires a "
                    f"registry-root-relative path. Pass copy_artifact=True "
                    f"to copy it into the version dir instead."
                ) from None

        manifest: dict[str, Any] = {
            "artifact": artifact_ref,
            "version": version,
            "sha256": actual_sha,
            "hash_policy": hash_policy,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha if git_sha is not None else current_git_sha(),
            "train_window": train_window,
            "data_snapshot": data_snapshot,
            "training_script": training_script,
            "spec_version": spec_version,
            "validation_results_ref": validation_results_ref,
        }
        if notes:
            manifest["notes"] = notes
        self._manifest_path(name, version).write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Registered %s %s -> %s", name, version, artifact_ref)

        index = self._read_index()
        entry = index.setdefault(
            name, {"production": None, "frozen_validated": None, "history": []}
        )
        entry.setdefault("history", []).append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "register",
                "version": version,
                "sha256": actual_sha,
                "hash_policy": hash_policy,
            }
        )
        self._write_index_atomic(index)
        return manifest

    def get_manifest_v2(self, name: str, version: str) -> dict[str, Any]:
        """Read a versioned manifest (WS2.1 layout).

        Named ``_v2`` to avoid clashing with the legacy sidecar
        :meth:`get_metadata` API, which remains supported.
        """
        path = self._manifest_path(name, version)
        if not path.exists():
            raise FileNotFoundError(f"No manifest at {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_versions(self, name: str) -> list[str]:
        """All registered versions for *name*, sorted ascending."""
        model_dir = self.registry_dir / name
        if not model_dir.exists():
            return []
        return sorted(
            p.name
            for p in model_dir.iterdir()
            if p.is_dir() and (p / MANIFEST_NAME).exists()
        )

    def artifact_path_for(self, name: str, version: str) -> Path:
        """Resolve the artifact file a registered version points at."""
        manifest = self.get_manifest_v2(name, version)
        artifact_ref = manifest["artifact"]
        in_dir = self._version_dir(name, version) / artifact_ref
        if in_dir.exists():
            return in_dir
        return self.registry_dir / artifact_ref

    def get_alias(self, name: str, alias: str) -> Optional[str]:
        """Version string an alias currently points at (None if unset)."""
        if alias not in REGISTRY_ALIASES:
            raise ValueError(
                f"Invalid alias {alias!r}; must be one of {REGISTRY_ALIASES}"
            )
        entry = self._read_index().get(name)
        if not entry:
            return None
        return entry.get(alias)

    def set_alias(self, name: str, alias: str, version: str) -> dict[str, Any]:
        """Atomically point ``registry.json``'s *alias* at *version*.

        The version must already be registered.  The change is appended to
        the model's history.  This is the ONLY mutation the registry
        performs on existing state — manifests and version dirs are
        write-once.
        """
        if alias not in REGISTRY_ALIASES:
            raise ValueError(
                f"Invalid alias {alias!r}; must be one of {REGISTRY_ALIASES}"
            )
        # Must exist (also validates the manifest parses).
        manifest = self.get_manifest_v2(name, version)
        if alias == "frozen_validated" and manifest.get("hash_policy") != "pinned":
            raise ValueError(
                f"frozen_validated must reference a hash_policy='pinned' "
                f"version; {name} {version} is "
                f"{manifest.get('hash_policy')!r}."
            )
        index = self._read_index()
        entry = index.setdefault(
            name, {"production": None, "frozen_validated": None, "history": []}
        )
        previous = entry.get(alias)
        entry[alias] = version
        entry.setdefault("history", []).append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "alias",
                "alias": alias,
                "from": previous,
                "to": version,
            }
        )
        self._write_index_atomic(index)
        logger.info("Alias %s.%s: %s -> %s", name, alias, previous, version)
        return entry

    def resolve_alias(self, name: str, alias: str) -> Path:
        """Artifact path for the version an alias points at.

        Raises:
            KeyError: If the model/alias is not set in ``registry.json``.
            FileNotFoundError: If the manifest or artifact file is missing.
        """
        version = self.get_alias(name, alias)
        if version is None:
            raise KeyError(f"No {alias!r} alias set for model {name!r}")
        path = self.artifact_path_for(name, version)
        if not path.exists():
            raise FileNotFoundError(
                f"{name} {version} ({alias}) artifact missing: {path}"
            )
        return path

    def verify(self, name: Optional[str] = None) -> list[dict[str, Any]]:
        """Integrity-check every registered version + alias.

        Checks, per version manifest:
          * manifest parses and the artifact file exists
            (pinned: FAIL if missing; advisory: WARN — gitignored in-season
            artifacts may legitimately be absent in a fresh clone);
          * recomputed sha256 matches the manifest
            (pinned: FAIL on mismatch; advisory: OK with drift noted —
            nightly retrains churn those artifacts by design).
        Per alias: it points at a registered version, and
        ``frozen_validated`` points at a *pinned* version.

        Returns:
            List of ``{model, version, check, status, detail}`` dicts with
            status in {"ok", "warn", "fail"}.
        """
        results: list[dict[str, Any]] = []

        def _add(model: str, version: str, check: str, status: str, detail: str) -> None:
            results.append(
                {
                    "model": model,
                    "version": version,
                    "check": check,
                    "status": status,
                    "detail": detail,
                }
            )

        try:
            index = self._read_index()
        except (json.JSONDecodeError, OSError) as exc:
            _add("<registry>", "-", "index_parse", "fail", f"registry.json unreadable: {exc}")
            return results

        names = [name] if name else sorted(index.keys())
        for model_name in names:
            entry = index.get(model_name)
            if entry is None:
                _add(model_name, "-", "index_entry", "fail", "model not in registry.json")
                continue

            for version in self.list_versions(model_name):
                try:
                    manifest = self.get_manifest_v2(model_name, version)
                except (json.JSONDecodeError, OSError, FileNotFoundError) as exc:
                    _add(model_name, version, "manifest_parse", "fail", str(exc))
                    continue
                policy = manifest.get("hash_policy", "pinned")
                artifact = self.artifact_path_for(model_name, version)
                if not artifact.exists():
                    status = "fail" if policy == "pinned" else "warn"
                    _add(
                        model_name, version, "artifact_exists", status,
                        f"missing: {artifact}"
                        + ("" if policy == "pinned" else " (advisory/gitignored artifact)"),
                    )
                    continue
                actual = sha256_of_file(artifact)
                expected = (manifest.get("sha256") or "").lower()
                if actual == expected:
                    _add(model_name, version, "sha256", "ok", f"{artifact.name} {actual[:12]}...")
                elif policy == "pinned":
                    _add(
                        model_name, version, "sha256", "fail",
                        f"MISMATCH {artifact}: manifest {expected[:12]}... "
                        f"actual {actual[:12]}...",
                    )
                else:
                    _add(
                        model_name, version, "sha256", "ok",
                        f"advisory drift on {artifact.name}: registered "
                        f"{expected[:12]}..., current {actual[:12]}... "
                        f"(expected for nightly in-season retrains)",
                    )

            for alias in REGISTRY_ALIASES:
                version = entry.get(alias)
                if version is None:
                    continue
                try:
                    manifest = self.get_manifest_v2(model_name, version)
                except (json.JSONDecodeError, OSError, FileNotFoundError) as exc:
                    _add(model_name, version, f"alias_{alias}", "fail", str(exc))
                    continue
                if alias == "frozen_validated" and manifest.get("hash_policy") != "pinned":
                    _add(
                        model_name, version, "alias_frozen_pinned", "fail",
                        "frozen_validated points at a non-pinned version",
                    )
                else:
                    _add(model_name, version, f"alias_{alias}", "ok", f"-> {version}")
        return results

    # ── Save ──────────────────────────────────────────────────────────────

    def save_model(
        self,
        model: Any,
        name: str,
        version: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        stage: Stage = "dev",
    ) -> dict[str, str]:
        """Persist a model and its metadata.

        Args:
            model: Any picklable model object.
            name: Short identifier (e.g. ``"stuff_plus"``).
            version: Optional explicit version string.  If omitted a
                     UTC-timestamp version is generated automatically.
            metadata: Arbitrary key-value metadata (training params,
                      metrics, data range, etc.).
            stage: Lifecycle stage for this model.  Must be one of
                   ``"dev"`` (default), ``"test"``, ``"staging"``, or
                   ``"production"``.

        Returns:
            Dictionary with ``model_path``, ``meta_path``, ``version``,
            and ``stage``.

        Raises:
            ValueError: If *stage* is not a recognised value.
        """
        if stage not in VALID_STAGES:
            raise ValueError(
                f"Invalid stage {stage!r}. Must be one of {VALID_STAGES}"
            )

        if version is None:
            version = _timestamp_version()

        stem = f"{name}_v{version}"
        model_path = self.registry_dir / f"{stem}.pkl"
        meta_path = self.registry_dir / f"{stem}_meta.json"

        # Persist artefact
        joblib.dump(model, model_path)
        logger.info("Model saved: %s", model_path)

        # Persist metadata
        meta = {
            "name": name,
            "version": version,
            "stage": stage,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        logger.info("Metadata saved: %s (stage=%s)", meta_path, stage)

        return {
            "model_path": str(model_path),
            "meta_path": str(meta_path),
            "version": version,
            "stage": stage,
        }

    # ── Load ──────────────────────────────────────────────────────────────

    def load_model(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Any:
        """Load a model artefact from the registry.

        When *version* is ``None`` the latest version (by filename sort)
        is loaded.

        Args:
            name: Model identifier.
            version: Specific version string, or None for latest.

        Returns:
            The deserialised model object.

        Raises:
            FileNotFoundError: If no matching artefact exists.
        """
        if version is not None:
            path = self.registry_dir / f"{name}_v{version}.pkl"
            if not path.exists():
                raise FileNotFoundError(
                    f"No model found at {path}"
                )
            return joblib.load(path)

        # Find latest version
        candidates = sorted(self.registry_dir.glob(f"{name}_v*.pkl"))
        # Filter out *_meta.json companion files that might match
        candidates = [p for p in candidates if not p.stem.endswith("_meta")]
        if not candidates:
            raise FileNotFoundError(
                f"No versions found for model '{name}' in {self.registry_dir}"
            )
        latest = candidates[-1]
        logger.info("Loading latest version: %s", latest.name)
        return joblib.load(latest)

    # ── List / metadata ───────────────────────────────────────────────────

    def list_models(self) -> list[dict[str, Any]]:
        """Return metadata for every registered model.

        Returns:
            List of metadata dictionaries, sorted by name then version.
            Each entry is guaranteed to contain a ``stage`` key (defaults
            to ``"dev"`` for legacy entries saved before stage tracking).
        """
        results: list[dict[str, Any]] = []
        for meta_path in sorted(self.registry_dir.glob("*_meta.json")):
            try:
                meta = json.loads(meta_path.read_text())
                meta.setdefault("stage", "dev")  # backward compat
                meta["meta_path"] = str(meta_path)
                results.append(meta)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping corrupt metadata %s: %s", meta_path, exc)
        return results

    def get_metadata(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> dict[str, Any]:
        """Retrieve metadata for a specific model version.

        When *version* is ``None`` the latest version's metadata is
        returned.

        The returned dictionary is guaranteed to contain a ``stage`` key
        (defaults to ``"dev"`` for legacy entries saved before stage
        tracking).

        Args:
            name: Model identifier.
            version: Specific version, or None for latest.

        Returns:
            Metadata dictionary.

        Raises:
            FileNotFoundError: If no matching metadata exists.
        """
        if version is not None:
            meta_path = self.registry_dir / f"{name}_v{version}_meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"No metadata at {meta_path}")
            meta = json.loads(meta_path.read_text())
            meta.setdefault("stage", "dev")  # backward compat
            return meta

        candidates = sorted(self.registry_dir.glob(f"{name}_v*_meta.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No metadata found for model '{name}' in {self.registry_dir}"
            )
        latest = candidates[-1]
        meta = json.loads(latest.read_text())
        meta.setdefault("stage", "dev")  # backward compat
        return meta

    # ── Stage management ──────────────────────────────────────────────────

    def promote(
        self,
        name: str,
        version: str,
        stage: Stage,
    ) -> dict[str, Any]:
        """Update the stage of an existing saved model.

        Reads the metadata sidecar JSON, sets the ``stage`` field, and
        writes the file back.  The previous stage (if any) is recorded
        under the ``previous_stage`` key, and a ``promoted_at`` timestamp
        is added.

        Args:
            name: Model identifier.
            version: The exact version string to promote.
            stage: Target stage — one of ``"dev"``, ``"test"``,
                   ``"staging"``, or ``"production"``.

        Returns:
            The updated metadata dictionary.

        Raises:
            ValueError: If *stage* is not a recognised value.
            FileNotFoundError: If the metadata file does not exist.
        """
        if stage not in VALID_STAGES:
            raise ValueError(
                f"Invalid stage {stage!r}. Must be one of {VALID_STAGES}"
            )

        meta_path = self.registry_dir / f"{name}_v{version}_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No metadata found at {meta_path}"
            )

        meta = json.loads(meta_path.read_text())
        old_stage = meta.get("stage", "dev")
        meta["previous_stage"] = old_stage
        meta["stage"] = stage
        meta["promoted_at"] = datetime.now(timezone.utc).isoformat()

        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        logger.info(
            "Model %s v%s promoted: %s -> %s", name, version, old_stage, stage
        )
        return meta

    def get_production_model(self, name: str) -> Any:
        """Load the latest model marked as ``"production"`` stage.

        Scans all metadata sidecars for the given model *name*, filters
        to those whose ``stage`` is ``"production"``, picks the latest
        version (by filename sort), and deserialises the artefact.

        Args:
            name: Model identifier.

        Returns:
            The deserialised model object.

        Raises:
            FileNotFoundError: If no production model exists for *name*.
        """
        production_meta: list[tuple[Path, dict[str, Any]]] = []
        for meta_path in sorted(self.registry_dir.glob(f"{name}_v*_meta.json")):
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if meta.get("stage") == "production":
                production_meta.append((meta_path, meta))

        if not production_meta:
            raise FileNotFoundError(
                f"No production model found for '{name}' in {self.registry_dir}"
            )

        # Latest by filename sort (timestamp-based versions sort correctly)
        latest_meta_path, latest_meta = production_meta[-1]
        version = latest_meta["version"]
        model_path = self.registry_dir / f"{name}_v{version}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Production metadata exists but artefact missing: {model_path}"
            )

        logger.info(
            "Loading production model: %s v%s", name, version
        )
        return joblib.load(model_path)


# ---------------------------------------------------------------------------
# WS2.1 module-level helpers (safe, best-effort resolution for loaders)
# ---------------------------------------------------------------------------


def resolve_registry_artifact(
    name: str,
    alias: str,
    registry_dir: Path | str | None = None,
) -> Optional[Path]:
    """Best-effort artifact resolution through ``registry.json``.

    Returns ``None`` — never raises — when the registry index is absent,
    the model/alias is unset, or the aliased artifact file is missing.
    Loaders use this as the first choice and fall back to the WS0.1
    constants, so a checkout without ``registry.json`` (or without the
    gitignored in-season artifacts) keeps working.
    """
    try:
        root = Path(registry_dir) if registry_dir else default_registry_dir()
        if not (root / REGISTRY_INDEX_NAME).exists():
            return None
        return ModelRegistry(registry_dir=root).resolve_alias(name, alias)
    except (KeyError, FileNotFoundError, ValueError, json.JSONDecodeError, OSError):
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Registry resolution failed for %s/%s: %s", name, alias, exc)
        return None


def artifact_is_frozen_validated(
    path: Path | str,
    registry_dir: Path | str | None = None,
) -> bool:
    """True if *path* is any model's ``frozen_validated`` artifact.

    Training code calls this as a registry-backed second layer on top of
    the WS0.1 frozen-path constants: even if a module constant is
    monkeypatched or a script passes an explicit path, a write aimed at a
    frozen_validated artifact is refused.  Fail-open (returns False) when
    the registry is absent/unreadable — the constant-based guards remain
    the primary protection there.
    """
    try:
        root = Path(registry_dir) if registry_dir else default_registry_dir()
        if not (root / REGISTRY_INDEX_NAME).exists():
            return False
        target = Path(path).resolve()
        reg = ModelRegistry(registry_dir=root)
        for model_name in reg._read_index():
            version = reg.get_alias(model_name, "frozen_validated")
            if version is None:
                continue
            try:
                frozen = reg.artifact_path_for(model_name, version).resolve()
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue
            if target == frozen:
                return True
        return False
    except Exception as exc:
        logger.warning("Frozen-artifact registry check failed: %s", exc)
        return False


def verify_registry(
    registry_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Run :meth:`ModelRegistry.verify` against a registry root.

    Shared by ``scripts/verify_artifacts.py`` and the pytest suite so both
    run the identical check.
    """
    root = Path(registry_dir) if registry_dir else default_registry_dir()
    if not (root / REGISTRY_INDEX_NAME).exists():
        return [
            {
                "model": "<registry>",
                "version": "-",
                "check": "index_exists",
                "status": "fail",
                "detail": f"no {REGISTRY_INDEX_NAME} at {root}",
            }
        ]
    return ModelRegistry(registry_dir=root).verify()
