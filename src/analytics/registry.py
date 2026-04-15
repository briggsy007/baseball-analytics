"""
Model registry for persisting, versioning, and loading analytics models.

Provides a centralised store for trained model artefacts alongside JSON
metadata so that any model can be reproduced, compared, or rolled back.

Each model can be assigned a *stage* — one of ``"dev"``, ``"test"``,
``"staging"``, or ``"production"`` — to track its lifecycle from initial
experimentation through to live serving.
"""

from __future__ import annotations

import json
import logging
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


def _timestamp_version() -> str:
    """Generate a timestamp-based version string like ``20260409T153012``."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


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
        self.registry_dir = Path(registry_dir) if registry_dir else DEFAULT_REGISTRY_DIR
        self.registry_dir.mkdir(parents=True, exist_ok=True)

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
