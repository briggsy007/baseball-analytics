"""
Base class for all analytics models in the baseball platform.

Every proprietary model (Stuff+, pitch sequencing, fatigue, etc.) inherits
from ``BaseAnalyticsModel`` to guarantee a consistent train/predict/evaluate
lifecycle, input/output validation, and serialisation with metadata tracking.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class BaseAnalyticsModel(ABC):
    """Abstract base class for all baseball analytics models.

    Subclasses must implement ``train``, ``predict``, ``evaluate``,
    ``model_name``, and ``version``.  The base class provides
    input/output validation, serialisation, and metadata tracking.
    """

    def __init__(self) -> None:
        self._metadata: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_date": None,
            "data_range": None,
            "params": {},
            "metrics": {},
        }

    # ── Abstract interface ────────────────────────────────────────────────

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Unique human-readable identifier for this model type."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version string (e.g. '1.0.0')."""

    @abstractmethod
    def train(self, conn, **kwargs) -> dict:
        """Train the model from the database and return training metrics.

        Args:
            conn: Open DuckDB connection.
            **kwargs: Model-specific training parameters.

        Returns:
            Dictionary of training metrics (loss, R-squared, etc.).
        """

    @abstractmethod
    def predict(self, conn, **kwargs) -> dict | pd.DataFrame:
        """Generate predictions.

        Args:
            conn: Open DuckDB connection.
            **kwargs: Model-specific prediction parameters.

        Returns:
            Predictions as a dictionary or DataFrame.
        """

    @abstractmethod
    def evaluate(self, conn, **kwargs) -> dict:
        """Evaluate model performance and return evaluation metrics.

        Args:
            conn: Open DuckDB connection.
            **kwargs: Model-specific evaluation parameters.

        Returns:
            Dictionary of evaluation metrics.
        """

    # ── Validation ────────────────────────────────────────────────────────

    def validate_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate an input DataFrame and return a cleaned copy.

        Checks for:
        - Non-empty DataFrame
        - No fully-null columns

        Subclasses may override or extend this method for model-specific
        validation (e.g. required columns, value ranges).

        Args:
            data: Input DataFrame.

        Returns:
            The validated DataFrame (may be a cleaned copy).

        Raises:
            ValueError: On validation failure.
        """
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("Input DataFrame is empty.")

        # Drop columns that are entirely null
        all_null_cols = [c for c in data.columns if data[c].isna().all()]
        if all_null_cols:
            logger.warning(
                "Dropping %d fully-null columns: %s",
                len(all_null_cols),
                all_null_cols,
            )
            data = data.drop(columns=all_null_cols)

        return data

    def validate_output(self, result: dict) -> dict:
        """Validate model output.

        Checks that the result is a non-empty dictionary.

        Args:
            result: Output from ``predict`` or ``evaluate``.

        Returns:
            The validated result.

        Raises:
            ValueError: On validation failure.
        """
        if result is None:
            raise ValueError("Model output is None.")
        if not isinstance(result, dict):
            raise ValueError(
                f"Model output must be a dict, got {type(result).__name__}."
            )
        if len(result) == 0:
            raise ValueError("Model output dictionary is empty.")
        return result

    # ── Metadata ──────────────────────────────────────────────────────────

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a copy of the model's metadata."""
        return dict(self._metadata)

    def update_metadata(self, **kwargs: Any) -> None:
        """Merge key-value pairs into the metadata store."""
        self._metadata.update(kwargs)

    def set_training_metadata(
        self,
        metrics: dict,
        params: dict | None = None,
        data_range: tuple[str, str] | None = None,
    ) -> None:
        """Record training results in metadata.

        Args:
            metrics: Dictionary of training metrics.
            params: Hyperparameters used for training.
            data_range: Tuple of (start_date, end_date) as ISO strings.
        """
        self._metadata["training_date"] = datetime.now(timezone.utc).isoformat()
        self._metadata["metrics"] = metrics
        if params is not None:
            self._metadata["params"] = params
        if data_range is not None:
            self._metadata["data_range"] = {
                "start": data_range[0],
                "end": data_range[1],
            }

    # ── Serialisation ─────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist the model to disk using joblib.

        Saves both the model artefact (``.pkl``) and a sidecar JSON
        metadata file (``_meta.json``).

        Args:
            path: Destination path for the ``.pkl`` file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, path)
        logger.info("Model saved to %s", path)

        meta_path = path.with_name(path.stem + "_meta.json")
        meta = {
            "model_name": self.model_name,
            "version": self.version,
            **self._metadata,
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        logger.info("Metadata saved to %s", meta_path)

    @classmethod
    def load(cls, path: Path) -> "BaseAnalyticsModel":
        """Load a persisted model from disk.

        Args:
            path: Path to the ``.pkl`` file.

        Returns:
            The deserialised model instance.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model = joblib.load(path)
        logger.info("Model loaded from %s", path)
        return model

    # ── Repr ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"name={self.model_name!r} version={self.version!r}>"
        )
