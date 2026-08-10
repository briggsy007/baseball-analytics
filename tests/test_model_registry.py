"""WS2.1 tests: versioned model manifests, alias index, artifact verification,
and the structural frozen-artifact write guards.

Three layers:

  A. ``ModelRegistry`` versioned-manifest mechanics against tmp registries
     (write-once, sha pinning, atomic alias index + history, verification).
  B. Loader resolution: registry ``production``/``frozen_validated`` aliases
     first, WS0.1 constant fallback when the registry is absent.
  C. Structural guards: live negative tests proving training entry points
     refuse to write any ``frozen_validated`` artifact — via the WS0.1
     constants AND via the registry layer — including against the real
     repo-level frozen artifacts.
  D. Repo integrity: the exact ``scripts/verify_artifacts.py`` check, run
     in-process and via the CLI, must be green for ``models/registry.json``.

All tmp registries live under pytest tmp dirs; the session-level
``models_dir_write_guard`` in conftest backstops that nothing here touches
the repo ``models/`` tree.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.analytics.registry import (
    ModelRegistry,
    REGISTRY_DIR_ENV,
    REGISTRY_INDEX_NAME,
    artifact_is_frozen_validated,
    resolve_registry_artifact,
    sha256_of_file,
    verify_registry,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REAL_MODELS_DIR = _PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(tmp_path: Path) -> tuple[ModelRegistry, Path]:
    """Tmp registry root with one artifact file inside it."""
    root = tmp_path / "registry"
    root.mkdir()
    artifact = root / "demo_model.pkl"
    artifact.write_bytes(b"frozen-model-bytes-v1")
    return ModelRegistry(registry_dir=root), artifact


# ═══════════════════════════════════════════════════════════════════════════
# A. Versioned-manifest mechanics
# ═══════════════════════════════════════════════════════════════════════════


class TestRegisterVersion:
    def test_register_creates_manifest_and_history(self, tmp_path):
        reg, artifact = _make_registry(tmp_path)
        manifest = reg.register_version(
            "demo", "v2026.01.01", artifact, train_window="2015-2022"
        )
        assert manifest["artifact"] == "demo_model.pkl"
        assert manifest["sha256"] == sha256_of_file(artifact)
        assert manifest["hash_policy"] == "pinned"
        assert manifest["train_window"] == "2015-2022"
        # Write-once dir + manifest on disk
        mpath = reg.registry_dir / "demo" / "v2026.01.01" / "manifest.json"
        assert mpath.exists()
        # registry.json exists, parses, and recorded the event
        index = json.loads(reg.index_path.read_text())
        assert index["demo"]["history"][0]["event"] == "register"
        assert index["demo"]["history"][0]["sha256"] == manifest["sha256"]
        assert index["demo"]["production"] is None

    def test_version_dir_is_write_once(self, tmp_path):
        reg, artifact = _make_registry(tmp_path)
        reg.register_version("demo", "v2026.01.01", artifact)
        with pytest.raises(FileExistsError, match="write-once"):
            reg.register_version("demo", "v2026.01.01", artifact)

    def test_sha256_mismatch_refused(self, tmp_path):
        reg, artifact = _make_registry(tmp_path)
        with pytest.raises(ValueError, match="sha256 mismatch"):
            reg.register_version(
                "demo", "v2026.01.01", artifact, sha256="0" * 64
            )
        # Nothing half-registered
        assert reg.list_versions("demo") == []

    def test_invalid_version_string_rejected(self, tmp_path):
        reg, artifact = _make_registry(tmp_path)
        for bad in ("2026.01.01", "v2026-01-01", "v1", "latest"):
            with pytest.raises(ValueError, match="Invalid version"):
                reg.register_version("demo", bad, artifact)

    def test_missing_artifact_rejected(self, tmp_path):
        reg, _ = _make_registry(tmp_path)
        with pytest.raises(FileNotFoundError):
            reg.register_version("demo", "v2026.01.01", "nope.pkl")

    def test_outside_artifact_requires_copy(self, tmp_path):
        reg, _ = _make_registry(tmp_path)
        outside = tmp_path / "outside.pkl"
        outside.write_bytes(b"outside-bytes")
        with pytest.raises(ValueError, match="copy_artifact=True"):
            reg.register_version("demo", "v2026.01.01", outside)
        # Failed pointer registration must not consume the version dir
        manifest = reg.register_version(
            "demo", "v2026.01.01", outside, copy_artifact=True
        )
        assert manifest["artifact"] == "outside.pkl"
        copied = reg.artifact_path_for("demo", "v2026.01.01")
        assert copied.parent.name == "v2026.01.01"
        assert copied.read_bytes() == b"outside-bytes"


class TestAliases:
    def test_alias_roundtrip_and_history(self, tmp_path):
        reg, artifact = _make_registry(tmp_path)
        reg.register_version("demo", "v2026.01.01", artifact)
        reg.set_alias("demo", "frozen_validated", "v2026.01.01")
        reg.set_alias("demo", "production", "v2026.01.01")
        assert reg.get_alias("demo", "frozen_validated") == "v2026.01.01"
        assert reg.resolve_alias("demo", "production") == artifact
        events = [e["event"] for e in reg._read_index()["demo"]["history"]]
        assert events == ["register", "alias", "alias"]
        # Index stays valid JSON after every atomic rewrite
        json.loads(reg.index_path.read_text())

    def test_alias_requires_registered_version(self, tmp_path):
        reg, artifact = _make_registry(tmp_path)
        with pytest.raises(FileNotFoundError):
            reg.set_alias("demo", "production", "v2026.01.01")

    def test_frozen_validated_must_be_pinned(self, tmp_path):
        reg, artifact = _make_registry(tmp_path)
        reg.register_version(
            "demo", "v2026.01.01", artifact, hash_policy="advisory"
        )
        with pytest.raises(ValueError, match="pinned"):
            reg.set_alias("demo", "frozen_validated", "v2026.01.01")
        # production may point at advisory (in-season) versions
        reg.set_alias("demo", "production", "v2026.01.01")

    def test_unknown_alias_rejected(self, tmp_path):
        reg, artifact = _make_registry(tmp_path)
        reg.register_version("demo", "v2026.01.01", artifact)
        with pytest.raises(ValueError, match="Invalid alias"):
            reg.set_alias("demo", "staging", "v2026.01.01")


class TestVerify:
    def test_green_registry_verifies_ok(self, tmp_path):
        reg, artifact = _make_registry(tmp_path)
        reg.register_version("demo", "v2026.01.01", artifact)
        reg.set_alias("demo", "frozen_validated", "v2026.01.01")
        results = reg.verify()
        assert results and all(r["status"] == "ok" for r in results)

    def test_pinned_tamper_is_a_failure(self, tmp_path):
        reg, artifact = _make_registry(tmp_path)
        reg.register_version("demo", "v2026.01.01", artifact)
        artifact.write_bytes(b"TAMPERED")
        fails = [r for r in reg.verify() if r["status"] == "fail"]
        assert len(fails) == 1
        assert fails[0]["check"] == "sha256"
        assert "MISMATCH" in fails[0]["detail"]

    def test_pinned_missing_is_a_failure(self, tmp_path):
        reg, artifact = _make_registry(tmp_path)
        reg.register_version("demo", "v2026.01.01", artifact)
        artifact.unlink()
        fails = [r for r in reg.verify() if r["status"] == "fail"]
        assert fails and fails[0]["check"] == "artifact_exists"

    def test_advisory_drift_is_ok_and_missing_is_warn(self, tmp_path):
        reg, artifact = _make_registry(tmp_path)
        reg.register_version(
            "demo", "v2026.01.01", artifact, hash_policy="advisory"
        )
        artifact.write_bytes(b"nightly-retrain-changed-me")
        results = reg.verify()
        assert all(r["status"] == "ok" for r in results)
        assert any("advisory drift" in r["detail"] for r in results)
        artifact.unlink()
        statuses = {r["status"] for r in reg.verify()}
        assert "warn" in statuses and "fail" not in statuses

    def test_verify_registry_absent_index_fails(self, tmp_path):
        results = verify_registry(registry_dir=tmp_path)
        assert results[0]["status"] == "fail"
        assert results[0]["check"] == "index_exists"


# ═══════════════════════════════════════════════════════════════════════════
# B. Loader resolution (registry alias first, WS0.1 constants as fallback)
# ═══════════════════════════════════════════════════════════════════════════


def _registry_with_dpi(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Tmp registry with frozen + in-season artifacts for defensive_pressing."""
    root = tmp_path / "reg"
    root.mkdir()
    frozen = root / "xout_frozen.pkl"
    frozen.write_bytes(b"frozen-xout")
    inseason = root / "xout_inseason.pkl"
    inseason.write_bytes(b"inseason-xout")
    reg = ModelRegistry(registry_dir=root)
    reg.register_version("defensive_pressing", "v2026.01.01", frozen)
    reg.register_version(
        "defensive_pressing", "v2026.02.02", inseason, hash_policy="advisory"
    )
    reg.set_alias("defensive_pressing", "frozen_validated", "v2026.01.01")
    reg.set_alias("defensive_pressing", "production", "v2026.02.02")
    return root, frozen, inseason


class TestLoaderResolution:
    def test_scoring_prefers_registry_production(self, tmp_path, monkeypatch):
        import src.analytics.defensive_pressing as dp

        root, frozen, inseason = _registry_with_dpi(tmp_path)
        monkeypatch.setenv(REGISTRY_DIR_ENV, str(root))
        assert dp.resolve_scoring_checkpoint() == inseason
        assert dp.resolve_validation_checkpoint() == frozen

    def test_validation_never_resolves_production(self, tmp_path, monkeypatch):
        import src.analytics.defensive_pressing as dp

        root, frozen, inseason = _registry_with_dpi(tmp_path)
        # Drop the frozen alias: validation must fall back to the FROZEN
        # constant, never to the registry's production/in-season artifact.
        index = json.loads((root / REGISTRY_INDEX_NAME).read_text())
        index["defensive_pressing"]["frozen_validated"] = None
        (root / REGISTRY_INDEX_NAME).write_text(json.dumps(index))
        monkeypatch.setenv(REGISTRY_DIR_ENV, str(root))
        assert dp.resolve_validation_checkpoint() == dp.FROZEN_XOUT_CHECKPOINT
        assert dp.resolve_validation_checkpoint() != inseason

    def test_fallback_without_registry_json(self, tmp_path, monkeypatch):
        import src.analytics.defensive_pressing as dp

        monkeypatch.setenv(REGISTRY_DIR_ENV, str(tmp_path))  # no registry.json
        frozen = tmp_path / "frozen.pkl"
        inseason = tmp_path / "inseason.pkl"
        monkeypatch.setattr(dp, "FROZEN_XOUT_CHECKPOINT", frozen)
        monkeypatch.setattr(dp, "INSEASON_XOUT_CHECKPOINT", inseason)
        # No in-season file -> frozen constant.
        assert dp.resolve_scoring_checkpoint() == frozen
        # In-season appears -> preferred (A1 behavior preserved).
        inseason.write_bytes(b"x")
        assert dp.resolve_scoring_checkpoint() == inseason

    def test_registry_missing_artifact_falls_back(self, tmp_path, monkeypatch):
        import src.analytics.stuff_model as sm

        root = tmp_path / "reg"
        root.mkdir()
        art = root / "stuff.pkl"
        art.write_bytes(b"stuff")
        reg = ModelRegistry(registry_dir=root)
        reg.register_version(
            "stuff_model", "v2026.01.01", art, hash_policy="advisory"
        )
        reg.set_alias("stuff_model", "production", "v2026.01.01")
        art.unlink()  # aliased artifact vanished (fresh clone / gitignore)
        monkeypatch.setenv(REGISTRY_DIR_ENV, str(root))
        assert resolve_registry_artifact("stuff_model", "production") is None
        assert sm.resolve_scoring_model_path() in (
            sm.INSEASON_MODEL_PATH, sm.FROZEN_MODEL_PATH
        )


# ═══════════════════════════════════════════════════════════════════════════
# C. Structural guards — live negative tests
# ═══════════════════════════════════════════════════════════════════════════
# The guards raise BEFORE any DB access or file write, so conn=None both
# proves ordering (no training ran) and keeps these tests hermetic.


class TestFrozenWriteGuards:
    def test_fit_xout_refuses_frozen_constant_path(self, tmp_path, monkeypatch):
        import src.analytics.defensive_pressing as dp

        frozen = tmp_path / "xout_frozen.pkl"
        frozen.write_bytes(b"frozen")
        monkeypatch.setattr(dp, "FROZEN_XOUT_CHECKPOINT", frozen)
        with pytest.raises(RuntimeError, match="frozen validated xOut"):
            dp.fit_xout(None, persist_path=frozen)
        assert frozen.read_bytes() == b"frozen"

    def test_fit_xout_refuses_registry_frozen_artifact(self, tmp_path, monkeypatch):
        """Registry layer refuses even when the constants point elsewhere."""
        import src.analytics.defensive_pressing as dp

        root, frozen, _ = _registry_with_dpi(tmp_path)
        monkeypatch.setenv(REGISTRY_DIR_ENV, str(root))
        assert frozen.resolve() != dp.FROZEN_XOUT_CHECKPOINT.resolve()
        assert artifact_is_frozen_validated(frozen)
        with pytest.raises(RuntimeError, match="registered as a frozen_validated"):
            dp.fit_xout(None, persist_path=frozen)
        assert frozen.read_bytes() == b"frozen-xout"

    def test_train_stuff_model_refuses_frozen_constant_path(
        self, tmp_path, monkeypatch
    ):
        import src.analytics.stuff_model as sm

        frozen = tmp_path / "stuff_frozen.pkl"
        frozen.write_bytes(b"frozen")
        monkeypatch.setattr(sm, "FROZEN_MODEL_PATH", frozen)
        with pytest.raises(RuntimeError, match="frozen Stuff"):
            sm.train_stuff_model(None, model_path=str(frozen))
        assert frozen.read_bytes() == b"frozen"

    def test_train_stuff_model_refuses_registry_frozen_artifact(
        self, tmp_path, monkeypatch
    ):
        import src.analytics.stuff_model as sm

        root = tmp_path / "reg"
        root.mkdir()
        art = root / "stuff_frozen.pkl"
        art.write_bytes(b"frozen-stuff")
        reg = ModelRegistry(registry_dir=root)
        reg.register_version("stuff_model", "v2026.01.01", art)
        reg.set_alias("stuff_model", "frozen_validated", "v2026.01.01")
        monkeypatch.setenv(REGISTRY_DIR_ENV, str(root))
        with pytest.raises(RuntimeError, match="registered as a frozen_validated"):
            sm.train_stuff_model(None, model_path=str(art))
        assert art.read_bytes() == b"frozen-stuff"

    def test_live_refusal_on_real_repo_frozen_artifacts(self, monkeypatch):
        """Acceptance proof: training entry points refuse the REAL frozen
        artifacts at their canonical repo paths (guards raise before any
        write; artifacts are untouched — models_dir_write_guard backstops).
        """
        import src.analytics.defensive_pressing as dp
        import src.analytics.stuff_model as sm

        real_xout = _REAL_MODELS_DIR / "defensive_pressing" / "xout_v1.pkl"
        real_stuff = _REAL_MODELS_DIR / "stuff_model.pkl"
        if not (real_xout.exists() and real_stuff.exists()):
            pytest.skip("repo frozen artifacts not present in this checkout")

        monkeypatch.setenv(REGISTRY_DIR_ENV, str(_REAL_MODELS_DIR))
        monkeypatch.setattr(dp, "FROZEN_XOUT_CHECKPOINT", real_xout)
        monkeypatch.setattr(sm, "FROZEN_MODEL_PATH", real_stuff)

        before_xout = sha256_of_file(real_xout)
        before_stuff = sha256_of_file(real_stuff)
        with pytest.raises(RuntimeError):
            dp.fit_xout(None, persist_path=real_xout)
        with pytest.raises(RuntimeError):
            sm.train_stuff_model(None, model_path=str(real_stuff))
        assert sha256_of_file(real_xout) == before_xout
        assert sha256_of_file(real_stuff) == before_stuff

    def test_allow_frozen_overwrite_bypasses_registry_guard(
        self, tmp_path, monkeypatch
    ):
        """The deliberate-re-freeze escape hatch still exists (and only it).

        With allow_frozen_overwrite=True the guard is skipped, so with
        conn=None the call proceeds into training and fails on the null
        connection — proving the refusal above came from the guard, not
        from the bogus conn.
        """
        import src.analytics.defensive_pressing as dp

        root, frozen, _ = _registry_with_dpi(tmp_path)
        monkeypatch.setenv(REGISTRY_DIR_ENV, str(root))
        with pytest.raises((AttributeError, TypeError)):
            dp.fit_xout(None, persist_path=frozen, allow_frozen_overwrite=True)


# ═══════════════════════════════════════════════════════════════════════════
# D. Repo integrity — the verify_artifacts check itself
# ═══════════════════════════════════════════════════════════════════════════


class TestRepoRegistryIntegrity:
    """Same check the nightly runs first and last (WS2.1 acceptance)."""

    def test_repo_registry_verifies_green(self):
        if not (_REAL_MODELS_DIR / REGISTRY_INDEX_NAME).exists():
            pytest.skip("models/registry.json not present in this checkout")
        results = verify_registry(registry_dir=_REAL_MODELS_DIR)
        fails = [r for r in results if r["status"] == "fail"]
        assert not fails, (
            "Registered artifact integrity failure — a pinned "
            f"(frozen-validated) artifact no longer matches its manifest: {fails}"
        )
        # Every model with a frozen_validated alias resolved it cleanly.
        checked = {r["check"] for r in results}
        assert "alias_frozen_validated" in checked

    def test_verify_artifacts_cli_exit_zero(self):
        if not (_REAL_MODELS_DIR / REGISTRY_INDEX_NAME).exists():
            pytest.skip("models/registry.json not present in this checkout")
        proc = subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "verify_artifacts.py"),
                "--registry-dir", str(_REAL_MODELS_DIR),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(_PROJECT_ROOT),
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["verdict"] == "ok"
        assert payload["n_fail"] == 0
