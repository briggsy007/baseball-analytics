"""
PitchGPT v3 — the §6 gate-suite statistics.

Spec: ``PITCHGPT_V2_SPEC.md`` §6 (FROZEN at ``b61e05b``).  Every estimator
here is the one the spec names, with its [FIXED] parameters as module
constants so a threshold can never drift silently:

* **G1** classwise-ECE (Nixon et al. 2019) + TACE — :func:`classwise_ece`,
  :func:`tace`
* **G2** kernel calibration error (Widmann, Lindsten & Zachariah 2019),
  unbiased SKCE with a Laplacian kernel and a bootstrap p-value —
  :func:`skce_test`
* **G3** randomized PIT (Gneiting, Balabdaoui & Raftery 2007) + the Phase-0.6
  marginal tolerances — :func:`randomized_pit`, :func:`gate_pct`, :func:`gate_abs`
* **G4** per-count-state binned calibration + the position gate —
  :func:`top1_ece`, :func:`position_marginal_gap`
* **G5** decision calibration (Zhao, Ma & Ermon 2021) — :func:`decision_deciles`

The module computes numbers only.  Thresholds live in
:data:`G1_THRESHOLDS` and friends and are applied by
``scripts/pitchgpt_v3_dev_gates.py``; nothing here decides a verdict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ── §6.1 [FIXED] ─────────────────────────────────────────────────────────────
CWECE_BINS: int = 15
TACE_BINS: int = 15
TACE_THRESHOLD: float = 1e-3
#: §6.1 — classes with fewer than this many observations are excluded from the
#: head's classwise mean and reported separately with their counts.
MIN_CLASS_OBS: int = 100

#: §6.1 threshold ladder: ``max(0.010, 0.010 * sqrt(C / 7))`` rounded to 0.005.
G1_THRESHOLDS: dict[str, dict[str, float]] = {
    "type": {"C": 17, "cwECE": 0.015, "TACE": 0.015},
    "zone": {"C": 26, "cwECE": 0.020, "TACE": 0.020},
    "velo": {"C": 5, "cwECE": 0.010, "TACE": 0.010},
    "outcome": {"C": 7, "cwECE": 0.010, "TACE": 0.010},
}

# ── §6.3 [FIXED] ─────────────────────────────────────────────────────────────
KCE_ALPHA: float = 0.01           # FAIL if p < 0.01 for any head
KCE_BOOTSTRAP: int = 1000
#: §9 entry 13 (NOT authorized by §6.3 as written — a logged deviation).
#: The unbiased SKCE is a U-statistic over pairs, so both the estimator and
#: each of the 1,000 bootstrap replicates are O(n^2); at the full dev n =
#: 149,949 a single head's kernel matrix alone is ~1.7 TB.  Every SKCE and
#: p-value this module produces is therefore a statistic on a fixed, seeded
#: draw of this many rows, and every consumer must publish ``n_used``
#: alongside the number.  This is an engineering concession, not a spec
#: provision; see §9 entry 13 for the lockbox posture.
KCE_SUBSAMPLE: int = 2000         # U-statistic is O(n^2); fixed, seeded draw
KCE_SEED: int = 42
#: §6.3 power probe — multiply each head's logits by this and re-test.
KCE_PROBE_LOGIT_SCALE: float = 1.10

#: §6.3 requires the kernel bandwidth to be "set by the median heuristic on the
#: dev cohort (**fixed before the contact and recorded**)".  Fitting it per
#: cohort would refit it on 2026 at the one graded contact, and fitting it per
#: model makes the §6.2 side-by-side compare SKCE values under different
#: kernels.  The dev-fitted values are therefore pinned to this artifact and
#: replayed thereafter.  See §9 entry 14.
KCE_PINNED_BANDWIDTH_FILENAME: str = "kce_bandwidths_pitchgpt_v3_dev2024.json"
#: The cohort the pinned bandwidths were fitted on.  A pinned artifact
#: declaring anything else is refused on load (2025/2026 are never legal).
KCE_BANDWIDTH_FIT_SEASON: int = 2024
#: The §6.3 rule the pinned value is produced by, recorded verbatim inside the
#: artifact so the pin can be re-derived without reading this module.
KCE_BANDWIDTH_RULE: str = (
    "§6.3 median heuristic: the median pairwise L1 distance on the probability "
    "simplex, evaluated on the model-under-test's (v3) 2024 dev-cohort "
    "probabilities for that head, over 20,000 seeded random pairs (seed 42) "
    "drawn from the same fixed seeded 2,000-row SKCE subsample (§9 entry 13). "
    "ONE bandwidth per head, shared by the v3 statistic, the §6.3 power probe "
    "and the §6.2 frozen-v2 reference."
)


class BandwidthNotPinnedError(RuntimeError):
    """Raised when a graded SKCE run has no pinned bandwidth to replay.

    §6.3 fixes the bandwidth *before* the contact.  A graded run that fell
    through to the median heuristic would fit the kernel on the cohort it is
    grading — on the sealed 2026 cohort that is a direct spec violation, so
    the graded path raises instead of silently refitting.
    """

# ── §6.4 [FIXED] ─────────────────────────────────────────────────────────────
PIT_BINS: int = 20
PIT_KS_MAX: float = 0.03
PIT_BIN_LO: float = 0.67 * 0.05   # 0.0335
PIT_BIN_HI: float = 1.5 * 0.05    # 0.0750
MARGINAL_ABS_TOL: float = 0.01    # 1.00pp
MARGINAL_REL_TOL: float = 0.10
WOBA_ABS_TOL: float = 0.015
PA_LEN_ABS_TOL: float = 0.5
CAL_VALID_COVERAGE_MIN: float = 0.95

# ── §6.5 [FIXED] ─────────────────────────────────────────────────────────────
COUNT_STATE_ECE_BINS: int = 10
COUNT_STATE_ECE_MAX: float = 0.02
COUNT_STATE_MIN_OBS: int = 500
POSITION_GAP_MAX: float = 0.01    # 1.0pp — the quantity 0.6.2 died on

# ── §6.6 [FIXED] ─────────────────────────────────────────────────────────────
DECISION_DECILES: int = 10
DECISION_MIN_N: int = 500
DECISION_DECILE_GAP_MAX: float = 0.01      # 1.0pp per decile
DECISION_WEIGHTED_GAP_MAX: float = 0.005   # 0.5pp n-weighted mean

# ── §6.7 ─────────────────────────────────────────────────────────────────────
ANTI_UNFAILABILITY_FACTOR: float = 0.6
ANTI_UNFAILABILITY_FLOOR: float = 0.005


def _equal_mass_bins(values: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Index groups of roughly equal mass, ordered by ``values``."""
    n = len(values)
    if n == 0:
        return []
    order = np.argsort(values, kind="stable")
    return [b for b in np.array_split(order, min(n_bins, n)) if len(b) > 0]


# ── G1 ───────────────────────────────────────────────────────────────────────


def classwise_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = CWECE_BINS,
    min_class_obs: int = MIN_CLASS_OBS,
) -> dict:
    """Classwise-ECE with equal-mass binning (§6.1).

    For each class *k*: bin every sample by its predicted ``p_k``, and
    accumulate ``(n_bin / N) * |mean(p_k) - freq(y == k)|``.  The head statistic
    is the mean over classes with at least ``min_class_obs`` observations;
    excluded classes are reported with their counts, never silently dropped.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n, C = probs.shape
    per_class: dict[int, float] = {}
    counts: dict[int, int] = {}
    for k in range(C):
        counts[k] = int((labels == k).sum())
        pk = probs[:, k]
        yk = (labels == k).astype(np.float64)
        e = 0.0
        for idx in _equal_mass_bins(pk, n_bins):
            e += (len(idx) / n) * abs(float(pk[idx].mean()) - float(yk[idx].mean()))
        per_class[k] = e
    included = [k for k in range(C) if counts[k] >= min_class_obs]
    excluded = {k: counts[k] for k in range(C) if counts[k] < min_class_obs}
    mean = (
        float(np.mean([per_class[k] for k in included])) if included else float("nan")
    )
    return {
        "classwise_ece": mean,
        "per_class": {int(k): float(v) for k, v in per_class.items()},
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "included_classes": [int(k) for k in included],
        "excluded_classes": {int(k): int(v) for k, v in excluded.items()},
        "n": int(n),
        "n_bins": int(n_bins),
        "binning": "equal-mass",
    }


def tace(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = TACE_BINS,
    threshold: float = TACE_THRESHOLD,
    min_class_obs: int = MIN_CLASS_OBS,
) -> dict:
    """Thresholded Adaptive Calibration Error (§6.1).

    Same as :func:`classwise_ece` but each class only counts the samples whose
    predicted ``p_k`` exceeds ``threshold`` (so the vast mass of
    near-zero predictions cannot dilute the statistic), with adaptive
    (equal-mass) bins over the surviving samples.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n, C = probs.shape
    per_class: dict[int, float] = {}
    counts: dict[int, int] = {}
    kept: dict[int, int] = {}
    for k in range(C):
        counts[k] = int((labels == k).sum())
        sel = probs[:, k] > threshold
        kept[k] = int(sel.sum())
        if kept[k] == 0:
            per_class[k] = 0.0
            continue
        pk = probs[sel, k]
        yk = (labels[sel] == k).astype(np.float64)
        m = len(pk)
        e = 0.0
        for idx in _equal_mass_bins(pk, n_bins):
            e += (len(idx) / m) * abs(float(pk[idx].mean()) - float(yk[idx].mean()))
        per_class[k] = e
    included = [k for k in range(C) if counts[k] >= min_class_obs]
    mean = (
        float(np.mean([per_class[k] for k in included])) if included else float("nan")
    )
    return {
        "tace": mean,
        "per_class": {int(k): float(v) for k, v in per_class.items()},
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "n_above_threshold": {int(k): int(v) for k, v in kept.items()},
        "included_classes": [int(k) for k in included],
        "excluded_classes": {
            int(k): int(counts[k]) for k in range(C) if counts[k] < min_class_obs
        },
        "threshold": float(threshold),
        "n_bins": int(n_bins),
        "binning": "adaptive-equal-mass",
    }


def top1_ece(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = COUNT_STATE_ECE_BINS,
) -> float:
    """Top-1 ECE over equal-mass bins (§6.5's per-count-state statistic)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if n == 0:
        return float("nan")
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    e = 0.0
    for idx in _equal_mass_bins(conf, n_bins):
        e += (len(idx) / n) * abs(float(conf[idx].mean()) - float(correct[idx].mean()))
    return float(e)


# ── G2: kernel calibration error ─────────────────────────────────────────────


def _laplacian_kernel_matrix(p: np.ndarray, bandwidth: float) -> np.ndarray:
    from scipy.spatial.distance import cdist

    d = cdist(p, p, metric="cityblock")
    return np.exp(-d / max(bandwidth, 1e-12))


def median_heuristic_bandwidth(
    p: np.ndarray, n_pairs: int = 20_000, seed: int = KCE_SEED
) -> float:
    """Median pairwise L1 distance on the probability simplex."""
    rng = np.random.default_rng(seed)
    n = len(p)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    d = np.abs(p[i[keep]] - p[j[keep]]).sum(axis=1)
    m = float(np.median(d))
    return m if m > 0 else 1.0


def _skce_from_matrices(
    K: np.ndarray, P: np.ndarray, p: np.ndarray, y: np.ndarray
) -> float:
    """Unbiased SKCE U-statistic for a fixed kernel matrix.

    ``h(Z_i, Z_j) = k(p_i, p_j) · <e_{y_i} - p_i, e_{y_j} - p_j>`` and the
    inner product expands to ``δ(y_i, y_j) - p_i[y_j] - p_j[y_i] + <p_i, p_j>``,
    which keeps the bootstrap cheap: only the label-dependent terms are
    recomputed per replicate.
    """
    n = len(y)
    A = p[:, y]                       # A[i, j] = p_i[y_j]
    D = (y[:, None] == y[None, :]).astype(np.float64) - A - A.T + P
    M = K * D
    total = M.sum() - np.trace(M)
    return float(total / (n * (n - 1)))


def skce_test(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    bandwidth: float | None = None,
    n_boot: int = KCE_BOOTSTRAP,
    subsample: int = KCE_SUBSAMPLE,
    seed: int = KCE_SEED,
    allow_bandwidth_fit: bool = True,
) -> dict:
    """SKCE hypothesis test (§6.3).

    Null hypothesis: the model is calibrated, i.e. ``y_i ~ Categorical(p_i)``.
    The null distribution is obtained by re-drawing labels from the model's own
    predictions ``n_boot`` times and recomputing the statistic; the p-value is
    the fraction of null replicates at least as extreme as the observed one.
    A small p-value therefore *rejects* calibration, which is exactly the
    direction the §6.3 gate fires in.

    ``bandwidth`` should be the **pinned** dev-fitted value (§6.3: "fixed
    before the contact and recorded"; §9 entry 14).  Two SKCE values are only
    comparable when they were computed under the same kernel, so any
    side-by-side (e.g. the §6.2 frozen-v2 reference) must pass the same
    number.  Set ``allow_bandwidth_fit=False`` on any graded path: it turns a
    missing bandwidth into :class:`BandwidthNotPinnedError` rather than a
    silent refit on the cohort being graded.

    ``subsample`` caps the U-statistic at a fixed, seeded draw (§9 entry 13).
    The returned ``n_used`` / ``n_available`` MUST be published with the
    statistic — it is not computed on the full cohort.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n_all = len(labels)
    if n_all > subsample:
        idx = rng.choice(n_all, size=subsample, replace=False)
        p = probs[idx]
        y = labels[idx]
    else:
        p = probs
        y = labels
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)

    if bandwidth is None:
        if not allow_bandwidth_fit:
            raise BandwidthNotPinnedError(
                "§6.3 fixes the kernel bandwidth on the DEV cohort before the "
                "contact and records it. This graded call supplied no pinned "
                "bandwidth, so the median heuristic would be refitted on the "
                "cohort being graded. Load the pinned artifact with "
                "load_pinned_bandwidths() and pass bandwidth=... instead."
            )
        bandwidth = median_heuristic_bandwidth(p, seed=seed)
        bandwidth_source = "median-heuristic-fitted-on-this-cohort"
    else:
        bandwidth = float(bandwidth)
        bandwidth_source = "pinned"
    K = _laplacian_kernel_matrix(p, bandwidth)
    P = p @ p.T

    observed = _skce_from_matrices(K, P, p, y)

    cum = np.cumsum(p, axis=1)
    n = len(y)
    ge = 0
    null_vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        u = rng.random(n)
        y_null = (u[:, None] > cum).sum(axis=1).astype(np.int64)
        y_null = np.clip(y_null, 0, p.shape[1] - 1)
        v = _skce_from_matrices(K, P, p, y_null)
        null_vals[b] = v
        if v >= observed:
            ge += 1
    p_value = (ge + 1) / (n_boot + 1)
    return {
        "skce": float(observed),
        "p_value": float(p_value),
        "bandwidth": float(bandwidth),
        "bandwidth_source": bandwidth_source,
        "n_used": int(n),
        "n_available": int(n_all),
        "subsample_cap": int(subsample),
        "subsampled": bool(n_all > subsample),
        "subsample_seed": int(seed),
        "n_bootstrap": int(n_boot),
        "null_mean": float(null_vals.mean()),
        "null_q99": float(np.quantile(null_vals, 0.99)),
        "rejects_calibration_at_1pct": bool(p_value < KCE_ALPHA),
    }


def save_pinned_bandwidths(
    path, bandwidths: dict[str, float], provenance: dict
) -> dict:
    """Write the §6.3 "fixed before the contact and recorded" artifact.

    Write-once, like every other artifact in this program: an existing file is
    never overwritten, so the pinned kernel cannot drift between the dev
    measurement and the one graded contact.
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"{path} exists. The pinned §6.3 bandwidths are write-once: "
            "re-pinning after any result exists would let the kernel be "
            "chosen against a number. Delete deliberately or use a new path."
        )
    season = int(provenance.get("fit_cohort_season", -1))
    if season != KCE_BANDWIDTH_FIT_SEASON:
        raise ValueError(
            f"§6.3 pins the bandwidth on the dev cohort "
            f"({KCE_BANDWIDTH_FIT_SEASON}); refusing to write an artifact "
            f"declaring fit_cohort_season={season}."
        )
    payload = {
        "schema": "pitchgpt_v3_kce_bandwidths/1",
        "spec_section": "§6.3 (bandwidth fixed before the contact and recorded)",
        "deviations_log_entry": 14,
        "superseding_deviations_log_entry": 16,
        "rule": KCE_BANDWIDTH_RULE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bandwidths": {str(k): float(v) for k, v in bandwidths.items()},
        **provenance,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    logger.info("pinned §6.3 bandwidths -> %s", path)
    return payload


def load_pinned_bandwidths(path) -> tuple[dict[str, float], dict]:
    """Read the pinned §6.3 bandwidths, refusing a tainted provenance."""
    import json
    from pathlib import Path

    path = Path(path)
    payload = json.loads(path.read_text())
    season = int(payload.get("fit_cohort_season", -1))
    if season != KCE_BANDWIDTH_FIT_SEASON:
        raise ValueError(
            f"{path} declares fit_cohort_season={season}; §6.3 fixes the "
            f"bandwidth on the {KCE_BANDWIDTH_FIT_SEASON} dev cohort. A "
            "bandwidth fitted on a budgeted or sealed cohort is exactly the "
            "refit-on-the-graded-cohort failure this artifact prevents."
        )
    bw = {str(k): float(v) for k, v in payload["bandwidths"].items()}
    return bw, payload


def graded_skce_test(
    head: str,
    probs: np.ndarray,
    labels: np.ndarray,
    pinned: dict[str, float],
    **kwargs,
) -> dict:
    """SKCE on a graded cohort: pinned bandwidth only, never refitted.

    This is the entry point the §5.5 lockbox grading run must use.  A head
    missing from ``pinned`` raises rather than falling back to the median
    heuristic on the cohort under grade.
    """
    if head not in pinned:
        raise BandwidthNotPinnedError(
            f"no pinned §6.3 bandwidth for head {head!r}; pinned heads are "
            f"{sorted(pinned)}. Refusing to fit the kernel on a graded cohort."
        )
    kwargs.pop("allow_bandwidth_fit", None)
    return skce_test(
        probs, labels, bandwidth=pinned[head], allow_bandwidth_fit=False, **kwargs,
    )


# ── G3: PIT and marginal tolerances ──────────────────────────────────────────


def randomized_pit(
    sample_matrix: np.ndarray,
    observed: np.ndarray,
    seed: int = KCE_SEED,
) -> np.ndarray:
    """Randomized PIT for a discrete predictive distribution (§6.4).

    ``sample_matrix`` is ``(n_pa, n_samples)`` of simulated PA-terminal wOBA;
    ``observed`` is ``(n_pa,)``.  For a discrete forecast the PIT needs the
    randomisation of Gneiting et al.: ``u = F(y-) + V · (F(y) - F(y-))``.
    NaN rows (no terminal outcome sampled / no observed value) are dropped.
    """
    rng = np.random.default_rng(seed)
    out: list[float] = []
    for row, y in zip(sample_matrix, observed):
        vals = row[~np.isnan(row)]
        if vals.size == 0 or np.isnan(y):
            continue
        lo = float((vals < y).mean())
        hi = float((vals <= y).mean())
        out.append(lo + float(rng.random()) * (hi - lo))
    return np.asarray(out, dtype=np.float64)


def ks_uniform(u: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance from U(0, 1)."""
    if u.size == 0:
        return float("nan")
    x = np.sort(u)
    n = x.size
    i = np.arange(1, n + 1)
    return float(np.max(np.maximum(i / n - x, x - (i - 1) / n)))


def pit_bin_masses(u: np.ndarray, n_bins: int = PIT_BINS) -> np.ndarray:
    if u.size == 0:
        return np.full(n_bins, np.nan)
    counts, _ = np.histogram(u, bins=n_bins, range=(0.0, 1.0))
    return counts / u.size


def gate_pct(sampled: float, empirical: float) -> dict:
    """Phase-0.6 rate tolerance: tighter of ±10% relative and ±1.00pp absolute."""
    tol = min(MARGINAL_REL_TOL * empirical, MARGINAL_ABS_TOL)
    delta = sampled - empirical
    return {
        "sampled": float(sampled),
        "empirical": float(empirical),
        "delta_abs": float(delta),
        "delta_rel": float(delta / empirical) if empirical else float("nan"),
        "tol_used": float(tol),
        "tol_rule": "tighter of ±10% relative and ±1.00pp absolute",
        "pass": bool(abs(delta) <= tol),
    }


def gate_abs(sampled: float, empirical: float, tol: float) -> dict:
    delta = sampled - empirical
    return {
        "sampled": float(sampled),
        "empirical": float(empirical),
        "delta_abs": float(delta),
        "tol_used": float(tol),
        "tol_rule": "absolute",
        "pass": bool(abs(delta) <= tol),
    }


# ── G4: position gate ────────────────────────────────────────────────────────


def position_marginal_gap(
    rollout_marginals: np.ndarray, empirical_marginals: np.ndarray
) -> dict:
    """Max ``|rollout − empirical|`` over positions × classes (§6.5).

    This is the identical quantity Phase 0.6.2 died on (16.37pp raw-T,
    2.625pp after two reweighting rounds, against a 1.0pp threshold).
    """
    delta = np.abs(np.asarray(rollout_marginals) - np.asarray(empirical_marginals))
    flat = int(np.argmax(delta))
    pos, cls = np.unravel_index(flat, delta.shape)
    return {
        "max_abs_gap": float(delta.max()),
        "max_at_position": int(pos),
        "max_at_class": int(cls),
        "per_position_max": [float(x) for x in delta.max(axis=1)],
        "delta": [[float(v) for v in row] for row in delta],
        "threshold": POSITION_GAP_MAX,
        "pass": bool(delta.max() <= POSITION_GAP_MAX),
    }


def position_kl_spearman(
    rollout_marginals: np.ndarray, empirical_marginals: np.ndarray
) -> dict:
    """Secondary (reported, not gated): rho(position, KL from empirical)."""
    from scipy.stats import spearmanr

    r = np.clip(np.asarray(rollout_marginals, dtype=np.float64), 1e-12, None)
    e = np.clip(np.asarray(empirical_marginals, dtype=np.float64), 1e-12, None)
    kl = (e * np.log(e / r)).sum(axis=1)
    positions = np.arange(len(kl))
    if len(kl) < 3:
        return {"kl_per_position": kl.tolist(), "spearman_rho": float("nan")}
    rho, p = spearmanr(positions, kl)
    return {
        "kl_per_position": [float(x) for x in kl],
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "v2_era_reference_rho": 0.822,
    }


# ── G5: decision calibration ─────────────────────────────────────────────────


def decision_deciles(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_deciles: int = DECISION_DECILES,
    min_n: int = DECISION_MIN_N,
) -> dict:
    """Decile calibration of a decision function (§6.6).

    ``predicted`` is the model's per-PA ``P(d)``; ``actual`` is the 0/1
    realisation.  Deciles with ``n < min_n`` are reported, not gated.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    keep = ~np.isnan(predicted) & ~np.isnan(actual)
    predicted, actual = predicted[keep], actual[keep]
    rows = []
    gaps, weights = [], []
    for idx in _equal_mass_bins(predicted, n_deciles):
        n = len(idx)
        mp = float(predicted[idx].mean())
        ma = float(actual[idx].mean())
        gap = abs(mp - ma)
        gated = n >= min_n
        rows.append({
            "n": int(n),
            "mean_predicted": mp,
            "empirical": ma,
            "abs_gap": gap,
            "gated": bool(gated),
            "pass": bool(gap <= DECISION_DECILE_GAP_MAX) if gated else None,
        })
        if gated:
            gaps.append(gap)
            weights.append(n)
    weighted = (
        float(np.average(gaps, weights=weights)) if gaps else float("nan")
    )
    all_pass = all(r["pass"] for r in rows if r["gated"]) if gaps else False
    return {
        "deciles": rows,
        "weighted_mean_abs_gap": weighted,
        "max_gated_gap": float(max(gaps)) if gaps else float("nan"),
        "decile_threshold": DECISION_DECILE_GAP_MAX,
        "weighted_threshold": DECISION_WEIGHTED_GAP_MAX,
        "pass": bool(all_pass and weighted <= DECISION_WEIGHTED_GAP_MAX),
        "n_used": int(len(predicted)),
        "n_gated_deciles": int(len(gaps)),
        # No decile cleared n >= min_n, so the gate could not be evaluated.
        # It is still reported as not-passed (never as a pass), but the
        # distinction is explicit rather than hidden behind a NaN.
        "insufficient_data": bool(not gaps),
    }


# ── §6.7 anti-unfailability ──────────────────────────────────────────────────


def tighten_threshold(spec_threshold: float, frozen_v2_dev_value: float) -> dict:
    """Apply the §6.7 rule.

    If the FROZEN v2 stack already satisfies a threshold on dev, tighten to
    ``0.6 × (frozen-v2 dev value)``, floored at 0.005.  Thresholds are never
    loosened: if frozen v2 fails, the spec threshold stands as written.
    """
    v2_satisfies = bool(frozen_v2_dev_value <= spec_threshold)
    if not v2_satisfies:
        return {
            "spec_threshold": float(spec_threshold),
            "frozen_v2_dev_value": float(frozen_v2_dev_value),
            "frozen_v2_satisfies": False,
            "effective_threshold": float(spec_threshold),
            "tightened": False,
        }
    tightened = max(
        ANTI_UNFAILABILITY_FACTOR * frozen_v2_dev_value, ANTI_UNFAILABILITY_FLOOR,
    )
    return {
        "spec_threshold": float(spec_threshold),
        "frozen_v2_dev_value": float(frozen_v2_dev_value),
        "frozen_v2_satisfies": True,
        "effective_threshold": float(min(tightened, spec_threshold)),
        "tightened": bool(min(tightened, spec_threshold) < spec_threshold),
        "rule": "0.6 × frozen-v2 dev value, floored at 0.005, never loosened",
    }


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: dict = field(default_factory=dict)
    void: bool = False

    def verdict(self) -> str:
        if self.void:
            return "VOID"
        return "PASS" if self.passed else "FAIL"
